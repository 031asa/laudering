"""Exact, label-free, causal transaction motifs for a target edge.

Only the explicitly named input arrays are retained; labels are never accepted
as an algorithm input. Adjacency indexes must be sorted by timestamp within
each account, as in the frozen GNN archive. Every historical edge is external
and belongs to [target_time - window, target_time), including neither the target
nor any other edge at the target timestamp. Paths are directed and strictly
time ordered. Amounts remain scaled integers throughout.

Incoming receipts use the recipient-side currency/amount. Multi-edge paths
require every historical edge to have both currencies equal to the target's
payment currency. Rings and parallel two-hop paths are missing for a target
whose two currencies differ. A ring's amount match includes its closing join;
a parallel path's match concerns its two historical edges, not the target's
amount. A parallel-path count is a structural signal, not proof of a split of
one particular transfer. No degree or path-count cap is used.
"""

from collections import OrderedDict
import numpy as np


HOUR_US = 3600_000000
REQUIRED = ("src", "dst", "timestamp", "paid_scaled", "received_scaled",
            "payment_currency_cat", "receiving_currency_cat", "in_ptr",
            "out_ptr", "in_order", "out_order")
COUNT_FEATURES = (
    "prior_in_count_72h", "prior_in_same_currency_count_72h",
    "prior_in_amount_match_count_72h", "chain3_path_count_72h",
    "chain3_amount_match_path_count_72h", "cycle2_count_72h",
    "cycle2_amount_match_count_72h", "cycle3_path_count_72h",
    "cycle3_amount_match_path_count_72h", "parallel2_path_count_72h",
    "parallel2_middle_accounts_72h", "parallel2_amount_match_path_count_72h",
    "parallel2_amount_match_middle_accounts_72h",
)
AMOUNT_FEATURES = ("prior_in_match_min_gap_minutes_72h",)
FEATURE_NAMES = ("eligible", "not_applicable", "full_window_72h",
                 "same_currency_target") + COUNT_FEATURES + AMOUNT_FEATURES


def amount_bounds(next_paid):
    """Inclusive exact bounds for previous receipt within 10% of next payment."""
    amount = int(next_paid)
    if amount < 0:
        raise ValueError("Negative scaled amount is outside the archive contract")
    return (9 * amount + 9) // 10, 11 * amount // 10


def amount_match(previous_received, next_paid):
    lo, hi = amount_bounds(next_paid)
    return lo <= int(previous_received) <= hi


def _between(values, lo, hi):
    """Compare int64 data against arbitrary Python bounds without float casts."""
    limit = np.iinfo(values.dtype)
    if lo > int(limit.max) or hi < int(limit.min):
        return np.zeros(len(values), dtype=bool)
    return ((values >= max(lo, int(limit.min))) &
            (values <= min(hi, int(limit.max))))


def _lower(values, value):
    """Binary search that converts NumPy integers before comparing huge ints."""
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(values[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


class _Fenwick:
    def __init__(self, n):
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, index):
        index += 1
        while index < len(self.tree):
            self.tree[index] += 1
            index += index & -index

    def prefix(self, end):
        total = 0
        while end:
            total += int(self.tree[end])
            end -= end & -end
        return total

    def rank_of(self, ordinal):
        """Return rank containing a one-based cumulative occurrence number."""
        index = 0
        bit = 1 << ((len(self.tree) - 1).bit_length() - 1)
        while bit:
            nxt = index + bit
            if nxt < len(self.tree) and int(self.tree[nxt]) < ordinal:
                ordinal -= int(self.tree[nxt])
                index = nxt
            bit >>= 1
        return index


class CausalStructures:
    def __init__(self, arrays, window_us=72 * HOUR_US,
                 max_cache_bytes=64 * 1024 * 1024):
        if not 0 <= max_cache_bytes <= 128 * 1024 * 1024:
            raise ValueError("Cache budget must be between 0 and 128 MiB")
        if int(window_us) <= 0:
            raise ValueError("window_us must be positive")
        # Do not iterate over the input mapping: a caller may hold other arrays,
        # but this engine only accesses this fixed, label-free whitelist.
        self.a = {key: arrays[key] for key in REQUIRED}
        self.window_us = int(window_us)
        self.max_cache_bytes = int(max_cache_bytes)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        size = len(self.a["timestamp"])
        if not size or any(len(self.a[k]) != size for k in REQUIRED[:7]):
            raise ValueError("Edge columns must have equal, nonzero lengths")
        self.history_start_us = int(np.min(self.a["timestamp"]))
        self.work = {}

    def _bound(self, order, lo, hi, cutoff):
        ts = self.a["timestamp"]
        while lo < hi:
            mid = (lo + hi) // 2
            if int(ts[int(order[mid])]) < cutoff:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _incident(self, node, direction, t):
        key = (int(node), direction, int(t))
        if key in self.cache:
            self.work["cache_hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        ptr, order = self.a[direction + "_ptr"], self.a[direction + "_order"]
        if not 0 <= node < len(ptr) - 1:
            raise ValueError("Node outside adjacency pointer range")
        begin, end = int(ptr[node]), int(ptr[node + 1])
        end = self._bound(order, begin, end, t)
        begin = self._bound(order, begin, end, t - self.window_us)
        ids = np.asarray(order[begin:end], dtype=np.int64)
        self.work["adjacency_rows_examined"] += len(ids)
        ids = ids[self.a["src"][ids] != self.a["dst"][ids]]
        # Include a conservative fixed allowance for Python key/LRU overhead.
        needed = ids.nbytes + 512
        if needed <= self.max_cache_bytes:
            while self.cache and self.cache_bytes + needed > self.max_cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.nbytes + 512
            self.cache[key] = ids
            self.cache_bytes += needed
        return ids

    def _currency(self, ids, currency):
        a = self.a
        return ids[(a["payment_currency_cat"][ids] == currency) &
                   (a["receiving_currency_cat"][ids] == currency)]

    def _groups(self, ids):
        if not len(ids):
            return
        nodes = self.a["src"][ids]
        order = np.argsort(nodes, kind="stable")
        sorted_nodes = nodes[order]
        cuts = np.r_[0, np.flatnonzero(sorted_nodes[1:] != sorted_nodes[:-1]) + 1, len(ids)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            self.work["middle_accounts"] += 1
            yield int(sorted_nodes[start]), ids[order[start:end]]

    def _pairs(self, first, second, allowed_second=None):
        """Count t1<t2, plus exact amount matches, without a Cartesian product."""
        if not len(first) or not len(second):
            return 0, 0, None, None
        a = self.a
        self.work["pair_first_rows"] += len(first)
        self.work["pair_second_rows"] += len(second)
        first_times = a["timestamp"][first]
        earlier = np.searchsorted(first_times, a["timestamp"][second], side="left")
        total = int(earlier.sum())
        if not total:
            return 0, 0, None, None
        first_valid = int(np.flatnonzero(earlier)[0])
        witness = [int(first[0]), int(second[first_valid])]
        if allowed_second is not None:
            second = second[allowed_second]
        if not len(second):
            return total, 0, witness, None
        # Many targets have no amount-compatible final receipt. Do not build a
        # Fenwick index at all in that case, and never insert first edges that
        # occur after the final eligible second edge.
        stop = int(np.searchsorted(first_times, a["timestamp"][int(second[-1])], side="left"))
        first = first[:stop]
        if not len(first):
            return total, 0, witness, None
        values, ranks = np.unique(a["received_scaled"][first], return_inverse=True)
        tree = _Fenwick(len(values))
        witnesses = np.full(len(values), -1, dtype=np.int64)
        pos = matched = 0
        match_witness = None
        for second_id in second:
            second_id = int(second_id)
            while pos < len(first) and int(a["timestamp"][int(first[pos])]) < int(a["timestamp"][second_id]):
                rank = int(ranks[pos])
                tree.add(rank)
                witnesses[rank] = int(first[pos])
                pos += 1
            if not pos:
                continue
            lower, upper = amount_bounds(a["paid_scaled"][second_id])
            lo, hi = _lower(values, lower), _lower(values, upper + 1)
            before = tree.prefix(lo)
            count = tree.prefix(hi) - before
            self.work["amount_range_queries"] += 1
            matched += count
            if count and match_witness is None:
                rank = tree.rank_of(before + 1)
                match_witness = [int(witnesses[rank]), second_id]
        return total, matched, witness, match_witness

    def extract(self, edge_id):
        a, edge_id = self.a, int(edge_id)
        if not 0 <= edge_id < len(a["timestamp"]):
            raise ValueError("Target edge ID outside archive")
        u, v, t = int(a["src"][edge_id]), int(a["dst"][edge_id]), int(a["timestamp"][edge_id])
        currency = int(a["payment_currency_cat"][edge_id])
        paid, received = int(a["paid_scaled"][edge_id]), int(a["received_scaled"][edge_id])
        amount_bounds(paid)
        amount_bounds(received)
        same_currency = currency == int(a["receiving_currency_cat"][edge_id])
        self.work = dict(adjacency_rows_examined=0, pair_first_rows=0,
                         pair_second_rows=0, middle_accounts=0,
                         amount_range_queries=0, cache_hits=0)
        f = {key: 0 for key in COUNT_FEATURES}
        f.update(eligible=u != v, not_applicable=u == v,
                 full_window_72h=t - self.window_us >= self.history_start_us,
                 same_currency_target=same_currency,
                 prior_in_match_min_gap_minutes_72h=None)
        w = dict(prior_in=None, prior_in_amount_match=None, chain3=None,
                 chain3_amount_match=None, cycle2=None, cycle2_amount_match=None,
                 cycle3=None, cycle3_amount_match=None, parallel2=None,
                 parallel2_amount_match=None)
        if u == v:
            for key in COUNT_FEATURES:
                f[key] = None
            return dict(features=f, witnesses=w, work=dict(self.work))

        incoming = self._incident(u, "in", t)
        f["prior_in_count_72h"] = len(incoming)
        same_in = incoming[a["receiving_currency_cat"][incoming] == currency]
        f["prior_in_same_currency_count_72h"] = len(same_in)
        if len(same_in):
            w["prior_in"] = [int(same_in[-1]), edge_id]
        lo, hi = amount_bounds(paid)
        matches = same_in[_between(a["received_scaled"][same_in], lo, hi)]
        f["prior_in_amount_match_count_72h"] = len(matches)
        if len(matches):
            recent = int(matches[-1])
            f["prior_in_match_min_gap_minutes_72h"] = (t - int(a["timestamp"][recent])) / 60_000000
            w["prior_in_amount_match"] = [recent, edge_id]

        same_in = self._currency(incoming, currency)
        for middle, second in self._groups(same_in):
            if middle in (u, v):
                continue
            first = self._currency(self._incident(middle, "in", t), currency)
            first = first[(a["src"][first] != u) & (a["src"][first] != v)]
            count, match, witness, match_witness = self._pairs(
                first, second, _between(a["received_scaled"][second], lo, hi))
            f["chain3_path_count_72h"] += count
            f["chain3_amount_match_path_count_72h"] += match
            if witness and w["chain3"] is None:
                w["chain3"] = witness + [edge_id]
            if match_witness and w["chain3_amount_match"] is None:
                w["chain3_amount_match"] = match_witness + [edge_id]

        if not same_currency:
            for key in COUNT_FEATURES:
                if key.startswith(("cycle", "parallel")):
                    f[key] = None
            return dict(features=f, witnesses=w, work=dict(self.work))

        direct = same_in[a["src"][same_in] == v]
        f["cycle2_count_72h"] = len(direct)
        if len(direct):
            w["cycle2"] = [int(direct[0]), edge_id]
        # Closing join uses current received versus the earliest historical paid.
        direct_matches = [int(e) for e in direct if amount_match(a["received_scaled"][e], paid)
                          and amount_match(received, a["paid_scaled"][e])]
        f["cycle2_amount_match_count_72h"] = len(direct_matches)
        if direct_matches:
            w["cycle2_amount_match"] = [direct_matches[0], edge_id]

        for middle, second in self._groups(same_in):
            if middle in (u, v):
                continue
            first = self._currency(self._incident(middle, "in", t), currency)
            first = first[a["src"][first] == v]
            count, _, witness, _ = self._pairs(first, second)
            closing = np.fromiter((amount_match(received, a["paid_scaled"][e]) for e in first),
                                  dtype=bool, count=len(first))
            _, match, _, match_witness = self._pairs(
                first[closing], second, _between(a["received_scaled"][second], lo, hi))
            f["cycle3_path_count_72h"] += count
            f["cycle3_amount_match_path_count_72h"] += match
            if witness and w["cycle3"] is None:
                w["cycle3"] = witness + [edge_id]
            if match_witness and w["cycle3_amount_match"] is None:
                w["cycle3_amount_match"] = match_witness + [edge_id]

        # Start at the receiver's incoming edges, avoiding a hub payer's huge
        # outgoing list. Look for u among each middle account's incoming edges.
        receiver_in = self._currency(self._incident(v, "in", t), currency)
        for middle, second in self._groups(receiver_in):
            if middle in (u, v):
                continue
            first = self._currency(self._incident(middle, "in", t), currency)
            first = first[a["src"][first] == u]
            count, match, witness, match_witness = self._pairs(first, second)
            f["parallel2_path_count_72h"] += count
            f["parallel2_middle_accounts_72h"] += int(count > 0)
            f["parallel2_amount_match_path_count_72h"] += match
            f["parallel2_amount_match_middle_accounts_72h"] += int(match > 0)
            if witness and w["parallel2"] is None:
                w["parallel2"] = witness
            if match_witness and w["parallel2_amount_match"] is None:
                w["parallel2_amount_match"] = match_witness
        return dict(features=f, witnesses=w, work=dict(self.work))
