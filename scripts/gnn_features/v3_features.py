"""Causal, four-direction 24h distribution sidecars; frozen v1/v2 are read-only.

Integer prefix power sums evaluate central moments exactly before conversion to
float. Quantile ranks retain integer ordering, including values above 2**53.
Partitions include both historical events and opposite-direction target queries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import v2_features as v2

DAY = v2.DAY
SPECS = v2.SPECS
ROLES = ("src_in", "src_out", "dst_in", "dst_out")
AMOUNT_KINDS = ("mean", "min", "max", "std", "skew", "kurtosis")
TIME_KINDS = ("mean", "min", "max", "median", "std", "skew", "kurtosis")
AMOUNT_NAMES = [f"v3_{r}_amount_{k}_24h" for r in ROLES for k in AMOUNT_KINDS] + [
    "v3_src_in_amount_median_24h", "v3_dst_out_amount_median_24h"]
TIME_NAMES = [f"v3_{r}_age_min_{k}_24h" for r in ROLES for k in TIME_KINDS]
AMOUNT_COLUMNS = AMOUNT_NAMES + [n + "__valid" for n in AMOUNT_NAMES]
TIME_COLUMNS = TIME_NAMES + [n + "__valid" for n in TIME_NAMES]
EXTRA_COLUMNS = AMOUNT_COLUMNS + TIME_COLUMNS
EXTRA_NAMES = AMOUNT_NAMES + TIME_NAMES
SUPPORT_COLUMNS = [r + "_count_24h" for r in ROLES]
AMOUNT_DIM, EXTRA_DIM = 52, 108
assert len(AMOUNT_NAMES) == 26 and len(TIME_NAMES) == 28 and len(EXTRA_COLUMNS) == EXTRA_DIM
digest, file_sha, array_sha, read, inside, ensure_space = (
    v2.digest, v2.file_sha, v2.array_sha, v2.read, v2.inside, v2.ensure_space)


def contract():
    return dict(version="gnn-v3-distributions-1", roles=list(ROLES),
                amount_columns=AMOUNT_COLUMNS, time_columns=TIME_COLUMNS,
                columns=EXTRA_COLUMNS, amount_dim=AMOUNT_DIM, extra_dim=EXTRA_DIM,
                history="[t-24h,t); exclude every same-timestamp event; external only",
                currency="src queries payment_currency; dst queries receiving_currency; incoming history uses receiving_currency/received; outgoing uses payment_currency/paid",
                moments="population central moments; exact integer power sums before float conversion; std n>=1; skew n>=3 and variance>0; excess kurtosis n>=4 and variance>0",
                median="linear interpolation with exact integer ordering",
                time="historical age in minutes at target time; not inter-event gaps",
                transform="nonnegative log1p; skew/kurtosis signed log1p; missing zero plus valid mask",
                self_transfer="all new values NULL and support counts -1",
                coverage="reuse v2 full_24h; retain partial-window history",
                labels="never read; no Test or tail",
                support_columns=SUPPORT_COLUMNS,
                reused_counts=[f"r_{r.split('_')[0]}_ext_{r.split('_')[1]}_count_24h" for r in ROLES],
                reused_sums=[f"r_{r.split('_')[0]}_ext_{r.split('_')[1]}_sum_24h" for r in ROLES],
                reused_medians=["v2_src_amount_q50_24h", "v2_dst_amount_q50_24h"])


def encode_extra(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 54 or np.isinf(raw).any():
        raise ValueError("Invalid distribution raw shape/nonfinite value")
    parts = []
    for names, section in ((AMOUNT_NAMES, raw[:, :26]), (TIME_NAMES, raw[:, 26:])):
        valid = ~np.isnan(section)
        signed = np.asarray(["_skew_" in n or "_kurtosis_" in n for n in names])
        if np.any((section < 0) & ~signed):
            raise ValueError("Negative nonnegative distribution value")
        safe = np.where(valid, section, 0)
        values = np.sign(safe) * np.log1p(np.abs(safe))
        parts.extend([values, valid])
    encoded = np.column_stack(parts).astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("Nonfinite encoded distribution value")
    return encoded


def _exact_moments(values, starts, ends, scale):
    """Vectorized object-integer arithmetic prevents catastrophic cancellation."""
    a = np.asarray(values, dtype=np.int64).astype(object)
    n = np.asarray(ends - starts, dtype=np.int64)
    nn = np.maximum(n, 1).astype(object)
    sums = []
    power = np.ones(len(a), dtype=object)
    for degree in range(1, 5):
        power = power * a
        prefix = np.empty(len(a) + 1, dtype=object)
        prefix[0] = 0
        prefix[1:] = np.cumsum(power, dtype=object)
        sums.append(prefix[ends] - prefix[starts])
    s1, s2, s3, s4 = sums
    d2 = nn * s2 - s1 * s1
    d3 = nn * nn * s3 - 3 * nn * s1 * s2 + 2 * s1 ** 3
    d4 = nn ** 3 * s4 - 4 * nn ** 2 * s1 * s3 + 6 * nn * s1 ** 2 * s2 - 3 * s1 ** 4
    if np.any(d2 < 0) or np.any(d4 < 0):
        raise ValueError("Exact central moment is negative")
    mean = np.asarray(s1, dtype=np.float64) / np.maximum(n, 1) / scale
    std = np.sqrt(np.asarray(d2, dtype=np.float64)) / np.maximum(n, 1) / scale
    skew, kurt = np.full(len(n), np.nan), np.full(len(n), np.nan)
    s = (n >= 3) & (d2 > 0)
    k = (n >= 4) & (d2 > 0)
    skew[s] = np.asarray(d3[s], dtype=np.float64) / np.asarray(d2[s], dtype=np.float64) ** 1.5
    kurt[k] = np.asarray(d4[k], dtype=np.float64) / np.asarray(d2[k], dtype=np.float64) ** 2 - 3
    mean[n == 0], std[n == 0] = np.nan, np.nan
    return dict(count=n, sum_scaled=s1, mean=mean, std=std, skew=skew, kurtosis=kurt)


def query_statistics(history_time, history_amount, history_group, query_time, query_group):
    """Bounded sorted rolling queries, including opposite-direction targets.

    Return amount and age dictionaries for query order. No labels or future
    statistics are consulted. The median implementation uses C-level rolling
    integer ranks; moments use linear-work exact prefix sums per partition.
    """
    ht, ha, hg, qt, qg = map(np.asarray, (history_time, history_amount, history_group, query_time, query_group))
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (ht, ha, hg, qt, qg)) or ht.shape != ha.shape or ht.shape != hg.shape or qt.shape != qg.shape:
        raise ValueError("Distribution query requires aligned integer arrays")
    if np.any(ha < 0) or np.any(hg < 0) or np.any(qg < 0):
        raise ValueError("Invalid history amount/group")
    for a in (ht, ha, hg, qt, qg):
        if a.dtype.kind == "u" and len(a) and int(a.max()) > np.iinfo(np.int64).max:
            raise ValueError("Integer input exceeds int64")
    ht, ha, hg, qt, qg = [a.astype(np.int64, copy=False) for a in (ht, ha, hg, qt, qg)]
    if not len(qt):
        empty = {k: np.empty(0) for k in ("mean", "min", "max", "median", "std", "skew", "kurtosis")}
        empty.update(count=np.empty(0, dtype=np.int64), sum_scaled=np.empty(0, dtype=object))
        return empty, empty.copy()
    t, g = np.r_[ht, qt], np.r_[hg, qg]
    order = np.lexsort((np.arange(len(t)), t, g))
    t, g = t[order], g[order]
    clock = v2._clock(t, g)
    history = order < len(ht)
    query = ~history
    query_positions = np.flatnonzero(query)
    restore = np.argsort(order[query] - len(ht))
    all_starts = np.searchsorted(clock, clock - DAY, side="left")
    all_ends = np.searchsorted(clock, clock, side="left")
    prefix_n = np.r_[0, np.cumsum(history, dtype=np.int64)]
    starts = prefix_n[all_starts[query]][restore]
    ends = prefix_n[all_ends[query]][restore]
    hids = order[history]
    amounts, times = ha[hids], ht[hids]
    amount = _exact_moments(amounts, starts, ends, 1_000_000)
    # Translating timestamps is exact, then negate odd central moments for age.
    origin = int(t.min())
    age = _exact_moments(times - origin, starts, ends, 60_000_000)
    age["sum_scaled"] = age["count"].astype(object) * (qt - origin).astype(object) - age["sum_scaled"]
    age["mean"] = np.asarray(age["sum_scaled"], dtype=np.float64) / np.maximum(age["count"], 1) / 60_000_000
    age["mean"][age["count"] == 0] = np.nan
    age["skew"] *= -1
    n, useful = amount["count"], amount["count"] > 0
    unique, ranks = np.unique(amounts, return_inverse=True)
    rank_series = np.full(len(t), np.nan)
    rank_series[history] = ranks
    rolling = pd.Series(rank_series).rolling(v2._Bounds(all_starts, all_ends), min_periods=1)
    for name, operation in (("min", "min"), ("max", "max")):
        found = getattr(rolling, operation)().to_numpy()[query_positions][restore]
        values = np.full(len(qt), np.nan)
        values[useful] = unique[found[useful].astype(np.int64)].astype(np.float64) / 1_000_000
        amount[name] = values
    low = rolling.quantile(.5, interpolation="lower").to_numpy()[query_positions][restore]
    high = rolling.quantile(.5, interpolation="higher").to_numpy()[query_positions][restore]
    median = np.full(len(qt), np.nan)
    left, right = unique[low[useful].astype(np.int64)], unique[high[useful].astype(np.int64)]
    median[useful] = left.astype(np.float64) / 1_000_000 + (right - left).astype(np.float64) / 1_000_000 * ((n[useful] - 1) % 2) / 2
    amount["median"] = median
    for name in ("min", "max", "median"):
        values = np.full(len(qt), np.nan)
        if name == "min":
            values[useful] = (qt[useful] - times[ends[useful] - 1]) / 60_000_000
        elif name == "max":
            values[useful] = (qt[useful] - times[starts[useful]]) / 60_000_000
        else:
            lo = starts[useful] + (n[useful] - 1) // 2
            hi = starts[useful] + n[useful] // 2
            values[useful] = ((qt[useful] - times[lo]).astype(np.float64) + (qt[useful] - times[hi]).astype(np.float64)) / 120_000_000
        age[name] = values
    for result in (amount, age):
        if any(np.isinf(result[k]).any() for k in ("mean", "min", "max", "median", "std", "skew", "kurtosis")):
            raise ValueError("Nonfinite distribution statistic")
    return amount, age


def _role_fields(role):
    return [n for n in EXTRA_NAMES if n.startswith("v3_" + role + "_")]


def _role_values(role, amount, age):
    output = {}
    for kind in AMOUNT_KINDS:
        output[f"v3_{role}_amount_{kind}_24h"] = amount[kind]
    if role in ("src_in", "dst_out"):
        output[f"v3_{role}_amount_median_24h"] = amount["median"]
    for kind in TIME_KINDS:
        output[f"v3_{role}_age_min_{kind}_24h"] = age[kind]
    return output


def _role_spec(role):
    node, direction = role.split("_")
    return (node, direction, "payment_currency_cat" if node == "src" else "receiving_currency_cat",
            "payment_currency_cat" if direction == "out" else "receiving_currency_cat",
            "paid_scaled" if direction == "out" else "received_scaled",
            "out" if node == "src" else "in")


def compute_selected(arrays, data_start_us, target_ids, return_support=False):
    """Independent <=256-target GPU acceptance API, never a full store."""
    ids = np.asarray(target_ids, dtype=np.int64)
    if ids.ndim != 1 or not 1 <= len(ids) <= 256 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(arrays["timestamp"])):
        raise ValueError("Acceptance requires 1..256 unique valid targets")
    raw, support = np.full((len(ids), 54), np.nan), np.full((len(ids), 4), -1, dtype=np.int64)
    for row, edge in enumerate(ids):
        if arrays["src"][edge] == arrays["dst"][edge]:
            continue
        cutoff = int(arrays["timestamp"][edge])
        for j, role in enumerate(ROLES):
            node_key, direction, query_cur, hist_cur, hist_amount, _ = _role_spec(role)
            node, currency = int(arrays[node_key][edge]), int(arrays[query_cur][edge])
            if currency < 2:
                raise ValueError("Missing/UNK query currency")
            ptr = arrays[direction + "_ptr"]
            order = arrays.get(direction + "_order", arrays.get(direction + "_edge_ids"))
            if order is None:
                raise ValueError("Missing directional history index")
            hist = np.asarray(order[int(ptr[node]):int(ptr[node + 1])], dtype=np.int64)
            times = arrays["timestamp"][hist]
            if np.any(times[1:] < times[:-1]):
                raise ValueError("Unordered history index")
            lo, hi = np.searchsorted(times, [cutoff - DAY, cutoff], side="left")
            hist = hist[lo:hi]
            hist = hist[(arrays["src"][hist] != arrays["dst"][hist]) & (arrays[hist_cur][hist] == currency)]
            amount, age = query_statistics(arrays["timestamp"][hist], arrays[hist_amount][hist], np.zeros(len(hist), dtype=np.int64), np.array([cutoff]), np.array([0]))
            support[row, j] = amount["count"][0]
            for name, values in _role_values(role, amount, age).items():
                raw[row, EXTRA_NAMES.index(name)] = values[0]
    encoded = encode_extra(raw)
    return (encoded, support) if return_support else encoded


def _role_input(arrays, role, start, end, target_lo, target_hi):
    node_key, direction, query_cur, hist_cur, hist_amount, query_direction = _role_spec(role)
    hp, qp = arrays[direction + "_ptr"], arrays[query_direction + "_ptr"]
    hids = np.asarray(arrays[direction + "_edge_ids"][int(hp[start]):int(hp[end])], dtype=np.int64)
    qids = np.asarray(arrays[query_direction + "_edge_ids"][int(qp[start]):int(qp[end])], dtype=np.int64)
    qids = qids[(qids >= target_lo) & (qids < target_hi)]
    history_node = "src" if direction == "out" else "dst"
    hn, qn = arrays[history_node][hids], arrays[node_key][qids]
    for ids, nodes in ((hids, hn), (qids, qn)):
        if np.any(nodes < start) or np.any(nodes >= end) or np.any(arrays["src"][ids] == arrays["dst"][ids]):
            raise ValueError("External history/query account identity differs")
    hc, qc = arrays[hist_cur][hids], arrays[query_cur][qids]
    if np.any(hc < 2) or np.any(qc < 2):
        raise ValueError("Missing/UNK currency cannot be pooled")
    width = max(int(hc.max()) if len(hc) else 1, int(qc.max()) if len(qc) else 1) + 1
    return dict(ids=qids, history_time=arrays["timestamp"][hids], history_amount=arrays[hist_amount][hids],
                history_group=hn.astype(np.int64) * width + hc,
                query_time=arrays["timestamp"][qids], query_group=qn.astype(np.int64) * width + qc)


def _statistics(data):
    return query_statistics(**{k: v for k, v in data.items() if k != "ids"})


def _check_reference(data, amount, age, indices):
    """Independent Decimal-centered, explicit-history oracle for bounded checks."""
    from decimal import localcontext
    for i in np.asarray(indices, dtype=np.int64):
        chosen = ((data["history_group"] == data["query_group"][i]) &
                  (data["history_time"] >= int(data["query_time"][i]) - DAY) &
                  (data["history_time"] < data["query_time"][i]))
        with localcontext() as ctx:
            ctx.prec = 100
            for values, result, scale in ((data["history_amount"][chosen], amount, 1_000_000),
                                          (int(data["query_time"][i]) - data["history_time"][chosen], age, 60_000_000)):
                values = sorted(int(v) for v in values)
                n = len(values)
                if int(result["count"][i]) != n:
                    raise ValueError("Pilot count mismatch")
                if not n:
                    assert all(np.isnan(result[k][i]) for k in TIME_KINDS)
                    continue
                mean = sum(map(Decimal, values)) / n
                moments = [sum((Decimal(v) - mean) ** p for v in values) / n for p in (2, 3, 4)]
                m2, m3, m4 = moments
                expected = dict(mean=float(mean / scale), min=values[0] / scale, max=values[-1] / scale,
                                median=float((Decimal(values[(n - 1) // 2]) + values[n // 2]) / 2 / scale),
                                std=float(m2.sqrt() / scale),
                                skew=float(m3 / m2 ** Decimal("1.5")) if n >= 3 and m2 else np.nan,
                                kurtosis=float(m4 / m2 ** 2 - 3) if n >= 4 and m2 else np.nan)
                for key, expected_value in expected.items():
                    np.testing.assert_allclose(result[key][i], expected_value, rtol=2e-12, atol=2e-10, equal_nan=True)


class Paused(RuntimeError):
    pass


def _pause(project):
    path = project / "result/gnn_tuning_v3/pause_request.json"
    if path.exists() and read(path).get("enabled") is True:
        raise Paused("User requested v3 pause; verified parts/blocks preserved")


def _validate_sidecar(folder, config_hash, base_manifest_hash, ids, cutoffs):
    manifest = read(folder / "manifest.json")
    if (manifest.get("status") != "verified" or manifest.get("config_sha256") != config_hash
            or manifest.get("base_manifest_sha256") != base_manifest_hash or manifest.get("contract_sha256") != digest(contract())):
        raise ValueError("Distribution sidecar manifest/config identity mismatch")
    if manifest.get("target_ids_sha256") != array_sha(ids) or manifest.get("targets") != len(ids):
        raise ValueError("Distribution target coverage mismatch")
    path = folder / "features.parquet"
    if file_sha(path) != manifest["file_sha256"]:
        raise ValueError("Distribution sidecar checksum mismatch")
    table = pq.ParquetFile(path).read()
    if table.column_names != ["edge_id", "cutoff_us"] + EXTRA_COLUMNS + SUPPORT_COLUMNS or table.num_rows != len(ids):
        raise ValueError("Distribution sidecar schema differs")
    if not np.array_equal(table["edge_id"].to_numpy(), ids) or not np.array_equal(table["cutoff_us"].to_numpy(), cutoffs):
        raise ValueError("Distribution target/cutoff differs")
    matrix = np.column_stack([table[n].to_numpy() for n in EXTRA_COLUMNS])
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("Invalid distribution tensor")
    for names, values, mask in ((AMOUNT_NAMES, matrix[:, :26], matrix[:, 26:52]),
                                (TIME_NAMES, matrix[:, 52:80], matrix[:, 80:108])):
        signed = np.array(["_skew_" in n or "_kurtosis_" in n for n in names])
        if not np.isin(mask, [0, 1]).all() or np.any(values[mask == 0] != 0) or np.any((values < 0) & ~signed):
            raise ValueError("Invalid distribution masks/encoding")
    support = {key: table[key].to_numpy() for key in SUPPORT_COLUMNS}
    self_mask = None
    for role, key in zip(ROLES, SUPPORT_COLUMNS):
        counts = support[key]
        if counts.dtype != np.int64 or np.any(counts < -1):
            raise ValueError("Invalid support counts")
        if self_mask is None:
            self_mask = counts == -1
        elif not np.array_equal(self_mask, counts == -1):
            raise ValueError("Self applicability differs between roles")
        for names, prefix, offset, mask_offset in ((AMOUNT_NAMES, "amount", 0, 26), (TIME_NAMES, "age_min", 52, 80)):
            std_index = names.index(f"v3_{role}_{prefix}_std_24h")
            for name in (n for n in names if n.startswith("v3_" + role + "_")):
                idx = names.index(name)
                minimum = 3 if "_skew_" in name else 4 if "_kurtosis_" in name else 1
                expected_mask = counts >= minimum
                if minimum > 1:
                    expected_mask &= matrix[:, offset + std_index] > 0
                if not np.array_equal(matrix[:, mask_offset + idx], expected_mask.astype(np.float32)):
                    raise ValueError("Distribution support/variance mask differs")
    if np.any(matrix[self_mask] != 0):
        raise ValueError("Self transfer distribution is not wholly missing")
    return manifest, matrix, support


class SidecarStore(v2.SidecarStore):
    def __init__(self, root, base_cache_root, expected_split=None):
        self.root, self.base_cache_root = Path(root).resolve(), Path(base_cache_root).resolve()
        self.config = read(self.root / "config.json")
        self.config_sha256 = file_sha(self.root / "config.json")
        self.contract_sha256 = digest(contract())
        self.manifest = read(self.root / "manifest.json")
        self.manifest_sha256 = file_sha(self.root / "manifest.json")
        if self.manifest.get("status") != "completed" or self.manifest.get("config_sha256") != self.config_sha256:
            raise ValueError("Distribution sidecar incomplete/config mismatch")
        for name, key in (("pilot.json", "pilot_sha256"), ("reuse_crosscheck.json", "reuse_crosscheck_sha256"),
                          ("currency_verification.json", "currency_verification_sha256")):
            saved = read(self.root / name)
            if file_sha(self.root / name) != self.manifest.get(key) or saved.get("status") != "passed" or saved.get("config_sha256") != self.config_sha256:
                raise ValueError("Distribution acceptance identity differs")
        if self.config.get("contract_sha256") != self.contract_sha256 or self.config.get("contract") != contract() or self.config.get("code_sha256") != file_sha(Path(__file__)):
            raise ValueError("Distribution contract/code changed")
        if expected_split is not None and self.config.get("split") != expected_split:
            raise ValueError("Distribution split differs")
        if Path(self.config["cache_root"]).resolve() != self.base_cache_root or file_sha(self.base_cache_root / "manifest.json") != self.config["base_manifest_sha256"]:
            raise ValueError("Distribution frozen cache identity differs")
        self.base = read(self.base_cache_root / "manifest.json")
        self.blocks = {b["block"]: b for b in self.manifest["blocks"]}
        if (self.base.get("status") != "completed" or self.base.get("targets") != self.manifest.get("targets")
                or len(self.blocks) != len(self.manifest["blocks"])
                or list(self.blocks) != [b["block"] for b in self.base["blocks"]]
                or sum(b["targets"] for b in self.blocks.values()) != self.manifest["targets"]):
            raise ValueError("Distribution block coverage differs")
        self.identity = dict(config_sha256=self.config_sha256, contract_sha256=self.contract_sha256,
                             manifest_sha256=self.manifest_sha256)
        self._loaded = None

    def _lookup(self, block, target_ids, cutoff_us):
        ids, times = np.asarray(target_ids, dtype=np.int64), np.asarray(cutoff_us, dtype=np.int64)
        if times.ndim == 0:
            times = np.full(len(ids), int(times), dtype=np.int64)
        if ids.ndim != 1 or times.shape != ids.shape or len(np.unique(ids)) != len(ids) or block not in self.blocks:
            raise ValueError("Invalid distribution target request")
        if self._loaded != block:
            folder = inside(self.root, block)
            if file_sha(folder / "manifest.json") != self.blocks[block]["manifest_sha256"]:
                raise ValueError("Distribution block manifest changed")
            _, all_ids, all_times = v2._base_plan(self.base_cache_root, block, self.base["config_sha256"])
            base_hash = file_sha(inside(self.base_cache_root, block) / "manifest.json")
            _, matrix, support = _validate_sidecar(folder, self.config_sha256, base_hash, all_ids, all_times)
            order = np.argsort(all_ids)
            self._ids, self._times, self._matrix = all_ids[order], all_times[order], matrix[order]
            self._support = {k: v[order] for k, v in support.items()}
            self._loaded = block
        positions = np.searchsorted(self._ids, ids)
        if np.any(positions >= len(self._ids)) or not np.array_equal(self._ids[positions], ids) or not np.array_equal(self._times[positions], times):
            raise ValueError("Distribution requested target/cutoff differs")
        return positions


def verify_complete(root, cache, split, progress=None):
    store = SidecarStore(root, cache, split)
    seen = np.zeros(store.manifest["targets"], dtype=bool)
    total = 0
    for i, item in enumerate(store.manifest["blocks"]):
        _, ids, times = v2._base_plan(store.base_cache_root, item["block"], store.base["config_sha256"])
        pos = ids - store.config["offset"]
        if np.any(pos < 0) or np.any(pos >= len(seen)) or seen[pos].any():
            raise ValueError("Distribution completed coverage differs")
        store.block(item["block"], ids, times)
        seen[pos] = True
        total += len(ids)
        if progress:
            progress(stage="reuse_verified_blocks", completed_blocks=i + 1, verified_targets=total)
    if not seen.all():
        raise ValueError("Distribution completed store missing targets")
    return store.manifest


def _r_paths(features, date):
    """Resolve only the recorded frozen R paths; never guess a replacement."""
    folder = features / f"transaction_date={date}"
    daily, merge = read(folder / "daily_run.json"), read(folder / "merge_phase.json")
    if daily.get("status") != "verified":
        raise ValueError("Frozen daily R is not verified")
    seen = set()
    for part in merge["parts"]:
        path = Path(part["path"]).resolve()
        if not path.is_relative_to((folder / "features").resolve()) or path in seen:
            raise ValueError("Unsafe/duplicate frozen R part identity")
        seen.add(path)
        if pq.ParquetFile(path).metadata.num_rows != part["rows"]:
            raise ValueError("Frozen R part count differs")
        yield path


def _pilot(arrays, spec, update):
    import psutil
    began, checks, samples = time.perf_counter(), 0, []
    process = psutil.Process()
    peak = process.memory_info().rss
    lo, hi = spec["offset"], spec["offset"] + spec["targets"]
    for role in ROLES:
        _, direction, _, _, _, query_direction = _role_spec(role)
        degree = np.diff(arrays[direction + "_ptr"]) + np.diff(arrays[query_direction + "_ptr"])
        active = np.flatnonzero(degree)
        nodes = sorted(set([int(np.argmax(degree))] + active[:8].tolist()))
        for node in nodes:
            data = _role_input(arrays, role, node, node + 1, lo, hi)
            if not len(data["ids"]):
                continue
            amount, age = _statistics(data)
            peak = max(peak, process.memory_info().rss)
            indices = np.unique(np.linspace(0, len(data["ids"]) - 1, min(12, len(data["ids"])), dtype=np.int64))
            _check_reference(data, amount, age, indices)
            checks += len(indices)
            samples.append(dict(role=role, node=node, history_rows=len(data["history_time"]), targets=len(data["ids"])))
            update(stage="pilot", role=role, node=node, history_rows=len(data["history_time"]), reference_queries=checks)
    t = np.array([0, 0, DAY - 1, DAY, DAY, DAY + 1, DAY + 2], dtype=np.int64)
    a = np.array([2**60 + 1, 2**60 + 2, 2**60 + 3, 2**60 + 4, 2**60 + 5, 2**60 + 6, 2**60 + 7], dtype=np.int64)
    data = dict(history_time=t, history_amount=a, history_group=np.zeros(len(t), dtype=np.int64),
                query_time=t, query_group=np.zeros(len(t), dtype=np.int64))
    amount, age = _statistics(data)
    _check_reference(data, amount, age, np.arange(len(t)))
    seconds = time.perf_counter() - began
    if not checks:
        raise ValueError("No real target queries in highest-frequency-account pilot")
    return dict(status="passed", checks=checks + len(t), samples=samples, seconds=seconds,
                history_rows=sum(s["history_rows"] for s in samples),
                history_rows_per_second=sum(s["history_rows"] for s in samples) / seconds,
                peak_rss_sampled_bytes=peak)


def _part_table(role, data, amount, age):
    from decimal import localcontext
    values = _role_values(role, amount, age)
    with localcontext() as ctx:
        ctx.prec = 100
        sums = [Decimal(int(v)) for v in amount["sum_scaled"]]
    if any(v < 0 or v >= 10**38 for v in sums):
        raise ValueError("History sum exceeds decimal128(38,0)")
    return pa.table({"edge_id": data["ids"], "count_24h": amount["count"],
                     "sum_scaled": pa.array(sums, type=pa.decimal128(38, 0)),
                     "median_reuse": pa.array(amount["median"], mask=np.isnan(amount["median"])),
                     **{n: pa.array(values[n], mask=np.isnan(values[n])) for n in _role_fields(role)}})


def run(project, split, output=None, pilot_only=False, partition_edges=250_000, progress=None):
    from .training_state import ProcessLock, atomic_json
    from decimal import localcontext
    project = Path(project).resolve()
    if split not in SPECS or not 10_000 <= partition_edges <= 1_000_000:
        raise ValueError("Invalid distribution split/partition budget")
    spec = SPECS[split]
    archive, cache, features = [project / spec[k] for k in ("archive", "cache", "features")]
    allowed = project / "data/gnn_v3_distribution_features"
    out = Path(output or allowed / split).absolute()
    if out.resolve() != out or not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("v3 distribution output must stay in its dedicated root")
    ensure_space(project)
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status="running", pid=os.getpid(), child_pid=None, split=split,
                 started_at=datetime.now(timezone.utc).isoformat(), expected_targets=spec["targets"])
    started = time.perf_counter()

    def update(**fields):
        state.update(fields)
        state.update(updated_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", state)
        print(__import__("json").dumps(fields, ensure_ascii=True), flush=True)
        if progress:
            progress(**fields)

    with ProcessLock(out):
        try:
            _pause(project)
            update(stage="preflight")
            meta, base = read(archive / "build.json"), read(cache / "manifest.json")
            if meta["end_exclusive"] != spec["end"] or meta["rows"] != spec["offset"] + spec["targets"]:
                raise ValueError("Archive scope differs; Test/tail forbidden")
            if base.get("status") != "completed" or base["targets"] != spec["targets"] or file_sha(cache / "config.json") != base["config_sha256"]:
                raise ValueError("Frozen cache incomplete/config mismatch")
            if Path(base["edge_and_identity_tables"]).resolve() != archive.resolve() or Path(base["feature_tables"]).resolve() != features.resolve():
                raise ValueError("Frozen archive/features root differs")
            frozen = read(cache / "config.json")["hashes"]
            for i, (name, expected) in enumerate(frozen.items()):
                if file_sha(inside(project, name)) != expected:
                    raise ValueError("Frozen code/input checksum differs: " + name)
                if i % 25 == 0:
                    update(stage="frozen_hashes", files=i + 1, total_files=len(frozen))
                    _pause(project)
            old = v2.SidecarStore(project / "data/gnn_v2_target_features" / split, cache, split)
            names = ["timestamp", "src", "dst", "paid_scaled", "received_scaled", "payment_currency_cat", "receiving_currency_cat"]
            paths = {name: archive / "history_index" / (name + ".npy") for name in names}
            paths.update({d + "_" + s: archive / "graph/adj_index" / (d + "_" + s + ".npy") for d in ("in", "out") for s in ("ptr", "edge_ids")})
            snapshot = {str(path.relative_to(project)): file_sha(path) for path in paths.values()}
            # v2 already pins the exhaustive Train/Validation history identity.
            for name, actual in snapshot.items():
                if old.config["input_hashes"].get(name) != actual:
                    raise ValueError("History array differs from frozen v2: " + name)
            extra_paths = [archive / "build.json", archive / "metadata/vocabulary.json", Path(__file__),
                           Path(v2.__file__), Path(__file__).with_name("training_state.py"),
                           old.root / "config.json", old.root / "manifest.json"]
            for date in spec["dates"]:
                extra_paths.extend([features / f"transaction_date={date}" / "merge_phase.json",
                                    features / f"transaction_date={date}" / "daily_run.json"])
                extra_paths.extend(_r_paths(features, date))
            extra_paths.extend(sorted((archive / "graph/edges").rglob("*.parquet")))
            for i, path in enumerate(extra_paths):
                name = str(path.relative_to(project))
                actual = file_sha(path)
                snapshot[name] = actual
                old_expected = old.config["input_hashes"].get(name)
                if old_expected is not None and actual != old_expected:
                    raise ValueError("Distribution input differs from frozen v2: " + name)
                if i % 20 == 0:
                    update(stage="identity_source_hashes", files=i + 1, total_files=len(extra_paths))
                    _pause(project)
            config = dict(version=1, split=split, contract=contract(), contract_sha256=digest(contract()),
                          code_sha256=file_sha(Path(__file__)), cache_root=str(cache.resolve()), archive_root=str(archive.resolve()),
                          base_manifest_sha256=file_sha(cache / "manifest.json"), base_config_sha256=base["config_sha256"],
                          v2_config_sha256=old.config_sha256, v2_manifest_sha256=old.manifest_sha256,
                          input_hashes=snapshot, frozen_input_inventory_sha256=digest(frozen), partition_edges=partition_edges,
                          offset=spec["offset"], targets=spec["targets"],
                          runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__))
            if (out / "config.json").exists():
                if read(out / "config.json") != config:
                    raise ValueError("v3 feature code/input/config changed; impact review required")
            else:
                atomic_json(out / "config.json", config)
            config_hash = file_sha(out / "config.json")
            if (out / "manifest.json").exists():
                completed = verify_complete(out, cache, split, update)
                update(status="completed", stage="completed_reused", verified_targets=spec["targets"])
                return completed
            arrays = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
            for key in names:
                arrays[key] = arrays[key][:meta["rows"]]
            currency_path = out / "currency_verification.json"
            if currency_path.exists():
                saved = read(currency_path)
                if saved.get("status") != "passed" or saved.get("config_sha256") != config_hash:
                    raise ValueError("Currency verification identity differs")
            else:
                v2._validate_currencies(archive, arrays, read(archive / "metadata/vocabulary.json"), update)
                atomic_json(currency_path, dict(status="passed", config_sha256=config_hash, history_rows=meta["rows"]))
            if (out / "pilot.json").exists():
                pilot = read(out / "pilot.json")
                if pilot.get("status") != "passed" or pilot.get("config_sha256") != config_hash:
                    raise ValueError("Distribution pilot identity differs")
            else:
                pilot = dict(_pilot(arrays, spec, update), config_sha256=config_hash)
                atomic_json(out / "pilot.json", pilot)
            _pause(project)
            if pilot_only:
                update(status="pilot_completed", stage="pilot_completed", pilot=pilot)
                return pilot
            target_lo, target_hi = spec["offset"], spec["offset"] + spec["targets"]
            expected_external = sum(int(np.count_nonzero(arrays["src"][i:min(i + 65536, target_hi)] != arrays["dst"][i:min(i + 65536, target_hi)])) for i in range(target_lo, target_hi, 65536))
            parts = []
            for role in ROLES:
                _, direction, _, _, _, qdirection = _role_spec(role)
                combined_ptr = arrays[direction + "_ptr"] + arrays[qdirection + "_ptr"]
                role_count = 0
                for start, end in v2._partitions(combined_ptr, partition_edges):
                    folder = inside(out, f"_parts/{role}/{start:08d}_{end:08d}")
                    signature = dict(config_sha256=config_hash, role=role, start_node=start, end_node=end)
                    mp, pp = folder / "manifest.json", folder / "features.parquet"
                    if mp.exists():
                        part = read(mp)
                        if any(part.get(k) != v for k, v in signature.items()) or part.get("status") != "verified" or file_sha(pp) != part["file_sha256"]:
                            raise ValueError("Distribution saved partition identity differs")
                    else:
                        ensure_space(out)
                        data = _role_input(arrays, role, start, end, target_lo, target_hi)
                        amount, age = _statistics(data)
                        indices = np.unique(np.linspace(0, len(data["ids"]) - 1, min(3, len(data["ids"])), dtype=np.int64))
                        _check_reference(data, amount, age, indices)
                        table = _part_table(role, data, amount, age)
                        v2._atomic_parquet(pp, table)
                        part = dict(signature, status="verified", targets=table.num_rows, file_sha256=file_sha(pp))
                        atomic_json(mp, part)
                    parts.append((folder, part))
                    role_count += part["targets"]
                    update(stage="role_partition", role=role, start_node=start, end_node=end,
                           verified_role_targets=role_count, expected_external=expected_external)
                    _pause(project)
                    gc.collect()
                if role_count != expected_external:
                    raise ValueError("Distribution role target coverage differs")
            # Fully rebuild scratch from immutable verified partitions, never
            # delete or overwrite a verified partition/block on resume.
            n = spec["targets"]
            ensure_space(out, n * (54 * 8 + 4 * 8 + 4 * 16 + 2 * 8))
            raw = np.lib.format.open_memmap(inside(out, "raw_work.npy"), mode="w+", dtype=np.float64, shape=(n, 54))
            count = np.lib.format.open_memmap(inside(out, "count_work.npy"), mode="w+", dtype=np.int64, shape=(n, 4))
            sums = np.lib.format.open_memmap(inside(out, "sum_scaled_work.npy"), mode="w+", dtype=np.uint64, shape=(n, 4, 2))
            medians = np.lib.format.open_memmap(inside(out, "median_reuse_work.npy"), mode="w+", dtype=np.float64, shape=(n, 2))
            for j, role in enumerate(ROLES):
                seen = np.zeros(n, dtype=bool)
                fields = _role_fields(role)
                columns = np.array([EXTRA_NAMES.index(f) for f in fields])
                for folder, part in parts:
                    if part["role"] != role:
                        continue
                    if file_sha(folder / "features.parquet") != part["file_sha256"]:
                        raise ValueError("Distribution partition changed during assembly")
                    for batch in pq.ParquetFile(folder / "features.parquet").iter_batches(batch_size=65536):
                        ids = batch.column("edge_id").to_numpy().astype(np.int64)
                        pos = ids - target_lo
                        if np.any(pos < 0) or np.any(pos >= n) or len(np.unique(pos)) != len(pos) or seen[pos].any():
                            raise ValueError("Distribution partition duplicate/out-of-split target")
                        seen[pos] = True
                        raw[np.ix_(pos, columns)] = np.column_stack([batch.column(f).to_numpy(zero_copy_only=False) for f in fields])
                        count[pos, j] = batch.column("count_24h").to_numpy()
                        integers = np.array([int(v) for v in batch.column("sum_scaled").to_pylist()], dtype=object)
                        sums[pos, j, 0] = np.asarray(integers % (1 << 64), dtype=np.uint64)
                        sums[pos, j, 1] = np.asarray(integers // (1 << 64), dtype=np.uint64)
                        if role in ("src_out", "dst_in"):
                            medians[pos, 0 if role == "src_out" else 1] = batch.column("median_reuse").to_numpy(zero_copy_only=False)
                    for a in (raw, count, sums, medians):
                        a.flush()
                    update(stage="assemble", role=role, partition_end=part["end_node"])
                    _pause(project)
                for i in range(0, n, 65536):
                    stop = min(i + 65536, n)
                    ids = np.arange(target_lo + i, target_lo + stop)
                    external = arrays["src"][ids] != arrays["dst"][ids]
                    if not np.array_equal(seen[i:stop], external):
                        raise ValueError("Distribution assembled target coverage differs")
                    self_pos = np.arange(i, stop)[~external]
                    raw[np.ix_(self_pos, columns)] = np.nan
                    count[self_pos, j], sums[self_pos, j, :] = -1, 0
                    if role in ("src_out", "dst_in"):
                        medians[self_pos, 0 if role == "src_out" else 1] = np.nan
                for a in (raw, count, sums, medians):
                    a.flush()
            seen = np.zeros(n, dtype=bool)
            checked = 0
            count_names, sum_names = contract()["reused_counts"], contract()["reused_sums"]
            for date in spec["dates"]:
                for path in _r_paths(features, date):
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=["edge_id", "timestamp"] + count_names + sum_names):
                        ids = batch.column("edge_id").to_numpy().astype(np.int64)
                        pos = ids - target_lo
                        if np.any(pos < 0) or np.any(pos >= n) or len(np.unique(pos)) != len(pos) or seen[pos].any():
                            raise ValueError("v1 reuse cross-check target coverage differs")
                        seen[pos] = True
                        if not np.array_equal(batch.column("timestamp").cast(pa.int64()).to_numpy(), arrays["timestamp"][ids]):
                            raise ValueError("v1 reuse cross-check timestamp differs")
                        for j, (cn, sn) in enumerate(zip(count_names, sum_names)):
                            valid = count[pos, j] >= 0
                            c = batch.column(cn)
                            if not np.array_equal(c.is_valid().to_numpy(zero_copy_only=False), valid) or not np.array_equal(c.to_numpy(zero_copy_only=False)[valid], count[pos, j][valid]):
                                raise ValueError("Full four-direction v1 24h count cross-check failed")
                            s = batch.column(sn)
                            if not np.array_equal(s.is_valid().to_numpy(zero_copy_only=False), valid):
                                raise ValueError("Full four-direction v1 sum applicability differs")
                            with localcontext() as ctx:
                                ctx.prec = 100
                                expected = np.array([int(v * 1_000_000) for v in s.to_pylist() if v is not None], dtype=object)
                            actual = sums[pos, j, 1][valid].astype(object) * (1 << 64) + sums[pos, j, 0][valid].astype(object)
                            if not np.array_equal(actual, expected):
                                raise ValueError("Full four-direction exact v1 24h sum cross-check failed")
                        checked += len(ids)
                update(stage="crosscheck_v1_counts_sums", checked_targets=checked, date=date)
                _pause(project)
            if not seen.all():
                raise ValueError("Incomplete original count/sum cross-check")
            blocks, median_checked = [], 0
            seen[:] = False
            for item in base["blocks"]:
                rel = item["block"]
                _, ids, times = v2._base_plan(cache, rel, base["config_sha256"])
                pos = ids - target_lo
                if np.any(pos < 0) or np.any(pos >= n) or seen[pos].any() or not np.array_equal(times, arrays["timestamp"][ids]):
                    raise ValueError("Distribution block target/cutoff identity differs")
                seen[pos] = True
                old_values = old.block(rel, ids, times)
                for j, role in enumerate(("src", "dst")):
                    idx = v2.EXTRA_NAMES.index(f"v2_{role}_amount_q50_24h")
                    values = np.asarray(medians[pos, j])
                    valid = ~np.isnan(values)
                    np.testing.assert_array_equal(old_values[:, idx + 28], valid.astype(np.float32))
                    np.testing.assert_allclose(old_values[:, idx], np.log1p(np.where(valid, values, 0)).astype(np.float32), rtol=0, atol=0)
                full24 = (times - int(meta["data_start_us"]) >= DAY).astype(np.float32)
                np.testing.assert_array_equal(old_values[:, 56], full24)
                median_checked += len(ids)
                folder = inside(out, rel)
                base_hash = file_sha(inside(cache, rel) / "manifest.json")
                if not (folder / "manifest.json").exists():
                    encoded = encode_extra(raw[pos])
                    data = {"edge_id": ids, "cutoff_us": times, **{name: encoded[:, i] for i, name in enumerate(EXTRA_COLUMNS)},
                            **{name: np.asarray(count[pos, j]) for j, name in enumerate(SUPPORT_COLUMNS)}}
                    v2._atomic_parquet(folder / "features.parquet", pa.table(data))
                    atomic_json(folder / "manifest.json", dict(status="verified", block=rel, targets=len(ids),
                                config_sha256=config_hash, contract_sha256=digest(contract()), base_manifest_sha256=base_hash,
                                target_ids_sha256=array_sha(ids), file_sha256=file_sha(folder / "features.parquet")))
                _validate_sidecar(folder, config_hash, base_hash, ids, times)
                blocks.append(dict(block=rel, targets=len(ids), manifest_sha256=file_sha(folder / "manifest.json")))
                update(stage="block_verified", completed_blocks=len(blocks), verified_targets=median_checked)
                _pause(project)
            if not seen.all() or checked != n or median_checked != n:
                raise ValueError("Distribution final coverage/cross-check differs")
            atomic_json(out / "reuse_crosscheck.json", dict(status="passed", config_sha256=config_hash, targets=n,
                        four_counts_checked=checked, four_exact_sums_checked=checked, two_v2_medians_checked=median_checked,
                        coverage_checked=median_checked, self_applicability_checked=n))
            for a in (raw, count, sums, medians):
                a.flush()
            del raw, count, sums, medians
            manifest = dict(status="completed", split=split, targets=n, blocks=blocks, config_sha256=config_hash,
                            contract_sha256=digest(contract()), extra_dim=EXTRA_DIM,
                            completed_at=datetime.now(timezone.utc).isoformat(),
                            reuse_crosscheck_sha256=file_sha(out / "reuse_crosscheck.json"),
                            pilot_sha256=file_sha(out / "pilot.json"),
                            currency_verification_sha256=file_sha(out / "currency_verification.json"))
            atomic_json(out / "manifest.json", manifest)
            update(status="completed", stage="completed", verified_targets=n)
            return manifest
        except Paused as exc:
            update(status="paused", stage="paused", message=str(exc))
            raise
        except BaseException as exc:
            update(status="failed", stage="failed", error=str(exc), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=list(SPECS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--partition-edges", type=int, default=250_000)
    args = parser.parse_args()
    try:
        run(args.project, args.split, args.output, args.pilot_only, args.partition_edges)
    except Paused:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
