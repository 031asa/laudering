"""Independent, label-free target features; frozen v1 archives/caches are read-only.

Each role has 24h/72h/7d amount Q50/Q90 and current/quantile ratios.
The two existing v1 24h counts are checked, not duplicated in the tensor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DAY = 86_400_000_000
WINDOWS = {"24h": DAY, "72h": 3 * DAY, "7d": 7 * DAY}
ROLES = {"src": ("out", "src", "payment_currency_cat", "paid_scaled"),
         "dst": ("in", "dst", "receiving_currency_cat", "received_scaled")}
EXTRA_FIELDS = []
for _role in ROLES:
    for _window in WINDOWS:
        for _kind in (["count"] if _window != "24h" else []) + [
                "amount_q50", "amount_q90", "current_to_q50", "current_to_q90"]:
            EXTRA_FIELDS.append(dict(name=f"v2_{_role}_{_kind}_{_window}", role=_role,
                                     kind=_kind, window=_window, nullable=True,
                                     transform="log1p"))
EXTRA_NAMES = [f["name"] for f in EXTRA_FIELDS]
EXTRA_COLUMNS = EXTRA_NAMES + [name + "__valid" for name in EXTRA_NAMES] + [
    "full_24h", "full_72h", "full_7d"]
EXTRA_DIM = 59
assert len(EXTRA_FIELDS) == 28 and len(EXTRA_COLUMNS) == EXTRA_DIM
SPECS = {"test": dict(archive="data/gnn_final_test_archive", cache="data/gnn_final_test_cache",
    features="result/gnn_final_test/features", offset=25_007_998, targets=6_883_233,
    dates=[f"2022-09-{d:02d}" for d in range(14,17)], end="2022-09-17T00:00:00")}



def contract():
    return dict(version="gnn-v2-target-amount-1", fields=EXTRA_FIELDS, columns=EXTRA_COLUMNS,
                extra_dim=EXTRA_DIM, history="[t-window,t); every same-timestamp event excluded",
                source="external outgoing; payment currency; paid amount",
                destination="external incoming; receiving currency; received amount",
                quantile="linear interpolation; exact scaled-int order; final float64 interpolation",
                count="n=0 is valid; Q50/Q90 require n>=1; ratio requires positive denominator",
                self_transfer="all 28 values NULL; valid masks zero",
                coverage="global archive start, not account first seen; partial windows retained",
                labels="never read for calculation; no Test data",
                reused_counts=["r_src_ext_out_count_24h", "r_dst_ext_in_count_24h"])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inside(root, relative):
    root = Path(root).resolve()
    path = root / relative
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root or path.absolute() != resolved:
        raise ValueError(f"Unsafe path within root: {relative}")
    return path


def ensure_space(path, required=0):
    if shutil.disk_usage(path).free - int(required) < 20 * 1024 ** 3:
        raise RuntimeError("Disk reserve below 20 GiB; no new output may be created")


def encode_extra(raw, full_flags):
    raw = np.asarray(raw, dtype=np.float64)
    full_flags = np.asarray(full_flags)
    if raw.ndim != 2 or raw.shape[1] != 28 or full_flags.shape != (len(raw), 3):
        raise ValueError("Invalid extra feature/coverage shape")
    valid = ~np.isnan(raw)
    if np.isinf(raw).any() or np.any(raw[valid] < 0) or not np.isin(full_flags, [0, 1]).all():
        raise ValueError("Invalid/nonfinite extra values or coverage flags")
    result = np.column_stack([np.log1p(np.where(valid, raw, 0)), valid, full_flags]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite extra tensor")
    return result


class _Bounds(BaseIndexer):
    def __init__(self, starts, ends):
        super().__init__()
        self.starts, self.ends = starts, ends

    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None, step=None):
        if num_values != len(self.starts) or step is not None:
            raise ValueError("Unexpected rolling window invocation")
        return self.starts, self.ends


def _clock(timestamp, group):
    """Separate groups with gaps longer than any window; retain timestamp ties."""
    t, group = np.asarray(timestamp, dtype=np.int64), np.asarray(group, dtype=np.int64)
    if t.ndim != 1 or group.shape != t.shape:
        raise ValueError("Invalid timestamp/group shape")
    if not len(t):
        return t.copy()
    new = np.r_[True, group[1:] != group[:-1]]
    if np.any(group[1:] < group[:-1]) or np.any((~new[1:]) & (t[1:] < t[:-1])):
        raise ValueError("Groups and timestamps must be ordered")
    span = int(t.max()) - int(t.min())
    spacing = span + max(WINDOWS.values()) + 1
    ordinal = np.cumsum(new, dtype=np.int64) - 1
    if int(ordinal[-1]) * spacing + span > np.iinfo(np.int64).max:
        raise ValueError("Grouped timestamp index overflow; reduce partition size")
    return (t - int(t.min())) + ordinal * spacing


def window_statistics(timestamp, amount_scaled, group=None):
    """C-level rolling order statistics, no repeated per-target history scans.

    Sorted groups contain all external events. Each event is queried *before*
    all events with its timestamp. Integer rank compression preserves ordering
    even for scaled amounts greater than 2**53. No vocabulary is learned.
    """
    timestamp = np.asarray(timestamp, dtype=np.int64)
    amount = np.asarray(amount_scaled)
    if amount.shape != timestamp.shape or amount.dtype.kind not in "iu" or np.any(amount < 0):
        raise ValueError("Amounts must be nonnegative scaled integers")
    if amount.dtype.kind == "u" and len(amount) and int(amount.max()) > np.iinfo(np.int64).max:
        raise ValueError("Amount exceeds archive int64 range")
    amount = amount.astype(np.int64, copy=False)
    group = np.zeros(len(amount), dtype=np.int64) if group is None else np.asarray(group)
    clock = _clock(timestamp, group)
    if not len(amount):
        return {w: dict(count=np.empty(0, dtype=np.int64), q50=np.empty(0), q90=np.empty(0)) for w in WINDOWS}
    unique, ranks = np.unique(amount, return_inverse=True)
    series = pd.Series(ranks.astype(np.float64, copy=False))
    ends = np.searchsorted(clock, clock, side="left")
    result = {}
    for window, duration in WINDOWS.items():
        starts = np.searchsorted(clock, clock - duration, side="left")
        counts = ends - starts
        rolling = series.rolling(_Bounds(starts, ends), min_periods=1)
        current = dict(count=counts)
        usable = counts > 0
        for name, numerator, denominator in [("q50", 1, 2), ("q90", 9, 10)]:
            low = rolling.quantile(numerator / denominator, interpolation="lower").to_numpy()
            high = rolling.quantile(numerator / denominator, interpolation="higher").to_numpy()
            values = np.full(len(amount), np.nan, dtype=np.float64)
            left = unique[low[usable].astype(np.int64)]
            right = unique[high[usable].astype(np.int64)]
            fraction = (((counts[usable] - 1) * numerator) % denominator) / denominator
            # Subtract as integers first. Nonnegative ordered int64 amounts make
            # this safe; convert only final numeric values, not the sort keys.
            values[usable] = left.astype(np.float64) / 1_000_000 + (right - left).astype(np.float64) / 1_000_000 * fraction
            current[name] = values
        result[window] = current
    return result


def role_features(timestamp, amount_scaled, group=None):
    stats = window_statistics(timestamp, amount_scaled, group)
    amount = np.asarray(amount_scaled, dtype=np.float64) / 1_000_000
    values = []
    for window in WINDOWS:
        s = stats[window]
        if window != "24h":
            values.append(s["count"])
        values.extend([s["q50"], s["q90"]])
        for key in ("q50", "q90"):
            ratio = np.full(len(amount), np.nan)
            np.divide(amount, s[key], out=ratio, where=s[key] > 0)
            values.append(ratio)
    return np.column_stack(values), stats["24h"]["count"]


def compute_selected(arrays, data_start_us, target_ids, return_count24=False):
    """Independent small acceptance path, never a substitute full feature store.

    Supports the existing builder's all-history in/out_order indices. At most
    256 targets; exact integer sorting before linear quantile interpolation.
    """
    ids=np.asarray(target_ids,dtype=np.int64)
    if ids.ndim!=1 or not 1<=len(ids)<=256 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(arrays["timestamp"])):
        raise ValueError("Selected acceptance requires 1..256 unique valid targets")
    raw=np.full((len(ids),28),np.nan);counts=np.full((len(ids),2),-1,dtype=np.int64)
    for row,edge in enumerate(ids):
        if arrays["src"][edge]==arrays["dst"][edge]:
            continue
        t=int(arrays["timestamp"][edge])
        for role_index,(role,(direction,node_key,currency_key,amount_key)) in enumerate(ROLES.items()):
            node=int(arrays[node_key][edge]);cur=int(arrays[currency_key][edge]);paid=int(arrays[amount_key][edge])
            if cur<2 or paid<0:
                raise ValueError("Invalid acceptance currency/amount")
            ptr=arrays[direction+"_ptr"]
            order=arrays.get(direction+"_order",arrays.get(direction+"_edge_ids"))
            if order is None:
                raise ValueError("Missing history order")
            hist=order[int(ptr[node]):int(ptr[node+1])]
            history_times=arrays["timestamp"][hist]
            lo,hi=np.searchsorted(history_times,[t-7*DAY,t],side="left")
            history=np.asarray(hist[lo:hi],dtype=np.int64)
            history=history[(arrays["src"][history]!=arrays["dst"][history])&(arrays[currency_key][history]==cur)]
            result=[]
            for window,duration in WINDOWS.items():
                selected=history[arrays["timestamp"][history]>=t-duration]
                amount=np.sort(np.asarray(arrays[amount_key][selected],dtype=np.int64));n=len(amount)
                if np.any(amount<0):
                    raise ValueError("Negative history amount")
                if window=="24h":counts[row,role_index]=n
                else:result.append(n)
                quantiles=[]
                for numerator,denominator in [(1,2),(9,10)]:
                    if not n:quantiles.append(np.nan);continue
                    k,rem=divmod((n-1)*numerator,denominator)
                    left,right=int(amount[k]),int(amount[min(k+1,n-1)])
                    quantiles.append((left*denominator+(right-left)*rem)/denominator/1_000_000)
                result.extend(quantiles)
                result.extend([paid/1_000_000/q if q>0 else np.nan for q in quantiles])
            raw[row,role_index*14:(role_index+1)*14]=result
    t=arrays["timestamp"][ids]
    flags=np.column_stack([t-int(data_start_us)>=w for w in WINDOWS.values()])
    encoded=encode_extra(raw,flags)
    return (encoded,counts) if return_count24 else encoded


def _check_reference(timestamp, amount_scaled, group, values, count24, indices):
    """Independent bounded acceptance, exact integer sorting before interpolation."""
    t, a, group = map(np.asarray, (timestamp, amount_scaled, group))
    for i in np.asarray(indices, dtype=np.int64):
        expected = []
        for window, duration in WINDOWS.items():
            selected = np.sort(a[(group == group[i]) & (t >= int(t[i]) - duration) & (t < t[i])])
            n = len(selected)
            if window == "24h" and int(count24[i]) != n:
                raise ValueError("Pilot reference 24h count mismatch")
            if window != "24h":
                expected.append(n)
            quantiles = []
            for numerator, denominator in [(1, 2), (9, 10)]:
                if not n:
                    quantiles.append(np.nan)
                    continue
                k, rem = divmod((n - 1) * numerator, denominator)
                left, right = int(selected[k]), int(selected[min(k + 1, n - 1)])
                quantiles.append((left * denominator + (right - left) * rem) / denominator / 1_000_000)
            expected.extend(quantiles)
            expected.extend([float(a[i]) / 1_000_000 / q if q > 0 else np.nan for q in quantiles])
        np.testing.assert_allclose(values[i], expected, rtol=2e-14, atol=1e-12, equal_nan=True)


def _partitions(ptr, budget=250_000):
    start, n = 0, len(ptr) - 1
    while start < n:
        end = int(np.searchsorted(ptr, int(ptr[start]) + budget, side="right") - 1)
        end = max(start + 1, min(n, end))
        yield start, end
        start = end


def _role_input(arrays, role, start, end):
    direction, node_key, currency_key, amount_key = ROLES[role]
    ptr = arrays[direction + "_ptr"]
    ids = np.asarray(arrays[direction + "_edge_ids"][int(ptr[start]):int(ptr[end])], dtype=np.int64)
    nodes, currencies = arrays[node_key][ids], arrays[currency_key][ids]
    timestamps, amounts = arrays["timestamp"][ids], arrays[amount_key][ids]
    if len(ids) and (np.any(nodes < start) or np.any(nodes >= end) or np.any(arrays["src"][ids] == arrays["dst"][ids])):
        raise ValueError("External history index node identity mismatch")
    if np.any(currencies < 2):
        raise ValueError("Missing/UNK currency cannot be pooled")
    order = np.lexsort((ids, timestamps, currencies, nodes))
    # Archive currency vocabulary is validated independently against raw strings.
    width = int(currencies.max()) + 1 if len(ids) else 2
    group = nodes.astype(np.int64) * width + currencies
    return ids[order], timestamps[order], amounts[order], group[order]


def _atomic_parquet(path, table):
    from gnn_features.training_state import _replace
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = inside(path.parent, path.name + ".tmp")
    ensure_space(path.parent, table.nbytes)
    pq.write_table(table, pending, compression="zstd")
    with pending.open("r+b") as stream:
        os.fsync(stream.fileno())
    _replace(pending, path)


def _base_plan(cache, rel, config_hash):
    folder = inside(cache, rel)
    manifest = read(folder / "manifest.json")
    if manifest.get("status") != "verified" or manifest.get("config_sha256") != config_hash or manifest.get("block") != rel:
        raise ValueError("Frozen block identity/config mismatch")
    path = folder / "plans.parquet"
    if file_sha(path) != manifest["file_sha256"]["plans.parquet"]:
        raise ValueError("Frozen plans checksum mismatch")
    plans = pq.ParquetFile(path).read(columns=["targets", "cutoff_us"]).to_pylist()
    ids = np.concatenate([np.asarray(p["targets"], dtype=np.int64) for p in plans])
    times = np.concatenate([np.full(len(p["targets"]), p["cutoff_us"], dtype=np.int64) for p in plans])
    if len(ids) != manifest["targets"] or len(np.unique(ids)) != len(ids) or array_sha(ids) != manifest["target_ids_sha256"]:
        raise ValueError("Frozen target coverage/hash mismatch")
    return manifest, ids, times


def _validate_sidecar(folder, config_hash, base_manifest_hash, ids, cutoffs):
    manifest = read(folder / "manifest.json")
    if manifest.get("status") != "verified" or manifest.get("config_sha256") != config_hash or manifest.get("base_manifest_sha256") != base_manifest_hash:
        raise ValueError("Sidecar manifest/config identity mismatch")
    if manifest.get("target_ids_sha256") != array_sha(ids) or manifest.get("targets") != len(ids):
        raise ValueError("Sidecar target hash/count mismatch")
    path = folder / "features.parquet"
    if file_sha(path) != manifest["file_sha256"]:
        raise ValueError("Sidecar checksum mismatch")
    table = pq.ParquetFile(path).read()
    expected = ["edge_id", "cutoff_us"] + EXTRA_COLUMNS + ["src_count_7d", "dst_count_7d"]
    if table.column_names != expected or table.num_rows != len(ids):
        raise ValueError("Sidecar schema/row count differs")
    if not np.array_equal(table["edge_id"].to_numpy(), ids) or not np.array_equal(table["cutoff_us"].to_numpy(), cutoffs):
        raise ValueError("Sidecar target/cutoff identity differs")
    matrix = np.column_stack([table[n].to_numpy() for n in EXTRA_COLUMNS])
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all() or not np.isin(matrix[:, 28:], [0, 1]).all():
        raise ValueError("Invalid sidecar values/masks")
    if np.any(matrix[:, :28] < 0) or np.any(matrix[:, :28][matrix[:, 28:56] == 0] != 0):
        raise ValueError("Invalid sidecar missing/value encoding")
    support = {r + "_count_7d": table[r + "_count_7d"].to_numpy() for r in ROLES}
    for role, counts in support.items():
        if counts.dtype != np.int64 or np.any(counts < -1):
            raise ValueError("Invalid support counts")
        position = EXTRA_NAMES.index("v2_" + role)
        if not np.array_equal((counts >= 0).astype(np.float32), matrix[:, 28 + position]):
            raise ValueError("Support count mask differs")
        np.testing.assert_array_equal(np.log1p(np.maximum(counts, 0)).astype(np.float32), matrix[:, position])
    return manifest, matrix, support


class SidecarStore:
    """One verified block in RAM, with subset/reordered target lookups."""
    def __init__(self, root, base_cache_root, expected_split=None):
        self.root, self.base_cache_root = Path(root).resolve(), Path(base_cache_root).resolve()
        self.config = read(self.root / "config.json")
        self.config_sha256 = file_sha(self.root / "config.json")
        self.contract_sha256 = digest(contract())
        self.manifest = read(self.root / "manifest.json")
        self.manifest_sha256 = file_sha(self.root / "manifest.json")
        if self.manifest.get("status") != "completed" or self.manifest.get("config_sha256") != self.config_sha256:
            raise ValueError("Sidecar is incomplete or config checksum differs")
        for name,key in [("pilot.json","pilot_sha256"),("count_crosscheck.json","count_crosscheck_sha256"),
                         ("currency_verification.json","currency_verification_sha256")]:
            if file_sha(self.root/name)!=self.manifest.get(key):
                raise ValueError("Sidecar acceptance checksum differs")
            saved=read(self.root/name)
            if saved.get("status")!="passed" or saved.get("config_sha256")!=self.config_sha256:
                raise ValueError("Sidecar acceptance identity differs")
        if self.config.get("contract_sha256") != self.contract_sha256 or self.config.get("contract") != contract():
            raise ValueError("Sidecar feature contract differs")
        if self.config.get("code_sha256") != file_sha(Path(__file__)):
            raise ValueError("Sidecar feature implementation checksum differs")
        if expected_split is not None and self.config.get("split") != expected_split:
            raise ValueError("Sidecar split differs")
        if Path(self.config["cache_root"]).resolve() != self.base_cache_root:
            raise ValueError("Sidecar base cache root differs")
        if file_sha(self.base_cache_root / "manifest.json") != self.config["base_manifest_sha256"]:
            raise ValueError("Frozen cache manifest changed")
        self.base = read(self.base_cache_root / "manifest.json")
        if self.base.get("status") != "completed" or self.base.get("targets") != self.manifest.get("targets"):
            raise ValueError("Frozen/sidecar total targets differ")
        self.blocks = {b["block"]: b for b in self.manifest["blocks"]}
        if len(self.blocks) != len(self.manifest["blocks"]) or list(self.blocks) != [b["block"] for b in self.base["blocks"]]:
            raise ValueError("Sidecar block coverage differs")
        if sum(b["targets"] for b in self.blocks.values()) != self.manifest["targets"]:
            raise ValueError("Sidecar target coverage differs")
        self._loaded = None

    def _lookup(self, block, target_ids, cutoff_us):
        ids, times = np.asarray(target_ids, dtype=np.int64), np.asarray(cutoff_us, dtype=np.int64)
        if times.ndim == 0:
            times = np.full(len(ids), int(times), dtype=np.int64)
        if ids.ndim != 1 or times.shape != ids.shape or len(np.unique(ids)) != len(ids):
            raise ValueError("Invalid sidecar target request")
        if block not in self.blocks:
            raise ValueError("Unknown sidecar block")
        if self._loaded != block:
            folder = inside(self.root, block)
            if file_sha(folder / "manifest.json") != self.blocks[block]["manifest_sha256"]:
                raise ValueError("Sidecar block manifest checksum differs")
            base, all_ids, all_times = _base_plan(self.base_cache_root, block, self.base["config_sha256"])
            base_hash = file_sha(inside(self.base_cache_root, block) / "manifest.json")
            _, matrix, support = _validate_sidecar(folder, self.config_sha256, base_hash, all_ids, all_times)
            order = np.argsort(all_ids)
            self._ids, self._times = all_ids[order], all_times[order]
            self._matrix = matrix[order]
            self._support = {k: v[order] for k, v in support.items()}
            self._loaded = block
        positions = np.searchsorted(self._ids, ids)
        if np.any(positions >= len(self._ids)) or not np.array_equal(self._ids[positions], ids):
            raise ValueError("Missing target in sidecar block")
        if not np.array_equal(self._times[positions], times):
            raise ValueError("Sidecar request cutoff differs")
        return positions

    def block(self, block, target_ids, cutoff_us):
        positions = self._lookup(block, target_ids, cutoff_us)
        return self._matrix[positions].copy()

    def support(self, block, target_ids, cutoff_us):
        positions = self._lookup(block, target_ids, cutoff_us)
        return {k: v[positions].copy() for k, v in self._support.items()}


class Paused(RuntimeError):
    pass


def _pause(project):
    path = project / "result/gnn_final_test/pause_request.json"
    if path.exists() and read(path).get("enabled") is True:
        raise Paused("User requested v2 pause; verified parts/blocks preserved")


def _validate_currencies(archive, arrays, vocab, update):
    values = read(Path(archive)/"metadata/history_identity_vocabulary.json")["values"]["currency"]
    if len(set(values)) != len(values) or vocab.get("offset") != 2:
        raise ValueError("Currency vocabulary identity is invalid")
    total = 0
    for path in sorted((archive / "graph/edges").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=["edge_id", "payment_currency", "receiving_currency"]):
            ids = batch.column("edge_id").to_numpy().astype(np.int64)
            if not np.array_equal(ids, np.arange(total, total + len(ids))):
                raise ValueError("Raw currency edge identity differs")
            for key in ("payment_currency", "receiving_currency"):
                column = batch.column(key)
                mapped = pc.index_in(column, value_set=pa.array(values))
                if column.null_count or mapped.null_count:
                    raise ValueError("Missing/unknown raw currency; cannot pool category identities")
                if not np.array_equal(mapped.to_numpy() + 2, arrays[key + "_cat"][ids]):
                    raise ValueError("Raw/encoded currency identity mismatch")
            total += len(ids)
        update(stage="currency_identity", checked_history_rows=total)
    if total != len(arrays["timestamp"]):
        raise ValueError("Raw currency coverage differs")


def _pilot(arrays, spec, update):
    import psutil
    began, checks, samples = time.perf_counter(), 0, []
    process=psutil.Process();peak_rss=process.memory_info().rss
    for role, (direction, *_) in ROLES.items():
        ptr = arrays[direction + "_ptr"]
        degree = np.diff(ptr)
        active = np.flatnonzero(degree)
        nodes = sorted(set([int(np.argmax(degree))] + active[:8].tolist()))
        for node in nodes:
            ids, t, a, group = _role_input(arrays, role, node, node + 1)
            values, counts = role_features(t, a, group)
            peak_rss=max(peak_rss,process.memory_info().rss)
            indices = np.unique(np.linspace(0, len(ids) - 1, min(12, len(ids)), dtype=np.int64))
            _check_reference(t, a, group, values, counts, indices)
            checks += len(indices)
            samples.append(dict(role=role, node=node, history_rows=len(ids), reference_queries=len(indices)))
            update(stage="pilot", role=role, node=node, history_rows=len(ids), reference_queries=checks)
    # Coverage, exact boundary and tied-time cases are deterministic and cheap.
    t = np.array([0, 0, DAY, DAY, 3 * DAY, 7 * DAY], dtype=np.int64)
    a = np.array([0, 1, 2**53 + 1, 2**53 + 5, 5_000_000, 9_000_000], dtype=np.int64)
    group = np.zeros(len(t), dtype=np.int64)
    values, counts = role_features(t, a, group)
    _check_reference(t, a, group, values, counts, np.arange(len(t)))
    seconds=time.perf_counter()-began
    rows=sum(s["history_rows"] for s in samples)
    return dict(status="passed", checks=checks + len(t), samples=samples, seconds=seconds,
                history_rows=rows,history_rows_per_second=rows/seconds,peak_rss_sampled_bytes=peak_rss)


def verify_complete(root, cache, split, progress=None):
    """Recheck a completed store without changing any bound artifact bytes."""
    store=SidecarStore(root,cache,split)
    root=Path(root)
    offset=store.config["offset"];seen=np.zeros(store.manifest["targets"],dtype=bool)
    total=0
    for i,item in enumerate(store.manifest["blocks"]):
        _,ids,times=_base_plan(store.base_cache_root,item["block"],store.base["config_sha256"])
        positions=ids-offset
        if np.any(positions<0) or np.any(positions>=len(seen)) or seen[positions].any():
            raise ValueError("Completed sidecar split coverage differs")
        store.block(item["block"],ids,times)
        seen[positions]=True;total+=len(ids)
        if progress:progress(stage="reuse_verified_blocks",completed_blocks=i+1,verified_targets=total)
    if not seen.all():
        raise ValueError("Completed sidecar has missing target identities")
    return store.manifest


def run(project, split, output=None, pilot_only=False, partition_edges=250_000, progress=None):
    """Prepare only new sidecars. Full runs require caller authorization."""
    from gnn_features.training_state import ProcessLock, atomic_json
    project = Path(project).resolve()
    if split not in SPECS or not 10_000 <= partition_edges <= 1_000_000:
        raise ValueError("Invalid split/partition size")
    spec = SPECS[split]
    archive, cache, features = [project / spec[k] for k in ("archive", "cache", "features")]
    out = Path(output or project / "data/gnn_final_test_A_features" / split).absolute()
    allowed = project / "data/gnn_final_test_A_features"
    if out.resolve() != out or not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("v2 outputs must be a dedicated subdirectory of data/gnn_final_test_A_features")
    ensure_space(project)
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status="running", pid=os.getpid(), child_pid=None, split=split,
                 started_at=datetime.now(timezone.utc).isoformat(), expected_targets=spec["targets"])
    started = time.perf_counter()

    def update(**fields):
        state.update(fields)
        state.update(updated_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", state)
        print(json.dumps(fields, ensure_ascii=True), flush=True)
        if progress:
            progress(**fields)

    with ProcessLock(out):
        try:
            _pause(project)
            update(stage="preflight")
            meta, base = read(archive / "build.json"), read(cache / "manifest.json")
            if meta["end_exclusive"] != spec["end"] or meta["rows"] != spec["offset"] + spec["targets"]:
                raise ValueError("Archive scope differs; only declared Test targets may be processed; tail forbidden")
            if base.get("status") != "completed" or base["targets"] != spec["targets"]:
                raise ValueError("Frozen base cache incomplete/wrong split")
            if Path(base["edge_and_identity_tables"]).resolve() != archive.resolve() or Path(base["feature_tables"]).resolve() != features.resolve():
                raise ValueError("Frozen archive/features root mismatch")
            if file_sha(cache / "config.json") != base["config_sha256"]:
                raise ValueError("Frozen cache config checksum differs")
            frozen = read(cache / "config.json")["hashes"]
            for i, (name, expected) in enumerate(frozen.items()):
                path = inside(project, name)
                if file_sha(path) != expected:
                    raise ValueError(f"Frozen input/code checksum differs: {name}")
                if i % 25 == 0:
                    update(stage="frozen_hashes", files=i + 1, total_files=len(frozen))
            names = ["timestamp", "src", "dst", "paid_scaled", "received_scaled", "payment_currency_cat", "receiving_currency_cat"]
            paths = {name: archive / "history_index" / (name + ".npy") for name in names}
            paths.update({d + "_" + s: archive / "graph/adj_index" / (d + "_" + s + ".npy") for d in ("in", "out") for s in ("ptr", "edge_ids")})
            snapshot = {str(path.relative_to(project)): file_sha(path) for path in paths.values()}
            identity_sources=sorted((archive/"graph/edges").rglob("*.parquet"))
            for date in spec["dates"]:
                for part in read(features/f"transaction_date={date}"/"merge_phase.json")["parts"]:
                    path=Path(part["path"]).resolve()
                    if not path.is_relative_to(features.resolve()):
                        raise ValueError("v1 R input path escapes root")
                    identity_sources.append(path)
            for i,path in enumerate(identity_sources):
                snapshot[str(path.relative_to(project))]=file_sha(path)
                if i%20==0:
                    update(stage="identity_source_hashes",files=i+1,total_files=len(identity_sources))
                    _pause(project)
            for path in (archive / "build.json", archive / "metadata/vocabulary.json", Path(__file__), project/"scripts/gnn_features/training_state.py"):
                snapshot[str(path.relative_to(project))] = file_sha(path)
            if split=="train":
                # The original Train cache predates exhaustive array hashes.
                # Pin these to the already completed v1 winner's training input
                # inventory, never establish a new baseline from current bytes.
                reference_dir=project/"result/gnn_tuning_v1/trials/lr_only"
                reference=read(reference_dir/"config.json")
                result=read(reference_dir/"training_result.json")
                if result.get("status")!="completed" or result.get("targets")!=spec["targets"] or result.get("config_sha256")!=file_sha(reference_dir/"config.json"):
                    raise ValueError("Frozen v1 training reference identity differs")
                expected={str(Path(k)):v for k,v in reference["input_hashes"].items()}
                for path in list(paths.values())+[archive/"build.json",archive/"metadata/vocabulary.json"]:
                    name=str(path.relative_to(project))
                    if expected.get(name)!=snapshot[name]:
                        raise ValueError(f"Frozen v1 Train history changed: {name}")
                for path in (reference_dir/"config.json",reference_dir/"training_result.json"):
                    snapshot[str(path.relative_to(project))]=file_sha(path)
            config = dict(version=1, split=split, contract=contract(), contract_sha256=digest(contract()),
                          code_sha256=file_sha(Path(__file__)),
                          cache_root=str(cache.resolve()), archive_root=str(archive.resolve()),
                          base_manifest_sha256=file_sha(cache / "manifest.json"), base_config_sha256=base["config_sha256"],
                          input_hashes=snapshot, frozen_input_inventory_sha256=digest(frozen),
                          partition_edges=partition_edges, offset=spec["offset"], targets=spec["targets"],
                          runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__))
            if (out / "config.json").exists():
                if read(out / "config.json") != config:
                    raise ValueError("v2 feature code/input/config changed; explicit impact review required")
            else:
                atomic_json(out / "config.json", config)
            config_hash = file_sha(out / "config.json")
            if (out/"manifest.json").exists():
                completed=verify_complete(out,cache,split,update)
                update(status="completed",stage="completed_reused",verified_targets=spec["targets"])
                return completed
            arrays = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
            for k in names:
                arrays[k] = arrays[k][:meta["rows"]]
            currency_path=out/"currency_verification.json"
            if currency_path.exists():
                saved=read(currency_path)
                if saved.get("config_sha256")!=config_hash or saved.get("status")!="passed":
                    raise ValueError("Currency verification identity differs")
            else:
                _validate_currencies(archive, arrays, read(archive / "metadata/vocabulary.json"), update)
                atomic_json(currency_path,dict(status="passed",config_sha256=config_hash,history_rows=meta["rows"]))
            if (out/"pilot.json").exists():
                pilot=read(out/"pilot.json")
                if pilot.get("status")!="passed" or pilot.get("config_sha256")!=config_hash:
                    raise ValueError("Existing pilot identity differs")
            else:
                pilot = dict(_pilot(arrays, spec, update),config_sha256=config_hash)
                atomic_json(out / "pilot.json", pilot)
            _pause(project)
            if pilot_only:
                update(status="pilot_completed", stage="pilot_completed", pilot=pilot)
                return pilot
            target_lo, target_hi = spec["offset"], spec["offset"] + spec["targets"]
            expected_external = sum(int(np.count_nonzero(arrays["src"][i:min(i+65536,target_hi)] != arrays["dst"][i:min(i+65536,target_hi)])) for i in range(target_lo,target_hi,65536))
            parts = []
            for role, (direction, *_) in ROLES.items():
                role_count = 0
                for start, end in _partitions(arrays[direction + "_ptr"], partition_edges):
                    folder = inside(out, f"_parts/{role}/{start:08d}_{end:08d}")
                    signature = dict(config_sha256=config_hash, role=role, start_node=start, end_node=end)
                    mp, pp = folder / "manifest.json", folder / "features.parquet"
                    if mp.exists():
                        part = read(mp)
                        if any(part.get(k) != v for k, v in signature.items()) or part.get("status") != "verified" or file_sha(pp) != part["file_sha256"]:
                            raise ValueError("Saved role partition identity/checksum differs")
                    else:
                        ensure_space(out)
                        ids, t, a, group = _role_input(arrays, role, start, end)
                        selected = (ids >= target_lo) & (ids < target_hi)
                        if selected.any():
                            values, count = role_features(t, a, group)
                            queries = np.flatnonzero(selected)
                            check = queries[np.unique(np.linspace(0, len(queries)-1, min(3,len(queries)),dtype=np.int64))]
                            _check_reference(t,a,group,values,count,check)
                            data = {"edge_id":ids[selected], "count_24h":count[selected]}
                            own_fields = [f for f in EXTRA_FIELDS if f["role"] == role]
                            data.update({f["name"]:pa.array(values[selected, j], mask=np.isnan(values[selected,j])) for j,f in enumerate(own_fields)})
                            table = pa.table(data)
                        else:
                            table = pa.table({"edge_id":pa.array([],type=pa.int64()),"count_24h":pa.array([],type=pa.int64()), **{f["name"]:pa.array([],type=pa.float64()) for f in EXTRA_FIELDS if f["role"]==role}})
                        _atomic_parquet(pp,table)
                        part = dict(signature,status="verified",targets=table.num_rows,file_sha256=file_sha(pp))
                        atomic_json(mp,part)
                    parts.append((folder,part))
                    role_count += part["targets"]
                    update(stage="role_partition",role=role,start_node=start,end_node=end,verified_role_targets=role_count,expected_external=expected_external)
                    _pause(project)
                    gc.collect()
                if role_count != expected_external:
                    raise ValueError("Role partition total coverage differs")
            ensure_space(out, spec["targets"] * 30 * 8)
            # Scratch is fully reconstructed from hash-verified, immutable parts;
            # no published partition or block is ever deleted on resume.
            raw = np.lib.format.open_memmap(inside(out,"raw_work.npy"),mode="w+",dtype=np.float64,shape=(spec["targets"],28))
            count24 = np.lib.format.open_memmap(inside(out,"count24_work.npy"),mode="w+",dtype=np.int64,shape=(spec["targets"],2))
            for role_index,role in enumerate(ROLES):
                seen = np.zeros(spec["targets"],dtype=bool)
                own_fields = [f["name"] for f in EXTRA_FIELDS if f["role"]==role]
                columns = np.array([EXTRA_NAMES.index(f) for f in own_fields])
                for folder,part in parts:
                    if part["role"] != role:
                        continue
                    if file_sha(folder/"features.parquet") != part["file_sha256"]:
                        raise ValueError("Partition changed during assembly")
                    for batch in pq.ParquetFile(folder/"features.parquet").iter_batches(batch_size=65536):
                        ids = batch.column("edge_id").to_numpy().astype(np.int64)
                        positions = ids-target_lo
                        if np.any(positions<0) or np.any(positions>=len(seen)) or len(np.unique(positions))!=len(positions) or seen[positions].any():
                            raise ValueError("Partition duplicate/out-of-split targets")
                        seen[positions]=True
                        vals=np.column_stack([batch.column(f).to_numpy(zero_copy_only=False) for f in own_fields])
                        raw[np.ix_(positions,columns)]=vals
                        count24[positions,role_index]=batch.column("count_24h").to_numpy()
                    raw.flush();count24.flush()
                    update(stage="assemble",role=role,partition_end=part["end_node"])
                    _pause(project)
                for i in range(0,spec["targets"],65536):
                    stop=min(i+65536,spec["targets"]);ids=np.arange(target_lo+i,target_lo+stop)
                    external=arrays["src"][ids]!=arrays["dst"][ids]
                    if not np.array_equal(seen[i:stop],external):
                        raise ValueError("Assembled role exact target coverage differs")
                    self_positions=np.arange(i,stop)[~external]
                    raw[np.ix_(self_positions,columns)]=np.nan
                    count24[self_positions,role_index]=-1
                raw.flush();count24.flush()
            # Every existing v1 24h own-direction count must match, including
            # NULL applicability. This does not load labels or full-day R.
            seen=np.zeros(spec["targets"],dtype=bool)
            checked=0
            for date in spec["dates"]:
                folder=features/f"transaction_date={date}"
                if read(folder/"daily_run.json")["status"]!="verified":
                    raise ValueError("v1 R is not verified")
                for part in read(folder/"merge_phase.json")["parts"]:
                    path=Path(part["path"]).resolve()
                    if not path.is_relative_to(features.resolve()):
                        raise ValueError("v1 R path escapes root")
                    cols=["edge_id","timestamp","r_src_ext_out_count_24h","r_dst_ext_in_count_24h"]
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=cols):
                        ids=batch.column("edge_id").to_numpy().astype(np.int64);positions=ids-target_lo
                        if np.any(positions<0) or np.any(positions>=len(seen)) or len(np.unique(positions))!=len(positions) or seen[positions].any():
                            raise ValueError("v1 R target coverage differs")
                        seen[positions]=True
                        if not np.array_equal(batch.column("timestamp").cast(pa.int64()).to_numpy(),arrays["timestamp"][ids]):
                            raise ValueError("v1 R timestamp differs")
                        for j,key in enumerate(cols[2:]):
                            col=batch.column(key);valid=col.is_valid().to_numpy(zero_copy_only=False)
                            actual=np.asarray(count24[positions,j])
                            if not np.array_equal(valid,actual>=0) or not np.array_equal(col.to_numpy(zero_copy_only=False)[valid],actual[valid]):
                                raise ValueError("Full v1 24h count cross-check failed")
                        checked+=len(ids)
                update(stage="crosscheck_v1_counts",checked_targets=checked,date=date)
                _pause(project)
            if not seen.all():
                raise ValueError("Incomplete v1 count cross-check")
            atomic_json(out/"count_crosscheck.json",dict(status="passed",targets=checked,config_sha256=config_hash))
            blocks=[];seen[:]=False
            for item in base["blocks"]:
                rel=item["block"];base_block,ids,times=_base_plan(cache,rel,base["config_sha256"])
                positions=ids-target_lo
                if np.any(positions<0) or np.any(positions>=len(seen)) or seen[positions].any() or not np.array_equal(times,arrays["timestamp"][ids]):
                    raise ValueError("Frozen block identity/time/coverage differs")
                seen[positions]=True
                folder=inside(out,rel);base_hash=file_sha(inside(cache,rel)/"manifest.json")
                if not (folder/"manifest.json").exists():
                    values=np.asarray(raw[positions]);full=np.column_stack([times-int(meta["data_start_us"])>=w for w in WINDOWS.values()])
                    encoded=encode_extra(values,full)
                    data={"edge_id":ids,"cutoff_us":times,**{n:encoded[:,i] for i,n in enumerate(EXTRA_COLUMNS)}}
                    for role in ROLES:
                        col=values[:,EXTRA_NAMES.index(f"v2_{role}_count_7d")]
                        data[role+"_count_7d"]=np.where(np.isnan(col),-1,col).astype(np.int64)
                    _atomic_parquet(folder/"features.parquet",pa.table(data))
                    manifest=dict(status="verified",block=rel,config_sha256=config_hash,contract_sha256=digest(contract()),
                                  base_manifest_sha256=base_hash,target_ids_sha256=array_sha(ids),targets=len(ids),
                                  file_sha256=file_sha(folder/"features.parquet"))
                    atomic_json(folder/"manifest.json",manifest)
                verified,_,_=_validate_sidecar(folder,config_hash,base_hash,ids,times)
                blocks.append(dict(block=rel,targets=len(ids),manifest_sha256=file_sha(folder/"manifest.json")))
                update(stage="block_verified",completed_blocks=len(blocks),verified_targets=sum(b["targets"] for b in blocks))
                _pause(project)
            if not seen.all():
                raise ValueError("Incomplete final block coverage")
            raw.flush();count24.flush();del raw,count24
            manifest=dict(status="completed",split=split,targets=spec["targets"],blocks=blocks,config_sha256=config_hash,
                          contract_sha256=digest(contract()),extra_dim=EXTRA_DIM,completed_at=datetime.now(timezone.utc).isoformat(),
                          count_crosscheck_sha256=file_sha(out/"count_crosscheck.json"),pilot_sha256=file_sha(out/"pilot.json"),
                          currency_verification_sha256=file_sha(out/"currency_verification.json"))
            atomic_json(out/"manifest.json",manifest)
            update(status="completed",stage="completed",verified_targets=spec["targets"])
            return manifest
        except Paused as exc:
            update(status="paused",stage="paused",message=str(exc))
            raise
        except BaseException as exc:
            update(status="failed",stage="failed",error=str(exc),traceback=traceback.format_exc())
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path.cwd())
    parser.add_argument("--split",choices=list(SPECS),required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--pilot-only",action="store_true")
    parser.add_argument("--partition-edges",type=int,default=250_000)
    args=parser.parse_args()
    try:
        run(args.project,args.split,args.output,args.pilot_only,args.partition_edges)
    except Paused:
        raise SystemExit(75)


if __name__=="__main__":
    main()
