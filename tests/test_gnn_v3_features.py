import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gnn_features import v3_features as f


def test_contract_dimensions_roles_and_no_reused_input_duplication():
    assert f.ROLES == ("src_in", "src_out", "dst_in", "dst_out")
    assert f.AMOUNT_DIM == len(f.AMOUNT_COLUMNS) == 52
    assert f.EXTRA_DIM == len(f.EXTRA_COLUMNS) == 108
    assert len(set(f.EXTRA_COLUMNS)) == 108
    assert f.EXTRA_COLUMNS == f.AMOUNT_COLUMNS + f.TIME_COLUMNS
    assert len(f.TIME_COLUMNS) == 56
    assert f.AMOUNT_NAMES[-2:] == ["v3_src_in_amount_median_24h", "v3_dst_out_amount_median_24h"]
    assert all("count" not in n and "sum" not in n and "full_" not in n for n in f.EXTRA_COLUMNS)


def run_queries(t, a, groups=None, queries=None, query_groups=None):
    t, a = np.asarray(t, dtype=np.int64), np.asarray(a, dtype=np.int64)
    groups = np.zeros(len(t), dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
    queries = t if queries is None else np.asarray(queries, dtype=np.int64)
    query_groups = groups if query_groups is None else np.asarray(query_groups, dtype=np.int64)
    data = dict(history_time=t, history_amount=a, history_group=groups, query_time=queries, query_group=query_groups)
    amount, age = f._statistics(data)
    f._check_reference(data, amount, age, np.arange(len(queries)))
    return amount, age


def test_boundary_and_all_timestamp_ties_excluded():
    t = [0, 0, f.DAY - 1, f.DAY, f.DAY, f.DAY + 1]
    a, age = run_queries(t, [1, 3, 7, 99, 109, 200])
    np.testing.assert_array_equal(a["count"], [0, 0, 2, 3, 3, 3])
    np.testing.assert_array_equal(a["sum_scaled"], [0, 0, 4, 11, 11, 215])
    assert a["median"][3] == 3e-6
    assert age["max"][3] == 1440


def test_no_history_single_zero_and_constant_validity():
    a, age = run_queries([0, 1, 2, 3, 4, 5], [0] * 6)
    assert np.isnan(a["mean"][0])
    np.testing.assert_array_equal(a["std"][1:], 0)
    assert np.isnan(a["skew"]).all() and np.isnan(a["kurtosis"]).all()
    assert np.isnan(age["skew"][:3]).all() and np.isnan(age["kurtosis"][:4]).all()
    assert age["std"][1] == 0 and age["std"][2] > 0


def test_population_skew_kurtosis_and_age_skew_sign():
    amounts = np.array([0, 0, 0, 8, 100], dtype=np.int64) * 1_000_000
    a, age = run_queries([0, 1, 2, 10, 20], amounts)
    assert a["mean"][-1] == 2
    np.testing.assert_allclose(a["std"][-1], np.sqrt(12))
    np.testing.assert_allclose(a["skew"][-1], 2 / np.sqrt(3))
    np.testing.assert_allclose(a["kurtosis"][-1], -2 / 3)
    assert age["skew"][-1] < 0


def test_large_integer_order_and_centered_moments_precision():
    # These differences disappear if integers are converted before centering.
    base = 2**60
    amounts = [base + x for x in [1, 2, 3, 4, 5, 9]]
    a, _ = run_queries(np.arange(6), amounts)
    np.testing.assert_allclose(a["std"][4], np.sqrt(1.25) / 1_000_000, rtol=1e-15)
    np.testing.assert_allclose(a["kurtosis"][4], -1.36, rtol=1e-15)
    assert a["sum_scaled"][4] == 4 * base + 10
    # An early extreme disappears from the rolling window without cancellation.
    a, _ = run_queries([0, f.DAY, f.DAY + 1, f.DAY + 2, f.DAY + 3, f.DAY + 4], [0] + amounts[:5])
    np.testing.assert_allclose(a["std"][-1], np.sqrt(1.25) / 1_000_000, rtol=1e-15)


@pytest.mark.parametrize("seed", [0, 7, 20260920])
def test_randomized_unsorted_groups_queries_and_independent_decimal_oracle(seed):
    rng = np.random.default_rng(seed)
    n = 180
    times = rng.integers(0, 4 * 24, n) * f.DAY // 24
    amounts = rng.integers(0, 1000000, n, dtype=np.int64)
    groups = rng.integers(0, 9, n)
    run_queries(times, amounts, groups, rng.integers(0, 4 * 24, 35) * f.DAY // 24, rng.integers(0, 11, 35))


def test_future_amount_and_new_group_cannot_influence_past():
    t, a, group = np.array([0, 1, 2, 3, 4, 5]), np.array([1, 3, 5, 7, 9, 11]), np.zeros(6, dtype=np.int64)
    before = run_queries(t, a, group, [2, 3], [0, 0])
    after = run_queries(np.r_[t, 6], np.r_[a[:3], [2**60, 0, 1], 99], np.r_[group, 7], [2, 3], [0, 0])
    for old, new in zip(before, after):
        for key in f.TIME_KINDS:
            np.testing.assert_array_equal(old[key], new[key])


def test_encoding_signed_moments_and_masks():
    raw = np.ones((2, 54))
    raw[0] = np.nan
    raw[1, f.EXTRA_NAMES.index("v3_src_in_amount_skew_24h")] = -2
    raw[1, f.EXTRA_NAMES.index("v3_src_in_age_min_kurtosis_24h")] = -1.36
    encoded = f.encode_extra(raw)
    assert encoded.dtype == np.float32 and encoded.shape == (2, 108)
    np.testing.assert_array_equal(encoded[0], 0)
    assert encoded[1, 4] == np.float32(-np.log(3))
    assert encoded[1, 52 + 6] == np.float32(-np.log1p(1.36))
    np.testing.assert_array_equal(encoded[1, 26:52], 1)
    np.testing.assert_array_equal(encoded[1, 80:], 1)


@pytest.mark.parametrize("problem", ["negative", "float", "shape", "overflow", "infinity", "negative_encode"])
def test_invalid_inputs_fail_stop(problem):
    if problem in ("infinity", "negative_encode"):
        raw = np.ones((1, 54))
        raw[0, 0] = np.inf if problem == "infinity" else -1
        with pytest.raises(ValueError):
            f.encode_extra(raw)
        return
    t, a, g = np.array([0, 1]), np.array([1, 2]), np.zeros(2, dtype=np.int64)
    if problem == "negative": a[0] = -1
    if problem == "float": a = a.astype(float)
    if problem == "shape": a = a[:1]
    if problem == "overflow": a = np.array([1, 2**63], dtype=np.uint64)
    with pytest.raises(ValueError):
        f.query_statistics(t, a, g, t, g)


def make_arrays():
    # Target 6: src=1 queries currency2; dst=2 queries currency3. Incoming
    # and outgoing history have deliberately different currencies and values.
    src = np.array([3, 1, 4, 2, 1, 2, 1, 1, 1, 1])
    dst = np.array([1, 4, 2, 4, 5, 1, 2, 2, 1, 2])
    t = np.array([0, 1, 2, 3, 4, 5, 6, 6, 7, 8], dtype=np.int64) * 60_000_000
    pc = np.array([3, 2, 2, 3, 3, 2, 2, 2, 2, 2])
    rc = np.array([2, 3, 3, 2, 2, 3, 3, 3, 2, 3])
    a = dict(src=src, dst=dst, timestamp=t, payment_currency_cat=pc, receiving_currency_cat=rc,
             paid_scaled=np.arange(10, 20, dtype=np.int64) * 1_000_000,
             received_scaled=np.arange(100, 110, dtype=np.int64) * 1_000_000,
             is_laundering=np.ones(10, dtype=np.int64))
    for direction, nodes in (("out", src), ("in", dst)):
        # The all-history acceptance path must explicitly exclude self edges.
        order = np.lexsort((np.arange(len(t)), t, nodes))
        a[direction + "_order"] = order
        a[direction + "_ptr"] = np.r_[0, np.cumsum(np.bincount(nodes, minlength=6))]
    return a


def test_selected_opposite_direction_currency_and_self_exclusion():
    a = make_arrays()
    encoded, support = f.compute_selected(a, 0, [6, 7, 8, 9], return_support=True)
    np.testing.assert_array_equal(support[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(support[0], support[1])
    np.testing.assert_array_equal(encoded[0], encoded[1])
    for j, expected in enumerate([100, 11, 102, 13]):
        assert encoded[0, j * 6] == np.float32(np.log1p(expected))
    np.testing.assert_array_equal(encoded[2], 0)
    np.testing.assert_array_equal(support[2], -1)
    # Self target at minute7 is not history for target9 at minute8.
    np.testing.assert_array_equal(support[3], [1, 3, 3, 1])


def test_labels_and_same_timestamp_future_amounts_never_change_selected():
    a = make_arrays()
    before = f.compute_selected(a, 0, [6, 7])
    a["is_laundering"][:] = 0
    a["paid_scaled"][6:] = 2**60 + 1
    a["received_scaled"][6:] = 2**60 + 3
    after = f.compute_selected(a, 0, [6, 7])
    np.testing.assert_array_equal(before, after)


def external_indices(arrays):
    result = arrays.copy()
    external = arrays["src"] != arrays["dst"]
    for direction, nodes in (("out", arrays["src"]), ("in", arrays["dst"])):
        ids = np.flatnonzero(external)
        ids = ids[np.lexsort((ids, arrays["timestamp"][ids], nodes[ids]))]
        result[direction + "_edge_ids"] = ids
        result[direction + "_ptr"] = np.r_[0, np.cumsum(np.bincount(nodes[external], minlength=6))]
    return result


def test_bounded_partition_path_matches_selected_for_all_four_roles():
    a = make_arrays()
    expected, expected_count = f.compute_selected(a, 0, [6, 7, 9], return_support=True)
    a = external_indices(a)
    raw, count = np.full((3, 54), np.nan), np.empty((3, 4), dtype=np.int64)
    for j, role in enumerate(f.ROLES):
        d = f._role_input(a, role, 0, 6, 6, 10)
        amount, age = f._statistics(d)
        f._check_reference(d, amount, age, np.arange(len(d["ids"])))
        table = f._part_table(role, d, amount, age)
        order = np.argsort(d["ids"])
        for name, vals in f._role_values(role, amount, age).items():
            raw[:, f.EXTRA_NAMES.index(name)] = vals[order]
        count[:, j] = amount["count"][order]
        assert table.num_rows == 3
    np.testing.assert_array_equal(f.encode_extra(raw), expected)
    np.testing.assert_array_equal(count, expected_count)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def make_store(tmp_path):
    cache, root, rel = tmp_path / "cache", tmp_path / "sidecar", "transaction_date=2022-09-11/hour=00/block=0000"
    ids, times = np.array([5, 3, 4]), np.array([100, 200, 200])
    cp = cache / rel
    cp.mkdir(parents=True)
    pq.write_table(pa.table({"targets": [[5], [3, 4]], "cutoff_us": [100, 200]}), cp / "plans.parquet")
    write_json(cp / "manifest.json", dict(status="verified", config_sha256="basehash", block=rel, targets=3,
              target_ids_sha256=f.array_sha(ids), file_sha256={"plans.parquet": f.file_sha(cp / "plans.parquet")}))
    write_json(cache / "manifest.json", dict(status="completed", targets=3, config_sha256="basehash", blocks=[dict(block=rel, targets=3)]))
    config = dict(split="validation", offset=3, code_sha256=f.file_sha(Path(f.__file__)), cache_root=str(cache.resolve()),
                  contract=f.contract(), contract_sha256=f.digest(f.contract()), base_manifest_sha256=f.file_sha(cache / "manifest.json"))
    write_json(root / "config.json", config)
    ch = f.file_sha(root / "config.json")
    raw = np.ones((3, 54))
    raw[0] = np.nan
    encoded = f.encode_extra(raw)
    data = {"edge_id": ids, "cutoff_us": times, **{n: encoded[:, i] for i, n in enumerate(f.EXTRA_COLUMNS)},
            **{n: np.array([-1, 5, 8], dtype=np.int64) for n in f.SUPPORT_COLUMNS}}
    folder = root / rel
    folder.mkdir(parents=True)
    pq.write_table(pa.table(data), folder / "features.parquet")
    write_json(folder / "manifest.json", dict(status="verified", block=rel, targets=3, config_sha256=ch,
              contract_sha256=f.digest(f.contract()), base_manifest_sha256=f.file_sha(cp / "manifest.json"),
              target_ids_sha256=f.array_sha(ids), file_sha256=f.file_sha(folder / "features.parquet")))
    for name in ("pilot.json", "reuse_crosscheck.json", "currency_verification.json"):
        write_json(root / name, dict(status="passed", config_sha256=ch))
    write_json(root / "manifest.json", dict(status="completed", targets=3, config_sha256=ch,
              pilot_sha256=f.file_sha(root / "pilot.json"), reuse_crosscheck_sha256=f.file_sha(root / "reuse_crosscheck.json"),
              currency_verification_sha256=f.file_sha(root / "currency_verification.json"),
              blocks=[dict(block=rel, targets=3, manifest_sha256=f.file_sha(folder / "manifest.json"))]))
    return root, cache, rel, ids, times, encoded


def test_store_identity_subset_reorder_support_and_immutable_copy(tmp_path):
    root, cache, rel, ids, times, expected = make_store(tmp_path)
    store = f.SidecarStore(root, cache, "validation")
    np.testing.assert_array_equal(store.block(rel, [4, 5], [200, 100]), expected[[2, 0]])
    np.testing.assert_array_equal(store.block(rel, [3, 4], 200), expected[[1, 2]])
    support = store.support(rel, [3, 4], 200)
    assert list(support) == f.SUPPORT_COLUMNS
    np.testing.assert_array_equal(support["src_in_count_24h"], [5, 8])
    changed = store.block(rel, [3], 200)
    changed[:] = 0
    np.testing.assert_array_equal(store.block(rel, [3], 200), expected[[1]])
    assert store.identity["manifest_sha256"] == store.manifest_sha256
    assert f.verify_complete(root, cache, "validation")["targets"] == 3


@pytest.mark.parametrize("problem", ["id", "cutoff", "duplicate", "block", "split"])
def test_store_bad_requests_stop(tmp_path, problem):
    root, cache, rel, *_ = make_store(tmp_path)
    if problem == "split":
        with pytest.raises(ValueError): f.SidecarStore(root, cache, "train")
        return
    store = f.SidecarStore(root, cache, "validation")
    with pytest.raises(ValueError):
        if problem == "id": store.block(rel, [77], 200)
        if problem == "cutoff": store.block(rel, [3], 100)
        if problem == "duplicate": store.block(rel, [3, 3], 200)
        if problem == "block": store.block("absent", [3], 200)


@pytest.mark.parametrize("problem", ["checksum", "code", "contract", "acceptance", "base", "incomplete"])
def test_store_tamper_and_incomplete_stop(tmp_path, problem):
    root, cache, rel, ids, times, _ = make_store(tmp_path)
    if problem == "checksum":
        with (root / rel / "features.parquet").open("ab") as out: out.write(b"tamper")
    elif problem in ("code", "contract"):
        config = f.read(root / "config.json")
        config[problem + "_sha256"] = "wrong"
        write_json(root / "config.json", config)
    elif problem == "acceptance":
        write_json(root / "pilot.json", {"status": "passed", "config_sha256": "wrong"})
    elif problem == "base":
        write_json(cache / "manifest.json", {"status": "completed", "targets": 4})
    elif problem == "incomplete":
        manifest = f.read(root / "manifest.json")
        manifest["status"] = "running"
        write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError):
        f.SidecarStore(root, cache, "validation").block(rel, ids, times)


def test_pilot_does_not_read_labels_and_reports_runtime_memory():
    a = external_indices(make_arrays())
    del a["is_laundering"]
    events = []
    report = f._pilot(a, {"offset": 0, "targets": 10}, lambda **kw: events.append(kw))
    assert report["status"] == "passed" and report["checks"] > 7
    assert report["seconds"] > 0 and report["peak_rss_sampled_bytes"] > 0
    assert {e["role"] for e in events} == set(f.ROLES)


def test_pause_targets_only_v3_and_preserves_files(tmp_path):
    path = tmp_path / "result/gnn_tuning_v3/pause_request.json"
    write_json(path, dict(enabled=True))
    with pytest.raises(f.Paused): f._pause(tmp_path)
    assert path.exists()
    write_json(path, dict(enabled=False))
    f._pause(tmp_path)


def test_disk_reserve_blocks_new_output(monkeypatch, tmp_path):
    import shutil
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(100, 90, 10 * 1024**3))
    with pytest.raises(RuntimeError): f.ensure_space(tmp_path)


def test_unicode_manifest_path_is_read_as_utf8_without_fallback(tmp_path):
    features = tmp_path / "features"
    folder = features / "transaction_date=2022-09-01"
    path = folder / "features/part-0000-25.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"id": [1]}), path)
    write_json(folder / "daily_run.json", {"status": "verified"})
    write_json(folder / "merge_phase.json", {"parts": [{"partition": 0, "path": str(path), "rows": 1}]})
    assert list(f._r_paths(features, "2022-09-01")) == [path.resolve()]
    write_json(folder / "merge_phase.json", {"parts": [{"partition": 0, "path": str(tmp_path / "wrong" / path.name), "rows": 1}]})
    with pytest.raises(ValueError): list(f._r_paths(features, "2022-09-01"))


def test_full_builder_small_fixture_crosschecks_and_reuses_verified_outputs(tmp_path, monkeypatch):
    """Exercise partition->scratch->four R sums/counts->v2 medians->publication.

    Only the pre-existing v2 Store construction is stubbed: all local R values
    come from an independent explicit-window oracle, and every v3 hash/reader
    check runs normally. No production archive is accessed by this test.
    """
    from decimal import Decimal
    import shutil
    code = tmp_path / "scripts/gnn_features"
    code.mkdir(parents=True)
    for source in (Path(f.__file__), Path(f.v2.__file__), Path(f.__file__).with_name("training_state.py")):
        shutil.copy2(source, code / source.name)
    monkeypatch.setattr(f, "__file__", str(code / "v3_features.py"))
    monkeypatch.setattr(f.v2, "__file__", str(code / "v2_features.py"))
    a = external_indices(make_arrays())
    a.pop("in_order"); a.pop("out_order")
    spec = dict(archive="data/gnn_v1_train", cache="data/gnn_train_cache_parquet", features="result/gnn_train_v1",
                offset=0, targets=10, dates=["2022-09-01"], end="2022-09-11T00:00:00")
    monkeypatch.setitem(f.SPECS, "train", spec)
    archive, cache, features = [tmp_path / spec[k] for k in ("archive", "cache", "features")]
    array_hashes = {}
    for name in ("timestamp", "src", "dst", "paid_scaled", "received_scaled", "payment_currency_cat", "receiving_currency_cat", "in_ptr", "out_ptr", "in_edge_ids", "out_edge_ids"):
        path = archive / ("graph/adj_index" if name.endswith(("_ptr", "_edge_ids")) else "history_index") / (name + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, a[name])
        array_hashes[str(path.relative_to(tmp_path))] = f.file_sha(path)
    write_json(archive / "build.json", dict(rows=10, end_exclusive=spec["end"], data_start_us=0))
    write_json(archive / "metadata/vocabulary.json", dict(offset=2, values=dict(currency=["USD", "EUR"])))
    graph = archive / "graph/edges/part.parquet"
    graph.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"edge_id": np.arange(10),
                           "payment_currency": ["USD" if c == 2 else "EUR" for c in a["payment_currency_cat"]],
                           "receiving_currency": ["USD" if c == 2 else "EUR" for c in a["receiving_currency_cat"]]}), graph)
    folder = features / "transaction_date=2022-09-01"
    path = folder / "features/part-0000-25.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"edge_id": np.arange(10), "timestamp": pa.array(a["timestamp"], type=pa.timestamp("us"))}
    for role in f.ROLES:
        node, direction, qcur, hcur, amount, _ = f._role_spec(role)
        hnode = "src" if direction == "out" else "dst"
        counts, sums = [], []
        for i in range(10):
            if a["src"][i] == a["dst"][i]:
                counts.append(None); sums.append(None); continue
            use = ((a["src"] != a["dst"]) & (a[hnode] == a[node][i]) & (a[hcur] == a[qcur][i]) &
                   (a["timestamp"] >= a["timestamp"][i] - f.DAY) & (a["timestamp"] < a["timestamp"][i]))
            counts.append(int(use.sum()))
            sums.append(Decimal(sum(int(x) for x in a[amount][use])) / 1_000_000)
        data[f"r_{node}_ext_{direction}_count_24h"] = pa.array(counts, type=pa.int64())
        data[f"r_{node}_ext_{direction}_sum_24h"] = pa.array(sums, type=pa.decimal128(38, 6))
    pq.write_table(pa.table(data), path)
    write_json(folder / "daily_run.json", dict(status="verified"))
    write_json(folder / "merge_phase.json", dict(parts=[dict(partition=0, path=str(path.resolve()), rows=10)]))
    write_json(cache / "config.json", dict(hashes={str(path.relative_to(tmp_path)): f.file_sha(path)}))
    cache_hash = f.file_sha(cache / "config.json")
    rel = "transaction_date=2022-09-01/hour=00/block=0000"
    cp = cache / rel
    cp.mkdir(parents=True)
    pq.write_table(pa.table({"targets": [[i] for i in range(10)], "cutoff_us": a["timestamp"]}), cp / "plans.parquet")
    write_json(cp / "manifest.json", dict(status="verified", block=rel, config_sha256=cache_hash, targets=10,
              target_ids_sha256=f.array_sha(np.arange(10)), file_sha256={"plans.parquet": f.file_sha(cp / "plans.parquet")}))
    write_json(cache / "manifest.json", dict(status="completed", targets=10, config_sha256=cache_hash,
              edge_and_identity_tables=str(archive.resolve()), feature_tables=str(features.resolve()), blocks=[dict(block=rel, targets=10)]))
    old_root = tmp_path / "data/gnn_v2_target_features/train"
    write_json(old_root / "config.json", dict(input_hashes=array_hashes))
    write_json(old_root / "manifest.json", dict(status="completed"))

    class OldStore:
        def __init__(self, root, base, split):
            self.root = Path(root)
            self.config = f.read(self.root / "config.json")
            self.config_sha256 = f.file_sha(self.root / "config.json")
            self.manifest_sha256 = f.file_sha(self.root / "manifest.json")

        def block(self, block, ids, times):
            return f.v2.compute_selected(a, 0, ids)

    monkeypatch.setattr(f.v2, "SidecarStore", OldStore)
    output = tmp_path / "data/gnn_v3_distribution_features/train"
    result = f.run(tmp_path, "train", partition_edges=10000)
    assert result["status"] == "completed" and result["targets"] == 10
    store = f.SidecarStore(output, cache, "train")
    expected, count = f.compute_selected(a, 0, np.arange(10), return_support=True)
    np.testing.assert_array_equal(store.block(rel, np.arange(10), a["timestamp"]), expected)
    for j, name in enumerate(f.SUPPORT_COLUMNS):
        np.testing.assert_array_equal(store.support(rel, np.arange(10), a["timestamp"])[name], count[:, j])
    before = {str(p.relative_to(output)): f.file_sha(p) for p in output.rglob("*") if p.is_file() and p.name not in ("status.json", ".lock")}
    assert f.run(tmp_path, "train", partition_edges=10000)["status"] == "completed"
    after = {str(p.relative_to(output)): f.file_sha(p) for p in output.rglob("*") if p.is_file() and p.name not in ("status.json", ".lock")}
    assert before == after
