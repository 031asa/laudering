"""Independent, read-only verification of the completed retrospective chain audit.

No producer algorithm is imported. Only a new verification JSON is written, with
exclusive creation. Historical-label diagnostics do not establish common funds,
causality, or a deployable feature. Sealed Test sources are never opened.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
DAY = 86_400_000_000
START = int(np.datetime64("2022-09-01", "us").astype(np.int64))
VALIDATION_START = START + 10 * DAY
END = START + 13 * DAY
NTRAIN, NVALID = 20_318_992, 4_689_006
P80 = 0.945133626461029
EXPECTED = {"ACH_FN": 772, "ACH_TP": 1508, "G_FN": 662}
FEATURES = ("prior_positive_any_7d", "src_temporal_walk_ge2_7d", "mixed_2hop_positive_any_7d")
STAGES = ("positive_history_features", "full_history_neighbors", "sampling_audit")


def require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, value):
    path = (root / value).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes declared root: " + str(value))
    return path


def close(actual, expected, message, atol=1e-12):
    require(np.isclose(float(actual), float(expected), rtol=0, atol=atol), message)


def list_value(value):
    result = json.loads(value)
    require(isinstance(result, list), "Expected a JSON list")
    return result


def verify_stage_inputs(project, root, name, audit, inputs, verified):
    """Accept only the explicitly reviewed sampling-parser-only migration."""
    if audit["inputs"] == inputs:
        return dict(stage=name, provenance="original_inputs_equal_current")
    require(name in ("positive_history_features", "full_history_neighbors"), "Unexpected recovered stage")
    review_path = root / "sampling_format_recovery.json"
    review = read(review_path)
    reuse_path = root / (name + ".reuse_audit.json")
    reuse = read(reuse_path)
    require(verified.get(review_path.name) == sha(review_path) and verified.get(reuse_path.name) == sha(reuse_path),
            "Recovery/reuse review missing from run manifest")
    require(review["status"] == "reviewed_unchanged_completed_computation", "Recovery has not been reviewed")
    require(review["old_inputs"] == audit["inputs"] and review["new_inputs"] == inputs, "Recovery input transition differs")
    require(review["unchanged_stage_outputs"][name] == audit["output_sha256"], "Recovered stage bytes differ")
    require(reuse == dict(original_audit_sha256=sha(root / (name + ".audit.json")), review_sha256=sha(review_path),
                          original_inputs=audit["inputs"], current_inputs=inputs, output_sha256=audit["output_sha256"]),
            "Stage reuse audit differs from byte-bound provenance")
    old = json.loads(json.dumps(audit["inputs"]))
    sampler = "gnn_chain_sampling_audit.py"
    old["modules"][sampler] = inputs["modules"][sampler]
    require(old == inputs, "Recovery changed more than the sampling diagnostic module")
    before = root / "recovery/sampling_manifest_format/gnn_chain_sampling_audit.before.py"
    before_meta = read(before.parent / "before.json")
    old_sha, new_sha = sha(before), sha(project / "scripts" / sampler)
    require(old_sha == audit["inputs"]["modules"][sampler] == before_meta["source_sha256"], "Archived original sampler SHA differs")
    require(new_sha == inputs["modules"][sampler], "Recovered sampler SHA differs")
    require(review["old_sampling_script_sha256"] == old_sha and review["new_sampling_script_sha256"] == new_sha,
            "Recovery declared source SHA differs")
    require(review["frozen_inputs_modified"] is False, "Recovery declared a frozen input change")
    old_tree = ast.parse(before.read_text(encoding="utf-8-sig"))
    new_tree = ast.parse((project / "scripts" / sampler).read_text(encoding="utf-8-sig"))
    for tree in (old_tree, new_tree):
        nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for key, expected in review["unchanged_ast_sha256"].items():
            require(key in nodes and hashlib.sha256(ast.dump(nodes[key], include_attributes=False).encode()).hexdigest() == expected,
                    "Recovery AST proof differs: " + key)
    def without_sampling(tree):
        require(sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "audit_saved_sampling" for n in tree.body) == 1,
                "Expected exactly one sampling function")
        tree.body = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "audit_saved_sampling")]
        return ast.dump(tree, include_attributes=False)
    old_ast, new_ast = without_sampling(old_tree), without_sampling(new_tree)
    require(old_ast == new_ast, "Non-sampling computation changed during recovery")
    return dict(stage=name, provenance="explicit_reviewed_sampling_parser_recovery",
                original_stage_audit_sha256=reuse["original_audit_sha256"], review_sha256=sha(review_path),
                old_sampler_sha256=old_sha, new_sampler_sha256=new_sha,
                independently_verified_non_sampling_ast_sha256=hashlib.sha256(old_ast.encode()).hexdigest())


class Progress:
    def __init__(self):
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage = None

    def __call__(self, stage, completed=None, total=None, **extra):
        if stage != self.stage:
            self.stage, self.stage_started = stage, time.monotonic()
        elapsed = time.monotonic() - self.stage_started
        message = dict(stage=stage, elapsed_seconds=round(time.monotonic()-self.started, 2), **extra)
        if completed is not None:
            message.update(completed=completed, total=total)
            if completed and total:
                message["estimated_remaining_stage_seconds"] = round(elapsed * (total-completed) / completed, 1)
        print(json.dumps(message, ensure_ascii=True), flush=True)


def verify_labels(project, labels, positives, candidates, progress):
    """Read only edge_id/label columns; full graph payloads are not reopened."""
    require(labels["status"] == "verified", "Label manifest is not verified")
    require(len(labels["supervision_files"]) == 102, "Unexpected supervision file count")
    seen = np.zeros(NTRAIN + NVALID, dtype=bool)
    positive_ids = []
    totals = {s: dict(rows=0, positives=0, files=0) for s in ("train", "validation")}
    sources = {}
    for index, item in enumerate(labels["supervision_files"], 1):
        path = safe_path(project, item["path"])
        relative = path.relative_to(project).as_posix()
        require(relative.startswith("data/gnn_v1_validation/supervision/targets/transaction_date="),
                "Unexpected supervision source: " + relative)
        dates = [part.split("=", 1)[1] for part in path.parts if part.startswith("transaction_date=")]
        require(len(dates) == 1 and "2022-09-01" <= dates[0] <= "2022-09-13", "Forbidden supervision date")
        expected_split = "train" if dates[0] <= "2022-09-10" else "validation"
        require(item["split"] == expected_split, "Supervision split/date mismatch")
        require(relative not in sources, "Duplicate supervision file")
        actual_sha = sha(path)
        require(actual_sha == item["sha256"], "Supervision SHA differs: " + relative)
        sources[relative] = actual_sha
        frame = pq.read_table(path, columns=["edge_id", "label"]).to_pandas()
        ids = frame.edge_id.to_numpy(dtype=np.int64)
        y = frame.label.to_numpy()
        require(len(ids) == item["rows"] and int((y == 1).sum()) == item["positives"], "Supervision counts differ")
        require(np.isin(y, [0, 1]).all(), "Nonbinary supervision label")
        require(np.all((ids >= 0) & (ids < len(seen))) and len(np.unique(ids)) == len(ids), "Invalid supervision IDs")
        require(not seen[ids].any(), "Repeated supervision ID across files")
        require(np.all(ids < NTRAIN) if expected_split == "train" else np.all(ids >= NTRAIN), "Train/Validation ID prefix differs")
        seen[ids] = True
        positive_ids.append(ids[y == 1])
        entry = totals[expected_split]
        entry["rows"] += len(ids); entry["positives"] += int((y == 1).sum()); entry["files"] += 1
        if index % 10 == 0 or index == 102:
            progress("supervision_labels", index, 102)
    require(seen.all(), "Supervision ID coverage is incomplete")
    require(totals == {"train": dict(rows=NTRAIN, positives=17794, files=78),
                       "validation": dict(rows=NVALID, positives=6420, files=24)}, "Global supervision totals differ")
    ids = np.sort(np.concatenate(positive_ids))
    saved_ids = positives.edge_id.to_numpy(dtype=np.int64)
    require(len(saved_ids) == 24214 and np.array_equal(ids, np.sort(saved_ids)), "Positive-edge label identity differs")
    require(hashlib.sha256(saved_ids.tobytes()).hexdigest() == labels["positive_edges"]["edge_ids_sha256"],
            "Positive-edge ordered ID digest differs")
    require(np.array_equal(np.isin(candidates.edge_id, ids).astype(np.int64), candidates.label.to_numpy()), "Candidate labels differ")
    require(set(candidates.loc[candidates.label == 1, "edge_id"]) == set(ids[ids >= NTRAIN]), "Validation positive coverage differs")
    return dict(counts=totals, supervision_files=102, positive_ids=len(ids), source_sha256=sources,
                read_columns=["edge_id", "label"], full_unique_edge_coverage=True)


def verify_cohorts(candidates, targets, positives):
    require(len(candidates) == 106420 and not candidates.edge_id.duplicated().any(), "Candidate count/identity")
    require(int(candidates.label.sum()) == 6420, "Candidate positive count")
    require(candidates.uniform_negative_control.isin([True, False]).all(), "Uniform-negative flag is not binary")
    require(np.array_equal(candidates.uniform_negative_control.to_numpy(bool), (candidates.label == 0).to_numpy()),
            "Uniform controls must be exactly the 100000 negatives")
    require(np.all((candidates.edge_id >= NTRAIN) & (candidates.edge_id < NTRAIN + NVALID)), "Candidate split identity")
    require(np.all((candidates.cutoff_us >= VALIDATION_START) & (candidates.cutoff_us < END)), "Candidate timestamp includes forbidden split")
    require(not positives.edge_id.duplicated().any(), "Duplicate positive identity")
    require(np.all((positives.timestamp_us >= START) & (positives.timestamp_us < END)), "Positive-edge time includes Test")
    require(np.array_equal((positives.edge_id < NTRAIN).to_numpy(), (positives.timestamp_us < VALIDATION_START).to_numpy()),
            "Positive time and split prefixes differ")
    require(np.array_equal(positives.split.astype(str).to_numpy(), np.where(positives.edge_id < NTRAIN, "train", "validation")),
            "Positive split strings differ")
    pred = candidates.score_lr_only.to_numpy(float) >= P80
    truth = candidates.label.to_numpy() == 1
    outcomes = np.where(truth, np.where(pred, "TP", "FN"), np.where(pred, "FP", "TN"))
    require(np.array_equal(outcomes, candidates.outcome_lr_only.to_numpy()), "Saved outcome differs from frozen global P80 threshold")
    ach = ((candidates.payment_format == "ACH") & (candidates.r_pair_fwd_count_72h == 0)
           & (candidates.v2_dst_current_to_q90_7d > 1)).to_numpy()
    masks = {"ACH_FN": ach & truth & ~pred, "ACH_TP": ach & truth & pred,
             "G_FN": (candidates.from_bank.astype(str) == "70").to_numpy()
                     & (candidates.payment_format != "ACH").to_numpy() & truth & ~pred}
    expected_labels = np.full(len(candidates), "candidate", dtype=object)
    for cohort, mask in masks.items():
        require(int(mask.sum()) == EXPECTED[cohort], "Predeclared cohort count differs: " + cohort)
        expected_labels[mask] = cohort
    require(np.array_equal(expected_labels, candidates.cohort.to_numpy()), "Candidate cohort assignments differ")
    require(not targets.edge_id.duplicated().any(), "Duplicate target IDs")
    source = candidates.set_index("edge_id").loc[targets.edge_id].reset_index()
    columns = [c for c in source if c != "cohort"]
    pd.testing.assert_frame_equal(source[columns].reset_index(drop=True), targets[columns].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    selected_pos = candidates[candidates.label == 1].set_index("edge_id")
    known = positives.set_index("edge_id").loc[selected_pos.index]
    for key in ("src", "dst"):
        require(np.array_equal(selected_pos[key], known[key]), "Positive/candidate endpoint identity mismatch: " + key)
    require(np.array_equal(selected_pos.cutoff_us, known.timestamp_us), "Positive/candidate timestamp mismatch")
    return masks


def support(x):
    if pd.isna(x): return "not_applicable"
    require(np.isfinite(x) and x >= 0 and x == math.floor(x), "Invalid matching support count")
    return "0" if x == 0 else "1-4" if x < 5 else ">=5"


def verify_matching(root, candidates, targets, progress):
    """Reconstruct strata and deterministic choices without producer imports."""
    d = candidates.set_index("edge_id", drop=False).copy()
    d.index.name = "_edge_index"
    d["amount_paid"] = pd.to_numeric(d.amount_paid, errors="raise").astype(float)
    require(np.isfinite(d.amount_paid).all() and (d.amount_paid >= 0).all(), "Invalid paid amount for matching")
    for side in ("src", "dst"):
        d["support_" + side] = d["v2_" + side + "_count_7d"].map(support)
    exact = ("payment_format", "payment_currency", "receiving_currency", "date", "support_src", "support_dst")
    tables, summaries = {}, {}
    chosen_ids = set(d.loc[d.cohort.isin(EXPECTED), "edge_id"])
    expected_target_cohort = d.loc[targets.edge_id, "cohort"].copy()
    for cohort in ("ACH_FN", "G_FN"):
        pairs = pd.read_csv(root / (cohort + "_pairs.csv"))
        unmatched = pd.read_csv(root / (cohort + "_unmatched.csv"))
        saved = read(root / (cohort + "_matching.json"))["summary"]
        require(not pairs.duplicated(["case_edge_id", "control_edge_id"]).any(), "Repeated case/control pair")
        require(pairs.cohort.eq(cohort).all(), "Pair cohort differs")
        require(not unmatched.case_edge_id.duplicated().any(), "Duplicate unmatched case")
        cases = d[d.cohort == cohort].sort_values("edge_id")
        negative = d[(d.label == 0) & d.uniform_negative_control]
        if cohort == "ACH_FN":
            negative = negative[(negative.payment_format == "ACH") & (negative.r_pair_fwd_count_72h == 0)
                                & (negative.v2_dst_current_to_q90_7d > 1)]
        groups = {tuple(map(str, key)): frame for key, frame in negative.groupby(list(exact), sort=False)}
        actual_groups = {int(k): g for k, g in pairs.groupby("case_edge_id", sort=False)}
        actual_unmatched = unmatched.set_index("case_edge_id").to_dict("index")
        require(set(actual_groups).isdisjoint(actual_unmatched), "Matched and unmatched cases overlap")
        require(set(actual_groups) | set(actual_unmatched) == set(cases.edge_id), "Case accounting differs")
        parents = {}
        def find(node):
            parents.setdefault(node, node)
            if parents[node] != node: parents[node] = find(parents[node])
            return parents[node]
        def union(a, b):
            a, b = find(a), find(b)
            if a != b: parents[max(a, b)] = min(a, b)
        for number, row in enumerate(cases.to_dict("records"), 1):
            eid, payer = int(row["edge_id"]), int(row["src"])
            expected_failure, n_exact = None, 0
            key = tuple(str(row[c]) for c in exact)
            pool = groups.get(key)
            if "not_applicable" in (row["support_src"], row["support_dst"]):
                expected_failure = "missing_history_support"
            elif pool is None:
                expected_failure = "no_exact_stratum"
            else:
                if cohort == "G_FN": pool = pool[pool.src == payer]
                n_exact = len(pool)
                if not n_exact: expected_failure = "no_same_payer_in_stratum"
                else:
                    amount = float(row["amount_paid"])
                    pool = pool[pool.amount_paid == 0] if amount == 0 else pool[(pool.amount_paid >= amount/2) & (pool.amount_paid <= amount*2)]
                    if pool.empty: expected_failure = "outside_amount_caliper"
            if expected_failure:
                require(eid in actual_unmatched, "Expected an unmatched case: " + str(eid))
                item = actual_unmatched[eid]
                require(item["reason"] == expected_failure and item["exact_stratum_candidates"] == (n_exact if expected_failure == "outside_amount_caliper" else 0)
                        and item["caliper_candidates"] == 0, "Unmatched reason/candidate accounting differs")
                continue
            require(eid in actual_groups, "Expected a matched case: " + str(eid))
            pool = pool.copy()
            amount_term = np.zeros(len(pool)) if amount == 0 else np.abs(np.log(pool.amount_paid.to_numpy(float)/amount))/math.log(2)
            count_term = sum(np.abs(np.log1p(pool["v2_"+s+"_count_7d"].to_numpy(float))-math.log1p(float(row["v2_"+s+"_count_7d"]))) for s in ("src", "dst")) / 2
            pool["rank_distance"] = amount_term + count_term
            pool["different_payer"] = pool.src != payer
            # Remove index name ambiguity without changing the ranking fields.
            selected = pool.reset_index(drop=True).sort_values(["different_payer", "rank_distance", "edge_id"]).head(3)
            actual = actual_groups[eid]
            require(actual.control_edge_id.tolist() == selected.edge_id.tolist(), "Deterministic chosen controls differ: " + str(eid))
            require(actual.controls_for_case.eq(len(selected)).all(), "Declared controls per case differ")
            require(actual.case_payer.eq(payer).all(), "Case payer differs")
            require(np.array_equal(actual.control_payer.to_numpy(), selected.src.to_numpy()), "Control payer differs")
            require(np.array_equal(actual.same_payer.to_numpy(bool), ~selected.different_payer.to_numpy()), "Same payer priority differs")
            require(np.allclose(actual.distance, selected.rank_distance, rtol=0, atol=1e-12), "Matching distance differs")
            require(np.allclose(actual.control_weight, 1/len(selected), rtol=0, atol=1e-12), "Controls do not share case weight equally")
            close(actual.control_weight.sum(), 1, "Case control weights do not sum to one")
            for control_payer in actual.control_payer: union(payer, int(control_payer))
            if number % 100 == 0: progress("matching_" + cohort, number, len(cases))
        require(np.array_equal(pairs.dependency_cluster.to_numpy(), np.array([find(int(p)) for p in pairs.case_payer])),
                "Account/control dependence components differ")
        cluster_cases = pairs[["case_edge_id", "dependency_cluster"]].drop_duplicates().groupby("dependency_cluster").size()
        dependency = dict(clusters=len(cluster_cases), largest_cluster_cases=int(cluster_cases.max()),
                          cluster_case_counts={str(k): int(v) for k, v in cluster_cases.items()})
        counts = pairs.groupby("case_edge_id").size()
        usage = pairs.control_edge_id.value_counts()
        summary = dict(cases_total=len(cases), matched_cases=len(counts), unmatched_cases=len(unmatched),
                       unique_controls=len(usage), pair_rows=len(pairs), controls_used_more_than_once=int((usage > 1).sum()),
                       max_control_reuse=int(usage.max()), cases_with_fewer_than_requested_controls=int((counts < 3).sum()),
                       controls_per_case={str(k): int(v) for k, v in counts.value_counts().items()},
                       case_payers_total=int(cases.src.nunique()), matched_case_payers=int(pairs.case_payer.nunique()),
                       control_payers=int(pairs.control_payer.nunique()), negative_pool_rows=len(negative), dependency=dependency,
                       unmatched_reasons=dict(Counter(unmatched.reason)))
        for key, value in summary.items(): require(saved[key] == value, "Matching summary differs: " + cohort + "/" + key)
        close(saved["matching_fraction"], len(counts)/len(cases), "Matching fraction differs")
        chosen_ids.update(pairs.control_edge_id)
        expected_target_cohort.loc[pairs.control_edge_id.unique()] = cohort + "_CONTROL"
        tables[cohort], summaries[cohort] = pairs, summary
        progress("matching_" + cohort, len(cases), len(cases))
    require(chosen_ids == set(targets.edge_id), "Target set differs from cases/reference/matched controls union")
    require(np.array_equal(expected_target_cohort.loc[targets.edge_id], targets.cohort), "Target control cohorts differ")
    return tables, summaries


def time_index_lengths(src, dst, timestamps, endpoints):
    """Independent max-plus recurrence via per-node strict-time step functions.

    A timestamp lookup uses bisect_left so records created at the current time
    cannot be consumed. This differs from the producer's simultaneous-update
    persistent witness links. Only target endpoint lengths are retained.
    """
    histories = {}
    for u, v, when in zip(src, dst, timestamps):
        u, v, when = int(u), int(v), int(when)
        before = histories.get(u)
        index = -1 if before is None else bisect_left(before[0], when)-1
        length = 1 if index < 0 else before[1][index]+1
        current = histories.get(v)
        if current is None:
            histories[v] = ([when], [length])
        elif length > current[1][-1]:
            if current[0][-1] == when: current[1][-1] = length
            else:
                current[0].append(when); current[1].append(length)
    return {int(node): histories[int(node)][1][-1] if int(node) in histories else 0 for node in endpoints}


def verify_witness(ids, edge_index, cutoff, endpoint, target_id, duration=7*DAY):
    require(len(ids) == len(set(ids)), "Witness repeats an edge")
    require(target_id not in ids, "Witness contains its target")
    if not ids: return [], [], []
    require(set(ids).issubset(edge_index.index), "Witness edge is not a verified positive")
    frame = edge_index.loc[ids]
    times = frame.timestamp_us.to_numpy(np.int64)
    src, dst = frame.src.to_numpy(np.int64), frame.dst.to_numpy(np.int64)
    require(np.all(src != dst), "Witness includes self-loop")
    require(np.all((times >= cutoff-duration) & (times < cutoff)), "Witness outside strict historical window")
    require(np.all(np.diff(times) > 0), "Witness timestamps are not strictly increasing")
    require(np.array_equal(dst[:-1], src[1:]), "Witness edges do not connect in the forward direction")
    require(int(dst[-1]) == endpoint, "Witness does not finish at target endpoint")
    return times.tolist(), [int(src[0])] + dst.tolist(), frame.index.tolist()


def verify_histories(positives, detail, progress):
    edge_index = positives.set_index("edge_id")
    external = positives[positives.src != positives.dst].sort_values(["timestamp_us", "edge_id"])
    all_ids, all_src, all_dst, all_times = (external[k].to_numpy(np.int64) for k in ("edge_id", "src", "dst", "timestamp_us"))
    completed, cutoffs, witness_edges, nonempty = 0, 0, 0, 0
    progress("historical_witnesses", 0, len(detail))
    for cutoff, group in detail.groupby("cutoff_us", sort=True):
        cutoff = int(cutoff); cutoffs += 1
        lo, hi = np.searchsorted(all_times, [cutoff-7*DAY, cutoff], side="left")
        ids, src, dst, times = all_ids[lo:hi], all_src[lo:hi], all_dst[lo:hi], all_times[lo:hi]
        recent = times >= cutoff-3*DAY
        graph_nodes = len(set(src) | set(dst))
        endpoints = set(group.src) | set(group.dst)
        independent_max = time_index_lengths(src, dst, times, endpoints)
        future_end = min(cutoff+3*DAY, END)
        flo = np.searchsorted(all_times, cutoff, side="right")
        fhi = np.searchsorted(all_times, future_end, side="left")
        for r in group.to_dict("records"):
            target = int(r["edge_id"])
            require(target not in ids, "Target is in historical positive graph")
            require(r["history_positive_unique_edges_7d"] == len(ids), "Window positive-edge count differs")
            require(r["history_positive_graph_nodes_7d"] == graph_nodes, "Window positive-node count differs")
            require(r["history_window_start_us_7d"] == cutoff-7*DAY and r["history_window_start_us_72h"] == cutoff-3*DAY,
                    "Historical window lower bound differs")
            for tag, width in (("7d", 7*DAY), ("72h", 3*DAY)):
                require(bool(r["history_window_complete_" + tag]) == (START <= cutoff-width), "Window completeness differs")
            union7, union72 = set(), set()
            for role in ("src", "dst"):
                endpoint = int(r[role]); prefix = role + "_"
                incident = (src == endpoint) | (dst == endpoint)
                for tag, mask in (("7d", np.ones(len(ids), bool)), ("72h", recent)):
                    for direction, direction_mask in (("incident_unique", incident), ("in", dst == endpoint), ("out", src == endpoint)):
                        require(int(r[prefix + "history_positive_"+direction+"_edges_"+tag]) == int((mask & direction_mask).sum()),
                                "Endpoint incident/in/out count differs: " + str(target))
                role_ids = sorted(ids[incident & recent].tolist())
                require(list_value(r[prefix+"history_positive_incident_edge_ids_72h"]) == role_ids, "72h endpoint witness IDs differ")
                union7.update(ids[incident]); union72.update(role_ids)
                witness = list_value(r[prefix + "temporal_walk_witness_edge_ids_7d"])
                wt, nodes, _ = verify_witness(witness, edge_index, cutoff, endpoint, target)
                length = len(witness)
                require(length == int(r[prefix+"temporal_walk_max_history_edges_7d"]) == independent_max[endpoint],
                        "Longest temporal walk differs from independent time-index DP: " + str(target))
                require(list_value(r[prefix+"temporal_walk_witness_timestamp_us_7d"]) == wt, "Witness recorded timestamps differ")
                require(list_value(r[prefix+"temporal_walk_witness_nodes_7d"]) == nodes, "Witness recorded nodes differ")
                require(int(r[prefix+"temporal_walk_witness_distinct_nodes_7d"]) == len(set(nodes)), "Distinct witness nodes differ")
                require(bool(r[prefix+"temporal_walk_witness_revisits_accounts_7d"]) == (len(set(nodes)) < len(nodes)), "Walk revisit flag differs")
                for k, tag in ((1, "1"), (2, "2"), (3, "3"), (4, "4plus")):
                    require(bool(r[prefix+"temporal_walk_exists_at_least_"+tag+"_edges_7d"]) == (length >= k), "Nested walk presence differs")
                context_column = prefix+"temporal_walk_witness_currency_context_7d"
                if context_column in r:
                    contexts = list_value(r[context_column]); require(len(contexts) == length, "Currency context length differs")
                    for eid, context in zip(witness, contexts):
                        for key, value in context.items(): require(edge_index.loc[eid, key] == value, "Witness currency context differs")
                node_buckets = sum(int(r[prefix+"positive_undirected_nodes_distance_"+k+"_7d"]) for k in ("1", "2", "3", "4plus"))
                edge_buckets = sum(int(r[prefix+"positive_undirected_edges_distance_"+k+"_7d"]) for k in ("1", "2", "3", "4plus"))
                present = bool(r[prefix+"present_in_positive_graph_7d"])
                require(present == bool(incident.any()), "Positive graph anchor presence differs")
                require(node_buckets+int(present) == r[prefix+"positive_component_nodes_7d"] and edge_buckets == r[prefix+"positive_component_unique_edges_7d"],
                        "Component/distance bucket conservation differs")
                witness_edges += length; nonempty += int(length > 0)
            require(bool(r["prior_positive_any_7d"]) == bool(union7), "7d prior-positive flag differs")
            require(bool(r["src_temporal_walk_ge2_7d"]) == (independent_max[int(r["src"])] >= 2), "Primary walk flag differs")
            require(int(r["endpoint_positive_incident_unique_7d"]) == len(union7), "Full-neighbor 7d count differs from independent scan")
            require(int(r["endpoint_positive_incident_unique_72h"]) == len(union72), "Full-neighbor 72h count differs from independent scan")
            require(list_value(r["positive_incident_ids_72h"]) == sorted(union72), "Positive graph 72h evidence differs")
            require(set(list_value(r["full_positive_direct_ids_72h"])) == union72, "Two full 72h evidence implementations disagree")
            for witness_key, flag_key in (("src_positive_walk2_witness_72h", "src_positive_walk_ge2_72h"),
                                          ("sampled_walk2_witness", "src_positive_walk_ge2_sampled")):
                if witness_key in r:
                    witness = list_value(r[witness_key])
                    require(len(witness) == (2 if bool(r[flag_key]) else 0), "72h length-two witness flag differs")
                    verify_witness(witness, edge_index, cutoff, int(r["src"]), target, 3*DAY)
            require(not bool(r["direct_positive_any_sampled"]) or bool(union72), "Sampled direct evidence exceeds full 72h evidence")
            require(not bool(r["src_positive_walk_ge2_sampled"]) or bool(r["src_positive_walk_ge2_72h"]), "Sampled walk exceeds full 72h walk evidence")
            positions = np.arange(flo, fhi)[all_src[flo:fhi] == int(r["dst"])]
            fids, ftimes = all_ids[positions].tolist(), all_times[positions].tolist()
            require(list_value(r["future_receiver_positive_out_edge_ids_observed_72h"]) == fids, "Future observation IDs differ")
            require(list_value(r["future_receiver_positive_out_timestamp_us_observed_72h"]) == ftimes, "Future observation timestamps differ")
            require(int(r["future_receiver_positive_out_edges_observed_72h"]) == len(fids), "Future observation count differs")
            require(int(r["future_receiver_positive_out_unique_destination_accounts_observed_72h"]) == len(set(all_dst[positions])), "Future destination count differs")
            gap = r["future_receiver_positive_out_first_gap_us_observed_72h"]
            require(pd.isna(gap) if not ftimes else gap == ftimes[0]-cutoff, "Future first gap differs")
            require(bool(r["future_window_complete_72h"]) == (END >= cutoff+3*DAY), "Future censor flag differs")
            require(int(r["future_observed_until_exclusive_us"]) == future_end and int(r["future_observed_duration_us"]) == future_end-cutoff,
                    "Future observation bounds differ")
            completed += 1
            if completed % 100 == 0 or completed == len(detail):
                progress("historical_witnesses", completed, len(detail), distinct_cutoffs=cutoffs,
                         verified_witness_edges=witness_edges)
    return dict(targets=completed, distinct_cutoffs=cutoffs, endpoint_witnesses=completed*2,
                nonempty_endpoint_witnesses=nonempty, verified_witness_edge_occurrences=witness_edges,
                independent_max_length_method="Per-node strict-time step functions and bisect_left, cached per cutoff; endpoint-only queries",
                direct_72h_evidence="Positive graph, full-neighbor audit, independent positive-edge scan: exact ID equality",
                static_graph_scope="Presence and bucket/component conservation checked; undirected BFS not independently rerun")


def verify_effects(root, detail, pairs):
    stats = read(root / "matched_statistics.json")
    require(stats["primary_test_family_size"] == 6 and len(stats["tests"]) == 6, "Primary test family differs")
    data = detail.set_index("edge_id")
    expected_tests = {(c, f) for c in ("ACH_FN", "G_FN") for f in FEATURES}
    require({(t["cohort"], t["feature"]) for t in stats["tests"]} == expected_tests, "Primary test identities differ")
    for test in stats["tests"]:
        p, feature = pairs[test["cohort"]].copy(), test["feature"]
        require(test["feature_column"] == feature, "Primary feature remapped unexpectedly")
        values = data[feature]
        require(values.isin([0, 1, False, True]).all(), "Primary feature is not binary")
        p["control"] = values.loc[p.control_edge_id].to_numpy(float)*p.control_weight
        control = p.groupby("case_edge_id").control.sum()
        case = values.loc[control.index].astype(float)
        delta = case-control
        close(test["case_rate"], case.mean(), "Case feature rate differs")
        close(test["matched_control_rate"], control.mean(), "Weighted control feature rate differs")
        close(test["effect_rate_difference"], delta.mean(), "Matched effect differs")
        case_payer = data.loc[delta.index, "src"]
        close(test["equal_case_payer_mean_difference"], delta.groupby(case_payer).mean().mean(), "Equal payer effect differs")
        require(test["matched_cases"] == len(case) and test["unique_controls"] == p.control_edge_id.nunique(), "Test unit counts differ")
        require(test["independent_clusters"] == p.dependency_cluster.nunique(), "Test dependence clusters differ")
    raw = [t["permutation_p"] for t in stats["tests"]]
    adjusted = [None]*6
    running = 0
    for rank, index in enumerate(sorted(range(6), key=lambda k: 1 if raw[k] is None else raw[k])):
        running = max(running, (6-rank)*(1 if raw[index] is None else raw[index]))
        adjusted[index] = None if raw[index] is None else min(1., running)
    for test, expected in zip(stats["tests"], adjusted):
        if expected is None: require(test["holm6_p"] is None, "Unavailable test Holm slot differs")
        else: close(test["holm6_p"], expected, "Holm family-six adjustment differs")
        require(test["holm6_below_005"] == (None if expected is None else expected < .05), "Holm decision differs")
    return dict(tests=6, weighted_effects="independently recomputed", holm6="independently recomputed",
                limitation="Bootstrap and sign-flip draws are hash-bound producer outputs; not independently rerun here")


def verify(project, root):
    project, root = project.resolve(), root.resolve()
    require(root.is_relative_to(project), "Output root must stay in the project")
    progress = Progress()
    manifest = read(root / "run_manifest.json")
    status = read(root / "audit_status.json")
    require(manifest["status"] == "completed" and status["status"] == "completed", "Audit has not completed; do not verify live stage outputs")
    manifest_sha = sha(root / "run_manifest.json")
    script_sha = sha(Path(__file__))
    destination = root / "verification.json"
    if destination.exists():
        old = read(destination)
        require(old["status"] == "passed" and old["run_manifest_sha256"] == manifest_sha and old["verifier_sha256"] == script_sha,
                "Existing independent verification has different identity; preserve it")
        for relative, expected in old["verified_output_sha256"].items():
            require(sha(safe_path(root, relative)) == expected, "Previously verified output changed: " + relative)
        progress("verified_existing", output_sha256=sha(destination))
        return old
    progress("output_hashes", 0, len(manifest["output_sha256"]))
    verified = {}
    for index, (name, expected) in enumerate(manifest["output_sha256"].items(), 1):
        path = safe_path(root, name)
        require(path.is_file() and sha(path) == expected, "Output SHA differs: " + name)
        verified[name] = expected
        if index % 10 == 0: progress("output_hashes", index, len(manifest["output_sha256"]))
    inputs = read(root / "audit_inputs.json")
    require(inputs == manifest["inputs"], "Run/audit input identity differs")
    for name, expected in inputs["modules"].items():
        require(sha(safe_path(project / "scripts", name)) == expected, "Producer module changed: " + name)
    cm, lm = read(root / "candidate_manifest.json"), read(root / "labels_manifest.json")
    require(inputs["candidates_sha256"] == cm["candidate_sha256"] == verified["candidates.parquet"], "Candidate SHA bindings differ")
    require(inputs["positive_edges_sha256"] == lm["positive_edges"]["sha256"] == verified["positive_edges.parquet"], "Positive SHA bindings differ")
    require(inputs["targets_sha256"] == verified["targets.parquet"], "Target SHA binding differs")
    small_sources = {}
    for name, expected in lm["source_sha256"].items():
        if not name.endswith(".json"):
            continue
        path = safe_path(project, name)
        require(path.stat().st_size < 10 * 1024**2, "Unexpectedly large metadata source")
        require(sha(path) == expected, "Frozen metadata source SHA differs: " + name)
        small_sources[path.relative_to(project).as_posix()] = expected
    evaluation = project / "result/gnn_tuning_v1/evaluations/lr_only"
    for name in ("config", "metrics"):
        expected = cm["prediction_audit"][name + "_sha256"]
        path = evaluation / (name + ".json")
        require(sha(path) == expected, "Candidate original evaluation identity differs: " + name)
        small_sources[path.relative_to(project).as_posix()] = expected
    candidates, targets, positives, detail = (pd.read_parquet(root / n) for n in
        ("candidates.parquet", "targets.parquet", "positive_edges.parquet", "transaction_audit.parquet"))
    require(len(detail) == len(targets) == status["targets"], "Final/target/status counts differ")
    require(not detail.edge_id.duplicated().any() and set(detail.edge_id) == set(targets.edge_id), "Final target IDs differ")
    aligned = detail.set_index("edge_id").loc[targets.edge_id].reset_index()
    pd.testing.assert_frame_equal(aligned[targets.columns], targets.reset_index(drop=True), check_exact=True, check_dtype=False)
    stage_frames, provenance = [], []
    for name in STAGES:
        audit = read(root / (name + ".audit.json"))
        require(audit["output_sha256"] == verified[name + ".parquet"] and audit["rows"] == len(targets),
                "Stage audit/input/count identity differs: " + name)
        provenance.append(verify_stage_inputs(project, root, name, audit, inputs, verified))
        frame = pd.read_parquet(root / (name + ".parquet"))
        require(not frame.edge_id.duplicated().any() and set(frame.edge_id) == set(targets.edge_id), "Stage target coverage differs: " + name)
        stage_frames.append(frame)
    merged = stage_frames[0].merge(stage_frames[1], on="edge_id", validate="one_to_one").merge(stage_frames[2], on="edge_id", validate="one_to_one")
    expected_detail = targets.merge(merged.drop(columns=[c for c in merged if c in targets and c != "edge_id"]), on="edge_id", validate="one_to_one")
    pd.testing.assert_frame_equal(detail, expected_detail, check_exact=True, check_dtype=False)
    progress("identity_verified", targets=len(targets), positive_edges=len(positives))
    verify_cohorts(candidates, targets, positives)
    label_result = verify_labels(project, lm, positives, candidates, progress)
    pairs, matching_result = verify_matching(root, candidates, targets, progress)
    history_result = verify_histories(positives, detail, progress)
    effects_result = verify_effects(root, detail, pairs)
    actual_cohorts = {str(k): int(v) for k, v in targets.cohort.value_counts().items()}
    require(status["cohort_counts"] == actual_cohorts, "Final status cohort totals differ")
    summary = read(root / "cohort_summary.json")
    require(set(summary) == set(actual_cohorts), "Cohort summary coverage differs")
    for cohort, group in detail.groupby("cohort"):
        item = summary[cohort]
        require(item["transactions"] == len(group) and item["payers"] == group.src.nunique() and item["receivers"] == group.dst.nunique(),
                "Cohort summary count differs")
        for flag, count in item["counts"].items(): require(count == int(group[flag].sum()), "Cohort flag summary differs")
    # Detect concurrent rewrites, not just identity at the beginning of the run.
    require(sha(root / "run_manifest.json") == manifest_sha, "Run manifest changed during verification")
    for name, expected in verified.items():
        require(sha(safe_path(root, name)) == expected, "Output changed during verification: " + name)
    result = dict(status="passed", verified_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.monotonic()-progress.started, 3), verifier_sha256=script_sha,
                  run_manifest_sha256=manifest_sha, verified_output_sha256=verified,
                  source_module_sha256=inputs["modules"], cohort_counts=actual_cohorts,
                  independently_verified_small_source_sha256=small_sources,
                  stage_provenance_verification=provenance,
                  global_predeclared_positive_cohort_counts=EXPECTED, label_verification=label_result,
                  matching_verification=matching_result, historical_verification=history_result,
                  statistics_verification=effects_result,
                  scope=["Retrospective final-label diagnostics only; no training or Test read.",
                         "Every longest endpoint temporal-walk witness checked, with independent maximum-length DP.",
                         "Direct 72h positive evidence exact IDs agree across both producers and independent scan.",
                         "Matching controls, amount calipers, strata, order, 1/k weights and payer dependence independently rebuilt.",
                         "No independent full unlabeled graph/BFS/sampling-plan or bootstrap replay; those outputs remain hash-bound."])
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    progress("completed", output=str(destination), output_sha256=sha(destination))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("result/gnn_chain_audit_20261003"))
    args = parser.parse_args()
    verify(args.project, args.output_root if args.output_root.is_absolute() else args.project / args.output_root)
