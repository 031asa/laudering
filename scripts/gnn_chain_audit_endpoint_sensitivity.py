"""Additive sensitivity audit of fixed matches; no rematching or new primary tests."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_chain_audit_statistics import PRIMARY_FEATURES, COHORTS, _cluster_inference, holm_adjust


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "result/gnn_chain_audit_20261003"
DESCRIPTIVE_FEATURES = (
    "endpoint_positive_density_7d", "endpoint_positive_incident_unique_7d",
    "mixed_2hop_positive_count_7d", "src_temporal_walk_max_history_edges_7d",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weighted_distribution(values, weights):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if values.shape != weights.shape or not np.isfinite(weights).all() or (weights <= 0).any() or np.isinf(values).any():
        raise ValueError("Invalid descriptive values or weights")
    finite = np.isfinite(values)
    result = dict(rows=len(values), nonmissing_rows=int(finite.sum()), missing_rows=int((~finite).sum()),
                  total_weight=float(weights.sum()), missing_weight=float(weights[~finite].sum()),
                  quantile_method="inverse weighted empirical CDF, no interpolation")
    if not finite.any():
        return dict(result, mean=None, min=None, max=None, quantiles=None)
    v, w = values[finite], weights[finite]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cumulative = np.cumsum(w)
    quantiles = {str(q): float(v[min(np.searchsorted(cumulative, q*cumulative[-1], side="left"), len(v)-1)])
                 for q in (.25, .5, .75, .9, .95)}
    return dict(result, mean=float(np.dot(v, w)/w.sum()), min=float(v[0]), max=float(v[-1]), quantiles=quantiles)


def continuous_descriptions(feature_index, pairs, cohort):
    """Exploratory distributions only; no additional p-values or primary tests."""
    case_ids = pairs.case_edge_id.drop_duplicates().to_numpy()
    rows = []
    for name in DESCRIPTIVE_FEATURES:
        a = feature_index.loc[case_ids, name].to_numpy(float)
        b = feature_index.loc[pairs.control_edge_id, name].to_numpy(float)
        if (a[np.isfinite(a)] < 0).any() or (b[np.isfinite(b)] < 0).any():
            raise ValueError("Negative chain count/density")
        p = pairs.copy()
        p["case_value"] = feature_index.loc[p.case_edge_id, name].to_numpy(float)
        p["control_value"] = b
        p["weighted_control"] = b*p.control_weight.to_numpy(float)
        p["valid_control"] = np.isfinite(b)
        c = p.groupby("case_edge_id").agg(case_value=("case_value", "first"),
                control_value=("weighted_control", "sum"), all_controls_valid=("valid_control", "all"))
        complete = np.isfinite(c.case_value) & c.all_controls_valid
        delta = c.loc[complete, "case_value"] - c.loc[complete, "control_value"]
        rows.append(dict(cohort=cohort, feature=name,
            cases=weighted_distribution(a, np.ones(len(a))),
            matched_controls=weighted_distribution(b, pairs.control_weight.to_numpy(float)),
            paired_complete_cases=int(complete.sum()), paired_incomplete_cases=int((~complete).sum()),
            matched_mean_difference=float(delta.mean()) if len(delta) else None,
            paired_difference_distribution=weighted_distribution(delta.to_numpy(float), np.ones(len(delta))),
            interpretation="Descriptive/exploratory only. Each case weight one, controls within case share 1/k. Reused controls remain repeated weighted references, not independent sample counts; missing density is not zero. Mean difference excludes a case if it or any assigned control has missing value."))
    return rows


def endpoint_clusters(pairs, identities):
    """Conservatively join all payer/receiver accounts in each fixed matched set."""
    if identities.edge_id.duplicated().any():
        raise ValueError("Duplicate transaction identity")
    ids = identities.set_index("edge_id")
    if not np.isin(np.r_[pairs.case_edge_id, pairs.control_edge_id], ids.index).all():
        raise ValueError("Missing endpoint identities")
    parent = {}

    def find(value):
        value = int(value)
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in pairs.itertuples(index=False):
        a, b = ids.loc[row.case_edge_id], ids.loc[row.control_edge_id]
        if int(a.src) != int(row.case_payer) or int(b.src) != int(row.control_payer):
            raise ValueError("Pair payer identity does not match frozen target metadata")
        for account in (a.dst, b.src, b.dst):
            union(a.src, account)
    result = pairs.copy()
    result["endpoint_dependency_cluster"] = [find(int(x)) for x in result.case_payer]
    # A stable case-based identifier makes the reported component inventory simple.
    case = result[["case_edge_id", "endpoint_dependency_cluster"]].drop_duplicates()
    if case.case_edge_id.duplicated().any():
        raise AssertionError("A matched set spans multiple endpoint components")
    rename = case.groupby("endpoint_dependency_cluster").case_edge_id.min().to_dict()
    result["endpoint_dependency_cluster"] = result.endpoint_dependency_cluster.map(rename)
    sizes = result.groupby("endpoint_dependency_cluster").case_edge_id.nunique()
    report = dict(clusters=int(len(sizes)), largest_cluster_cases=int(sizes.max()) if len(sizes) else 0,
                  cluster_case_counts={str(k): int(v) for k, v in sizes.items()},
                  construction="Transitive closure of case.src, case.dst, control.src, control.dst within fixed matches; raw node identity is shared across payer/receiver roles")
    return result, report


def sensitivity(feature_df, fixed_pairs, primary_statistics, *, seed=20261003, n_boot=5000, n_perm=19999):
    if set(fixed_pairs) != set(COHORTS):
        raise ValueError("Expected fixed ACH_FN and G_FN matches")
    f = feature_df.set_index("edge_id")
    if f.index.duplicated().any():
        raise ValueError("Duplicate feature identity")
    main = {(x["cohort"], x["feature"]): x for x in primary_statistics["tests"]}
    tests, per_payer, continuous = [], [], []
    components = {}
    for ci, cohort in enumerate(COHORTS):
        pairs, components[cohort] = endpoint_clusters(fixed_pairs[cohort], feature_df[["edge_id", "src", "dst"]])
        if not pairs.empty:
            continuous.extend(continuous_descriptions(f, pairs, cohort))
        for fi, feature in enumerate(PRIMARY_FEATURES):
            original = main[cohort, feature]
            if pairs.empty:
                tests.append(dict(cohort=cohort, feature=feature, matched_cases=0,
                                  effect_rate_difference=None, ci95=None, permutation_p=None))
                continue
            p = pairs.copy()
            for prefix in ("case", "control"):
                value = f.loc[p[f"{prefix}_edge_id"], feature]
                if value.isna().any() or not value.isin([0, 1, False, True]).all():
                    raise ValueError("Primary features must remain complete binary values")
                p[f"{prefix}_value"] = value.to_numpy(float)
            p["weighted_control"] = p.control_value * p.control_weight
            case = p.groupby("case_edge_id").agg(case_payer=("case_payer", "first"),
                    case_value=("case_value", "first"), control_value=("weighted_control", "sum"),
                    endpoint_cluster=("endpoint_dependency_cluster", "first"), weights=("control_weight", "sum"))
            if not np.allclose(case.weights, 1, rtol=0, atol=1e-12):
                raise ValueError("Controls do not retain total weight one per case")
            case["delta"] = case.case_value - case.control_value
            effect = float(case.delta.mean())
            if len(case) != original["matched_cases"] or not math_close(effect, original["effect_rate_difference"]):
                raise AssertionError("Sensitivity must not change matched cases or effect estimate")
            for key, value in [("case_rate", case.case_value.mean()), ("matched_control_rate", case.control_value.mean())]:
                if not math_close(float(value), original[key]):
                    raise AssertionError(f"Sensitivity changed {key}")
            infer = _cluster_inference(case.delta.to_numpy(), case.endpoint_cluster.to_numpy(),
                    seed=np.random.SeedSequence([seed, ci, fi]), n_boot=n_boot, n_perm=n_perm)
            infer["uncertainty_unit"] = "connected components of all case/control payer AND receiver identities, including shared accounts across roles and reused controls"
            tests.append(dict(cohort=cohort, feature=feature, matched_cases=len(case),
                 case_rate=float(case.case_value.mean()), matched_control_rate=float(case.control_value.mean()),
                 effect_rate_difference=effect, primary_payer_cluster_ci95=original["ci95"],
                 primary_payer_cluster_p=original["permutation_p"], primary_payer_cluster_holm6_p=original["holm6_p"],
                 primary_independent_clusters=original.get("independent_clusters"), **infer))
            if cohort == "G_FN":
                for payer, group in case.groupby("case_payer"):
                    pair = pairs[pairs.case_payer == payer]
                    per_payer.append(dict(feature=feature, payer_node_id=int(payer), cases=len(group),
                        case_share=float(len(group)/len(case)), unique_controls=int(pair.control_edge_id.nunique()),
                        pair_rows=len(pair), case_rate=float(group.case_value.mean()),
                        matched_control_rate=float(group.control_value.mean()),
                        rate_difference=float(group.delta.mean()),
                        contribution_to_overall_rate_difference=float(group.delta.sum()/len(case))))
    for test, adjusted in zip(tests, holm_adjust([x["permutation_p"] for x in tests])):
        test["sensitivity_holm6_p"] = adjusted
    return dict(status="completed", analysis_role="Prespecified conservative endpoint-dependence sensitivity for the existing six comparisons; not new primary tests",
                seed=seed, components=components, tests=tests, g_per_payer=per_payer,
                exploratory_continuous_descriptions=continuous,
                invariants=dict(no_rematching=True, no_new_primary_features=True,
                                unchanged_case_control_sets_weights_and_effects=True),
                limitations=["Target endpoint components still need not capture dependence through shared earlier graph neighbors.",
                    "G has only 15 payer accounts and unequal case counts; treat nominal significance cautiously.",
                    "ACH matches are all cross-payer: numerical balance does not remove unobserved account differences.",
                    "Already-used Validation, retrospective label-oracle chain evidence, association only; not deployable causal risk evidence."])


def math_close(a, b):
    return bool(np.isclose(a, b, rtol=0, atol=1e-12))


def main():
    output = OUT / "endpoint_cluster_sensitivity.json"
    if output.exists():
        raise FileExistsError("Preserve earlier sensitivity result")
    manifest_path = OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise RuntimeError("Main audit is not complete")
    for name, expected in manifest["inputs"]["modules"].items():
        if sha(PROJECT / "scripts" / name) != expected:
            raise ValueError(f"Main audit module changed: {name}")
    names = ["transaction_audit.parquet", "matched_statistics.json", "ACH_FN_pairs.csv", "G_FN_pairs.csv",
             "ACH_FN_matching.json", "G_FN_matching.json"]
    hashes = {}
    for name in names:
        hashes[name] = sha(OUT / name)
        if hashes[name] != manifest["output_sha256"][name]:
            raise ValueError(f"Completed audit output changed: {name}")
    hashes[manifest_path.name] = sha(manifest_path)
    features = pd.read_parquet(OUT / "transaction_audit.parquet", columns=["edge_id", "src", "dst"] + list(PRIMARY_FEATURES) + list(DESCRIPTIVE_FEATURES))
    pairs = {c: pd.read_csv(OUT / f"{c}_pairs.csv") for c in COHORTS}
    primary = json.loads((OUT / "matched_statistics.json").read_text(encoding="utf-8"))
    result = sensitivity(features, pairs, primary)
    for name, digest in hashes.items():
        if sha(OUT / name) != digest:
            raise ValueError(f"Input changed during sensitivity: {name}")
    result["audit"] = dict(created_at=datetime.now(timezone.utc).isoformat(), inputs_sha256=hashes,
                           script_sha256=sha(Path(__file__)), statistics_module_sha256=sha(PROJECT/"scripts/gnn_chain_audit_statistics.py"))
    if shutil.disk_usage(OUT).free < 20 * 1024**3:
        raise RuntimeError("Less than 20 GiB available")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), sha256=sha(output), components=result["components"]), ensure_ascii=True))


if __name__ == "__main__":
    main()
