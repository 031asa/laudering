"""Read-only matching/statistics for the predeclared Validation chain audit.

No training, feature extraction, file writes, or threshold selection occurs here.
Each case has weight one; its (at most three) controls share that weight. Controls
may be reused, but account-linked matching sets are one dependence component for
inference. These observational comparisons cannot establish causation.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd


PRIMARY_FEATURES = (
    "prior_positive_any_7d",
    "src_temporal_walk_ge2_7d",
    "mixed_2hop_positive_any_7d",
)
COHORTS = ("ACH_FN", "G_FN")
EXACT_COLUMNS = ("payment_format", "payment_currency", "receiving_currency", "date",
                 "support_src_7d", "support_dst_7d")
PAIR_COLUMNS = ("cohort", "case_edge_id", "control_edge_id", "case_payer", "control_payer",
                "same_payer", "distance", "control_weight", "controls_for_case",
                "dependency_cluster")


def _fisher_two_sided(a, b, c, d):
    """Probability-ordering Fisher exact p, without an optional SciPy dependency."""
    n, successes, population = a + b, a + c, a + b + c + d
    if population == 0:
        return 1.0

    def lc(total, take):
        return math.lgamma(total + 1) - math.lgamma(take + 1) - math.lgamma(total-take + 1)

    denominator = lc(population, n)

    def lp(x):
        return lc(successes, x) + lc(population-successes, n-x) - denominator

    observed = lp(a)
    probabilities = [math.exp(value) for x in range(max(0, n-(population-successes)), min(n, successes)+1)
                     if (value := lp(x)) <= observed + 1e-10]
    return min(1.0, math.fsum(probabilities))


def _support(values):
    x = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if np.isinf(x).any() or np.any(np.isfinite(x) & ((x < 0) | (x != np.floor(x)))):
        raise ValueError("History counts must be nonnegative integers or missing")
    return np.select([~np.isfinite(x), x == 0, x < 5],
                     ["not_applicable", "0", "1-4"], default=">=5")


def _flag(values):
    mapping = {True: True, False: False, "True": True, "False": False,
               "true": True, "false": False, 1: True, 0: False}
    if values.isna().any() or not values.isin(list(mapping)).all():
        raise ValueError("Uniform-negative indicator is not Boolean")
    return values.map(mapping).astype(bool)


def _prepare(data, *, uniform_negative_pool_verified=False):
    needed = ["edge_id", "label", "src", "date", "payment_format", "payment_currency",
              "receiving_currency", "amount_paid", "v2_src_count_7d", "v2_dst_count_7d",
              "r_pair_fwd_count_72h", "v2_dst_current_to_q90_7d"]
    missing = set(needed) - set(data.columns)
    if missing:
        raise ValueError(f"Missing matching columns: {sorted(missing)}")
    extra = [c for c in ["from_bank", "outcome_lr_only", "uniform_negative_control",
                         "is_uniform_negative_control"] if c in data]
    d = data[needed + extra].copy()
    for col in ["edge_id", "src"]:
        x = pd.to_numeric(d[col], errors="raise")
        if x.isna().any() or not np.isfinite(x.to_numpy(dtype=float)).all() or (x < 0).any() or (x != np.floor(x)).any():
            raise ValueError(f"Invalid integer identity: {col}")
        d[col] = x.astype(np.int64)
    if d.edge_id.duplicated().any():
        raise ValueError("Duplicate edge_id")
    if not d.label.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    for col in ["payment_format", "payment_currency", "receiving_currency", "date"]:
        if d[col].isna().any() or (d[col].astype(str).str.len() == 0).any():
            raise ValueError(f"Missing exact-match identity: {col}")
        d[col] = d[col].astype(str)
    if not d.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("date must retain source YYYY-MM-DD; no timezone conversion")
    pd.to_datetime(d.date, format="%Y-%m-%d", errors="raise")
    d["amount"] = pd.to_numeric(d.amount_paid, errors="raise").astype(float)
    if not np.isfinite(d.amount).all() or (d.amount < 0).any():
        raise ValueError("Matching amount must be finite and nonnegative")
    for side in ("src", "dst"):
        col = f"v2_{side}_count_7d"
        d[col] = pd.to_numeric(d[col], errors="raise").astype(float)
        d[f"support_{side}_7d"] = _support(d[col])
    for col in ["r_pair_fwd_count_72h", "v2_dst_current_to_q90_7d"]:
        d[col] = pd.to_numeric(d[col], errors="raise").astype(float)
        if np.isinf(d[col]).any():
            raise ValueError(f"Infinite matching value: {col}")
    d["ach_condition"] = ((d.payment_format == "ACH") &
                          (d.r_pair_fwd_count_72h == 0) &
                          (d.v2_dst_current_to_q90_7d > 1))
    flag_cols = [c for c in ["uniform_negative_control", "is_uniform_negative_control"] if c in d]
    if flag_cols:
        uniform = _flag(d[flag_cols[0]])
        if len(flag_cols) == 2 and not uniform.equals(_flag(d[flag_cols[1]])):
            raise ValueError("Conflicting uniform-negative indicators")
        if ((d.label == 1) & uniform).any():
            raise ValueError("A positive is marked as a uniform negative")
    elif uniform_negative_pool_verified:
        uniform = d.label == 0
    else:
        raise ValueError("A verified uniform-negative indicator/declaration is required")
    d["eligible_negative"] = (d.label == 0) & uniform
    d = d.set_index("edge_id", drop=False)
    d.index.name = "_edge_index"
    return d


def dependency_clusters(pairs):
    """Join case/control payer identities, including cross-case control reuse.

    Shared *accounts*, not just repeated transaction rows, induce dependence.
    Hence a reused control can never inflate the number of independent clusters.
    """
    p = pairs.copy()
    if p.empty:
        p["dependency_cluster"] = pd.Series(dtype="int64")
        return p, dict(clusters=0, largest_cluster_cases=0, cluster_case_counts={})
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    control_owner = {}
    for r in p.itertuples(index=False):
        union(int(r.case_payer), int(r.control_payer))
        old = control_owner.setdefault(int(r.control_edge_id), int(r.control_payer))
        if old != int(r.control_payer):
            raise ValueError("One control transaction has inconsistent payer identity")
    p["dependency_cluster"] = [find(int(x)) for x in p.case_payer]
    cases = p[["case_edge_id", "dependency_cluster"]].drop_duplicates()
    if cases.case_edge_id.duplicated().any():
        raise ValueError("A case spans more than one dependence component")
    sizes = cases.groupby("dependency_cluster").size()
    return p, dict(clusters=int(len(sizes)), largest_cluster_cases=int(sizes.max()),
                   cluster_case_counts={str(k): int(v) for k, v in sizes.items()})


def _balance(d, pairs):
    if pairs.empty:
        return dict(status="no_matches", numeric=[], categorical=[])
    ids = pairs.case_edge_id.drop_duplicates().to_numpy()
    case = d.loc[ids]
    controls = d.loc[pairs.control_edge_id]
    w = pairs.control_weight.to_numpy(float, copy=True)
    w /= w.sum()
    numeric = []
    for col in ["amount", "v2_src_count_7d", "v2_dst_count_7d"]:
        a = np.log1p(case[col].to_numpy(float))
        b = np.log1p(controls[col].to_numpy(float))
        ma, mb = float(a.mean()), float(np.dot(w, b))
        va, vb = float(np.mean((a - ma) ** 2)), float(np.dot(w, (b - mb) ** 2))
        scale = math.sqrt((va + vb) / 2)
        smd = (ma - mb) / scale if scale > 0 else (0.0 if ma == mb else None)
        numeric.append(dict(feature=f"log1p({col})", case_mean=ma,
                            weighted_control_mean=mb, standardized_mean_difference=smd,
                            abs_smd_exceeds_point1=smd is None or abs(smd) > .1))
    categorical = []
    for col in EXACT_COLUMNS:
        for value in sorted(set(case[col]) | set(controls[col])):
            a = float((case[col] == value).mean())
            b = float(np.dot(w, (controls[col] == value).to_numpy(float)))
            categorical.append(dict(feature=col, level=value, case_fraction=a,
                                    weighted_control_fraction=b, difference=a - b))
    mismatch = sum(int(np.count_nonzero(d.loc[pairs.case_edge_id, col].to_numpy() !=
                                       controls[col].to_numpy())) for col in EXACT_COLUMNS)
    if mismatch:
        raise AssertionError("Exact-match covariates differ")
    return dict(status="computed", weighting="One per case; each control receives 1/k within its case",
                numeric=numeric, categorical=categorical, exact_covariate_mismatches=mismatch,
                same_payer_pair_fraction=float(pairs.same_payer.mean()),
                same_payer_case_weighted_fraction=float(np.dot(w, pairs.same_payer.to_numpy(float))))


def match_controls(data, case_ids, cohort, max_controls=3, *,
                   uniform_negative_pool_verified=False):
    """Deterministic exact-stratum matching, factor-two paid-amount caliper.

    ACH_FN controls satisfy the identical ACH/pair-zero/receiver-P90 condition;
    same-payer matches rank first. G_FN requires the identical payer. Neither
    model scores, chain features, nor historical labels enter matching.
    """
    if cohort not in COHORTS or max_controls not in (1, 2, 3):
        raise ValueError("Expected ACH_FN/G_FN and 1..3 controls per case")
    d = _prepare(data, uniform_negative_pool_verified=uniform_negative_pool_verified)
    ids = np.asarray(list(case_ids), dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or not np.isin(ids, d.index).all():
        raise ValueError("Duplicate or unknown case IDs")
    cases = d.loc[np.sort(ids)]
    if not (cases.label == 1).all():
        raise ValueError("Cases must be labeled positives")
    if "outcome_lr_only" in cases and not (cases.outcome_lr_only == "FN").all():
        raise ValueError("Target case IDs must be current-P80 false negatives")
    if cohort == "ACH_FN" and not cases.ach_condition.all():
        raise ValueError("An ACH case does not satisfy the predeclared condition")
    if cohort == "G_FN":
        if "from_bank" not in cases or not ((cases.from_bank.astype(str) == "70") &
                                             (cases.payment_format != "ACH")).all():
            raise ValueError("G cases must use raw payer bank 70 and non-ACH")
    pool = d[d.eligible_negative & (d.ach_condition if cohort == "ACH_FN" else True)]
    grouped = {key: value for key, value in pool.groupby(list(EXACT_COLUMNS), sort=False)}
    records, unmatched = [], []
    for r in cases.itertuples(index=False):
        if r.support_src_7d == "not_applicable" or r.support_dst_7d == "not_applicable":
            unmatched.append(dict(case_edge_id=int(r.edge_id), reason="missing_history_support",
                                  exact_stratum_candidates=0, caliper_candidates=0))
            continue
        key = tuple(getattr(r, c) for c in EXACT_COLUMNS)
        cand = grouped.get(key)
        if cand is None:
            unmatched.append(dict(case_edge_id=int(r.edge_id), reason="no_exact_stratum",
                                  exact_stratum_candidates=0, caliper_candidates=0))
            continue
        cand = cand if cohort == "ACH_FN" else cand[cand.src == r.src]
        n_exact = len(cand)
        if not n_exact:
            unmatched.append(dict(case_edge_id=int(r.edge_id), reason="no_same_payer_in_stratum",
                                  exact_stratum_candidates=0, caliper_candidates=0))
            continue
        # Zero is matched only to zero: its multiplicative caliper is undefined.
        within = (cand.amount == 0) if r.amount == 0 else ((cand.amount >= r.amount / 2) &
                                                                          (cand.amount <= r.amount * 2))
        cand = cand[within].copy()
        if cand.empty:
            unmatched.append(dict(case_edge_id=int(r.edge_id), reason="outside_amount_caliper",
                                  exact_stratum_candidates=n_exact, caliper_candidates=0))
            continue
        amount_gap = np.zeros(len(cand)) if r.amount == 0 else np.abs(np.log(cand.amount.to_numpy() / r.amount)) / math.log(2)
        count_gap = sum(np.abs(np.log1p(cand[f"v2_{side}_count_7d"].to_numpy()) -
                              math.log1p(getattr(r, f"v2_{side}_count_7d"))) for side in ("src", "dst")) / 2
        cand["distance"] = amount_gap + count_gap
        cand["same_payer"] = cand.src == r.src
        picked = cand.sort_values(["same_payer", "distance", "edge_id"],
                                  ascending=[False, True, True]).head(max_controls)
        for q in picked.itertuples(index=False):
            records.append(dict(cohort=cohort, case_edge_id=int(r.edge_id), control_edge_id=int(q.edge_id),
                                case_payer=int(r.src), control_payer=int(q.src), same_payer=bool(q.same_payer),
                                distance=float(q.distance), control_weight=1 / len(picked),
                                controls_for_case=len(picked)))
    pairs = pd.DataFrame(records, columns=PAIR_COLUMNS[:-1])
    pairs, dep = dependency_clusters(pairs)
    unmatched = pd.DataFrame(unmatched, columns=["case_edge_id", "reason", "exact_stratum_candidates", "caliper_candidates"])
    usage = pairs.control_edge_id.value_counts()
    counts = pairs.groupby("case_edge_id").size()
    matched = int(len(counts))
    if matched + len(unmatched) != len(cases):
        raise AssertionError("Case accounting failed")
    summary = dict(cohort=cohort, cases_total=len(cases), matched_cases=matched,
                   unmatched_cases=len(unmatched), matching_fraction=matched / len(cases) if len(cases) else None,
                   unique_controls=int(len(usage)), pair_rows=len(pairs),
                   controls_used_more_than_once=int((usage > 1).sum()), max_control_reuse=int(usage.max()) if len(usage) else 0,
                   cases_with_fewer_than_requested_controls=int((counts < max_controls).sum()),
                   controls_per_case={str(k): int(v) for k, v in counts.value_counts().items()},
                   case_payers_total=int(cases.src.nunique()), matched_case_payers=int(pairs.case_payer.nunique()),
                   control_payers=int(pairs.control_payer.nunique()),
                   negative_pool_rows=len(pool), dependency=dep,
                   unmatched_reasons=dict(Counter(unmatched.reason)),
                   matching_covariates=list(EXACT_COLUMNS) + ["payer priority/exact G", "paid amount factor 2", "continuous src/dst 7d counts"],
                   distance="abs(log(control_amount/case_amount))/log(2) + 0.5*sum(abs(log1p(control_count)-log1p(case_count)))",
                   forbidden_matching_inputs=["scores", "chain features", "historical labels"],
                   g_independence_warning="G has only 15 possible case payers in the full cohort; fewer may match" if cohort == "G_FN" else None)
    return dict(pairs=pairs, unmatched=unmatched, summary=summary, balance=_balance(d, pairs))


def _cluster_inference(delta, clusters, *, seed, n_boot, n_perm):
    if n_boot < 5000 or n_perm < 5000:
        raise ValueError("At least 5000 bootstrap and Monte Carlo permutation draws are required")
    x = pd.DataFrame(dict(delta=delta, cluster=clusters))
    g = x.groupby("cluster").delta.agg(["sum", "size", "mean"])
    k = len(g)
    info = dict(independent_clusters=k, largest_cluster_cases=int(g["size"].max()) if k else 0,
                equal_dependency_cluster_mean_difference=float(g["mean"].mean()) if k else None,
                uncertainty_unit="connected components of all case/control payer identities and reused controls",
                bootstrap_replicates=n_boot, permutation_replicates=n_perm,
                sign_flip_assumption="Dependence-component summed paired differences are symmetric under the null; observational association, not causation",
                few_cluster_warning=k < 20,
                inference_status="unavailable_single_component" if k < 2 else ("very_few_clusters_no_strong_conclusion" if k < 10 else "few_clusters_caution" if k < 20 else "computed"))
    if k < 2:
        return dict(info, ci95=None, permutation_p=None, permutation_method="unavailable")
    sums, sizes = g["sum"].to_numpy(float), g["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, 256):
        ix = rng.integers(0, k, size=(min(256, n_boot-start), k))
        boot[start:start+len(ix)] = sums[ix].sum(axis=1) / sizes[ix].sum(axis=1)
    ci = np.quantile(boot, [.025, .975]).tolist()
    observed = abs(float(sums.sum()))
    extreme = 0
    exact = k <= 18
    count = 2 ** k if exact else n_perm
    for start in range(0, count, 1024):
        num = min(1024, count-start)
        if exact:
            masks = np.arange(start, start+num, dtype=np.uint64)
            signs = (((masks[:, None] >> np.arange(k, dtype=np.uint64)) & 1).astype(np.int8) * 2 - 1)
        else:
            signs = rng.integers(0, 2, size=(num, k), dtype=np.int8) * 2 - 1
        extreme += int(np.count_nonzero(np.abs(signs @ sums) >= observed - 1e-12))
    p = extreme / count if exact else (extreme + 1) / (count + 1)
    return dict(info, ci95=ci, permutation_p=float(p),
                permutation_method="exact_two_sided_cluster_sign_flip" if exact else "Monte_Carlo_two_sided_cluster_sign_flip",
                permutation_replicates=count)


def holm_adjust(pvalues):
    """Holm across the whole predeclared family; unavailable tests retain a slot."""
    values = np.array([1.0 if p is None else float(p) for p in pvalues])
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid p-value")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(values)-rank) * values[i])
        adjusted[i] = min(1.0, running)
    return [None if p is None else float(adjusted[i]) for i, p in enumerate(pvalues)]


def run_primary_statistics(feature_df, match_results, feature_columns=None, *,
                           seed=20261003, n_boot=5000, n_perm=19999):
    """Return exactly 2 cohorts x 3 prespecified binary-feature comparisons.

    feature_columns maps each PRIMARY_FEATURES concept to an actual binary column.
    match_results is {"ACH_FN": match_controls(...), "G_FN": ...}.
    """
    columns = dict(zip(PRIMARY_FEATURES, PRIMARY_FEATURES)) if feature_columns is None else dict(feature_columns)
    if set(columns) != set(PRIMARY_FEATURES) or set(match_results) != set(COHORTS):
        raise ValueError("The primary family is exactly two cohorts times three features")
    if feature_df.edge_id.duplicated().any():
        raise ValueError("Duplicate feature edge_id")
    f = feature_df.set_index("edge_id")
    tests = []
    for ci, cohort in enumerate(COHORTS):
        pairs, dep = dependency_clusters(match_results[cohort]["pairs"])
        for fi, concept in enumerate(PRIMARY_FEATURES):
            column = columns[concept]
            test = dict(cohort=cohort, feature=concept, feature_column=column,
                        matching=match_results[cohort]["summary"], dependency=dep)
            if pairs.empty:
                tests.append(dict(test, matched_cases=0, effect_rate_difference=None,
                                  ci95=None, permutation_p=None, inference_status="no_matches"))
                continue
            ids = np.unique(np.r_[pairs.case_edge_id, pairs.control_edge_id])
            if column not in f or not np.isin(ids, f.index).all():
                raise ValueError(f"Incomplete feature coverage: {column}")
            v = f.loc[ids, column]
            if v.isna().any() or not v.isin([0, 1, False, True]).all():
                raise ValueError(f"Primary feature must be complete binary: {column}")
            p = pairs.copy()
            p["case_value"] = f.loc[p.case_edge_id, column].to_numpy(float)
            p["control_value"] = f.loc[p.control_edge_id, column].to_numpy(float)
            p["weighted_control"] = p.control_value * p.control_weight
            c = p.groupby("case_edge_id").agg(case_value=("case_value", "first"),
                    control_value=("weighted_control", "sum"), case_payer=("case_payer", "first"),
                    dependency_cluster=("dependency_cluster", "first"), weight_sum=("control_weight", "sum"))
            if not np.allclose(c.weight_sum, 1, rtol=0, atol=1e-12):
                raise ValueError("Control weights do not sum to one per case")
            delta = c.case_value.to_numpy() - c.control_value.to_numpy()
            infer = _cluster_inference(delta, c.dependency_cluster.to_numpy(),
                    seed=np.random.SeedSequence([seed, ci, fi]), n_boot=n_boot, n_perm=n_perm)
            unique_control = p.drop_duplicates("control_edge_id").control_value
            a, b = int(c.case_value.sum()), int(unique_control.sum())
            fisher = _fisher_two_sided(a, len(c)-a, b, len(unique_control)-b)
            payer_delta = pd.Series(delta).groupby(c.case_payer.to_numpy()).mean()
            tests.append(dict(test, matched_cases=len(c), unique_controls=len(unique_control),
                    case_rate=float(c.case_value.mean()), matched_control_rate=float(c.control_value.mean()),
                    effect_rate_difference=float(delta.mean()),
                    equal_case_payer_mean_difference=float(payer_delta.mean()),
                    effect_estimand="Transaction-weighted mean(case binary value - within-case mean of its matched controls)",
                    fisher_exploratory=dict(p=fisher, cases_with_feature=a,
                        unique_controls_with_feature=b, warning="Exploratory only: row independence is violated by payer dependence and control reuse; never the primary significance conclusion"), **infer))
    adjusted = holm_adjust([t["permutation_p"] for t in tests])
    for t, p in zip(tests, adjusted):
        t["holm6_p"] = p
        t["holm6_below_005"] = None if p is None else p < .05
    return dict(primary_test_family_size=6, seed=seed, tests=tests,
                multiplicity="Holm across the six prespecified dependence-cluster sign-flip tests; missing tests retain a slot",
                conclusion_limit="Matched observational association on already-used Validation; no causal claim or independent Test evidence. Too few dependence components or poor balance precludes a strong conclusion.")


def describe_ach_reference(data, feature_df=None, feature_columns=PRIMARY_FEATURES):
    """ACH FN versus same-condition TP, stratified by both 7d support bins.

    Descriptive only: this is not a matched/inferential comparison with TP.
    """
    if "outcome_lr_only" not in data:
        raise ValueError("Current global-P80 outcome labels are required")
    d = _prepare(data, uniform_negative_pool_verified=True)
    d = d[(d.label == 1) & d.ach_condition & d.outcome_lr_only.isin(["FN", "TP"])].copy()
    extra = []
    if feature_df is not None:
        if feature_df.edge_id.duplicated().any():
            raise ValueError("Duplicate feature IDs")
        f = feature_df.set_index("edge_id")
        for col in feature_columns:
            if not np.isin(d.edge_id, f.index).all():
                raise ValueError("Incomplete ACH reference feature coverage")
            d[col] = f.loc[d.edge_id, col].to_numpy()
            if d[col].isna().any() or not d[col].isin([0, 1, False, True]).all():
                raise ValueError("Reference feature must be complete binary")
            extra.append(col)
    rows = []
    for key, group in d.groupby(["outcome_lr_only", "support_src_7d", "support_dst_7d"], sort=True):
        rows.append(dict(outcome=key[0], support_src_7d=key[1], support_dst_7d=key[2],
                         transactions=len(group), payer_accounts=int(group.src.nunique()),
                         feature_rates={c: float(group[c].mean()) for c in extra}))
    return dict(scope="ACH positives, forward pair count 72h zero, receiver current/P90 7d > 1",
                totals={str(k): int(v) for k, v in d.outcome_lr_only.value_counts().items()},
                strata=rows, inference="Descriptive support-stratified FN/TP comparison only; no unadjusted significance claim")
