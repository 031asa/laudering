"""Matched, Train-only screening of causal structural features.

This module never reads history, fits a model, chooses new matching controls, or
writes files. Targets carry labels; the structural feature frame must not.
The matching runner owns deterministic control selection from the full pool.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

PERIODS = {
    'discovery_sep01_07': tuple(range(1, 8)),
    'sep08': (8,),
    'later_sep09_10': (9, 10),
    'all_train': tuple(range(1, 11)),
}
COHORTS = ('ACH', 'bank70_nonACH', 'other_nonACH')
COVERAGE_COLUMNS = ('full_window_72h', 'history_coverage', 'history_complete')
CONTEXT_COLUMNS = ('eligible', 'not_applicable', 'same_currency_target')


def holm_adjust(pvalues):
    """Unavailable tests retain their slot in the fixed comparison family."""
    p = np.array([1. if x is None else float(x) for x in pvalues])
    if not len(p):
        return []
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty(len(p))
    out[order] = adjusted
    return out.tolist()


def _seed(seed, *keys):
    digest = hashlib.sha256(('|'.join(map(str, (seed,) + keys))).encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def _inference(delta, clusters, *, seed, n_boot, n_perm):
    """Case-weighted cluster bootstrap and two-sided cluster sign flipping.

    Sign flipping assumes symmetric, independent cluster effects under the
    null. Shared historical paths can still connect apparently separate target
    endpoint clusters, so these are exploratory uncertainty calculations.
    """
    k = len(np.unique(clusters))
    result = dict(clusters=k, cases=len(delta), ci95=None, signflip_p=None,
                  largest_cluster_cases=0, bootstrap_replicates=n_boot,
                  signflip_replicates=0, equal_cluster_auc=None,
                  inference_status='insufficient_clusters')
    if not len(delta):
        return result
    _, ci = np.unique(clusters, return_inverse=True)
    sums = np.bincount(ci, weights=delta)
    sizes = np.bincount(ci).astype(float)
    result.update(largest_cluster_cases=int(sizes.max()),
                  equal_cluster_auc=float(.5 + np.mean(sums / sizes)))
    # Leave-one-payer/endpoint-component-out effect is a sensitivity, not a CI.
    if k > 1:
        loo = .5 + (sums.sum() - sums) / (sizes.sum() - sizes)
        result['leave_one_cluster_out_auc_range'] = [float(loo.min()), float(loo.max())]
    else:
        result['leave_one_cluster_out_auc_range'] = None
    if k < 2:
        return result
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot)
    for begin in range(0, n_boot, 64):
        count = min(64, n_boot - begin)
        index = rng.integers(0, k, size=(count, k))
        bootstrap[begin:begin + count] = .5 + sums[index].sum(1) / sizes[index].sum(1)
    exact = 2**k <= n_perm if k < 63 else False
    reps = 2**k if exact else n_perm
    observed = abs(float(sums.sum()))
    extreme = 0
    for begin in range(0, reps, 64):
        count = min(64, reps - begin)
        if exact:
            masks = np.arange(begin, begin + count, dtype=np.uint64)
            signs = 2 * ((masks[:, None] >> np.arange(k, dtype=np.uint64)) & 1).astype(np.int8) - 1
        else:
            signs = 2 * rng.integers(0, 2, size=(count, k), dtype=np.int8) - 1
        extreme += int((np.abs(signs @ sums) >= observed - 1e-12).sum())
    p = extreme / reps if exact else (extreme + 1) / (reps + 1)
    result.update(ci95=[float(x) for x in np.quantile(bootstrap, [.025, .975])],
                  signflip_p=float(p), signflip_replicates=reps,
                  signflip_method='exact' if exact else 'monte_carlo',
                  inference_status='few_clusters_caution' if k < 20 else 'computed')
    return result


def _endpoint_components(case_ids, control_sets, targets):
    parent = {}

    def find(a):
        a = int(a)
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for eid, controls in zip(case_ids, control_sets):
        src = int(targets.at[eid, 'src'])
        union(src, targets.at[eid, 'dst'])
        for cid in controls:
            union(src, targets.at[cid, 'src'])
            union(src, targets.at[cid, 'dst'])
    return np.array([find(targets.at[eid, 'src']) for eid in case_ids], dtype=np.int64)


def _rank_and_null_variance(positive, negatives):
    """One positive against its controls; ties are exactly half wins."""
    n = len(negatives)
    rank = (np.count_nonzero(positive > negatives) + .5 * np.count_nonzero(positive == negatives)) / n
    _, counts = np.unique(np.r_[positive, negatives], return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2
    ss = np.sum(counts * (midranks - (n + 2) / 2)**2)
    return float(rank), float(ss / ((n + 1) * n * n))


def _validate(targets, pairs, features):
    required = {'edge_id', 'label', 'day', 'src', 'dst', 'cohort'}
    pair_required = {'case_edge_id', 'control_edge_id', 'day', 'src', 'cohort'}
    if not required <= set(targets) or not pair_required <= set(pairs) or 'edge_id' not in features:
        raise ValueError('Missing required target, pair, or feature identity columns')
    if targets.edge_id.duplicated().any() or features.edge_id.duplicated().any():
        raise ValueError('Duplicate edge identity')
    if not targets.label.isin([0, 1]).all() or not targets.day.isin(range(1, 11)).all():
        raise ValueError('Only binary labels and Train days 1..10 are accepted')
    if not targets.cohort.isin(COHORTS).all():
        raise ValueError('Unknown cohort')
    if set(targets.edge_id) != set(features.edge_id):
        raise ValueError('Feature identities must exactly cover the supplied target inventory')
    forbidden = {'label', 'is_laundering', 'src', 'dst', 'day', 'cohort', 'score', 'prediction'}
    if forbidden & set(features):
        raise ValueError('Features must not contain labels, identities, or model scores')
    if pairs.duplicated(['case_edge_id', 'control_edge_id']).any():
        raise ValueError('Duplicate control within a case')
    if pairs.groupby('case_edge_id').size().max() > 5:
        raise ValueError('At most five controls per case')
    ids = set(targets.edge_id)
    if not set(pairs.case_edge_id) <= ids or not set(pairs.control_edge_id) <= ids:
        raise ValueError('Pair references an unknown target')
    t = targets.set_index('edge_id', drop=False)
    cases = t.loc[pairs.case_edge_id].reset_index(drop=True)
    controls = t.loc[pairs.control_edge_id].reset_index(drop=True)
    p = pairs.reset_index(drop=True)
    if not cases.label.eq(1).all() or not controls.label.eq(0).all():
        raise ValueError('Cases must be positive and controls negative')
    for column in ('day', 'cohort'):
        if not cases[column].eq(controls[column]).all() or not p[column].eq(cases[column]).all():
            raise ValueError('Pair matching/identity mismatch: ' + column)
    if not p.src.eq(cases.src).all():
        raise ValueError('Pair case-payer identity mismatch')
    cross_payer = ~cases.src.eq(controls.src)
    if cross_payer.any() and not cases.loc[cross_payer, 'cohort'].eq('ACH').all():
        raise ValueError('Cross-payer matching is only authorized for ACH')
    if 'matching_level' in p:
        if not p.matching_level.isin(['payer', 'payer_bank']).all():
            raise ValueError('Unknown matching level')
        if (cross_payer & ~p.matching_level.eq('payer_bank')).any():
            raise ValueError('Cross-payer control requires payer_bank matching level')
        if (p.matching_level.eq('payer_bank') & ~cases.cohort.eq('ACH')).any():
            raise ValueError('Bank matching is only authorized for ACH')
    if not (cases.src.eq(cases.dst)).eq(controls.src.eq(controls.dst)).all():
        raise ValueError('Self-transfer matching mismatch')
    checked = ['day', 'case_src_identity', 'cohort', 'self_transfer']
    for column in ('payer_bank_key', 'payer_bank'):
        if column in t:
            if not cases[column].eq(controls[column]).all():
                raise ValueError('Pair payer-bank mismatch: ' + column)
            checked.append(column)
    for column in ('payment_format_cat', 'payment_currency_cat', 'receiving_currency_cat',
                   'payment_format', 'payment_currency', 'receiving_currency'):
        if column in t:
            if not cases[column].eq(controls[column]).all():
                raise ValueError('Pair matching mismatch: ' + column)
            checked.append(column)
    for column in ('paid_scaled', 'received_scaled'):
        if column in t:
            # Python integer arithmetic avoids int64 overflow near currency limits.
            for a, b in zip(cases[column], controls[column]):
                if pd.isna(a) or pd.isna(b) or a < 0 or b < 0 or not (int(a) <= 2*int(b) and int(b) <= 2*int(a)):
                    raise ValueError('Amount factor-two matching mismatch: ' + column)
            checked.append(column)
    return t, checked


def _summarize(values, case_ids, controls, payer, endpoint, t, *, seed, n_boot, n_perm, eligible=None, applicable=None):
    n = len(case_ids)
    applicable = np.ones(n, bool) if applicable is None else applicable
    good = applicable.copy()
    if eligible is not None:
        good &= eligible
    rank = np.full(n, np.nan)
    variance = np.zeros(n)
    missing_cases = missing_controls = 0
    for j, (eid, group) in enumerate(zip(case_ids, controls)):
        pv = float(values.at[eid])
        nv = values.loc[group].to_numpy(dtype=float)
        if not np.isfinite(pv):
            missing_cases += 1
            good[j] = False
        if not np.isfinite(nv).all():
            missing_controls += 1
            good[j] = False
        if good[j]:
            rank[j], variance[j] = _rank_and_null_variance(pv, nv)
    chosen = case_ids[good]
    selected_controls = [g for g, ok in zip(controls, good) if ok]
    unique_controls = np.unique(np.concatenate(selected_controls)) if len(chosen) else np.array([], dtype=np.int64)
    positive = values.loc[chosen].to_numpy(dtype=float)
    negative_rates = [float(np.mean(values.loc[g].to_numpy(dtype=float) != 0)) for g in selected_controls]
    control_mean = [float(values.loc[g].mean()) for g in selected_controls]
    delta = rank[good] - .5
    null_variance = float(variance[good].sum())
    normal_p = math.erfc(abs(float(delta.sum())) / math.sqrt(2*null_variance)) if null_variance > 0 else 1.
    payer_inf = _inference(delta, payer[good], seed=seed, n_boot=n_boot, n_perm=n_perm)
    endpoint_inf = _inference(delta, endpoint[good], seed=seed+1, n_boot=n_boot, n_perm=n_perm)
    for inference in (payer_inf, endpoint_inf):
        raw = inference['signflip_p']
        inference['conservative_p'] = max(raw, normal_p) if raw is not None else None
    pos_rate = float(np.mean(positive != 0)) if len(positive) else None
    neg_rate = float(np.mean(negative_rates)) if negative_rates else None
    return dict(ncase=len(chosen), matched_cases_before_validity=n,
        excluded_case_missing=missing_cases, excluded_any_control_missing=missing_controls,
        missing_counts_overlap=True,
        excluded_not_applicable=int((~applicable).sum()),
        excluded_coverage=int((applicable & ~eligible).sum()) if eligible is not None else 0,
        ncontrols_unique=len(unique_controls), control_references=sum(map(len, selected_controls)),
        conditional_auc=float(np.mean(rank[good])) if good.any() else None,
        positive_nonzero_rate=pos_rate, matched_negative_nonzero_rate=neg_rate,
        nonzero_rate_difference=pos_rate-neg_rate if pos_rate is not None else None,
        nonzero_rate_ratio=pos_rate/neg_rate if neg_rate is not None and neg_rate > 0 else None,
        positive_nonzero_count=int(np.count_nonzero(positive)),
        unique_control_nonzero_count=int(np.count_nonzero(values.loc[unique_controls].to_numpy(dtype=float))),
        positive_mean=float(np.mean(positive)) if len(positive) else None,
        matched_negative_mean=float(np.mean(control_mean)) if control_mean else None,
        case_payers=int(t.loc[chosen, 'src'].nunique()), case_receivers=int(t.loc[chosen, 'dst'].nunique()),
        control_payers=int(t.loc[unique_controls, 'src'].nunique()), control_receivers=int(t.loc[unique_controls, 'dst'].nunique()),
        nonzero_case_payers=int(t.loc[chosen[positive != 0], 'src'].nunique()),
        conditional_rank_normal_p=float(normal_p), payer=payer_inf, endpoint=endpoint_inf)


def analyze(targets_df, pairs_df, features_df, *, seed=20260920, n_boot=2000, n_perm=2000):
    """Return JSON-safe development screening; inputs are never mutated.

    targets: edge_id,label,day(1..10),src,dst,cohort. Additional exact matching
      categories and paid_scaled/received_scaled are independently rechecked.
    pairs: case_edge_id,control_edge_id,day,src,cohort, max 5 distinct controls.
    features: edge_id plus finite numeric or NaN feature values. Coverage flags
      are full_window_72h/history_coverage/history_complete (1 means complete).
    Missing a case OR any of its fixed controls excludes that whole case for
      this feature. It never silently rematches or shrinks a control set.
    """
    if not 1 <= n_boot <= 5000 or not 1 <= n_perm <= 5000:
        raise ValueError('Require 1..5000 bootstrap and sign-flip replicates')
    t, checked = _validate(targets_df, pairs_df, features_df)
    f = features_df.set_index('edge_id', drop=True).copy()
    metadata = [name for name in COVERAGE_COLUMNS if name in f]
    context = [name for name in CONTEXT_COLUMNS if name in f]
    columns = sorted(name for name in f if name not in metadata + context)
    for name in columns:
        if not pd.api.types.is_numeric_dtype(f[name]):
            raise ValueError('Non-numeric structural feature: ' + name)
        if np.isinf(f[name].to_numpy(dtype=float)).any():
            raise ValueError('Infinite structural value: ' + name)
    definitions = {}
    # Fixed predicates: all nonnegative motif counts have a presence indicator;
    # parallel-path middle-account counts additionally have a two-account flag.
    for name in list(columns):
        if name.endswith('count_72h') or name.endswith('accounts_72h'):
            if (f[name].dropna() < 0).any():
                raise ValueError('Negative structural count: ' + name)
            pred = name + '__present'
            if pred in f:
                raise ValueError('Generated predicate collides with an input column')
            f[pred] = np.where(f[name].isna(), np.nan, (f[name] > 0).astype(float))
            definitions[pred] = dict(source=name, operator='>', threshold=0)
            columns.append(pred)
            if name.startswith('parallel2') and 'middle_accounts' in name:
                pred = name + '__at_least_2'
                if pred in f:
                    raise ValueError('Generated predicate collides with an input column')
                f[pred] = np.where(f[name].isna(), np.nan, (f[name] >= 2).astype(float))
                definitions[pred] = dict(source=name, operator='>=', threshold=2)
                columns.append(pred)
    columns = sorted(columns)
    target_applicable = pd.Series(True, index=f.index)
    for name in context:
        if not pd.api.types.is_numeric_dtype(f[name]) or not f[name].dropna().isin([0, 1]).all():
            raise ValueError('Context flag must be binary: ' + name)
    if 'eligible' in f:
        target_applicable &= f.eligible.eq(1)
    if 'not_applicable' in f:
        target_applicable &= f.not_applicable.eq(0)
    complete = None
    if metadata:
        for name in metadata:
            if not pd.api.types.is_numeric_dtype(f[name]) or not f[name].dropna().between(0, 1).all():
                raise ValueError('Coverage must be numeric in [0,1], with 1 meaning full')
        complete = f[metadata].eq(1).all(axis=1)
    grouped = {int(eid): group.control_edge_id.to_numpy(dtype=np.int64)
               for eid, group in pairs_df.groupby('case_edge_id', sort=True)}
    positive = t[t.label == 1]
    results = []
    for cohort in COHORTS:
        all_cases = positive[positive.cohort == cohort]
        for period, days in PERIODS.items():
            inventory = all_cases[all_cases.day.isin(days)]
            case_ids = np.array(sorted(set(inventory.index) & set(grouped)), dtype=np.int64)
            controls = [grouped[int(eid)] for eid in case_ids]
            payer = t.loc[case_ids, 'src'].to_numpy(dtype=np.int64)
            endpoint = _endpoint_components(case_ids, controls, t)
            applicable = np.array([bool(target_applicable.at[eid] and target_applicable.loc[g].all())
                                   for eid, g in zip(case_ids, controls)], dtype=bool)
            full = np.array([bool(complete.at[eid] and complete.loc[g].all())
                             for eid, g in zip(case_ids, controls)], dtype=bool) if complete is not None else None
            for name in columns:
                common = dict(seed=_seed(seed, cohort, period, name), n_boot=n_boot, n_perm=n_perm,
                              applicable=applicable)
                stats = _summarize(f[name], case_ids, controls, payer, endpoint, t, **common)
                stats.update(total_positive_inventory=len(inventory), unmatched_cases=len(inventory)-len(case_ids))
                sensitivity = _summarize(f[name], case_ids, controls, payer, endpoint, t, eligible=full, **common) if full is not None else None
                results.append(dict(cohort=cohort, period=period, feature=name, **stats,
                                    full_window_sensitivity=sensitivity))
    for period in PERIODS:
        subset = [r for r in results if r['period'] == period]
        for kind in ('payer', 'endpoint'):
            for r, q in zip(subset, holm_adjust([r[kind]['conservative_p'] for r in subset])):
                r[kind]['holm_p'] = q
            if complete is not None:
                qvals = holm_adjust([r['full_window_sensitivity'][kind]['conservative_p'] for r in subset])
                for r, q in zip(subset, qvals):
                    r['full_window_sensitivity'][kind]['holm_p'] = q
    indexed = {(r['cohort'], r['feature'], r['period']): r for r in results}
    screening = []
    for cohort in COHORTS:
        for name in columns:
            d, mid, late = [indexed[cohort, name, p] for p in ('discovery_sep01_07', 'sep08', 'later_sep09_10')]
            direction = None if d['conditional_auc'] is None or d['conditional_auc'] == .5 else (1 if d['conditional_auc'] > .5 else -1)
            failures = []
            if direction is None:
                failures.append('no_discovery_direction')
            for label, row in (('discovery', d), ('later', late)):
                auc = row['conditional_auc']
                ci = row['endpoint']['ci95']
                if direction is None or auc is None or direction*(auc-.5) < .03:
                    failures.append(label+'_effect_below_0.03')
                if row['endpoint']['clusters'] < 5:
                    failures.append(label+'_fewer_than_five_endpoint_clusters')
                if row['endpoint']['holm_p'] >= .05:
                    failures.append(label+'_endpoint_holm_not_below_0.05')
                if ci is None or direction is None or not (ci[0] > .5 if direction == 1 else ci[1] < .5):
                    failures.append(label+'_endpoint_ci_not_directional')
                loo = row['endpoint'].get('leave_one_cluster_out_auc_range')
                if loo is None or direction is None or not (loo[0] > .5 if direction == 1 else loo[1] < .5):
                    failures.append(label+'_leave_one_component_not_stable')
            if direction is None or mid['conditional_auc'] is None or direction*(mid['conditional_auc']-.5) <= 0:
                failures.append('sep08_not_same_direction')
            if complete is not None:
                row = late['full_window_sensitivity']
                if direction is None or row['conditional_auc'] is None or direction*(row['conditional_auc']-.5) < .03:
                    failures.append('later_complete_window_effect_not_reproduced')
            sparse = (d['nonzero_rate_difference'] is not None and late['nonzero_rate_difference'] is not None
                and d['nonzero_rate_difference']*late['nonzero_rate_difference'] > 0
                and abs(d['nonzero_rate_difference']) >= .01 and abs(late['nonzero_rate_difference']) >= .01
                and d['positive_nonzero_count'] >= 10 and late['positive_nonzero_count'] >= 5
                and d['nonzero_case_payers'] >= 3 and late['nonzero_case_payers'] >= 3)
            screening.append(dict(cohort=cohort, feature=name,
                discovery_direction='higher' if direction == 1 else 'lower' if direction == -1 else None,
                promising=not failures, reasons_not_promising=failures,
                sparse_descriptive_candidate=bool(sparse and failures),
                sparse_candidate_is_not_significance_or_model_improvement=True))
    return dict(status='completed', feature_names=columns, cohorts=list(COHORTS), periods=list(PERIODS),
        positive_inventory=len(positive), matched_cases=len(grouped), unique_controls=int(pairs_df.control_edge_id.nunique()),
        input_targets=len(t), matching_fields_rechecked=checked, coverage_columns=metadata,
        context_columns=context, predicate_definitions=definitions,
        cross_payer_control_references=int((t.loc[pairs_df.case_edge_id, 'src'].to_numpy() != t.loc[pairs_df.control_edge_id, 'src'].to_numpy()).sum()),
        cross_payer_matching_caution='ACH may use same-bank cross-payer controls; matching_level and full-pool sampling are independently verified by the runner, not inferred by this statistic.',
        test_family_size_per_period=len(columns)*len(COHORTS), results=results, screening=screening,
        promising=[r for r in screening if r['promising']],
        sparse_candidates=[r for r in screening if r['sparse_descriptive_candidate']],
        method=dict(case_weight='Each case weight one; fixed controls share equal weight within case',
            missing='Whole matched case excluded when its value or any fixed control value is missing; no rematching',
            inference='Case-weighted cluster bootstrap and symmetric cluster sign flip; maximum with tie-corrected conditional-rank normal p as sparse-flag sensitivity',
            multiplicity='Holm over every feature/cohort separately per period; missing tests retain slots; complete-window sensitivity is a separately reported family',
            dependency='Payer-only plus conservative connected components of case/control payer and receiver accounts; hidden/shared historical paths can induce further dependence',
            presence='Nonzero feature values; not an optimized detection rule; controls weighted within each case',
            sparse_candidate='Heuristic descriptive screen: repeated nonzero rate difference >=1 percentage point; >=10 discovery and >=5 later nonzero positives over >=3 payers in each period; never automatically selected'),
        limitations=['Development-only matched association, not causal inference, held-out generalization, population AP/precision/recall, or demonstrated GNN improvement.',
            'Negative controls are sampled and can be reused; unique control counts differ from weighted references.',
            'The positive inventory count does not imply all positives were matchable; excluded cases are reported by cohort/period.',
            'Endpoint components may merge to one, making robust inference unavailable; small numbers of clusters and rank-normal approximation require caution.'],
        history_labels_as_features=False, validation_test_read=False, model_trained=False)
