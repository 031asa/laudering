"""Independent paired-statistic replay, without importing the analysis code.

Only completed derived Train inputs are read. This verifies identities,
coverage, fixed expressions, occurrence rates, and paired ranking effects.
It deliberately does not replay bootstrap confidence intervals or p-values.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/gnn_causal_structure_probe_20261006'
COUNT_COLUMNS = (
    'prior_in_count_72h', 'prior_in_same_currency_count_72h',
    'prior_in_amount_match_count_72h', 'chain3_path_count_72h',
    'chain3_amount_match_path_count_72h', 'cycle2_count_72h',
    'cycle2_amount_match_count_72h', 'cycle3_path_count_72h',
    'cycle3_amount_match_path_count_72h', 'parallel2_path_count_72h',
    'parallel2_middle_accounts_72h', 'parallel2_amount_match_path_count_72h',
    'parallel2_amount_match_middle_accounts_72h',
)
GAP_COLUMN = 'prior_in_match_min_gap_minutes_72h'
TWO_COLUMNS = ('parallel2_middle_accounts_72h', 'parallel2_amount_match_middle_accounts_72h')
CONTEXT = ('eligible', 'not_applicable', 'full_window_72h', 'same_currency_target')
BASELINE = frozenset(('prior_in_count_72h', 'prior_in_same_currency_count_72h'))
COHORTS = ('ACH', 'bank70_nonACH', 'other_nonACH')
PERIODS = {'discovery_sep01_07': (1, 7), 'sep08': (8, 8),
           'later_sep09_10': (9, 10), 'all_train': (1, 10)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(base, rel):
    base = Path(base).resolve()
    path = base/rel
    resolved = path.resolve()
    if resolved != path.absolute() or not resolved.is_relative_to(base) or resolved == base or path.is_symlink():
        raise ValueError('Unsafe bound path: ' + str(rel))
    return resolved


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def verifier_lock():
    path = inside(OUT, 'verification_statistics.lock')
    stream = path.open('a+b')
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'\0'); stream.flush(); os.fsync(stream.fileno())
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


def independent_expressions(frame):
    expected = {'edge_id', *CONTEXT, *COUNT_COLUMNS, GAP_COLUMN}
    require(set(frame) == expected, 'Unexpected structural feature schema')
    require(frame.edge_id.is_unique, 'Duplicate structural target identity')
    indexed = frame.set_index('edge_id').sort_index()
    for key in CONTEXT:
        require(indexed[key].notna().all() and indexed[key].isin([0, 1]).all(), 'Invalid context flag: '+key)
    values = {}
    for key in COUNT_COLUMNS+(GAP_COLUMN,):
        x = indexed[key].to_numpy(dtype=float)
        require(not np.isinf(x).any(), 'Infinite value: '+key)
        require(np.all(x[np.isfinite(x)] >= 0), 'Negative structural value: '+key)
        values[key] = x
    for key in COUNT_COLUMNS:
        raw = values[key]
        presence = np.full(len(raw), np.nan)
        present = np.isfinite(raw)
        presence[present] = np.greater(raw[present], 0).astype(float)
        values[key+'__present'] = presence
    for key in TWO_COLUMNS:
        raw = values[key]
        predicate = np.full(len(raw), np.nan)
        valid = np.isfinite(raw)
        predicate[valid] = np.greater_equal(raw[valid], 2).astype(float)
        values[key+'__at_least_2'] = predicate
    require(len(values) == 29, 'The fixed expression family must contain 29 expressions')
    return indexed, values


def prepare_matches(targets, pairs, indexed_features):
    require(targets.edge_id.is_unique, 'Duplicate target identity')
    require(targets.label.isin([0, 1]).all() and targets.day.isin(range(1, 11)).all(), 'Invalid Train label/day')
    require(targets.cohort.isin(COHORTS).all(), 'Unknown target cohort')
    require(set(targets.edge_id) <= set(indexed_features.index), 'A target has no structural record')
    require(not pairs.duplicated(['case_edge_id', 'control_edge_id']).any(), 'Duplicate control in a matched set')
    t = targets.set_index('edge_id')
    cases, group = np.unique(pairs.case_edge_id.to_numpy(dtype=np.int64), return_inverse=True)
    require(set(cases) <= set(t.index), 'Unknown case identity')
    control_ids = pairs.control_edge_id.to_numpy(dtype=np.int64)
    require(set(control_ids) <= set(t.index), 'Unknown control identity')
    ncontrol = np.bincount(group, minlength=len(cases))
    require(np.all((ncontrol >= 1) & (ncontrol <= 5)), 'Control set size outside 1..5')
    c = t.loc[cases]
    controls = t.loc[control_ids]
    require(c.label.eq(1).all() and controls.label.eq(0).all(), 'Case/control label mismatch')
    pair_cases = t.loc[pairs.case_edge_id]
    for name in ('day', 'cohort'):
        require(np.array_equal(pair_cases[name].to_numpy(), controls[name].to_numpy()), 'Pair target matching mismatch: '+name)
        require(np.array_equal(pair_cases[name].to_numpy(), pairs[name].to_numpy()), 'Pair metadata mismatch: '+name)
    require(np.array_equal(pair_cases.src.to_numpy(), pairs.src.to_numpy()), 'Pair payer identity mismatch')
    cpos = indexed_features.index.get_indexer(cases)
    npos = indexed_features.index.get_indexer(control_ids)
    flags = indexed_features.eligible.eq(1).to_numpy() & indexed_features.not_applicable.eq(0).to_numpy()
    full = indexed_features.full_window_72h.eq(1).to_numpy()
    applicable = flags[cpos] & (np.bincount(group, weights=(~flags[npos]).astype(int), minlength=len(cases)) == 0)
    complete = full[cpos] & (np.bincount(group, weights=(~full[npos]).astype(int), minlength=len(cases)) == 0)
    return dict(targets=targets, case_ids=cases, case_days=c.day.to_numpy(),
        case_cohorts=c.cohort.to_numpy(), case_src=c.src.to_numpy(), case_dst=c.dst.to_numpy(),
        control_ids=control_ids, control_src=controls.src.to_numpy(), control_dst=controls.dst.to_numpy(),
        group=group, ncontrol=ncontrol, case_positions=cpos, control_positions=npos,
        applicable=applicable, complete=complete)


def case_arrays(values, match):
    """Compute one equally weighted win fraction for each complete matched set."""
    group, ncontrol = match['group'], match['ncontrol']
    px, nx = values[match['case_positions']], values[match['control_positions']]
    bad_case = ~np.isfinite(px)
    bad_control = np.bincount(group, weights=(~np.isfinite(nx)).astype(int), minlength=len(px)) > 0
    # This differs from the production loop: wins are formed per pair and then
    # reduced by a case inverse index, independent of control-set size.
    wins = (px[group] > nx).astype(float) + .5*(px[group] == nx)
    ranks = np.bincount(group, weights=wins, minlength=len(px))/ncontrol
    negative_nonzero = np.bincount(group, weights=(nx != 0).astype(float), minlength=len(px))/ncontrol
    negative_mean = np.bincount(group, weights=np.where(np.isfinite(nx), nx, 0.), minlength=len(px))/ncontrol
    return dict(px=px, nx=nx, bad_case=bad_case, bad_control=bad_control,
                ranks=ranks, negative_nonzero=negative_nonzero, negative_mean=negative_mean)


def replay_row(arrays, match, cohort, period, full=False):
    lo, hi = PERIODS[period]
    base = (match['case_cohorts'] == cohort) & (match['case_days'] >= lo) & (match['case_days'] <= hi)
    valid = base & match['applicable'] & ~arrays['bad_case'] & ~arrays['bad_control']
    if full:
        valid &= match['complete']
    pair_valid = valid[match['group']]
    unique_controls = np.unique(match['control_ids'][pair_valid])
    px = arrays['px'][valid]
    groups = match['group'][pair_valid]
    # Deduplicated control presence is calculated separately from the
    # case-weighted negative occurrence rate, never substituted for it.
    _, first = np.unique(match['control_ids'][pair_valid], return_index=True)
    unique_control_values = arrays['nx'][pair_valid][first]
    positive_rate = float(np.count_nonzero(px)/len(px)) if len(px) else None
    negative_rate = float(np.mean(arrays['negative_nonzero'][valid])) if len(px) else None
    selected_targets = match['targets']
    inventory = int(((selected_targets.label == 1) & (selected_targets.cohort == cohort)
                     & selected_targets.day.between(lo, hi)).sum())
    return dict(ncase=int(valid.sum()), matched_cases_before_validity=int(base.sum()),
        excluded_case_missing=int((base & arrays['bad_case']).sum()),
        excluded_any_control_missing=int((base & arrays['bad_control']).sum()),
        excluded_not_applicable=int((base & ~match['applicable']).sum()),
        excluded_coverage=int((base & match['applicable'] & ~match['complete']).sum()) if full else 0,
        ncontrols_unique=len(unique_controls), control_references=int(pair_valid.sum()),
        conditional_auc=float(np.mean(arrays['ranks'][valid])) if len(px) else None,
        positive_nonzero_rate=positive_rate, matched_negative_nonzero_rate=negative_rate,
        positive_nonzero_count=int(np.count_nonzero(px)),
        unique_control_nonzero_count=int(np.count_nonzero(unique_control_values)),
        nonzero_rate_difference=positive_rate-negative_rate if len(px) else None,
        nonzero_rate_ratio=positive_rate/negative_rate if negative_rate is not None and negative_rate > 0 else None,
        positive_mean=float(np.mean(px)) if len(px) else None,
        matched_negative_mean=float(np.mean(arrays['negative_mean'][valid])) if len(px) else None,
        case_payers=len(np.unique(match['case_src'][valid])), case_receivers=len(np.unique(match['case_dst'][valid])),
        control_payers=len(np.unique(match['control_src'][pair_valid])), control_receivers=len(np.unique(match['control_dst'][pair_valid])),
        nonzero_case_payers=len(np.unique(match['case_src'][valid][px != 0])),
        total_positive_inventory=inventory, unmatched_cases=inventory-int(base.sum()))


def compare_fields(expected, observed, *, include_inventory):
    maximum = 0.
    for key, actual in expected.items():
        if key in ('total_positive_inventory', 'unmatched_cases') and not include_inventory:
            continue
        require(key in observed, 'Missing published statistic: '+key)
        original = observed[key]
        if actual is None:
            require(original is None, 'Missing-value mismatch: '+key)
        elif isinstance(actual, int):
            require(original == actual, 'Count mismatch: '+key)
        else:
            require(original is not None and math.isfinite(original), 'Nonfinite/missing published effect: '+key)
            error = abs(float(original)-actual)
            maximum = max(maximum, error)
            require(math.isclose(float(original), actual, rel_tol=2e-12, abs_tol=2e-12), 'Effect mismatch: '+key)
    return maximum


def verify_report(report, targets, pairs, indexed, expressions):
    require(report['status'] == 'completed', 'Statistics are incomplete')
    require(set(report['feature_names']) == set(expressions), 'Published expression inventory differs')
    require(report['test_family_size_per_period'] == 87, 'Unexpected multiple-comparison family size')
    match = prepare_matches(targets, pairs, indexed)
    require(report['positive_inventory'] == int(targets.label.sum()), 'Positive inventory mismatch')
    require(report['matched_cases'] == len(match['case_ids']), 'Matched case inventory mismatch')
    require(report['unique_controls'] == pairs.control_edge_id.nunique(), 'Unique control inventory mismatch')
    observed = {(r['cohort'], r['feature'], r['period']): r for r in report['results']}
    wanted = {(c, f, p) for c in COHORTS for f in expressions for p in PERIODS}
    require(len(observed) == len(report['results']) and set(observed) == wanted, 'Published result grid differs')
    maximum = 0.
    checks = []
    for name, values in expressions.items():
        arrays = case_arrays(values, match)
        for cohort in COHORTS:
            for period in PERIODS:
                original = observed[cohort, name, period]
                row = replay_row(arrays, match, cohort, period)
                maximum = max(maximum, compare_fields(row, original, include_inventory=True))
                full = replay_row(arrays, match, cohort, period, full=True)
                require(original.get('full_window_sensitivity') is not None, 'Missing complete-window sensitivity')
                maximum = max(maximum, compare_fields(full, original['full_window_sensitivity'], include_inventory=False))
                checks.append(dict(cohort=cohort, feature=name, period=period,
                    ncase=row['ncase'], conditional_auc=row['conditional_auc'],
                    positive_nonzero_rate=row['positive_nonzero_rate'],
                    matched_negative_nonzero_rate=row['matched_negative_nonzero_rate'],
                    full_window_ncase=full['ncase'], full_window_auc=full['conditional_auc']))
    return dict(positive_inventory=int(targets.label.sum()), matched_cases=len(match['case_ids']),
        expressions=29, cohorts=3, periods=4, ordinary_rows=len(checks),
        complete_window_rows=len(checks), maximum_absolute_numeric_discrepancy=maximum,
        independently_recomputed=checks)


def check_selected_file(path):
    """Optional post-export check; this can also be invoked by the main agent."""
    path = Path(path)
    if not path.exists():
        return dict(status='not_present', reason='Export has not run; check selected_features.json after export')
    data = read(path)
    require(isinstance(data.get('selected'), list), 'Exported selection lacks selected list')
    seen = set()
    for row in data['selected']:
        require(isinstance(row, dict) and row.get('cohort') in COHORTS and isinstance(row.get('feature'), str), 'Malformed exported feature selection')
        key = (row['cohort'], row['feature'])
        require(key not in seen, 'Duplicate exported cohort/feature')
        seen.add(key)
        require(row['feature'].split('__', 1)[0] not in BASELINE, 'Existing baseline duplicate exported as a new feature')
    return dict(status='passed', selected=len(seen), sha256=sha(path), baseline_duplicates_excluded=True)


def verify_summary(summary, reports):
    checks = {}
    for scope, report in reports.items():
        section = summary[scope]
        screens = {(s['cohort'], s['feature']): s for s in report['screening']}
        rows = section['rows_ranked_by_discovery_effect']
        row_keys = {(r['cohort'], r['feature']) for r in rows}
        require(len(rows) == len(row_keys) and row_keys == set(screens), 'Compact summary screening inventory mismatch')
        expected = []
        duplicate_promising = []
        for row in rows:
            screen = screens[row['cohort'], row['feature']]
            require(row['promising'] == screen['promising'], 'Summary promising flag mismatch')
            if screen['promising']:
                if row['feature'].split('__', 1)[0] in BASELINE:
                    duplicate_promising.append(dict(cohort=row['cohort'], feature=row['feature']))
                else:
                    expected.append(row['feature'])
        require(section['promising_new_features'] == expected, 'New-feature recommendation list differs or includes a baseline duplicate')
        require(all(name.split('__', 1)[0] not in BASELINE for name in expected), 'Baseline duplicate selected')
        checks[scope] = dict(baseline_duplicates_excluded_from_new_feature_list=True,
            internal_baseline_promising_only_for_reference=duplicate_promising,
            recommended_new_feature_entries=len(expected))
    return checks


def run():
    started = time.monotonic()
    manifest_path = OUT/'analysis_manifest.json'
    require(manifest_path.exists(), 'Analysis has not completed')
    with verifier_lock():
        manifest = read(manifest_path)
        require(manifest.get('status') == 'completed', 'Analysis is not completed')
        hashes = {manifest_path.relative_to(ROOT).as_posix(): sha(manifest_path)}
        for rel, digest in manifest['binding']['inputs_and_sources'].items():
            path = inside(ROOT, rel)
            require(sha(path) == digest, 'Bound input/source SHA mismatch: '+rel)
            hashes[rel] = digest
        for name, digest in manifest['outputs'].items():
            path = inside(OUT, name)
            require(sha(path) == digest, 'Published analysis SHA mismatch: '+name)
            hashes[path.relative_to(ROOT).as_posix()] = digest
        structures = pd.read_parquet(OUT/'structures.parquet')
        indexed, expressions = independent_expressions(structures)
        final_targets = pd.read_parquet(OUT/'final_targets.parquet')
        final_pairs = pd.read_parquet(OUT/'final_pairs.parquet')
        strict_targets = pd.read_parquet(OUT/'targets.parquet')
        strict_pairs = pd.read_parquet(OUT/'pairs.parquet')
        strict_targets = strict_targets[strict_targets.cohort.eq('ACH')].copy()
        strict_pairs = strict_pairs[strict_pairs.cohort.eq('ACH')].copy()
        require(int(final_targets.label.sum()) == 17794, 'Main Train-positive inventory must be 17794')
        require(int(strict_targets.label.sum()) == 15067, 'Strict ACH-positive inventory must be 15067')
        reports = dict(main=read(OUT/'statistics.json'), strict_ach=read(OUT/'statistics_strict_ach.json'))
        for report in reports.values():
            require(report.get('analysis_binding_sha256') == manifest['binding_sha256'], 'Statistics bind another analysis')
        main_result = verify_report(reports['main'], final_targets, final_pairs, indexed, expressions)
        strict_result = verify_report(reports['strict_ach'], strict_targets, strict_pairs, indexed, expressions)
        summary_check = verify_summary(read(OUT/'statistics_summary.json'), reports)
        selected_check = check_selected_file(OUT/'selected_features.json')
        if selected_check['status'] == 'passed':
            hashes[(OUT/'selected_features.json').relative_to(ROOT).as_posix()] = selected_check['sha256']
        hashes[Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
        # Fail closed if a source or result was changed while replaying it.
        for rel, digest in hashes.items():
            require(sha(inside(ROOT, rel)) == digest, 'Input/output changed during verification: '+rel)
        result = dict(status='passed', completed_at=datetime.now(timezone.utc).isoformat(),
            analysis_manifest_sha256=hashes[manifest_path.relative_to(ROOT).as_posix()],
            elapsed_seconds=time.monotonic()-started, inputs_and_outputs_sha256=hashes,
            main=main_result, strict_ach=strict_result, summary_selection=summary_check,
            optional_export_selection=selected_check,
            independently_recomputed=['Fixed 14 base features plus 13 count>0 and 2 parallel-middle>=2 expressions',
                'Each cohort/period matched-case coverage and missing/applicability exclusions',
                'Case-equal conditional win rates with half-credit ties',
                'Positive occurrence and case-equal negative occurrence; unique controls and account support',
                'Complete-72h-window versions of the same descriptive statistics'],
            not_replayed=['History graph extraction or path witnesses',
                'Payer/endpoint component construction, bootstrap confidence intervals, sign-flip/randomization p-values, and Holm adjustment'],
            limitation='This verifies descriptive paired statistics and selection-list hygiene; it does not independently reproduce statistical significance, generalization, or model improvement.',
            stats_module_imported=False, validation_test_read=False, models_trained=False)
        destination = OUT/'verification_statistics.json'
        if destination.exists():
            old = read(destination)
            for key in ('status', 'analysis_manifest_sha256', 'inputs_and_outputs_sha256', 'main', 'strict_ach', 'summary_selection', 'optional_export_selection'):
                require(old.get(key) == result[key], 'Existing verification differs; preserve and review: '+key)
            print(json.dumps(dict(status='passed', verified_reuse=True, sha256=sha(destination)), ensure_ascii=True), flush=True)
            return
        payload = json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False).encode('utf-8')
        require(shutil.disk_usage(ROOT).free-len(payload) >= 20*1024**3, 'Disk reserve below 20 GiB')
        require(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(payload) <= 2*1024**3, 'Diagnostic output exceeds 2 GiB budget')
        temporary = inside(OUT, 'verification_statistics.json.tmp')
        with temporary.open('wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
        print(json.dumps(dict(status='passed', rows=main_result['ordinary_rows']+strict_result['ordinary_rows'],
            complete_window_rows=main_result['complete_window_rows']+strict_result['complete_window_rows'],
            seconds=result['elapsed_seconds'], sha256=sha(destination)), ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    run()


if __name__ == '__main__':
    main()
