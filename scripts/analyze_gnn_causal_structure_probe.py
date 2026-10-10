"""Verify completed causal structures, then run the fixed matched analyses.

No automatic waiting/start, history traversal, model fitting, or extraction
status mutation. The only input history array read is the frozen bank mapping.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
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

from gnn_causal_structure_stats import analyze

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/gnn_causal_structure_probe_20261006'
SEED = 20260920
BOOT = PERM = 2000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_path(base, name):
    base = Path(base).resolve()
    path = base / name
    resolved = path.resolve()
    if resolved != path.absolute() or not resolved.is_relative_to(base) or resolved == base or path.is_symlink():
        raise ValueError('Unsafe relative input/output path: ' + str(name))
    return resolved


def _bytes(value):
    return json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False).encode('utf-8')


def space(next_bytes=0):
    if shutil.disk_usage(ROOT).free - next_bytes < 20*1024**3:
        raise RuntimeError('Disk reserve would fall below 20 GiB')
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if used + next_bytes > 2*1024**3:
        raise RuntimeError('Diagnostic output would exceed its 2 GiB budget')


def atomic_write(name, value):
    payload = _bytes(value)
    space(len(payload))
    path = safe_path(OUT, name)
    temporary = safe_path(OUT, name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def progress(stage, **kwargs):
    record = dict(pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                  stage=stage, **kwargs)
    atomic_write('status_analysis.json', record)
    print(json.dumps(record, ensure_ascii=True, allow_nan=False), flush=True)


@contextmanager
def analysis_lock():
    """Separate, process-owned byte lock; never touch extraction process.lock."""
    path = safe_path(OUT, 'analysis.lock')
    stream = path.open('a+b')
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'\0')
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Causal-structure analysis is already locked') from exc
        acquired = True
        try:
            yield
        except BaseException as exc:
            # Only the lock owner may change the independent analysis status.
            try:
                progress('failed', status='failed', error=repr(exc))
            except BaseException:
                print(json.dumps(dict(status='failed', error=repr(exc)), ensure_ascii=True), flush=True)
            raise
    finally:
        if acquired:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


def _checked_hash(path, expected, hashes):
    actual = sha(path)
    if actual != expected:
        raise ValueError('SHA mismatch: ' + str(path))
    hashes[path.relative_to(ROOT).as_posix()] = actual


def _load_targets(prefix, hashes):
    name = prefix + 'targets_manifest.json'
    manifest_path = OUT / name
    manifest = read(manifest_path)
    if manifest.get('status') != 'verified':
        raise ValueError('Target manifest is not verified: ' + name)
    hashes[manifest_path.relative_to(ROOT).as_posix()] = sha(manifest_path)
    expected_outputs = {prefix+'targets.parquet', prefix+'pairs.parquet', prefix+'unmatched_positives.csv'}
    if set(manifest['outputs']) != expected_outputs:
        raise ValueError('Unexpected target payload inventory')
    for rel, digest in manifest['outputs'].items():
        _checked_hash(safe_path(OUT, rel), digest, hashes)
    design_name = 'design_bank_ach.json' if prefix else 'design.json'
    _checked_hash(OUT/design_name, manifest['design_sha256'], hashes)
    targets = pd.read_parquet(OUT/(prefix+'targets.parquet'))
    pairs = pd.read_parquet(OUT/(prefix+'pairs.parquet'))
    counts = manifest['counts']
    if not targets.edge_id.is_unique or len(targets) != counts['targets'] or int(targets.label.sum()) != 17794:
        raise ValueError('Target inventory/count mismatch')
    if len(pairs) != counts['pairs'] or pairs.case_edge_id.nunique() != counts['matched_positives']:
        raise ValueError('Pair inventory/count mismatch')
    verifier_name = 'verification_targets_final.json' if prefix else 'verification_targets_strict.json'
    verification = read(OUT/verifier_name)
    if verification.get('status') != 'passed' or verification.get('targets_manifest_sha256') != sha(manifest_path):
        raise ValueError('Independent target verification is missing or unbound')
    hashes[(OUT/verifier_name).relative_to(ROOT).as_posix()] = sha(OUT/verifier_name)
    return manifest, targets, pairs, verification


def verified_inputs():
    structure_path = OUT/'structures_manifest.json'
    if not structure_path.exists():
        raise RuntimeError('Extraction is incomplete; do not start analysis until structures_manifest.json exists')
    structures_manifest = read(structure_path)
    if structures_manifest.get('status') != 'completed':
        raise RuntimeError('Extraction is not completed')
    hashes = {structure_path.relative_to(ROOT).as_posix(): sha(structure_path)}
    final_manifest, targets, pairs, final_verification = _load_targets('final_', hashes)
    strict_manifest, strict_targets, strict_pairs, strict_verification = _load_targets('', hashes)
    binding = structures_manifest['binding']
    for key, name in [('targets_sha256', 'final_targets.parquet'), ('strict_targets_sha256', 'targets.parquet')]:
        if binding.get(key) != sha(OUT/name):
            raise ValueError('Structural features do not bind the target inventory: ' + key)
    _checked_hash(ROOT/'scripts/gnn_causal_structures.py', binding['engine_sha256'], hashes)
    _checked_hash(OUT/'structures.parquet', structures_manifest['sha256'], hashes)
    expected_parts = {f'day_{day:02d}.json' for day in range(1, 11)}
    if set(structures_manifest['parts']) != expected_parts:
        raise ValueError('Structural partition inventory must be exactly Train days 1..10')
    frames = []
    for name in sorted(expected_parts):
        path = OUT/'parts'/name
        _checked_hash(path, structures_manifest['parts'][name], hashes)
        part = read(path)
        if part.get('status') != 'verified' or part.get('binding') != binding:
            raise ValueError('Structural partition identity mismatch: ' + name)
        stem = Path(name).stem
        if set(part['outputs']) != {stem+'.parquet', stem+'_witnesses.json'}:
            raise ValueError('Unexpected structural partition payload')
        for rel, digest in part['outputs'].items():
            _checked_hash(safe_path(OUT/'parts', rel), digest, hashes)
        frame = pd.read_parquet(OUT/'parts'/(stem+'.parquet'))
        if len(frame) != part['rows']:
            raise ValueError('Partition count mismatch')
        frames.append(frame)
    features = pd.read_parquet(OUT/'structures.parquet')
    joined = pd.concat(frames, ignore_index=True)
    if not features.edge_id.is_unique or not joined.edge_id.is_unique:
        raise ValueError('Duplicate structural target identity')
    if list(features) != structures_manifest['columns'] or len(features) != structures_manifest['rows']:
        raise ValueError('Aggregate feature shape mismatch')
    pd.testing.assert_frame_equal(features.sort_values('edge_id').reset_index(drop=True),
                                  joined.sort_values('edge_id').reset_index(drop=True), check_exact=True)
    del frames, joined
    union = pd.concat([targets, strict_targets], ignore_index=True)
    # Overlapping target metadata must be byte-equivalent in its typed columns.
    shared = sorted(set(targets.edge_id) & set(strict_targets.edge_id))
    pd.testing.assert_frame_equal(targets.set_index('edge_id').loc[shared].sort_index(),
                                  strict_targets.set_index('edge_id').loc[shared].sort_index(), check_exact=True)
    if set(features.edge_id) != set(union.edge_id):
        raise ValueError('Structural aggregate differs from the final/strict target union')

    config_path = ROOT/'result/gnn_train_gpu_v1/config.json'
    training = read(ROOT/'result/gnn_train_gpu_v1/training_result.json')
    _checked_hash(config_path, training['config_sha256'], hashes)
    for verification in (final_verification, strict_verification):
        if verification.get('frozen_training_config_sha256') != sha(config_path):
            raise ValueError('Target verification binds another frozen configuration')
    frozen = {k.replace('\\', '/'): v for k, v in read(config_path)['input_hashes'].items()}
    for manifest in (final_manifest, strict_manifest):
        for rel, digest in manifest['source_hashes'].items():
            if frozen.get(rel.replace('\\', '/')) != digest:
                raise ValueError('Target provenance differs from a frozen source anchor')
    for rel, digest in binding['history_hashes'].items():
        if frozen.get(rel.replace('\\', '/')) != digest:
            raise ValueError('Structural history provenance differs from a frozen anchor')
    bank_rel = 'data/gnn_v1_train/history_index/bank_key.npy'
    bank_path = ROOT/bank_rel
    for manifest in (final_manifest, strict_manifest):
        if manifest['source_hashes'].get(bank_rel) != frozen[bank_rel]:
            raise ValueError('Target manifest lacks the frozen payer-bank mapping')
    if binding['history_hashes'].get(bank_rel) != frozen[bank_rel]:
        raise ValueError('Structural manifest lacks the frozen payer-bank mapping')
    _checked_hash(bank_path, frozen[bank_rel], hashes)
    bank_keys = np.load(bank_path, mmap_mode='r', allow_pickle=False)
    for frame in (targets, strict_targets):
        src = frame.src.to_numpy(dtype=np.int64)
        if (src < 0).any() or (src >= len(bank_keys)).any():
            raise ValueError('Target payer outside bank mapping')
        frame['payer_bank_key'] = np.asarray(bank_keys[src])
    del bank_keys, union
    for path in (Path(__file__), ROOT/'scripts/gnn_causal_structure_stats.py'):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return targets, pairs, strict_targets, strict_pairs, features, hashes


def feature_role(name):
    base = name.split('__', 1)[0]
    if base in ('prior_in_count_72h', 'prior_in_same_currency_count_72h'):
        return 'existing_feature_duplicate'
    if base.startswith('cycle2'):
        return 'existing_reverse_pair_refinement'
    if base.startswith('prior_in'):
        return 'new_target_conditioned_single_account_statistic'
    if base.startswith(('chain3', 'cycle3', 'parallel2')):
        return 'new_temporal_topology'
    return 'unclassified'


def compact_summary(report):
    """Keep every feature; rank only by discovery effect, never late-period fit."""
    indexed = {(r['cohort'], r['feature'], r['period']): r for r in report['results']}
    rows = []
    for screen in report['screening']:
        cohort, name = screen['cohort'], screen['feature']
        row = dict(cohort=cohort, feature=name, role=feature_role(name),
                   promising=screen['promising'], sparse_descriptive_candidate=screen['sparse_descriptive_candidate'],
                   discovery_direction=screen['discovery_direction'],
                   reasons_not_promising=screen['reasons_not_promising'], periods={})
        for period in ('discovery_sep01_07', 'sep08', 'later_sep09_10'):
            source = indexed[cohort, name, period]
            full = source.get('full_window_sensitivity')
            row['periods'][period] = dict(
                ncase=source['ncase'], total_positive_inventory=source['total_positive_inventory'],
                unmatched_cases=source['unmatched_cases'], conditional_auc=source['conditional_auc'],
                positive_nonzero_rate=source['positive_nonzero_rate'],
                matched_negative_nonzero_rate=source['matched_negative_nonzero_rate'],
                nonzero_rate_difference=source['nonzero_rate_difference'],
                positive_nonzero_count=source['positive_nonzero_count'], case_payers=source['case_payers'],
                ncontrols_unique=source['ncontrols_unique'],
                payer_clusters=source['payer']['clusters'], payer_holm_p=source['payer']['holm_p'],
                endpoint_clusters=source['endpoint']['clusters'], endpoint_holm_p=source['endpoint']['holm_p'],
                endpoint_ci95=source['endpoint']['ci95'],
                full_window=dict(ncase=full['ncase'], conditional_auc=full['conditional_auc'],
                    positive_nonzero_rate=full['positive_nonzero_rate'],
                    matched_negative_nonzero_rate=full['matched_negative_nonzero_rate'],
                    endpoint_clusters=full['endpoint']['clusters'], endpoint_holm_p=full['endpoint']['holm_p'],
                    endpoint_ci95=full['endpoint']['ci95']) if full is not None else None)
        rows.append(row)
    def key(row):
        auc = row['periods']['discovery_sep01_07']['conditional_auc']
        return (row['cohort'], -(abs(auc-.5) if auc is not None else -1), row['feature'])
    rows.sort(key=key)
    return dict(positive_inventory=report['positive_inventory'], matched_cases=report['matched_cases'],
        unique_controls=report['unique_controls'], feature_count=len(report['feature_names']),
        test_family_size_per_period=report['test_family_size_per_period'],
        promising_new_features=[r['feature'] for r in rows if r['promising'] and r['role']!='existing_feature_duplicate'],
        rows_ranked_by_discovery_effect=rows,
        interpretation=['Existing-feature duplicates are controls and must not be proposed as newly discovered inputs.',
            'cycle2 refines existing reverse-pair history; only chain3/cycle3/parallel2 are genuinely new topology here.',
            'Discovery full-window estimates must be inspected beside overall estimates because the first 72 hours lack a complete window.',
            'Sparse flags are descriptive candidates; a single endpoint component rules out a robust significance claim.',
            'Matched case-control rates are not deployment precision or recall; no model improvement is claimed.'])


def _feature_subset(features, targets):
    selected = features[features.edge_id.isin(targets.edge_id)].copy()
    if not selected.edge_id.is_unique or set(selected.edge_id) != set(targets.edge_id):
        raise ValueError('Selected features must exactly equal selected targets')
    return selected


def run():
    if not OUT.is_dir() or not (OUT/'structures_manifest.json').exists():
        raise RuntimeError('Extraction is incomplete; analysis was not started')
    with analysis_lock():
        started = time.monotonic()
        progress('verify_inputs', status='running')
        targets, pairs, strict_targets, strict_pairs, features, hashes = verified_inputs()
        binding = dict(version=1, inputs_and_sources=hashes, seed=SEED,
                       bootstrap_replicates=BOOT, signflip_replicates=PERM,
                       main='final_targets/final_pairs; ACH same payer bank; non-ACH same payer',
                       sensitivity='original strict-payer ACH only; full positive ACH inventory retained')
        binding_sha = hashlib.sha256(_bytes(binding)).hexdigest()
        manifest_path = OUT/'analysis_manifest.json'
        if manifest_path.exists():
            manifest = read(manifest_path)
            if manifest.get('binding') != binding:
                raise ValueError('Analysis identity changed; preserve prior result and review before migration')
            for name, digest in manifest.get('outputs', {}).items():
                if sha(safe_path(OUT, name)) != digest:
                    raise ValueError('Existing analysis output hash mismatch: ' + name)
            if manifest.get('status') == 'completed':
                progress('completed', status='completed', reused=True, manifest_sha256=sha(manifest_path))
                return
        else:
            manifest = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                            binding=binding, binding_sha256=binding_sha, outputs={}, phases={})
            atomic_write('analysis_manifest.json', manifest)
        phases = [
            ('main', 'statistics.json', targets, pairs),
            ('strict_ach', 'statistics_strict_ach.json',
             strict_targets[strict_targets.cohort.eq('ACH')].copy(),
             strict_pairs[strict_pairs.cohort.eq('ACH')].copy()),
        ]
        reports = {}
        for phase, name, target_frame, pair_frame in phases:
            if name in manifest['outputs']:
                reports[phase] = read(OUT/name)
                continue
            tick = time.monotonic()
            progress('analyze_'+phase, status='running', targets=len(target_frame),
                     positives=int(target_frame.label.sum()), matched_cases=int(pair_frame.case_edge_id.nunique()),
                     feature_columns=len(features.columns)-1)
            report = analyze(target_frame, pair_frame, _feature_subset(features, target_frame),
                             seed=SEED, n_boot=BOOT, n_perm=PERM)
            report['analysis_binding_sha256'] = binding_sha
            report['analysis_scope'] = phase
            if (OUT/name).exists() and read(OUT/name) != report:
                raise ValueError('Unmanifested result differs from deterministic replay; preserve it: '+name)
            manifest['outputs'][name] = atomic_write(name, report)
            manifest['phases'][phase] = dict(seconds=time.monotonic()-tick,
                targets=len(target_frame), positives=int(target_frame.label.sum()),
                matched_cases=report['matched_cases'], promising=len(report['promising']),
                sparse_candidates=len(report['sparse_candidates']))
            atomic_write('analysis_manifest.json', manifest)
            reports[phase] = report
            progress('completed_'+phase, status='running', **manifest['phases'][phase])
        summary = dict(status='completed', analysis_binding_sha256=binding_sha,
                       main=compact_summary(reports['main']),
                       strict_ach=compact_summary(reports['strict_ach']))
        manifest['outputs']['statistics_summary.json'] = atomic_write('statistics_summary.json', summary)
        # Inputs and executable sources must still be identical at publication.
        for rel, digest in hashes.items():
            if sha(safe_path(ROOT, rel)) != digest:
                raise ValueError('Bound input/source changed during analysis: ' + rel)
        manifest.update(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds_this_invocation=time.monotonic()-started,
                        validation_test_read=False, models_trained=False,
                        frozen_sources_modified=False, extraction_status_modified=False)
        atomic_write('analysis_manifest.json', manifest)
        progress('completed', status='completed', seconds=manifest['elapsed_seconds_this_invocation'],
                 main_promising=len(reports['main']['promising']),
                 strict_ach_promising=len(reports['strict_ach']['promising']),
                 manifest_sha256=sha(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    run()


if __name__ == '__main__':
    main()
