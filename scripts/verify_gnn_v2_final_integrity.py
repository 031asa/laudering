"""Independent final audit; reads frozen artifacts, writes only a new audit JSON."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/gnn_tuning_v2'
EXPECTED_COMPARISON = '7de3332e9c50e1603dc0b73e6a474057784146db47a3d3d22f132c1e17d0cf5f'
NAMES = ('lr_only', 'features_only', 'features_pruned', 'features_balanced')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def independent_threshold(y, score):
    order = np.argsort(-score, kind='stable')
    s = score[order]
    ends = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    counts = ends + 1
    good = np.flatnonzero((tp > 0) & (5 * tp >= 4 * counts))
    if not len(good):
        return None
    # Descending complete tied groups: earliest among maximum TP gives
    # minimum alerts / greatest precision, then greatest threshold.
    max_tp = tp[good].max()
    at = good[tp[good] == max_tp][0]
    return float(s[ends[at]])


def main():
    destination = OUT / 'final_integrity_verification.json'
    if destination.exists():
        previous = read(destination)
        assert previous['comparison_sha256'] == sha(OUT / 'comparison.json') == EXPECTED_COMPARISON
        for name, entry in previous['models'].items():
            assert sha(ROOT / entry['model_path']) == entry['model_sha256']
        print(json.dumps({'existing_audit_identity_verified': True, 'path': str(destination)}, ensure_ascii=True))
        return
    assert sha(OUT / 'comparison.json') == EXPECTED_COMPARISON
    engineering = read(OUT / 'engineering_tests.json')
    assert engineering['status'] == 'passed'
    for source, digest in engineering['source_sha256'].items():
        assert sha(ROOT / 'scripts/gnn_features' / source) == digest, source
    for rel, digest in engineering['frozen_v1_anchors'].items():
        assert sha(ROOT / rel) == digest, rel

    from gnn_features.v2_evaluate import verified_training, confusion, paired_changes, group_report
    from gnn_features.v2_compare import load_predictions
    from gnn_features.v2_adapter import projection_contract
    torch.set_num_threads(1)
    comparison = read(OUT / 'comparison.json')
    by_name = {r['name']: r for r in comparison['results']}
    assert set(by_name) == set(NAMES) and comparison['status'] == 'completed'
    cache = read(ROOT / 'data/gnn_validation_cache_parquet/config.json')
    arrays = {}
    input_sha = {}
    for field in ('timestamp', 'src', 'dst', 'payment_format_cat'):
        rel = Path('data/gnn_v1_validation/history_index') / (field + '.npy')
        digest = sha(ROOT / rel)
        assert digest == cache['hashes'][str(rel)], str(rel)
        arrays[field] = np.load(ROOT / rel, mmap_mode='r')[20318992:25007998]
        assert len(arrays[field]) == 4689006
        input_sha[str(rel)] = digest
    vocabulary = ROOT / 'data/gnn_v1_validation/metadata/vocabulary.json'
    assert sha(vocabulary) == cache['hashes'][str(vocabulary.relative_to(ROOT))]
    payment_values = read(vocabulary)['values']['payment_format']
    ach = payment_values.index('ACH') + 2
    non_ach = arrays['payment_format_cat'] != ach
    dates = arrays['timestamp'].astype('datetime64[us]').astype('datetime64[D]').astype(str)
    direct_groups = {'payment_format': arrays['payment_format_cat'],
                     'ach_group': np.where(non_ach, 'non_ACH', 'ACH'), 'date': dates}
    payment_labels = {'payment_format': {i + 2: name for i, name in enumerate(payment_values)}}
    models, scores, labels = {}, {}, None
    for name in NAMES:
        entry = by_name[name]
        training = verified_training(ROOT, name)
        cfg = training['config']
        assert training['model_sha256'] == entry['model_sha256']
        assert training['config_sha256'] == entry['training_config_sha256']
        assert training['training_result_sha256'] == entry['training_result_sha256']
        assert cfg['epochs'] == 1 and cfg['seed'] == 20260920
        assert cfg['positive_weight'] == 1140.9013150500168
        assert cfg['optimizer'] == {'name': 'AdamW', 'learning_rate': .0003, 'weight_decay': .0001, 'clip_norm': 5}
        assert cfg['model']['width'] == 64 and cfg['model']['dropout'] == .1
        assert cfg['dates'] == [f'2022-09-{d:02}' for d in range(1, 11)]
        assert cfg['runtime']['cuda'] and cfg['runtime']['gpu']
        acceptance = None
        if name != 'lr_only':
            assert cfg['projection'] == projection_contract(name)
            group_loss = cfg['group_loss']
            expected = {'ACH': .8285154011, 'non_ACH': 1.9474728463} if name == 'features_balanced' else {'ACH': 1., 'non_ACH': 1.}
            for group, factor in expected.items():
                assert math.isclose(group_loss['factors'][group], factor, abs_tol=5e-11, rel_tol=0)
            assert group_loss['negatives'] == 1 and group_loss['applied'] == (name == 'features_balanced')
            assert group_loss['positive_counts'] == {'ACH': 15067, 'non_ACH': 2727}
            avg = sum(group_loss['factors'][g] * group_loss['positive_counts'][g] for g in expected) / 17794
            assert math.isclose(avg, 1., abs_tol=1e-12)
            for key in ('config', 'manifest'):
                assert sha(ROOT / cfg['sidecar']['root'] / f'{key}.json') == cfg['sidecar'][f'{key}_sha256']
            acceptance_path = OUT / 'trials' / name / 'acceptance.json'
            acceptance = read(acceptance_path)
            assert acceptance['status'] == 'passed' and acceptance['config_sha256'] == training['config_sha256']
            for key in ('masked_A_equals_B', 'B_C_identical_initialization', 'identical_training_rng', 'checkpoint_resume_verified'):
                assert acceptance[key] is True
        del training['model']
        gc.collect()
        y, score, ev = load_predictions(ROOT, name, arrays['timestamp'], labels)
        if labels is None:
            labels = y
        assert len(y) == 4689006 and int(y.sum()) == 6420 and int(y[non_ach].sum()) == 715
        assert ev['config_sha256'] == entry['evaluation_config_sha256'] and ev['metrics_sha256'] == entry['metrics_sha256']
        assert ev['metrics'] == entry['metrics']
        threshold = independent_threshold(y, score)
        assert threshold == entry['precision80']['threshold']
        assert confusion(y, score >= threshold, threshold) == entry['precision80']
        assert confusion(y[non_ach], score[non_ach] >= threshold, threshold) == entry['non_ach_precision80']
        direct = group_report(y, score, entry['precision80'], direct_groups, arrays['src'], arrays['dst'], payment_labels)
        for group in direct:
            assert direct[group] == entry['groups'][group], (name, group)
        # Historical support categories are already verified by the canonical
        # sidecar-bound comparison; check exhaustive coverage and sums here.
        groups_verified = []
        for group in ('payment_format', 'ach_group', 'date', *[f'{role}_support_{window}' for role in ('payer', 'receiver') for window in ('24h', '72h', '7d')]):
            details = entry['groups'][group]
            assert sum(v['samples'] for v in details.values()) == 4689006
            assert sum(v['positives'] for v in details.values()) == 6420
            if '_support_' in group:
                assert set(details) == {'not_applicable', '0', '1-4', '>=5'}
            for key in ('tp', 'fp', 'fn', 'tn', 'alerts'):
                assert sum(v['operating_point'][key] for v in details.values()) == entry['precision80'][key]
            assert all(v['operating_point']['threshold'] == threshold for v in details.values())
            groups_verified.append(group)
        p = entry['precision80']
        q = {'precision': 5*p['tp'] >= 4*p['alerts'], 'overall_recall': p['tp'] >= 3469,
             'non_ach_recall': entry['non_ach_precision80']['tp'] >= 143}
        assert entry['qualification'] == {'passed': all(q.values()), **q}
        scores[name] = score
        models[name] = {'model_path': entry['model_path'], 'model_sha256': training['model_sha256'],
                        'strict_load_cpu_passed': True, 'finite_parameters': True, 'epoch': 1,
                        'train_targets': 20318992, 'train_positives': 17794,
                        'validation_targets': len(y), 'validation_positives': int(y.sum()),
                        'prediction_blocks_sha_identity_verified': len(ev['blocks']),
                        'metrics_reproduced': True, 'precision80': p, 'non_ach_precision80': entry['non_ach_precision80'],
                        'qualification': entry['qualification'], 'parameter_contract_passed': True,
                        'groups_coverage_and_sums_verified': groups_verified,
                        'payment_date_groups_recomputed': True, 'acceptance_bound_to_training': acceptance is not None}
        print(json.dumps({'verified': name, 'model_sha256': training['model_sha256']}, ensure_ascii=True), flush=True)
    for label, saved in comparison['paired_changes'].items():
        before, after = label.split(' -> ')
        actual = paired_changes(labels, scores[before], by_name[before]['precision80'],
                                scores[after], by_name[after]['precision80'], arrays['src'], arrays['dst'])
        assert actual == saved, label
    assert {'features_only -> features_pruned', 'features_pruned -> features_balanced'} <= set(comparison['paired_changes'])
    eligible = [n for n in NAMES[1:] if models[n]['qualification']['passed']]
    selected = max(eligible, key=lambda n: (by_name[n]['precision80']['recall'], by_name[n]['non_ach_precision80']['recall'], by_name[n]['metrics']['average_precision'])) if eligible else 'lr_only'
    best = read(OUT / 'best_model.json')
    assert best['comparison_sha256'] == EXPECTED_COMPARISON and best['name'] == selected == comparison['best']['name']
    assert best['path'] == by_name[selected]['model_path'] and best['sha256'] == by_name[selected]['model_sha256']
    assert best['validation_threshold'] == by_name[selected]['precision80']['threshold'] and best['test_evaluated'] is False
    assert sha(OUT / 'comparison.json') == EXPECTED_COMPARISON
    report = {'status': 'passed_with_separate_bank70_correction_required', 'verified_at': datetime.now(timezone.utc).isoformat(),
              'comparison_sha256': EXPECTED_COMPARISON, 'best_model_sha256': sha(OUT/'best_model.json'),
              'auditor_script_sha256': sha(__file__), 'models': models, 'selected': selected,
              'qualified_new_candidates': eligible, 'all_five_paired_changes_and_unique_account_counts_recomputed': True,
              'input_array_sha256': input_sha, 'engineering_record_and_frozen_anchors_verified': True,
              'historical_support_groups': 'All six groups present; buckets, coverage, global-threshold confusion sums verified against canonical comparison. Categories not independently regenerated in this audit.',
              'bank70_groups': 'Excluded: canonical raw-bank vs encoded-key error; separate correction required before final delivery.',
              'scope': 'Read-only verification of frozen Train/Validation artifacts; CPU strict weight loading only; no training, inference, Test access, or frozen artifact modification.'}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'audit': str(destination), 'status': report['status'], 'selected': selected}, ensure_ascii=True))


if __name__ == '__main__':
    main()
