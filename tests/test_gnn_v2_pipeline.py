import numpy as np
import pytest
import json

from gnn_features.v2_audit import summarize, check_ach_relationship
from gnn_features.contract import N_FIELDS
from gnn_features.v2_pipeline import inside, TRIALS, verify_early_acceptance
from gnn_features.cache_train import sha
from pathlib import Path


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        inside(tmp_path, '../escape')
    assert inside(tmp_path, 'safe/file') == (tmp_path/'safe/file').resolve()
    assert TRIALS == ('features_only', 'features_pruned', 'features_balanced')


def test_audit_distinguishes_missing_zero_and_constant():
    report = summarize(np.array([[np.nan, 0., 4.], [1., 0., 4.], [2., 0., 4.]]), ['partial','zero','constant'])
    assert report['columns'][0]['valid'] == 2
    assert report['columns'][1]['valid'] == 3
    assert report['columns'][1]['constant']
    assert report['columns'][2]['constant']
    assert report['correlation_matrix'][0][1] is None


def test_share_identity_includes_empty_history_and_encoded_roundoff():
    names = [f['name'] for f in N_FIELDS]
    x = np.zeros((3, len(names)))
    for direction in ('in','out'):
        x[:, names.index(f'n_{direction}_all_count_24h')] = np.log1p([0, 200, 700000]).astype(np.float32)
        x[:, names.index(f'n_{direction}_ach_count_24h')] = np.log1p([0, 30, 30000]).astype(np.float32)
        x[:, names.index(f'n_{direction}_ach_share_24h')] = [np.nan, .15, 30000/700000]
    assert check_ach_relationship(x)['in']['checked'] == 2
    x[1, names.index('n_in_ach_share_24h')] = .9
    with pytest.raises(ValueError):
        check_ach_relationship(x)


def test_early_cuda_gate_rejects_changed_source_and_incomplete_proof(tmp_path):
    source = tmp_path / 'scripts/gnn_features'
    source.mkdir(parents=True)
    for name in ('v2_train.py', 'v2_adapter.py', 'v2_features.py', 'tune_model.py'):
        (source / name).write_text('fixture', encoding='utf-8')
    inventory = {str(p.relative_to(tmp_path)): sha(p) for p in source.iterdir()}
    out = tmp_path / 'result/gnn_tuning_v2/early_gpu_acceptance'
    results = {}
    for trial in TRIALS:
        folder = out / trial
        folder.mkdir(parents=True)
        config = folder / 'config.json'
        config.write_text(json.dumps(dict(trial=trial, v2_code_sha256=inventory)), encoding='utf-8')
        results[trial] = dict(status='passed', config_sha256=sha(config), positive_groups=['ACH', 'non_ACH'],
            compared_batches=2, masked_A_equals_B=True, B_C_identical_initialization=True,
            identical_training_rng=True, checkpoint_resume_verified=True)
        (folder / 'acceptance.json').write_text(json.dumps(results[trial]), encoding='utf-8')
    report = out / 'acceptance.json'
    report.write_text(json.dumps(dict(status='passed', trials=results)), encoding='utf-8')
    verify_early_acceptance(tmp_path)
    (source / 'v2_train.py').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Source changed'):
        verify_early_acceptance(tmp_path)
    (source / 'v2_train.py').write_text('fixture', encoding='utf-8')
    results[TRIALS[0]]['checkpoint_resume_verified'] = False
    report.write_text(json.dumps(dict(status='passed', trials=results)), encoding='utf-8')
    (out / TRIALS[0] / 'acceptance.json').write_text(json.dumps(results[TRIALS[0]]), encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        verify_early_acceptance(tmp_path)
