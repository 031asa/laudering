"""V2 projection, fair initialization, positive-only weighting and resume checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gnn_features import v2_adapter as adapter
from gnn_features import v2_train as runner
from gnn_features.model import to_torch


def fixture_arrays():
    rng = np.random.default_rng(4)
    return dict(x_num=rng.random((3, 63), dtype=np.float32),
                x_cat=np.array([[1], [2], [3]], dtype=np.int64),
                edge_attr_num=rng.random((2, 9), dtype=np.float32),
                edge_attr_cat=np.ones((2, 4), dtype=np.int64),
                edge_index_in=np.array([[0, 1], [1, 2]], dtype=np.int64),
                edge_index_out=np.array([[1, 2], [0, 1]], dtype=np.int64),
                edge_label_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
                target_base_num=rng.random((2, 8), dtype=np.float32),
                target_base_cat=np.array([[2, 1, 1, 1], [3, 1, 1, 1]], dtype=np.int64),
                target_attr=rng.random((2, 110), dtype=np.float32),
                target_edge_id=np.array([7, 8], dtype=np.int64))


def model_config(trial):
    return dict(trial=trial, model=dict(bank_count=5, edge_category_sizes=[5] * 4,
                                      width=64, dropout=.1))


def test_prunes_all_graph_nodes_and_masks_without_touching_categories_or_cache():
    original = fixture_arrays()
    before = {k: v.copy() for k, v in original.items()}
    extra = np.arange(118, dtype=np.float32).reshape(2, 59)
    full = adapter.adapt_arrays(original, extra, 'features_only')
    pruned = adapter.adapt_arrays(original, extra, 'features_pruned')
    assert full['x_num'].shape == (3, 63) and full['target_attr'].shape == (2, 169)
    assert pruned['x_num'].shape == (3, 57) and pruned['target_attr'].shape == (2, 165)
    np.testing.assert_array_equal(pruned['x_num'], original['x_num'][:, adapter.NODE_KEEP])
    np.testing.assert_array_equal(pruned['target_attr'][:, :106], original['target_attr'][:, adapter.R_KEEP])
    np.testing.assert_array_equal(pruned['target_attr'][:, 106:], extra)
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])
        if key not in ('x_num', 'target_attr'):
            assert pruned[key] is original[key]
    removed = set(adapter.NODE_NAMES) - set(adapter.projection_contract('features_pruned')['node_columns'])
    assert len(removed) == 6
    assert 'n_in_ach_share_24h__valid' in removed
    assert 'n_out_ach_share_24h__valid' in removed
    assert len(set(adapter.R_NAMES) - set(adapter.projection_contract('features_pruned')['original_r_columns'])) == 4


@pytest.mark.parametrize('extra', [np.zeros((2, 58), np.float32), np.zeros((2, 59), np.float64),
                                 np.full((2, 59), np.nan, np.float32)])
def test_adapter_rejects_invalid_sidecar(extra):
    with pytest.raises(ValueError):
        adapter.adapt_arrays(fixture_arrays(), extra, 'features_only')


def test_balanced_and_pruned_initialization_share_values_and_rng():
    b = adapter.make_initial_model(model_config('features_pruned'))
    state_b = torch.get_rng_state().clone()
    c = adapter.make_initial_model(model_config('features_balanced'))
    assert torch.equal(state_b, torch.get_rng_state())
    for key, value in b.state_dict().items():
        assert torch.equal(value, c.state_dict()[key])
    a = adapter.make_initial_model(model_config('features_only'))
    assert torch.equal(state_b, torch.get_rng_state())
    expected = adapter.project_initial_state(a.state_dict())
    for key, value in b.state_dict().items():
        assert torch.equal(value, expected[key])
    assert torch.equal(b.node_encoder[0].weight[:, 57:], a.node_encoder[0].weight[:, 63:])


def test_masked_A_forward_equals_projected_B():
    a = adapter.make_initial_model(model_config('features_only')).eval()
    b = adapter.make_initial_model(model_config('features_pruned')).eval()
    arrays = fixture_arrays()
    extra = np.random.default_rng(1).random((2, 59), dtype=np.float32)
    pruned = adapter.adapt_arrays(arrays, extra, 'features_pruned')
    full = adapter.adapt_arrays(arrays, extra, 'features_only')
    full['x_num'] = full['x_num'].copy()
    full['x_num'][:, sorted(set(range(63)) - set(adapter.NODE_KEEP))] = 0
    full['target_attr'][:, sorted(set(range(110)) - set(adapter.R_KEEP))] = 0
    with torch.no_grad():
        torch.testing.assert_close(a(to_torch(full)), b(to_torch(pruned)), rtol=1e-5, atol=1e-6)


def test_weight_budget_and_positive_only_grouping():
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    codes = np.array([2, 3, 2, 3], dtype=np.int64)
    factors = runner.positive_factors(labels, codes, 2, 'features_balanced')
    np.testing.assert_array_equal(factors[:2], np.ones(2))
    np.testing.assert_allclose(factors[2:], [.8285154010864452, 1.947472846289157])
    for trial in ('features_only', 'features_pruned'):
        np.testing.assert_array_equal(runner.positive_factors(labels, codes, 2, trial), np.ones(4))
    assert sum(runner.GROUP_COUNTS[k] * runner.GROUP_FACTORS[k] for k in runner.GROUP_COUNTS) == pytest.approx(17794)


def test_loss_equals_unweighted_baseline_when_factors_one():
    logits = torch.tensor([-4., 0., 3.], requires_grad=True)
    labels = torch.tensor([0., 1., 1.])
    actual = runner.weighted_loss(logits, labels, torch.ones(3))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels,
                                        pos_weight=torch.tensor(runner.WEIGHT))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_actual, = torch.autograd.grad(actual, logits, retain_graph=True)
    grad_expected, = torch.autograd.grad(expected, logits)
    # The fused mean backward and none-then-mean use different float32 orders.
    torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-6, atol=1e-7)


def test_balanced_loss_plain_mean_not_weight_sum_normalization():
    logits = torch.tensor([-4., 0., 3.], requires_grad=True)
    labels = torch.tensor([0., 1., 1.])
    factors = torch.tensor([1., runner.GROUP_FACTORS['ACH'], runner.GROUP_FACTORS['non_ACH']])
    actual = runner.weighted_loss(logits, labels, factors)
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels,
                           pos_weight=torch.tensor(runner.WEIGHT), reduction='none')
    torch.testing.assert_close(actual, (base * factors).sum() / 3, rtol=0, atol=0)
    assert not torch.isclose(actual, (base * factors).sum() / factors.sum())
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_negative_only_loss_is_unchanged():
    logits = torch.tensor([-.3, .7])
    labels = torch.zeros(2)
    torch.testing.assert_close(runner.weighted_loss(logits, labels, torch.ones(2)),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))


@pytest.mark.parametrize('logits,labels,factors', [([float('nan')], [0.], [1.]),
    ([0.], [0.], [2.]), ([0.], [1.], [-1.]), ([0.], [2.], [1.])])
def test_invalid_loss_rejected(logits, labels, factors):
    with pytest.raises(FloatingPointError):
        runner.weighted_loss(torch.tensor(logits), torch.tensor(labels), torch.tensor(factors))


def checkpoint():
    blocks = [dict(block='a', targets=64, positives=2, batches=1)]
    # Add an uncommitted block so this fixture is not a full-Train completion.
    blocks.append(dict(block='b', targets=64, positives=1, batches=1))
    payload = dict(config_sha256='hash', next_block=1, metrics=dict(targets=64, positives=2,
        batches=1, weighted_loss_sum=1., last_order_key=[1, 63], positive_groups=dict(ACH=1, non_ACH=1),
        blocks=[dict(block='a', targets=64, positives=2, batches=1,
                     positive_groups=dict(ACH=1, non_ACH=1), feature_manifest_sha256='f')]))
    return payload, blocks


def test_resume_rejects_group_and_block_mismatches():
    payload, blocks = checkpoint()
    assert runner.validate_v2_resume(payload, blocks, 'hash') == 1
    broken = copy.deepcopy(payload)
    broken['metrics']['positive_groups']['ACH'] = 2
    with pytest.raises(ValueError): runner.validate_v2_resume(broken, blocks, 'hash')
    broken = copy.deepcopy(payload)
    broken['metrics']['blocks'][0]['targets'] = 63
    with pytest.raises(ValueError): runner.validate_v2_resume(broken, blocks, 'hash')
    with pytest.raises(ValueError): runner.validate_v2_resume(payload, blocks, 'different')
    broken = copy.deepcopy(payload)
    broken['metrics']['blocks'][0]['positive_groups'] = dict(ACH=2, non_ACH=0)
    with pytest.raises(ValueError): runner.validate_v2_resume(broken, blocks, 'hash')


def test_semantic_checkpoint_mismatch_never_falls_back(tmp_path):
    payload, blocks = checkpoint()
    runner.save_checkpoint(tmp_path, payload)
    changed = copy.deepcopy(payload)
    changed['config_sha256'] = 'wrong'
    runner.save_checkpoint(tmp_path, changed)
    loaded = runner.load_checkpoint(tmp_path)
    assert Path(loaded['_loaded_from']).name == 'checkpoint_latest.pt'
    with pytest.raises(ValueError):
        runner.validate_v2_resume(loaded, blocks, 'hash')
    assert (tmp_path / 'checkpoint_previous.pt').exists()


def test_later_v1_helper_is_bound_to_completed_trial(tmp_path):
    folder = tmp_path / 'result/gnn_tuning_v1/trials/lr_only'
    folder.mkdir(parents=True)
    path = folder / 'config.json'
    config = dict(trial_code_sha256=runner.sha(Path(runner.__file__).with_name('tune_model.py')))
    path.write_text(json.dumps(config), encoding='utf-8')
    result_path = folder / 'training_result.json'
    result_path.write_text(json.dumps(dict(status='completed', config_sha256=runner.sha(path))), encoding='utf-8')
    runner.verify_trial_helper(tmp_path)
    config['trial_code_sha256'] = 'changed'
    path.write_text(json.dumps(config), encoding='utf-8')
    result_path.write_text(json.dumps(dict(status='completed', config_sha256=runner.sha(path))), encoding='utf-8')
    with pytest.raises(ValueError): runner.verify_trial_helper(tmp_path)


@pytest.mark.parametrize('artifact', ['checkpoint_latest.pt', 'checkpoint_previous.pt',
                                     'final_model.pt', 'training_result.json'])
def test_production_artifacts_cannot_trigger_replacement_acceptance(tmp_path, artifact):
    path = tmp_path / artifact
    path.write_bytes(b'preserve')
    with pytest.raises(ValueError):
        runner.validate_production_artifacts(tmp_path, accepted=False)
    assert path.read_bytes() == b'preserve'


@pytest.mark.parametrize('artifact', ['final_model.pt', 'training_result.json'])
def test_final_artifacts_cannot_be_overwritten_from_incomplete_checkpoint(tmp_path, artifact):
    path = tmp_path / artifact
    path.write_bytes(b'preserve')
    with pytest.raises(ValueError):
        runner.validate_production_artifacts(tmp_path, accepted=True, cursor=796)
    runner.validate_production_artifacts(tmp_path, accepted=True, cursor=797)
    assert path.read_bytes() == b'preserve'


def test_invalid_output_cannot_touch_cuda_or_v1(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'require_cuda', lambda: pytest.fail('must fail before CUDA'))
    for path in (tmp_path / 'result/gnn_train_gpu_v1', tmp_path / 'result/gnn_tuning_v1/trials/lr_only',
                 tmp_path / 'result/gnn_tuning_v2/trials/features_balanced', tmp_path.parent / 'elsewhere'):
        with pytest.raises(ValueError):
            runner.run(tmp_path, 'features_only', output=path)


def test_disk_floor_stops_before_bulk_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 19 * 1024**3})())
    with pytest.raises(RuntimeError): runner.check_space(tmp_path)


def test_unrecognized_trial_rejected():
    for name in ('lr_only', 'sqrt_weight', 'extra_epoch'):
        with pytest.raises(ValueError): adapter.projection_contract(name)
