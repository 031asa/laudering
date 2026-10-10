"""V4 new-column projection, identical initial RNG, gradient and resume invariants."""
import copy
import numpy as np
import pytest
import torch

from gnn_features import v4_adapter as adapter, v4_train as runner
from gnn_features import v2_adapter
from gnn_features.model import to_torch
from gnn_features.training_state import capture_rng, restore_rng
from test_gnn_v2_train import fixture_arrays, model_config, checkpoint


def inputs():
    rng = np.random.default_rng(42)
    return fixture_arrays(), rng.random((2, 59), dtype=np.float32), np.column_stack((rng.random((2,3)),np.ones((2,3)))).astype(np.float32)


@pytest.mark.parametrize('trial,dimension', zip(adapter.TRIALS, (175,)))
def test_only_target_R_appended_no_cache_mutation(trial, dimension):
    original, v2, extra = inputs()
    before = {k:v.copy() for k,v in original.items()}
    actual = adapter.adapt_arrays(original, v2, extra, trial)
    assert actual['target_attr'].shape == (2, dimension)
    np.testing.assert_array_equal(actual['target_attr'][:, :110], original['target_attr'])
    np.testing.assert_array_equal(actual['target_attr'][:, 110:169], v2)
    np.testing.assert_array_equal(actual['target_attr'][:, 169:], extra[:, :dimension-169])
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])
        if key != 'target_attr': assert actual[key] is original[key]


@pytest.mark.parametrize('bad', [np.zeros((2,5),np.float32), np.zeros((2,6),np.float64),
                                np.full((2,6),np.nan,np.float32)])
def test_distribution_identity_validation(bad):
    original, v2, _ = inputs()
    with pytest.raises(ValueError): adapter.adapt_arrays(original, v2, bad, adapter.TRIALS[0])


def test_undeclared_trials_and_input_initialization_rejected():
    with pytest.raises(ValueError): adapter.check_trial('features_only')
    a = v2_adapter.make_initial_model(model_config('features_only'))
    with pytest.raises(ValueError): adapter.expand_initial_state(a.state_dict(), 176)
    wrong = dict(a.state_dict())
    wrong['r_encoder.0.weight'] = torch.zeros(64, 110)
    with pytest.raises(ValueError): adapter.expand_initial_state(wrong, 175)


@pytest.mark.parametrize('trial', adapter.TRIALS)
def test_rebuilds_A_exact_shared_tensors_and_all_rng(trial):
    a = v2_adapter.make_initial_model(model_config('features_only'))
    expected_rng = capture_rng()
    model = adapter.make_initial_model(model_config(trial))
    runner._assert_nested_equal(expected_rng, capture_rng())
    for name, value in a.state_dict().items():
        candidate = model.state_dict()[name]
        if name == 'r_encoder.0.weight':
            assert torch.equal(value, candidate[:, :169])
            assert not torch.count_nonzero(candidate[:, 169:])
        else: assert torch.equal(value, candidate)
    assert not runner.make_optimizer(model).state


@pytest.mark.parametrize('trial', adapter.TRIALS)
def test_initial_forward_same_A_with_nonzero_new_inputs(trial):
    original, v2, extra = inputs()
    a = v2_adapter.make_initial_model(model_config('features_only')).eval()
    new = adapter.make_initial_model(model_config(trial)).eval()
    with torch.no_grad():
        expected = a(to_torch(v2_adapter.adapt_arrays(original, v2, 'features_only')))
        actual = new(to_torch(adapter.adapt_arrays(original, v2, extra, trial)))
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('trial', adapter.TRIALS)
def test_new_inputs_receive_gradients_and_resume_training_exactly(tmp_path, trial):
    original, v2, extra = inputs()
    data = adapter.adapt_arrays(original, v2, extra, trial)
    labels = np.array([0,1],np.int8)
    model = adapter.make_initial_model(model_config(trial))
    optimizer = runner.make_optimizer(model)
    runner.train_step(model, optimizer, data, labels, 'cpu', 2, trial)
    weight = model.r_encoder[0].weight[:,169:]
    assert torch.count_nonzero(weight[:,:3])
    runner.save_checkpoint(tmp_path, dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng()))
    expected_loss, _ = runner.train_step(model, optimizer, data, labels, 'cpu',2,trial)
    expected_rng = capture_rng()
    payload = runner.load_checkpoint(tmp_path)
    resumed = adapter.make_initial_model(model_config(trial))
    resumed.load_state_dict(payload['model'], strict=True)
    resumed_opt = runner.make_optimizer(resumed)
    resumed_opt.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    actual_loss, _ = runner.train_step(resumed,resumed_opt,data,labels,'cpu',2,trial)
    assert actual_loss == pytest.approx(expected_loss,abs=1e-7)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
    runner._assert_nested_equal(optimizer.state_dict(),resumed_opt.state_dict())
    runner._assert_nested_equal(expected_rng,capture_rng())


@pytest.mark.parametrize('trial',adapter.TRIALS)
def test_no_payment_reweighting(trial):
    y=np.array([0,0,1,1],np.int8)
    np.testing.assert_array_equal(runner.positive_factors(y,np.array([2,3,2,3]),2,trial),np.ones(4))
    logits=torch.tensor([-.3,.2,1.,-1.])
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(y,dtype=torch.float32),
        pos_weight=torch.tensor(runner.WEIGHT))
    torch.testing.assert_close(runner.weighted_loss(logits,torch.tensor(y,dtype=torch.float32),torch.ones(4)),expected)


def test_resume_requires_both_sidecar_identities_and_original_config():
    payload,blocks=checkpoint()
    with pytest.raises(ValueError): runner.validate_v4_resume(payload,blocks,'hash')
    payload['metrics']['blocks'][0]['structure_manifest_sha256']='distribution'
    assert runner.validate_v4_resume(payload,blocks,'hash')==1
    with pytest.raises(ValueError): runner.validate_v4_resume(payload,blocks,'different')
    changed=copy.deepcopy(payload)
    changed['metrics']['blocks'][0]['positive_groups']={'ACH':2,'non_ACH':0}
    with pytest.raises(ValueError): runner.validate_v4_resume(changed,blocks,'hash')


def test_projection_contract_column_dimensions_and_preserves_all_A_columns():
    from gnn_features.v2_features import EXTRA_COLUMNS as OLD
    from gnn_features.v4_features import EXTRA_COLUMNS as NEW
    actual=adapter.projection_contract(adapter.TRIALS[0])
    assert len(actual['node_columns'])==63 and len(actual['r_columns'])==175
    assert actual['r_columns'][:169]==list(v2_adapter.R_NAMES)+list(OLD)
    assert actual['r_columns'][169:]==list(NEW)


def test_early_acceptance_never_mixes_adjacency_index_families(monkeypatch):
    from types import SimpleNamespace
    from gnn_features import v4_features
    original={'in_ptr':'full_history_pointer','in_order':'full_history_order'}
    builder=SimpleNamespace(a=original,adj={'in_ptr':'external_only_pointer','in_edge_ids':'external_only_order'})
    def compute(actual,start,ids):
        assert actual is original
        assert actual['in_ptr']=='full_history_pointer'
        return 'consistent'
    monkeypatch.setattr(v4_features,'compute_selected',compute)
    assert runner.acceptance_structure_inputs(builder,0,np.array([1]))=='consistent'
