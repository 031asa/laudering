"""The bounded trial runner preserves comparison and resume identities."""
import copy
from pathlib import Path

import pytest
import torch

from gnn_features import tune_model as runner


def test_only_two_fixed_candidates():
    for name, params in runner.TRIALS.items():
        assert runner.trial_parameters(params) == dict(name=name, **params)
    for params in [dict(learning_rate=1e-3, positive_weight=1),
                   dict(runner.TRIALS['lr_only'], epochs=2),
                   dict(runner.TRIALS['lr_only'], name='sqrt_weight')]:
        with pytest.raises(ValueError):
            runner.trial_parameters(params)


def test_make_model_uses_trial_learning_rate_and_weight():
    config = dict(model=dict(bank_count=5, edge_category_sizes=[4] * 4, width=64, dropout=.1),
                  optimizer=dict(learning_rate=.0003, weight_decay=.0001), positive_weight=33.)
    _, optimizer, loss = runner.make_model(config, 'cpu')
    assert optimizer.param_groups[0]['lr'] == .0003
    assert optimizer.param_groups[0]['weight_decay'] == .0001
    assert loss.pos_weight.item() == 33.


def test_config_copy_retains_fair_comparison(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'test-device')
    original = dict(model=dict(width=64, dropout=.1), optimizer=dict(learning_rate=.001,
                    weight_decay=.0001, clip_norm=5), seed=runner.SEED, epochs=1,
                    batch_size=64, dates=['2022-09-01'], input_hashes={'f': 'hash'})
    before = copy.deepcopy(original)
    first = runner.make_config(original, runner.TRIALS['lr_only'], {'baseline_config_sha256': 'hash'})
    second = runner.make_config(original, runner.TRIALS['sqrt_weight'], {'baseline_config_sha256': 'hash'})
    assert original == before
    assert first['model'] == second['model'] == original['model']
    assert first['optimizer'] == second['optimizer']
    assert first['positive_weight'] != second['positive_weight']
    assert first['seed'] == second['seed'] == runner.SEED


def test_resume_rejects_missing_batches_and_chronology():
    blocks = [dict(block='a', targets=64, positives=1, batches=1)]
    state = dict(config_sha256='hash', next_block=1, metrics=dict(targets=64, positives=1,
                 batches=1, weighted_loss_sum=1., last_order_key=[1, 63], blocks=[dict(block='a')]))
    assert runner.validate_resume(state, blocks, 'hash') == 1
    for field, invalid in [('batches', 2), ('last_order_key', None), ('targets', 65)]:
        altered = copy.deepcopy(state)
        altered['metrics'][field] = invalid
        with pytest.raises(ValueError):
            runner.validate_resume(altered, blocks, 'hash')
    with pytest.raises(ValueError):
        runner.validate_resume(state, blocks, 'wrong')


def test_reject_baseline_or_outside_output_before_cuda(tmp_path, monkeypatch):
    def no_cuda():
        pytest.fail('Invalid path must fail before touching CUDA')
    monkeypatch.setattr(runner, 'require_cuda', no_cuda)
    for output in [tmp_path / 'result', tmp_path / 'result/gnn_train_gpu_v1',
                   tmp_path / 'result/gnn_train_gpu_v1/child', tmp_path.parent / 'outside']:
        with pytest.raises(ValueError):
            runner.run_trial(tmp_path, output, runner.TRIALS['lr_only'])


def test_finite_loss_rejection_precedes_optimizer_update():
    class InvalidModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, arrays):
            return self.weight * torch.tensor([float('nan')])
    import numpy as np
    model = InvalidModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(FloatingPointError):
        runner.train_step(model, optimizer, torch.nn.BCEWithLogitsLoss(),
                          {}, np.array([0.], dtype=np.float32), 'cpu')
    assert model.weight.item() == 1.
    assert not optimizer.state
