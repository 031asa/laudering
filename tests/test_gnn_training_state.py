import copy
import json
import random

import numpy as np
import pytest
import torch

from gnn_features import training_state as state


def train_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(torch.tensor([[1., 2.], [3., -1.]])) - 0.5).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def rng_draw():
    return (random.random(), np.random.random(3), torch.rand(3),
            torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else None)


def assert_draw_equal(first, second):
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    if first[3] is not None:
        assert torch.equal(first[3], second[3])


def test_checkpoint_restores_model_optimizer_rng_and_counts(tmp_path):
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = train_step(model, optimizer)
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   rng=state.capture_rng(), next_block=1,
                   metrics=dict(targets=64, positives=1, loss=loss), config_sha256="frozen")
    with state.ProcessLock(tmp_path):
        state.save_checkpoint(tmp_path, payload)
    expected_draw = rng_draw()
    expected_loss = train_step(model, optimizer)
    expected_model = copy.deepcopy(model.state_dict())
    loaded = state.load_checkpoint(tmp_path)
    assert loaded["_loaded_from"].endswith("checkpoint_latest.pt")
    assert loaded["metrics"]["targets"] == 64 and loaded["next_block"] == 1
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_model.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    state.restore_rng(loaded["rng"])
    assert_draw_equal(expected_draw, rng_draw())
    assert train_step(restored_model, restored_optimizer) == expected_loss
    for name, tensor in expected_model.items():
        assert torch.equal(restored_model.state_dict()[name], tensor)


def test_rotation_and_corruption_fallback_preserves_good_previous(tmp_path):
    for index in range(1, 4):
        state.save_checkpoint(tmp_path, dict(next_block=index, config_sha256="fixed"))
    assert state.load_checkpoint(tmp_path)["next_block"] == 3
    previous = tmp_path / "checkpoint_previous.pt"
    assert torch.load(previous, weights_only=True)["next_block"] == 2
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"interrupted archive")
    with pytest.warns(RuntimeWarning, match="falling back"):
        assert state.load_checkpoint(tmp_path)["next_block"] == 2
    state.save_checkpoint(tmp_path, dict(next_block=3, config_sha256="fixed"))
    assert torch.load(previous, weights_only=True)["next_block"] == 2
    assert state.load_checkpoint(tmp_path)["next_block"] == 3
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "checkpoint_latest.pt", "checkpoint_previous.pt"]


def test_interrupted_publication_keeps_latest(tmp_path, monkeypatch):
    state.save_checkpoint(tmp_path, dict(next_block=1))
    replace = state.os.replace
    def interrupted(source, destination):
        if destination.name == "checkpoint_latest.pt":
            raise PermissionError("simulated Windows handle")
        return replace(source, destination)
    monkeypatch.setattr(state.os, "replace", interrupted)
    monkeypatch.setattr(state.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        state.save_checkpoint(tmp_path, dict(next_block=2))
    assert state.load_checkpoint(tmp_path)["next_block"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "checkpoint_pending.pt").exists()


def test_transient_replace_retry_and_json(tmp_path, monkeypatch):
    replace = state.os.replace
    calls = []
    def busy_once(source, destination):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("busy")
        return replace(source, destination)
    monkeypatch.setattr(state.os, "replace", busy_once)
    monkeypatch.setattr(state.time, "sleep", lambda seconds: None)
    state.atomic_json(tmp_path / "status.json", dict(status="running", targets=64))
    assert json.loads((tmp_path / "status.json").read_text())["targets"] == 64
    assert len(calls) == 2


def test_lock_excludes_other_writer_and_releases(tmp_path):
    with state.ProcessLock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with state.ProcessLock(tmp_path):
                pass
    with state.ProcessLock(tmp_path):
        pass


def test_semantic_mismatch_is_returned_to_trainer_without_fallback(tmp_path):
    state.save_checkpoint(tmp_path, dict(config_sha256="expected", next_block=1))
    state.save_checkpoint(tmp_path, dict(config_sha256="different", next_block=2))
    assert state.load_checkpoint(tmp_path)["config_sha256"] == "different"


def test_failed_serialization_does_not_modify_existing_checkpoints(tmp_path):
    state.save_checkpoint(tmp_path, dict(next_block=1))
    before = (tmp_path / "checkpoint_latest.pt").read_bytes()
    with pytest.raises((AttributeError, TypeError)):
        state.save_checkpoint(tmp_path, dict(unserializable=lambda: None))
    assert (tmp_path / "checkpoint_latest.pt").read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
