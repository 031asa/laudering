"""Atomic, bounded training checkpoints. Call writers under ``ProcessLock``."""
from __future__ import annotations

import json
import msvcrt
import os
from pathlib import Path
import pickle
import random
import shutil
import time
import warnings

import numpy as np
import torch


def _root(directory):
    path = Path(directory).absolute()
    resolved = path.resolve()
    if path != resolved or path.is_symlink():
        raise ValueError(f"Unsafe output directory: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _inside(root, name):
    path = root / name
    if path.is_symlink() or path.resolve().parent != root or path.parent != root:
        raise ValueError(f"Unsafe output file: {path}")
    return path


def _replace(source, destination):
    # A transient Windows file handle must not invalidate the previous checkpoint.
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (2 ** attempt))


class ProcessLock:
    """Nonblocking OS lock released automatically when the process exits."""

    def __init__(self, output_dir):
        self.root = _root(output_dir)
        self.stream = None

    def __enter__(self):
        if self.stream is not None:
            raise RuntimeError("ProcessLock already held")
        stream = _inside(self.root, "process.lock").open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            stream.close()
            raise RuntimeError(f"Training output is locked: {self.root}") from exc
        self.stream = stream
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                stream.close()


def _atomic_write(path, writer):
    path = Path(path)
    root = _root(path.parent)
    target = _inside(root, path.name)
    temporary = _inside(root, path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck paths before publication; never move/delete a directory.
        _inside(root, target.name)
        _inside(root, temporary.name)
        _replace(temporary, target)
    finally:
        if temporary.exists():
            _inside(root, temporary.name).unlink()
    return target


def atomic_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    return _atomic_write(path, lambda stream: stream.write(data))


def atomic_torch_save(path, payload):
    """Write one file atomically; final models can use this without rotation."""
    return _atomic_write(path, lambda stream: torch.save(payload, stream))


def capture_rng():
    """Serialize RNGs using primitives/tensors accepted by weights_only=True."""
    name, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": name,
            "state": torch.from_numpy(state.astype(np.int64)),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    cuda_states = state["torch_cuda"]
    if cuda_states and (
        not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()
    ):
        raise RuntimeError("CUDA RNG device count differs from checkpoint")
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((
        numpy_state["name"],
        numpy_state["state"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


class CheckpointReadError(RuntimeError):
    pass


def _read_checkpoint(path, map_location="cpu"):
    try:
        value = torch.load(path, map_location=map_location, weights_only=True)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError, IndexError) as exc:
        raise CheckpointReadError(f"Unreadable checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointReadError(f"Checkpoint payload must be a dict: {path}")
    return value


def load_checkpoint(out, map_location="cpu"):
    """Load latest, with an explicit warning only for missing/unreadable latest.

    Config, identity, model, and metric validation belongs to the caller and must
    never trigger fallback. I/O permission errors propagate instead of fallback.
    """
    root = _root(out)
    latest = _inside(root, "checkpoint_latest.pt")
    previous = _inside(root, "checkpoint_previous.pt")
    latest_error = None
    if latest.exists():
        try:
            value = _read_checkpoint(latest, map_location)
        except CheckpointReadError as exc:
            latest_error = exc
        else:
            value["_loaded_from"] = str(latest)
            return value
    if previous.exists():
        value = _read_checkpoint(previous, map_location)
        reason = str(latest_error) if latest_error else "latest checkpoint is missing"
        warnings.warn(
            f"Checkpoint recovery: {reason}; falling back to {previous}",
            RuntimeWarning, stacklevel=2,
        )
        value["_loaded_from"] = str(previous)
        return value
    if latest_error is not None:
        raise latest_error
    return None


def save_checkpoint(out, payload):
    """Publish latest while retaining a readable previous checkpoint.

    Copying the old latest to previous leaves latest present throughout rotation.
    If latest is corrupt after a fallback, do not replace the valid previous.
    Caller holds ProcessLock and validates semantic checkpoint identity.
    """
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dict")
    root = _root(out)
    latest = _inside(root, "checkpoint_latest.pt")
    previous = _inside(root, "checkpoint_previous.pt")
    # Fully serialize the new state before modifying either recovery checkpoint.
    pending = _inside(root, "checkpoint_pending.pt")
    atomic_torch_save(pending, payload)
    try:
        if latest.exists():
            try:
                _read_checkpoint(latest)
            except CheckpointReadError:
                # load_checkpoint has reported recovery; preserve the good backup.
                pass
            else:
                def copy_latest(stream):
                    with latest.open("rb") as source:
                        shutil.copyfileobj(source, stream, length=1024 * 1024)
                _atomic_write(previous, copy_latest)
        _inside(root, latest.name)
        _inside(root, pending.name)
        _replace(pending, latest)
    finally:
        if pending.exists():
            _inside(root, pending.name).unlink()
    return latest
