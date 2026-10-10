"""Predeclared seed repetitions of frozen lr_only and features_only recipes."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from .cache_train import sha
from .training_state import _atomic_write

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20261009, 20261010, 20261011)
TRIALS = ('lr_only', 'features_only')
OLD_SEED = 20260920
OUTPUT = 'result/gnn_seed_repeats'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_json(path, value):
    data = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False).encode('utf-8')
    return _atomic_write(path, lambda f: f.write(data))

def check_trial(trial):
    if trial not in TRIALS: raise ValueError('Undeclared repeat model')
    return trial

def check_seed(seed):
    if type(seed) is not int or seed not in SEEDS: raise ValueError('Undeclared repeat seed')
    return seed

def trial_root(project, trial, seed):
    return Path(project)/OUTPUT/'trials'/str(check_seed(seed))/check_trial(trial)

def eval_root(project, trial, seed):
    return Path(project)/OUTPUT/'evaluations'/str(check_seed(seed))/check_trial(trial)

def reference_root(project, trial):
    check_trial(trial)
    version = 'gnn_tuning_v1' if trial == 'lr_only' else 'gnn_tuning_v2'
    return Path(project)/'result'/version/'trials'/trial

def source_hashes(project):
    paths = sorted((Path(project)/'scripts/gnn_features').glob('*.py'))
    paths += [Path(project)/'scripts/start_gnn_seed_repeats.ps1', Path(project)/'scripts/test_gnn_seed_repeats.py']
    return {str(p.relative_to(project)): sha(p) for p in paths}

def verify_plan(project):
    root = Path(project)/OUTPUT
    plan = read(root/'config.json')
    if (plan['seeds'] != list(SEEDS) or plan['trials'] != list(TRIALS)
            or plan['source_sha256'] != source_hashes(project)):
        raise ValueError('Repeat plan/source identity mismatch')
    for rel, digest in plan['frozen_anchors'].items():
        if sha(Path(project)/rel) != digest: raise ValueError('Frozen reference changed: '+rel)
    return plan

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_initial_model(config, device='cpu'):
    # Faithfully reproduce each original architecture's fresh initialization.
    # A is NOT a zero-column extension of lr_only: that would change its recipe.
    from .model import BidirectionalGINE
    seed_all(config['seed'])
    return BidirectionalGINE(**config['model']).to(device)

def paused(project):
    p = Path(project)/OUTPUT/'pause_request.json'
    return p.exists() and read(p).get('enabled') is True
