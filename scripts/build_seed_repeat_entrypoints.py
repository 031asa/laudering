"""One-time, audited derivation of bounded loops from immutable v2 sources.

Never run after production starts. Originals remain byte-for-byte unchanged.
"""
from pathlib import Path

root = Path(__file__).resolve().parent/'gnn_features'
if any((root/name).exists() for name in ('repeat_train.py','repeat_evaluate.py')):
    raise RuntimeError('One-time derivation already applied; never overwrite reviewed entrypoints')
train = (root/'v2_train.py').read_text(encoding='utf-8')
loop = train[train.index('def _finish('):train.index("if __name__ == '__main__':")]

def sub(text, old, new, n=1):
    if text.count(old) != n: raise ValueError((old, text.count(old), n))
    return text.replace(old, new)

loop = sub(loop, 'seed=SEED', "seed=config['seed']")
loop = sub(loop, "report = dict(status='completed', trial=config['trial'],", "report = dict(status='completed', seed=config['seed'], trial=config['trial'],")
loop = sub(loop, "def run(project=ROOT, trial='features_only', mode='train', output=None, progress=None):", "def run(project=ROOT, trial='features_only', mode='train', output=None, progress=None, seed=None):")
loop = sub(loop, "    trial = check_trial(trial)", "    trial = check_trial(trial)\n    check_seed(seed)\n    verify_plan(project)")
loop = sub(loop, "destination = project / 'result/gnn_tuning_v2/trials' / trial", "destination = trial_root(project, trial, seed)")
loop = sub(loop, 'state = dict(status=\'preparing\', pid=os.getpid(), trial=trial,', "state = dict(status='preparing', pid=os.getpid(), trial=trial, seed=seed,")
loop = sub(loop, 'build_config(project, baseline, identities, sidecar, trial, ach_code, group_hashes)', 'build_config(project, baseline, identities, sidecar, trial, ach_code, group_hashes, seed)')
loop = loop.replace('V2', 'Repeat').replace('v2 configuration', 'repeat configuration').replace('Frozen v2', 'Frozen repeat').replace('independent v2 trial', 'independent repeat trial')
header = '''"""Full Train seed repeats; bounded loop derived from frozen v2_train.
Only two original recipes, three fixed seeds, one epoch each. No old writes.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from .model import BidirectionalGINE, to_torch
from .model_cache import ParquetBlockCache
from .train_model import (TOTAL, POSITIVES, sha, read_json, require_cuda,
    finite_parameters, fresh_metrics, checkpoint_payload, train_step as lr_step)
from .training_state import (ProcessLock, atomic_torch_save, save_checkpoint,
    load_checkpoint, capture_rng, restore_rng)
from .tune_model import verify_baseline
from .v2_train import (verify_trial_helper, verify_groups, check_space, make_optimizer,
    validate_v2_resume, validate_production_artifacts, _open_loader, _acceptance_data,
    _assert_nested_equal, GROUP_COUNTS, PAYMENT_COLUMN, WEIGHT, train_step as a_step)
from .v2_adapter import adapt_arrays
from .repeat_common import (ROOT, TRIALS, SEEDS, OUTPUT, atomic_json, check_trial,
    check_seed, trial_root, reference_root, verify_plan, make_initial_model,
    paused as _pause_requested)

def build_config(project, baseline, identities, sidecar, trial, ach_code, group_hashes, seed):
    reference = reference_root(project, trial)
    config = copy.deepcopy(read_json(reference/'config.json'))
    if config['seed'] != 20260920 or config['epochs'] != 1:
        raise ValueError('Reference recipe changed')
    config.update(version='seed_repeat_1', trial=trial, seed=check_seed(seed), **identities)
    config['reference_config_sha256'] = sha(reference/'config.json')
    config['repeat_plan_sha256'] = sha(project/OUTPUT/'config.json')
    config['input_hashes'].update(group_hashes)
    config['sidecar'] = dict(root=str(sidecar.root.relative_to(project)),
        config_sha256=sidecar.config_sha256, contract_sha256=sidecar.contract_sha256,
        manifest_sha256=sidecar.manifest_sha256, used_as_input=(trial == 'features_only'))
    config['group_loss'] = dict(ach_category_code=ach_code, positive_counts=GROUP_COUNTS,
        factors=dict(ACH=1., non_ACH=1.), applied=False)
    config['initialization'] = 'Fresh original architecture with declared seed; empty optimizer; frozen batch order'
    config['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        numpy=np.__version__, gpu=torch.cuda.get_device_name(0))
    config['scope'] = 'Paired seed replication; full Train one epoch; Test sealed; no deployment replacement'
    if (config['optimizer'] != dict(name='AdamW', learning_rate=.0003, weight_decay=.0001, clip_norm=5)
            or config['positive_weight'] != WEIGHT or config['batch_size'] != 64):
        raise ValueError('Original recipe hyperparameters differ')
    return config

def _adapt_batch(cached, index, sidecar, block, trial):
    arrays, labels, info = cached.batch(index)
    if trial == 'features_only':
        ids = arrays['target_edge_id']
        extra = sidecar.block(block, ids, cached.loader.builder.a['timestamp'][ids])
        arrays = adapt_arrays(arrays, extra, 'features_only')
    return arrays, labels, info

def train_step(model, optimizer, arrays, labels, device, ach_code, trial):
    if trial == 'features_only':
        return a_step(model, optimizer, arrays, labels, device, ach_code, 'features_only')
    if trial != 'lr_only': raise ValueError('Undeclared recipe')
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(WEIGHT, dtype=torch.float32, device=device))
    return lr_step(model, optimizer, loss_fn, arrays, labels, device)

def gpu_acceptance(project, output, config, config_hash, verified, sidecar, positive_ids, update, device):
    data = _acceptance_data(project, verified, sidecar, positive_ids, update)
    trial, ach = config['trial'], config['group_loss']['ach_category_code']
    adapted = [(adapt_arrays(a, e, 'features_only') if trial == 'features_only' else a, y) for a,e,y in data]
    groups = set()
    for a,y in adapted:
        codes = a['target_base_cat'][:, PAYMENT_COLUMN]
        if np.any((y == 1) & (codes == ach)): groups.add('ACH')
        if np.any((y == 1) & (codes != ach)): groups.add('non_ACH')
    if groups != set(GROUP_COUNTS): raise ValueError('Real acceptance needs both positive groups')
    model = make_initial_model(config, device); optimizer = make_optimizer(model)
    if optimizer.state: raise ValueError('Fresh optimizer is not empty')
    losses = []
    for a,y in adapted: losses.append(train_step(model, optimizer, a, y, device, ach, trial))
    finite_parameters(model)
    scratch = output/'acceptance_checkpoints'
    if scratch.exists() and any(scratch.iterdir()):
        raise ValueError('Interrupted acceptance artifacts require review, not overwrite')
    check_space(output)
    save_checkpoint(scratch, dict(config_sha256=config_hash, seed=config['seed'],
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng()))
    a,y = adapted[-1]
    expected = train_step(model, optimizer, a, y, device, ach, trial)
    expected_rng = capture_rng()
    loaded = load_checkpoint(scratch)
    if loaded['config_sha256'] != config_hash or loaded['seed'] != config['seed']:
        raise ValueError('Acceptance identity mismatch')
    resumed = make_initial_model(config, device); opt = make_optimizer(resumed)
    resumed.load_state_dict(loaded['model'], strict=True); opt.load_state_dict(loaded['optimizer'])
    restore_rng(loaded['rng'])
    actual = train_step(resumed, opt, a, y, device, ach, trial)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    _assert_nested_equal(model.state_dict(), resumed.state_dict())
    _assert_nested_equal(optimizer.state_dict(), opt.state_dict())
    _assert_nested_equal(expected_rng, capture_rng())
    finite_parameters(resumed)
    report = dict(status='passed', config_sha256=config_hash, seed=config['seed'], trial=trial,
        positive_groups=sorted(groups), real_batches=len(data), checkpoint_resume_verified=True,
        empty_optimizer=True, original_recipe=True, step_losses=losses)
    check_space(output); atomic_json(output/'acceptance.json', report)
    del model, optimizer, resumed, opt, loaded
    gc.collect(); torch.cuda.empty_cache()
    return report

'''
# Match the original adapter's known cache-loader interface, not an assumed one.
old_adapt = train[train.index('def _adapt_batch('):train.index('def _acceptance_data(')]
print(old_adapt)
cli = '''
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--trial', choices=TRIALS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--mode', choices=('accept','train'), default='train')
    a = p.parse_args()
    result = run(a.project, a.trial, a.mode, seed=a.seed)
    if result.get('status') == 'paused': raise SystemExit(75)
'''
(root/'repeat_train.py').write_text(header+loop+cli, encoding='utf-8')

ev = (root/'v2_evaluate.py').read_text(encoding='utf-8')
ev = ev[ev.index('def verified_training('):ev.index("if __name__ == '__main__':")]
ev = sub(ev, 'def verified_training(project, trial):', 'def verified_training(project, trial, seed):')
ev = sub(ev, "folder = project / ('result/gnn_tuning_v1' if trial == 'lr_only' else 'result/gnn_tuning_v2') / 'trials' / trial", 'folder = trial_root(project, trial, seed)\n    verify_plan(project)')
ev = sub(ev, "    if trial == 'lr_only' and digest != BASELINE_SHA:\n        raise ValueError('Frozen lr_only reference hash differs')", "    if config.get('seed') != seed or result.get('seed') != seed or config.get('trial') != trial:\n        raise ValueError('Repeat seed/trial identity mismatch')\n    if config.get('repeat_plan_sha256') != sha(project/OUTPUT/'config.json'):\n        raise ValueError('Repeat plan binding mismatch')")
ev = sub(ev, "    architecture = saved.get('config') or saved.get('model_config')", "    if saved.get('seed') != seed or saved.get('trial') != trial:\n        raise ValueError('Saved repeat seed/trial differs')\n    architecture = saved.get('config') or saved.get('model_config')")
ev = sub(ev, 'def run(project, trial, output=None, progress=None):', 'def run(project, trial, output=None, progress=None, seed=None):')
ev = sub(ev, "root = project/'result/gnn_tuning_v2'", "root = project/OUTPUT\n    check_seed(seed)")
ev = ev.replace("root/'evaluations'/trial", 'eval_root(project, trial, seed)')
ev = sub(ev, 'trained = verified_training(project, trial)', 'trained = verified_training(project, trial, seed)')
ev = sub(ev, "state = dict(status='preparing', pid=os.getpid(), trial=trial,", "state = dict(status='preparing', pid=os.getpid(), trial=trial, seed=seed,")
ev = sub(ev, 'config = dict(version=2, trial=trial,', "config = dict(version='seed_repeat_1', trial=trial, seed=seed,")
ev = sub(ev, "('v2_evaluate.py', 'v2_adapter.py', 'v2_features.py', 'evaluation_metrics.py',", "('repeat_evaluate.py', 'repeat_common.py', 'v4_metrics.py', 'v2_adapter.py', 'v2_features.py', 'evaluation_metrics.py',")
ev = sub(ev, 'batch = adapt_arrays(batch, extra, trial)', "batch = adapt_arrays(batch, extra, 'features_only') if trial == 'features_only' else batch")
ev = sub(ev, 'metrics=binary_metrics(labels, scores)', 'metrics=score_metrics(labels, scores)')
ev = sub(ev, "report = dict(status='completed', config_sha256=config_hash,", "report = dict(status='completed', seed=seed, trial=trial, config_sha256=config_hash,")
evheader = '''"""Independent complete Validation predictions for frozen seed repeats."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
import torch
from .build import load_archive
from .cache_train import array_sha, sha
from .engine import FeatureEngine
from .evaluate_validation import ValidationEdgeStore, prediction_table, verify_predictions
from .loader import TargetDay, TemporalBatchBuilder
from .model import BidirectionalGINE, to_torch
from .model_cache import ParquetBlockCache
from .training_state import ProcessLock
from .validation_cache import DATES, TOTAL, POSITIVES, END, read, validation_spec
from .v2_evaluate import checked_path, _disk, confusion, precision80_point
from .v4_metrics import score_metrics
from .repeat_common import (ROOT, TRIALS, SEEDS, OUTPUT, check_seed, trial_root,
    eval_root, verify_plan, atomic_json)
TRAIN_TOTAL, TRAIN_POSITIVES = 20318992, 17794
'''
evcli = '''
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--trial', choices=TRIALS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = p.parse_args()
    result = run(a.project, a.trial, seed=a.seed)
    if result.get('status') == 'paused': raise SystemExit(75)
'''
(root/'repeat_evaluate.py').write_text(evheader+ev+evcli, encoding='utf-8')
