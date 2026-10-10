"""Three fixed v2 Train trials; v1 source, caches, and artifacts stay immutable."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
import torch

from .loader import TemporalLoader, ECAT
from .model import BidirectionalGINE, to_torch
from .model_cache import ParquetBlockCache
from .train_model import (TOTAL, POSITIVES, SEED, sha, read_json, require_cuda,
                          finite_parameters, fresh_metrics, checkpoint_payload)
from .training_state import (ProcessLock, atomic_json, atomic_torch_save,
                             save_checkpoint, load_checkpoint, capture_rng, restore_rng)
from .tune_model import verify_baseline, validate_resume
from .v2_adapter import (TRIALS, NODE_KEEP, R_KEEP, R_NAMES, adapt_arrays, architecture,
                         projection_contract, check_trial, make_initial_model)

ROOT = Path(__file__).resolve().parents[2]
WEIGHT = 1140.9013150500168
GROUP_COUNTS = {'ACH': 15067, 'non_ACH': 2727}
_NORMALIZER = POSITIVES / sum(math.sqrt(x) for x in GROUP_COUNTS.values())
GROUP_FACTORS = {name: _NORMALIZER / math.sqrt(count) for name, count in GROUP_COUNTS.items()}
MIN_FREE_BYTES = 20 * 1024**3
PAYMENT_COLUMN = ECAT.index('payment_format_cat')


def check_space(path):
    if shutil.disk_usage(path).free < MIN_FREE_BYTES:
        raise RuntimeError('Fewer than 20 GiB free; no additional artifacts may be written')


def verify_trial_helper(project):
    """The original baseline predates tune_model; bind its later frozen helper."""
    folder = project / 'result/gnn_tuning_v1/trials/lr_only'
    config_path = folder / 'config.json'
    result = read_json(folder / 'training_result.json')
    config = read_json(config_path)
    if (result.get('status') != 'completed' or result.get('config_sha256') != sha(config_path)
            or sha(Path(__file__).with_name('tune_model.py')) != config.get('trial_code_sha256')):
        raise ValueError('Frozen v1 trial helper/config identity changed')


def positive_factors(labels, payment_codes, ach_code, trial):
    check_trial(trial)
    labels, codes = np.asarray(labels), np.asarray(payment_codes)
    if (labels.ndim != 1 or codes.shape != labels.shape or not np.isin(labels, [0, 1]).all()
            or codes.dtype.kind not in 'iu' or np.any(codes < 0)):
        raise ValueError('Group loss labels/payment identities invalid')
    factors = np.ones(labels.shape, dtype=np.float32)
    if trial == 'features_balanced':
        factors[(labels == 1) & (codes == ach_code)] = GROUP_FACTORS['ACH']
        factors[(labels == 1) & (codes != ach_code)] = GROUP_FACTORS['non_ACH']
    return factors


def weighted_loss(logits, labels, factors):
    if logits.shape != labels.shape or factors.shape != labels.shape or logits.ndim != 1:
        raise ValueError('Loss tensors must be aligned one-dimensional vectors')
    if (not torch.isfinite(logits).all() or not torch.isfinite(factors).all()
            or not ((labels == 0) | (labels == 1)).all() or not (factors > 0).all()
            or not (factors[labels == 0] == 1).all()):
        raise FloatingPointError('Invalid/nonfinite logits, labels, or positive-only factors')
    values = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels,
        pos_weight=logits.new_tensor(WEIGHT), reduction='none')
    return (values * factors).mean()


def train_step(model, optimizer, arrays, labels, device, ach_code, trial):
    factors = positive_factors(labels, arrays['target_base_cat'][:, PAYMENT_COLUMN], ach_code, trial)
    batch = to_torch(arrays, device=device)
    y = torch.as_tensor(labels.copy(), dtype=torch.float32, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = weighted_loss(model(batch), y, torch.as_tensor(factors, device=device))
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm.detach())


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


def verify_groups(project, baseline, update):
    """Read frozen labels/category IDs; no model feature ever receives labels."""
    archive = project / 'data/gnn_v1_train'
    vocabulary = read_json(archive / 'metadata/vocabulary.json')['values']['payment_format']
    ach_code = vocabulary.index('ACH') + 2
    category_path = archive / 'history_index/payment_format_cat.npy'
    categories = np.load(category_path, mmap_mode='r')
    if categories.shape != (TOTAL,):
        raise ValueError('Train category coverage differs')
    seen = np.zeros(TOTAL, dtype=bool)
    groups = dict(ACH=0, non_ACH=0)
    positive_batches = {}
    for date in baseline['dates']:
        folder = archive / 'supervision/targets' / ('transaction_date=' + date)
        paths = sorted(folder.glob('*.parquet'))
        if not paths:
            raise ValueError('Train supervision missing: ' + date)
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=131072, columns=['edge_id', 'label']):
                ids = batch.column('edge_id').to_numpy().astype(np.int64)
                labels = batch.column('label').to_numpy()
                if (np.any(ids < 0) or np.any(ids >= TOTAL) or len(np.unique(ids)) != len(ids)
                        or seen[ids].any() or not np.isin(labels, [0, 1]).all()):
                    raise ValueError('Train group audit target identity differs')
                seen[ids] = True
                pos_ids = ids[labels == 1]
                for name, mask in (('ACH', categories[pos_ids] == ach_code),
                                   ('non_ACH', categories[pos_ids] != ach_code)):
                    groups[name] += int(mask.sum())
                    if mask.any() and name not in positive_batches:
                        positive_batches[name] = int(pos_ids[mask][0])
        update(stage='verify_train_groups', targets_checked=int(seen.sum()), positive_groups=groups)
    if not seen.all() or groups != GROUP_COUNTS:
        raise ValueError('Frozen Train ACH/non-ACH positive counts differ')
    hashes = {str(path.relative_to(project)): sha(path) for path in
              (archive / 'metadata/vocabulary.json', category_path,
               project / 'data/gnn_tensor_cache_v1/manifest.json',
               project / 'data/gnn_tensor_cache_v1/numeric.npy',
               project / 'data/gnn_tensor_cache_v1/category.npy')}
    return ach_code, hashes, positive_batches


def build_config(project, baseline, identities, sidecar, trial, ach_code, group_hashes):
    from . import v2_features
    config = copy.deepcopy(baseline)
    config.update(version=2, trial=check_trial(trial), **identities)
    config['model'] = architecture(config['model'], trial)
    config['optimizer'].update(learning_rate=3e-4, weight_decay=1e-4, clip_norm=5)
    config['positive_weight'] = WEIGHT
    config['input_hashes'].update(group_hashes)
    config['v2_code_sha256'] = {str(path.relative_to(project)): sha(path) for path in
        (Path(__file__), Path(__file__).with_name('v2_adapter.py'), Path(v2_features.__file__),
         Path(__file__).with_name('tune_model.py'))}
    config['sidecar'] = dict(root=str(sidecar.root.relative_to(project)),
        config_sha256=sidecar.config_sha256, contract_sha256=sidecar.contract_sha256,
        manifest_sha256=sidecar.manifest_sha256)
    config['projection'] = projection_contract(trial)
    config['group_loss'] = dict(ach_category_code=ach_code, positive_counts=GROUP_COUNTS,
        factors=GROUP_FACTORS if trial == 'features_balanced' else dict(ACH=1., non_ACH=1.),
        formula='C/sqrt(P_g), C=17794/(sqrt(15067)+sqrt(2727)); positives only; batch arithmetic mean',
        negatives=1., applied=(trial == 'features_balanced'))
    config['initialization'] = ('Fresh A initialization seed 20260920; B/C project A input columns, '
        'preserving bank embedding offsets and all other parameters. Same post-initialization RNG.')
    config['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                            numpy=np.__version__, gpu=torch.cuda.get_device_name(0))
    config['scope'] = 'Entire Train 2022-09-01..10, exactly one epoch; no Validation or Test training.'
    return config


def _pause_requested(project):
    path = project / 'result/gnn_tuning_v2/pause_request.json'
    return path.exists() and read_json(path).get('enabled') is True


def validate_production_artifacts(output, accepted, cursor=None, block_count=797):
    """Fail closed for inconsistent existing artifacts; never overwrite them."""
    output = Path(output)
    production = ('checkpoint_latest.pt', 'checkpoint_previous.pt', 'final_model.pt', 'training_result.json')
    if not accepted and any((output / name).exists() for name in production):
        raise ValueError('Production artifacts exist without valid GPU acceptance; preserve for audit')
    if cursor is not None and cursor < block_count and any((output / name).exists() for name in
                                                          ('final_model.pt', 'training_result.json')):
        raise ValueError('Final artifacts conflict with incomplete checkpoint; preserve for audit')


def _open_loader(project, date):
    # Prevent frozen CompactEdgeStore from silently building a missing cache.
    cache = project / 'data/gnn_tensor_cache_v1'
    for name in ('manifest.json', 'numeric.npy', 'category.npy'):
        if not (cache / name).is_file():
            raise ValueError('Frozen Train E cache missing; repair separately: ' + name)
    return TemporalLoader(project / 'data/gnn_v1_train', project / 'result/gnn_train_v1', date,
                          edge_cache=cache)


def _adapt_batch(cached, index, sidecar, block, trial):
    arrays, labels, report = cached.batch(index)
    ids = arrays['target_edge_id']
    cutoffs = np.asarray(cached.loader.builder.a['timestamp'][ids], dtype=np.int64)
    extra = sidecar.block(block, ids, cutoffs)
    return adapt_arrays(arrays, extra, trial), labels, report


def _acceptance_data(project, verified, sidecar, positive_ids, update):
    """Select bounded real batches covering both Train-positive groups."""
    needed = set(positive_ids.values())
    selected = []
    # Reading only tiny plan columns avoids repeatedly constructing daily R.
    for entry in verified['blocks']:
        table = pq.ParquetFile(project / 'data/gnn_train_cache_parquet' / entry['block'] / 'plans.parquet').read(columns=['targets'])
        for i, ids in enumerate(table['targets'].to_pylist()):
            covered = needed.intersection(ids)
            if covered:
                selected.append((entry, i))
                needed.difference_update(covered)
        if not needed:
            break
    if needed:
        raise ValueError('Acceptance cannot locate both positive groups in frozen plans')
    data, loader, day = [], None, None
    for entry, index in selected:
        date = entry['block'].split('/')[0].split('=')[1]
        if date != day:
            if loader is not None:
                del loader
                gc.collect()
            loader, day = _open_loader(project, date), date
        cached = ParquetBlockCache(project / 'data/gnn_train_cache_parquet' / entry['block'],
                                   loader, verified['config_sha256'])
        arrays, labels, _ = cached.batch(index)
        ids = arrays['target_edge_id']
        if sidecar is None:
            from .v2_features import compute_selected
            start = read_json(project / 'data/gnn_v1_train/build.json')['data_start_us']
            extra, count24 = compute_selected(loader.builder.a, start, ids, return_count24=True)
            for column, name in enumerate(('r_src_ext_out_count_24h', 'r_dst_ext_in_count_24h')):
                valid = count24[:, column] >= 0
                expected = np.log1p(np.maximum(count24[:, column], 0)).astype(np.float32)
                np.testing.assert_array_equal(arrays['target_attr'][:, R_NAMES.index(name + '__valid')], valid)
                np.testing.assert_allclose(arrays['target_attr'][:, R_NAMES.index(name + '__encoded')], expected,
                                           rtol=1e-6, atol=1e-6)
        else:
            extra = sidecar.block(entry['block'], ids, np.asarray(loader.builder.a['timestamp'][ids]))
        data.append((arrays, extra, labels))
        update(stage='acceptance_inputs', selected_batches=len(data), block=entry['block'])
    del loader
    gc.collect()
    return data


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('Optimizer/RNG state keys differ')
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError('Optimizer/RNG state length differs')
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    elif left != right:
        raise ValueError('Optimizer/RNG state differs')


def gpu_acceptance(project, output, config, config_hash, verified, sidecar, positive_ids, update, device, data=None):
    if data is None:
        data = _acceptance_data(project, verified, sidecar, positive_ids, update)
    if not data:
        raise ValueError('No real acceptance batches')
    configs = {trial: dict(config, trial=trial, model=architecture(config['model'], trial)) for trial in TRIALS}
    a = make_initial_model(configs['features_only'], device)
    rng_a = capture_rng()
    b = make_initial_model(configs['features_pruned'], device)
    rng_b = capture_rng()
    c = make_initial_model(configs['features_balanced'], device)
    for key, value in b.state_dict().items():
        torch.testing.assert_close(value, c.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(rng_a['torch_cpu'], rng_b['torch_cpu'], rtol=0, atol=0)
    for left, right in zip(rng_a['torch_cuda'], rng_b['torch_cuda']):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    a.eval(); b.eval()
    with torch.no_grad():
        for arrays, extra, _ in data:
            full = adapt_arrays(arrays, extra, 'features_only')
            zeroed = dict(full)
            zeroed['x_num'] = full['x_num'].copy()
            zeroed['target_attr'] = full['target_attr'].copy()
            zeroed['x_num'][:, sorted(set(range(63)) - set(NODE_KEEP))] = 0
            zeroed['target_attr'][:, sorted(set(range(110)) - set(R_KEEP))] = 0
            small = adapt_arrays(arrays, extra, 'features_pruned')
            torch.testing.assert_close(a(to_torch(zeroed, device)), b(to_torch(small, device)), rtol=1e-4, atol=1e-5)
    del a, b, c
    model = make_initial_model(config, device)
    optimizer = make_optimizer(model)
    adapted = [(adapt_arrays(arrays, extra, config['trial']), labels) for arrays, extra, labels in data]
    ach_code = config['group_loss']['ach_category_code']
    observed = set()
    for arrays, labels in adapted:
        codes = arrays['target_base_cat'][:, PAYMENT_COLUMN]
        if np.any((labels == 1) & (codes == ach_code)): observed.add('ACH')
        if np.any((labels == 1) & (codes != ach_code)): observed.add('non_ACH')
    if observed != set(GROUP_COUNTS):
        raise ValueError('GPU acceptance must include both positive groups')
    stats = []
    for arrays, labels in adapted:
        stats.append(train_step(model, optimizer, arrays, labels, device, ach_code, config['trial']))
    finite_parameters(model)
    scratch = output / 'acceptance_checkpoints'
    if scratch.exists() and any(scratch.iterdir()):
        raise ValueError('Unreviewed acceptance checkpoint exists; preserve for review')
    check_space(output)
    save_checkpoint(scratch, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        rng=capture_rng(), config_sha256=config_hash))
    arrays, labels = adapted[-1]
    expected_loss, _ = train_step(model, optimizer, arrays, labels, device, ach_code, config['trial'])
    expected_rng = capture_rng()
    loaded = load_checkpoint(scratch)
    if loaded['config_sha256'] != config_hash:
        raise ValueError('Acceptance checkpoint configuration mismatch')
    resumed = make_initial_model(config, device)
    resumed.load_state_dict(loaded['model'], strict=True)
    resumed_opt = make_optimizer(resumed)
    resumed_opt.load_state_dict(loaded['optimizer'])
    restore_rng(loaded['rng'])
    actual_loss, _ = train_step(resumed, resumed_opt, arrays, labels, device, ach_code, config['trial'])
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-4, atol=1e-5)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=1e-4, atol=1e-5)
    _assert_nested_equal(optimizer.state_dict(), resumed_opt.state_dict())
    _assert_nested_equal(expected_rng, capture_rng())
    finite_parameters(resumed)
    report = dict(status='passed', config_sha256=config_hash, positive_groups=sorted(observed),
        compared_batches=len(data), masked_A_equals_B=True, B_C_identical_initialization=True,
        identical_training_rng=True, checkpoint_resume_verified=True, loss_grad_norm=stats,
        scope='Bounded real Train CUDA engineering acceptance; no full training result')
    check_space(output)
    atomic_json(output / 'acceptance.json', report)
    # Delete only our just-verified scratch checkpoint, never production files.
    path = scratch / 'checkpoint_latest.pt'
    if path.resolve().parent != scratch.resolve() or scratch.resolve().parent != output.resolve():
        raise ValueError('Unsafe acceptance scratch path')
    path.unlink()
    scratch.rmdir()
    del model, resumed, optimizer, resumed_opt, loaded, adapted, data
    gc.collect(); torch.cuda.empty_cache()
    return report


def run_early_acceptance(project=ROOT, progress=None):
    """Bounded real-input CUDA smoke before full sidecar construction.

    Results live outside production trials and cannot satisfy their full
    sidecar-bound acceptance. No completed sidecar manifest is fabricated.
    """
    from . import v2_features
    project = Path(project).resolve()
    output = project / 'result/gnn_tuning_v2/early_gpu_acceptance'
    check_space(project)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessLock(output):
        state = dict(status='preparing', pid=os.getpid(), scope='Bounded engineering only')
        def update(**fields):
            state.update(fields)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / 'status.json', state)
            if progress is not None: progress(**fields)
            else: print(json.dumps(state, ensure_ascii=True), flush=True)
        try:
            device = require_cuda()
            verify_trial_helper(project)
            baseline, verified, identities = verify_baseline(project, update)
            ach_code, group_hashes, positive_ids = verify_groups(project, baseline, update)
            data = _acceptance_data(project, verified, None, positive_ids, update)
            reports = {}
            for trial in TRIALS:
                destination = output / trial
                destination.mkdir(exist_ok=True)
                config = dict(version=2, trial=trial, model=architecture(baseline['model'], trial),
                    group_loss=dict(ach_category_code=ach_code), positive_groups=GROUP_COUNTS,
                    group_factors=GROUP_FACTORS if trial == 'features_balanced' else dict(ACH=1., non_ACH=1.),
                    positive_weight=WEIGHT, projection=projection_contract(trial),
                    feature_contract=v2_features.contract(), **identities,
                    input_hashes=group_hashes,
                    v2_code_sha256={str(p.relative_to(project)):sha(p) for p in
                                   (Path(__file__), Path(__file__).with_name('v2_adapter.py'), Path(v2_features.__file__),
                                    Path(__file__).with_name('tune_model.py'))},
                    scope='Bounded real Train GPU acceptance; no production checkpoint or sidecar publication')
                config_path = destination / 'config.json'
                if config_path.exists() and read_json(config_path) != config:
                    raise ValueError('Early acceptance identity changed; preserve results for audit')
                if not config_path.exists():
                    check_space(destination)
                    atomic_json(config_path, config)
                config_hash = sha(config_path)
                accepted = destination / 'acceptance.json'
                if accepted.exists():
                    report = read_json(accepted)
                    if report.get('status') != 'passed' or report.get('config_sha256') != config_hash:
                        raise ValueError('Early acceptance result identity mismatch')
                else:
                    report = gpu_acceptance(project, destination, config, config_hash, verified,
                        None, positive_ids, update, device, data=data)
                reports[trial] = report
                update(stage='trial_passed', trial=trial)
            atomic_json(output / 'acceptance.json', dict(status='passed', trials=reports,
                scope='Three bounded real Train CUDA checks only; full Train is not run'))
            update(status='completed', stage='early_acceptance_passed')
            return reports
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


def validate_v2_resume(payload, blocks, config_hash):
    cursor = validate_resume(payload, blocks, config_hash)
    metrics = payload['metrics']
    groups = metrics.get('positive_groups')
    if (not isinstance(groups, dict) or set(groups) != set(GROUP_COUNTS)
            or any(type(v) is not int or v < 0 or v > GROUP_COUNTS[k] for k, v in groups.items())
            or sum(groups.values()) != metrics['positives']):
        raise ValueError('Checkpoint positive-group accounting mismatch')
    for actual, expected in zip(metrics['blocks'], blocks[:cursor]):
        if any(actual.get(key) != expected[key] for key in ('block', 'targets', 'positives', 'batches')):
            raise ValueError('Checkpoint block coverage differs')
        if actual.get('feature_manifest_sha256') is None:
            raise ValueError('Checkpoint lacks sidecar identity')
        if (set(actual.get('positive_groups', {})) != set(GROUP_COUNTS)
                or any(type(v) is not int or v < 0 for v in actual['positive_groups'].values())
                or sum(actual['positive_groups'].values()) != actual['positives']):
            raise ValueError('Checkpoint per-block positive-group accounting differs')
    if any(sum(b['positive_groups'][name] for b in metrics['blocks']) != groups[name] for name in GROUP_COUNTS):
        raise ValueError('Checkpoint cumulative positive groups differ from blocks')
    if cursor == len(blocks) and groups != GROUP_COUNTS:
        raise ValueError('Completed Train group counts differ')
    return cursor


def _finish(output, config, config_hash, model, metrics, blocks, update):
    payload = checkpoint_payload(model, make_optimizer(model), config_hash, len(blocks), metrics)
    validate_v2_resume(payload, blocks, config_hash)
    if metrics['targets'] != TOTAL or metrics['positives'] != POSITIVES or len(blocks) != 797:
        raise ValueError('Incomplete Train; final publication prohibited')
    finite_parameters(model)
    check_space(output)
    final = output / 'final_model.pt'
    atomic_torch_save(final, dict(model=model.state_dict(), config=config['model'], trial=config['trial'],
        config_sha256=config_hash, epoch=1, targets=TOTAL, positives=POSITIVES, seed=SEED))
    saved = torch.load(final, map_location='cpu', weights_only=True)
    clone = BidirectionalGINE(**saved['config'])
    clone.load_state_dict(saved['model'], strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), clone.state_dict()[name], rtol=0, atol=0)
    report = dict(status='completed', trial=config['trial'], epoch=1, targets=TOTAL, positives=POSITIVES,
        positive_groups=metrics['positive_groups'], batches=metrics['batches'], blocks=len(blocks),
        online_weighted_bce=metrics['weighted_loss_sum'] / TOTAL,
        training_seconds=metrics['training_seconds'], config_sha256=config_hash,
        final_model_sha256=sha(final), checkpoint_load_verified=True,
        completed_at=datetime.now(timezone.utc).isoformat(),
        scope='Full Train only; weighted BCE is not comparable between differing loss weights.')
    check_space(output)
    atomic_json(output / 'training_result.json', report)
    update(status='completed', stage='complete', committed_targets=TOTAL, processed_targets=TOTAL,
           positives=POSITIVES, epoch=1, next_block=len(blocks))
    return report


def run(project=ROOT, trial='features_only', mode='train', output=None, progress=None):
    project = Path(project).resolve()
    trial = check_trial(trial)
    if mode not in ('accept', 'train'):
        raise ValueError('Unsupported execution mode')
    destination = project / 'result/gnn_tuning_v2/trials' / trial
    output = Path(output).absolute() if output is not None else destination
    if output != output.resolve() or output.resolve() != destination.resolve():
        raise ValueError('Output must be the exact independent v2 trial directory')
    from .v2_features import SidecarStore
    check_space(project)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessLock(output):
        state = dict(status='preparing', pid=os.getpid(), trial=trial,
                     started_at=datetime.now(timezone.utc).isoformat(), expected_targets=TOTAL)
        def update(**fields):
            state.update(fields)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            # Status updates remain possible when a disk floor stops bulk writes.
            atomic_json(output / 'status.json', state)
            if progress is not None: progress(**fields)
            else: print(json.dumps(state, ensure_ascii=True), flush=True)
        try:
            if _pause_requested(project):
                update(status='paused', stage='pause_requested')
                return dict(status='paused')
            device = require_cuda()
            verify_trial_helper(project)
            baseline, verified, identities = verify_baseline(project, update)
            ach_code, group_hashes, positive_ids = verify_groups(project, baseline, update)
            sidecar = SidecarStore(project / 'data/gnn_v2_target_features/train',
                                   project / 'data/gnn_train_cache_parquet', expected_split='train')
            config = build_config(project, baseline, identities, sidecar, trial, ach_code, group_hashes)
            config_path = output / 'config.json'
            if config_path.exists():
                if read_json(config_path) != config:
                    raise ValueError('Frozen v2 config/source/input identity changed')
            else:
                if any((output / name).exists() for name in ('checkpoint_latest.pt', 'checkpoint_previous.pt', 'final_model.pt', 'acceptance.json')):
                    raise ValueError('V2 artifacts exist without their frozen configuration')
                check_space(output)
                atomic_json(config_path, config)
            config_hash = sha(config_path)
            accepted_path = output / 'acceptance.json'
            if accepted_path.exists():
                accepted = read_json(accepted_path)
                if accepted.get('status') != 'passed' or accepted.get('config_sha256') != config_hash:
                    raise ValueError('V2 GPU acceptance identity mismatch')
            else:
                validate_production_artifacts(output, accepted=False)
                accepted = gpu_acceptance(project, output, config, config_hash, verified,
                                          sidecar, positive_ids, update, device)
            if mode == 'accept':
                update(status='ready', stage='acceptance_passed')
                return accepted
            blocks = verified['blocks']
            model = make_initial_model(config, device)
            optimizer = make_optimizer(model)
            loaded = load_checkpoint(output)
            cursor, metrics = 0, fresh_metrics()
            metrics.update(clipped_batches=0, positive_groups=dict(ACH=0, non_ACH=0))
            if loaded is not None:
                cursor = validate_v2_resume(loaded, blocks, config_hash)
                model.load_state_dict(loaded['model'], strict=True)
                optimizer.load_state_dict(loaded['optimizer'])
                restore_rng(loaded['rng'])
                metrics = loaded['metrics']
                for block in metrics['blocks']:
                    if block['feature_manifest_sha256'] != sidecar.manifest_sha256:
                        raise ValueError('Committed block sidecar identity differs')
                    # Verify every already-consumed feature block before resume;
                    # an empty request still performs SidecarStore file hashing.
                    sidecar.block(block['block'], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
                    update(stage='verify_committed_features', current_block=block['block'])
                del loaded
            elif (output / 'final_model.pt').exists():
                raise ValueError('Final model exists without recoverable checkpoint')
            finite_parameters(model)
            validate_production_artifacts(output, accepted=True, cursor=cursor, block_count=len(blocks))
            if cursor == len(blocks) and (output / 'training_result.json').exists():
                result = read_json(output / 'training_result.json')
                if (result.get('status') != 'completed' or result.get('epoch') != 1
                        or result.get('targets') != TOTAL or result.get('positives') != POSITIVES
                        or result.get('config_sha256') != config_hash
                        or result.get('final_model_sha256') != sha(output / 'final_model.pt')):
                    raise ValueError('Existing final model identity/count/hash differs')
                update(status='completed', stage='complete', committed_targets=TOTAL, epoch=1)
                return result
            loader, loaded_date = None, None
            clock = time.perf_counter()
            for index in range(cursor, len(blocks)):
                if _pause_requested(project):
                    update(status='paused', stage='pause_requested', next_block=index, committed_targets=metrics['targets'])
                    return dict(status='paused', next_block=index, targets=metrics['targets'])
                check_space(output)
                entry = blocks[index]
                date = entry['block'].split('/')[0].split('=')[1]
                update(status='running', stage='load_day' if date != loaded_date else 'load_block',
                       current_block=entry['block'], next_block=index, committed_targets=metrics['targets'])
                if date != loaded_date:
                    if loader is not None:
                        del loader
                        gc.collect()
                    loader, loaded_date = _open_loader(project, date), date
                cached = ParquetBlockCache(project / 'data/gnn_train_cache_parquet' / entry['block'],
                                          loader, verified['config_sha256'])
                count, positive, block_loss, clipped = 0, 0, 0., 0
                group_counts = dict(ACH=0, non_ACH=0)
                began, last_update = time.perf_counter(), time.monotonic()
                for batch_index in range(len(cached)):
                    arrays, labels, _ = _adapt_batch(cached, batch_index, sidecar, entry['block'], trial)
                    ids = arrays['target_edge_id']
                    stamp = int(loader.builder.a['timestamp'][ids[0]])
                    if metrics['last_order_key'] is not None and [stamp, int(ids[0])] <= metrics['last_order_key']:
                        raise ValueError('Trial targets duplicated or not chronological')
                    loss, norm = train_step(model, optimizer, arrays, labels, device, ach_code, trial)
                    codes = arrays['target_base_cat'][:, PAYMENT_COLUMN]
                    group_counts['ACH'] += int(((labels == 1) & (codes == ach_code)).sum())
                    group_counts['non_ACH'] += int(((labels == 1) & (codes != ach_code)).sum())
                    count += len(labels); positive += int(labels.sum())
                    block_loss += loss * len(labels); clipped += int(norm > 5)
                    metrics['batches'] += 1
                    metrics['last_order_key'] = [stamp, int(ids[-1])]
                    if time.monotonic() - last_update > 15:
                        update(stage='training', block_batch=batch_index + 1, block_batches=len(cached),
                               processed_targets=metrics['targets'] + count, last_batch_loss=loss, last_gradient_norm=norm)
                        last_update = time.monotonic()
                if count != entry['targets'] or positive != entry['positives'] or len(cached) != entry['batches']:
                    raise ValueError('V2 full block identity/class/batch coverage differs')
                finite_parameters(model)
                metrics['targets'] += count; metrics['positives'] += positive
                metrics['weighted_loss_sum'] += block_loss; metrics['clipped_batches'] += clipped
                for name in GROUP_COUNTS: metrics['positive_groups'][name] += group_counts[name]
                metrics['blocks'].append(dict(block=entry['block'], targets=count, positives=positive,
                    batches=len(cached), positive_groups=group_counts, weighted_bce=block_loss/count,
                    feature_manifest_sha256=sidecar.manifest_sha256, seconds=time.perf_counter()-began))
                metrics['training_seconds'] += time.perf_counter() - clock
                payload = checkpoint_payload(model, optimizer, config_hash, index + 1, metrics)
                validate_v2_resume(payload, blocks, config_hash)
                check_space(output)
                save_checkpoint(output, payload)
                update(stage='block_committed', next_block=index + 1, committed_targets=metrics['targets'],
                       processed_targets=metrics['targets'], positives=metrics['positives'], positive_groups=metrics['positive_groups'])
                clock = time.perf_counter()
                del payload, cached, arrays, labels
            return _finish(output, config, config_hash, model, metrics, blocks, update)
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--trial', choices=TRIALS)
    parser.add_argument('--mode', choices=('accept', 'train', 'early-accept'), default='train')
    args = parser.parse_args()
    if args.mode == 'early-accept':
        result = run_early_acceptance(args.project)
    else:
        if args.trial is None:
            parser.error('--trial is required for accept/train')
        result = run(args.project, args.trial, args.mode)
    if result.get('status') == 'paused':
        raise SystemExit(75)
