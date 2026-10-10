"""Full Train seed repeats; bounded loop derived from frozen v2_train.
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

def _finish(output, config, config_hash, model, metrics, blocks, update):
    payload = checkpoint_payload(model, make_optimizer(model), config_hash, len(blocks), metrics)
    validate_v2_resume(payload, blocks, config_hash)
    if metrics['targets'] != TOTAL or metrics['positives'] != POSITIVES or len(blocks) != 797:
        raise ValueError('Incomplete Train; final publication prohibited')
    finite_parameters(model)
    check_space(output)
    final = output / 'final_model.pt'
    atomic_torch_save(final, dict(model=model.state_dict(), config=config['model'], trial=config['trial'],
        config_sha256=config_hash, epoch=1, targets=TOTAL, positives=POSITIVES, seed=config['seed']))
    saved = torch.load(final, map_location='cpu', weights_only=True)
    clone = BidirectionalGINE(**saved['config'])
    clone.load_state_dict(saved['model'], strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), clone.state_dict()[name], rtol=0, atol=0)
    report = dict(status='completed', seed=config['seed'], trial=config['trial'], epoch=1, targets=TOTAL, positives=POSITIVES,
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


def run(project=ROOT, trial='features_only', mode='train', output=None, progress=None, seed=None):
    project = Path(project).resolve()
    trial = check_trial(trial)
    check_seed(seed)
    verify_plan(project)
    if mode not in ('accept', 'train'):
        raise ValueError('Unsupported execution mode')
    destination = trial_root(project, trial, seed)
    output = Path(output).absolute() if output is not None else destination
    if output != output.resolve() or output.resolve() != destination.resolve():
        raise ValueError('Output must be the exact independent repeat trial directory')
    from .v2_features import SidecarStore
    check_space(project)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessLock(output):
        state = dict(status='preparing', pid=os.getpid(), trial=trial, seed=seed,
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
            config = build_config(project, baseline, identities, sidecar, trial, ach_code, group_hashes, seed)
            config_path = output / 'config.json'
            if config_path.exists():
                if read_json(config_path) != config:
                    raise ValueError('Frozen repeat config/source/input identity changed')
            else:
                if any((output / name).exists() for name in ('checkpoint_latest.pt', 'checkpoint_previous.pt', 'final_model.pt', 'acceptance.json')):
                    raise ValueError('Repeat artifacts exist without their frozen configuration')
                check_space(output)
                atomic_json(config_path, config)
            config_hash = sha(config_path)
            accepted_path = output / 'acceptance.json'
            if accepted_path.exists():
                accepted = read_json(accepted_path)
                if accepted.get('status') != 'passed' or accepted.get('config_sha256') != config_hash:
                    raise ValueError('Repeat GPU acceptance identity mismatch')
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
                    raise ValueError('Repeat full block identity/class/batch coverage differs')
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
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--trial', choices=TRIALS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--mode', choices=('accept','train'), default='train')
    a = p.parse_args()
    result = run(a.project, a.trial, a.mode, seed=a.seed)
    if result.get('status') == 'paused': raise SystemExit(75)
