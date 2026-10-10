"""Two bounded, independent full-Train trials; all baseline artifacts stay read-only."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

from .loader import TemporalLoader
from .model import BidirectionalGINE
from .model_cache import ParquetBlockCache
from .train_model import (TOTAL, POSITIVES, SEED, sha, read_json, require_cuda,
                          seed_all, train_step, finite_parameters, fresh_metrics,
                          validate_cursor, checkpoint_payload)
from .training_state import (ProcessLock, atomic_json, atomic_torch_save,
                             save_checkpoint, load_checkpoint, capture_rng, restore_rng)

ROOT = Path(__file__).resolve().parents[2]
WEIGHT = (TOTAL - POSITIVES) / POSITIVES
TRIALS = {
    'lr_only': dict(learning_rate=3e-4, positive_weight=WEIGHT),
    'sqrt_weight': dict(learning_rate=3e-4, positive_weight=math.sqrt(WEIGHT)),
}


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path outside project: ' + str(path))
    return path


def trial_parameters(params):
    """Only the two predeclared candidates are permitted by this entrypoint."""
    value = dict(params)
    name = value.pop('name', None)
    matches = [key for key, candidate in TRIALS.items() if value == candidate]
    if len(matches) != 1 or (name is not None and name != matches[0]):
        raise ValueError('Only lr_only and sqrt_weight fixed trial parameters are permitted')
    return dict(name=matches[0], **TRIALS[matches[0]])


def verify_baseline(project, progress):
    """Recheck frozen bytes, reusing the completed exhaustive identity/graph audit.

    This does not merely trust timestamps or skip checks: every original frozen
    input is hashed again and each block is hashed again as it is consumed.
    The expensive repeated plan reconstruction is covered by the original audit.
    """
    project = Path(project).resolve()
    base = project / 'result/gnn_train_gpu_v1'
    config, result = read_json(base / 'config.json'), read_json(base / 'training_result.json')
    config_hash = sha(base / 'config.json')
    verified = read_json(base / 'cache_verification.json')
    if (result.get('status') != 'completed' or result.get('epoch') != 1
            or result.get('targets') != TOTAL or result.get('positives') != POSITIVES
            or result.get('config_sha256') != config_hash
            or result.get('final_model_sha256') != sha(base / 'final_model.pt')):
        raise ValueError('Completed baseline identity/count/hash mismatch')
    if (config.get('epochs') != 1 or config.get('seed') != SEED
            or config.get('positive_weight') != WEIGHT
            or config.get('dates') != [f'2022-09-{day:02d}' for day in range(1, 11)]
            or config.get('batch_size') != 64 or config.get('model', {}).get('width') != 64
            or config.get('model', {}).get('dropout') != .1):
        raise ValueError('Baseline is not the frozen comparison configuration')
    accepted = read_json(base / 'acceptance.json')
    if accepted.get('status') != 'passed' or accepted.get('config_sha256') != config_hash:
        raise ValueError('Baseline engineering acceptance identity mismatch')
    if (verified.get('targets') != TOTAL or verified.get('positives') != POSITIVES
            or verified.get('dates') != config['dates']
            or verified.get('config_sha256') != config['cache_config_sha256']
            or verified.get('manifest_sha256') != config['cache_manifest_sha256']):
        raise ValueError('Completed cache verification differs from baseline')
    code = config['code_sha256']
    for name, digest in code.items():
        if sha(_inside(project, 'scripts/gnn_features/' + name)) != digest:
            raise ValueError('Frozen baseline code changed: ' + name)
    inputs = config['input_hashes']
    if any(inputs.get(name) != digest for name, digest in verified['input_hashes'].items()):
        raise ValueError('Original full verification inputs are not covered by baseline freeze')
    last = 0.
    for i, (name, digest) in enumerate(inputs.items()):
        if sha(_inside(project, name)) != digest:
            raise ValueError('Frozen baseline input changed: ' + name)
        if time.monotonic() - last > 10:
            progress(stage='verify_baseline_inputs', files_checked=i + 1, files_total=len(inputs))
            last = time.monotonic()
    cache = project / 'data/gnn_train_cache_parquet'
    root, state = read_json(cache / 'manifest.json'), read_json(cache / 'status.json')
    if root.get('status') != 'completed' or state.get('status') != 'completed':
        raise ValueError('Train cache is not completed')
    active = cache / '_work/active'
    if active.exists() and any(active.iterdir()):
        raise ValueError('Completed cache has unreviewed active work')
    blocks = verified['blocks']
    if (len(blocks) != 797 or len(blocks) != len(root['blocks'])
            or len({b['block'] for b in blocks}) != len(blocks)
            or sum(b['targets'] for b in blocks) != TOTAL
            or sum(b['positives'] for b in blocks) != POSITIVES):
        raise ValueError('Full Train block coverage/classes differ')
    for block, summary in zip(blocks, root['blocks']):
        if any(block.get(k) != value for k, value in summary.items()):
            raise ValueError('Verified/root block order or identity differs')
        path = _inside(cache, block['block'] + '/manifest.json')
        # This hash is also in the immutable baseline config input inventory.
        if sha(path) != block['manifest_sha256']:
            raise ValueError('Verified block manifest changed')
        rel = str(path.relative_to(project))
        if inputs.get(rel) != block['manifest_sha256']:
            raise ValueError('Verified block hash is not bound to baseline config')
        manifest = read_json(path)
        block['batches'] = manifest['batches']
    if sum(b['batches'] for b in blocks) != result['batches']:
        raise ValueError('Frozen total batch count differs')
    return config, verified, dict(
        baseline_config_sha256=config_hash,
        baseline_final_model_sha256=result['final_model_sha256'],
        baseline_verification_sha256=sha(base / 'cache_verification.json'),
        baseline_result_sha256=sha(base / 'training_result.json'))


def make_config(baseline, params, identities):
    config = copy.deepcopy(baseline)
    config.update(version=1, trial=trial_parameters(params), **identities)
    config['optimizer']['learning_rate'] = config['trial']['learning_rate']
    config['positive_weight'] = config['trial']['positive_weight']
    config['trial_code_sha256'] = sha(Path(__file__))
    config['initialization'] = 'Fresh same seed as baseline; no trained weights or optimizer reused'
    config['scope'] = 'Complete Train 2022-09-01..10, one epoch. Validation is external; never Test.'
    config['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                             numpy=np.__version__, gpu=torch.cuda.get_device_name(0))
    return config


def make_model(config, device):
    model = BidirectionalGINE(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['optimizer']['learning_rate'],
                                  weight_decay=config['optimizer']['weight_decay'])
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        config['positive_weight'], dtype=torch.float32, device=device))
    return model, optimizer, loss


def validate_resume(payload, blocks, config_hash):
    cursor = validate_cursor(payload, blocks, config_hash)
    if payload['metrics']['batches'] != sum(b['batches'] for b in blocks[:cursor]):
        raise ValueError('Checkpoint batch count differs from committed prefix')
    if cursor and payload['metrics'].get('last_order_key') is None:
        raise ValueError('Nonempty checkpoint lacks chronological cursor')
    return cursor


def _accept(project, output, config, config_hash, verified, progress, device):
    selected = next(b for b in verified['blocks']
                    if b['block'] == 'transaction_date=2022-09-04/hour=12/block=0000')
    loader = TemporalLoader(project / 'data/gnn_v1_train', project / 'result/gnn_train_v1',
                            '2022-09-04', edge_cache=project / 'data/gnn_tensor_cache_v1')
    cache = ParquetBlockCache(project / 'data/gnn_train_cache_parquet' / selected['block'],
                              loader, verified['config_sha256'])
    positive_index = next(i for i, p in enumerate(cache.plans)
                          if loader.day.labels[loader.day.positions(p['targets'])].sum() > 0)
    indices = sorted({0, len(cache) // 2, positive_index})
    if len(indices) < 2:
        indices.append(len(cache) - 1)
    data = []
    for i in indices:
        arrays, labels, _ = cache.batch(i)
        loader.builder.clear_cache()
        direct, y, _ = loader.batch(cache.plans[i]['targets'])
        if set(arrays) != set(direct):
            raise ValueError('Acceptance cached/direct input names differ')
        for name in arrays:
            if name == 'x_num':
                np.testing.assert_allclose(arrays[name], direct[name], rtol=1e-6, atol=1e-6)
            else:
                np.testing.assert_array_equal(arrays[name], direct[name])
        np.testing.assert_array_equal(labels, y)
        data.append((arrays, labels))
        progress(stage='acceptance_inputs', compared_batches=len(data))
    if not any(int(y.sum()) > 0 for _, y in data):
        raise ValueError('Acceptance must include positive targets')
    del loader, cache, direct
    gc.collect()
    scratch = output / 'acceptance_checkpoints'
    scratch.mkdir(exist_ok=True)
    if any(scratch.iterdir()):
        raise ValueError('Unreviewed acceptance checkpoints exist; preserve for diagnosis')
    seed_all()
    model, opt, loss_fn = make_model(config, device)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    first_loss, norm0 = train_step(model, opt, loss_fn, *data[0], device)
    save_checkpoint(scratch, dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                  rng=capture_rng(), config_sha256=config_hash, next_batch=1))
    continuous, norm1 = train_step(model, opt, loss_fn, *data[1], device)
    expected = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
    loaded = load_checkpoint(scratch)
    if loaded['config_sha256'] != config_hash or loaded['next_batch'] != 1:
        raise ValueError('Acceptance checkpoint identity/cursor mismatch')
    resumed, resumed_opt, resumed_loss = make_model(config, device)
    resumed.load_state_dict(loaded['model'], strict=True)
    resumed_opt.load_state_dict(loaded['optimizer'])
    restore_rng(loaded['rng'])
    restored_loss, norm2 = train_step(resumed, resumed_opt, resumed_loss, *data[1], device)
    np.testing.assert_allclose(continuous, restored_loss, rtol=1e-4, atol=1e-5)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[name], rtol=1e-4, atol=1e-5)
    finite_parameters(resumed)
    changed = [name for name, p in resumed.named_parameters()
               if not torch.equal(p.detach().cpu(), initial[name])]
    for prefix in ('bank', 'node_encoder', 'history_encoder', 'target_encoder', 'r_encoder',
                   'in_layers.0', 'in_layers.1', 'out_layers.0', 'out_layers.1', 'head'):
        if not any(name.startswith(prefix + '.') for name in changed):
            raise ValueError('Acceptance model branch did not update: ' + prefix)
    # Ensure that the class-weight path really participates in an update.
    positive_arrays, positive_labels = next((a, y) for a, y in data if int(y.sum()) > 0)
    positive_loss, positive_norm = train_step(resumed, resumed_opt, resumed_loss,
                                             positive_arrays, positive_labels, device)
    finite_parameters(resumed)
    report = dict(status='passed', config_sha256=config_hash,
                  block=selected['block'], batch_indices=indices,
                  targets=sum(len(y) for _, y in data), positives=sum(int(y.sum()) for _, y in data),
                  first_loss=first_loss, continuous_next_loss=continuous, resumed_next_loss=restored_loss,
                  positive_batch_loss=positive_loss, gradient_norms=[norm0, norm1, norm2, positive_norm],
                  updated_parameter_tensors=len(changed), cached_inputs_match_direct=True,
                  resume_update_within_float32_tolerance=True,
                  scope='Bounded real Train GPU acceptance, not production training')
    atomic_json(output / 'acceptance.json', report)
    # Only the single explicitly created acceptance file is removed on success.
    checkpoint = scratch / 'checkpoint_latest.pt'
    if scratch.resolve().parent != output.resolve() or checkpoint.resolve().parent != scratch.resolve():
        raise ValueError('Unsafe acceptance cleanup')
    checkpoint.unlink()
    scratch.rmdir()
    del model, opt, resumed, resumed_opt, loaded, data
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _finish(output, config, config_hash, model, metrics, blocks, progress):
    if (metrics['targets'] != TOTAL or metrics['positives'] != POSITIVES
            or len(metrics['blocks']) != len(blocks)
            or metrics['batches'] != sum(b['batches'] for b in blocks)):
        raise ValueError('Incomplete full Train epoch; final publication prohibited')
    finite_parameters(model)
    final = output / 'final_model.pt'
    atomic_torch_save(final, dict(model=model.state_dict(), config=config['model'],
                                  config_sha256=config_hash, epoch=1, targets=TOTAL,
                                  positives=POSITIVES, seed=SEED))
    loaded = torch.load(final, map_location='cpu', weights_only=True)
    clone = BidirectionalGINE(**loaded['config'])
    clone.load_state_dict(loaded['model'], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), clone.state_dict()[name]):
            raise ValueError('Final model readback differs: ' + name)
    report = dict(status='completed', epoch=1, trial=config['trial'], targets=TOTAL,
                  positives=POSITIVES, negatives=TOTAL - POSITIVES,
                  batches=metrics['batches'], blocks=len(blocks),
                  positive_weight=config['positive_weight'],
                  online_weighted_bce=metrics['weighted_loss_sum'] / TOTAL,
                  clipped_batches=metrics.get('clipped_batches', 0),
                  training_seconds=metrics['training_seconds'],
                  gpu=config['runtime']['gpu'], config_sha256=config_hash,
                  final_model_sha256=sha(final), checkpoint_load_verified=True,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  scope='Online training objective only; different class weights are not comparable. '
                        'Select models using shared held-out Validation, never Test.')
    atomic_json(output / 'training_result.json', report)
    progress(status='completed', stage='complete', committed_targets=TOTAL,
             processed_targets=TOTAL, positives=POSITIVES, epoch=1, next_block=len(blocks))
    return report


def _run(project, output, params, progress=None, acceptance_only=False):
    project, output = Path(project).resolve(), Path(output).resolve()
    if (not output.is_relative_to(project / 'result')
            or output == project / 'result'
            or output.is_relative_to(project / 'result/gnn_train_gpu_v1')):
        raise ValueError('Trial output must be a separate directory under project result/')
    params = trial_parameters(params)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessLock(output):
        state = dict(status='preparing', pid=os.getpid(), trial=params['name'],
                     started_at=datetime.now(timezone.utc).isoformat(), expected_targets=TOTAL)
        def update(**fields):
            state.update(fields)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / 'status.json', state)
            if progress is not None:
                progress(**fields)
            else:
                print(json.dumps(state, ensure_ascii=False), flush=True)
        try:
            update(stage='cuda_check')
            device = require_cuda()
            baseline, verified, identities = verify_baseline(project, update)
            config = make_config(baseline, params, identities)
            config_path = output / 'config.json'
            if config_path.exists():
                if read_json(config_path) != config:
                    raise ValueError('Frozen trial code/configuration/input identity changed')
            else:
                config_path = output / '_acceptance_config.json'
                if config_path.exists() and read_json(config_path) != config:
                    raise ValueError('Unaccepted candidate configuration changed; preserve for review')
                atomic_json(config_path, config)
            config_hash = sha(config_path)
            accepted_path = output / 'acceptance.json'
            if accepted_path.exists():
                accepted = read_json(accepted_path)
                if accepted.get('status') != 'passed' or accepted.get('config_sha256') != config_hash:
                    raise ValueError('Trial GPU acceptance identity mismatch')
            else:
                if (output / 'checkpoint_latest.pt').exists() or (output / 'final_model.pt').exists():
                    raise ValueError('Acceptance cannot replace an existing production model')
                update(stage='gpu_acceptance')
                accepted = _accept(project, output, config, config_hash, verified, update, device)
            if config_path.name == '_acceptance_config.json':
                os.replace(config_path, output / 'config.json')
            if acceptance_only:
                update(status='ready', stage='acceptance_passed', committed_targets=0)
                return accepted
            blocks = verified['blocks']
            seed_all()  # Acceptance never contributes weights, RNG, or samples to the trial.
            model, optimizer, loss_fn = make_model(config, device)
            loaded = load_checkpoint(output)
            cursor, metrics = 0, fresh_metrics()
            metrics['clipped_batches'] = 0
            if loaded is not None:
                cursor = validate_resume(loaded, blocks, config_hash)
                model.load_state_dict(loaded['model'], strict=True)
                optimizer.load_state_dict(loaded['optimizer'])
                restore_rng(loaded['rng'])
                metrics = loaded['metrics']
                del loaded
            elif (output / 'final_model.pt').exists():
                raise ValueError('Final model exists without a recoverable checkpoint')
            finite_parameters(model)
            if cursor == len(blocks) and (output / 'training_result.json').exists():
                result = read_json(output / 'training_result.json')
                if (result.get('status') != 'completed' or result.get('epoch') != 1
                        or result.get('targets') != TOTAL or result.get('positives') != POSITIVES
                        or result.get('config_sha256') != config_hash
                        or result.get('final_model_sha256') != sha(output / 'final_model.pt')):
                    raise ValueError('Existing completed trial identity differs')
                update(status='completed', stage='complete', committed_targets=TOTAL,
                       processed_targets=TOTAL, epoch=1, next_block=len(blocks))
                return result
            loader, loaded_date = None, None
            clock = time.perf_counter()
            for index in range(cursor, len(blocks)):
                entry = blocks[index]
                date = entry['block'].split('/')[0].split('=')[1]
                update(status='running', stage='load_day' if date != loaded_date else 'load_block',
                       current_block=entry['block'], next_block=index, epoch=0,
                       committed_targets=metrics['targets'], processed_targets=metrics['targets'])
                if date != loaded_date:
                    if loader is not None:
                        loader.builder.clear_cache()
                        del loader
                        gc.collect()
                    loader = TemporalLoader(project / 'data/gnn_v1_train', project / 'result/gnn_train_v1',
                                            date, edge_cache=project / 'data/gnn_tensor_cache_v1')
                    loaded_date = date
                loader.builder.clear_cache()
                cached = ParquetBlockCache(project / 'data/gnn_train_cache_parquet' / entry['block'],
                                          loader, verified['config_sha256'])
                count, positive, block_loss, clipped, max_norm = 0, 0, 0., 0, 0.
                began, last_update = time.perf_counter(), time.monotonic()
                for batch_index in range(len(cached)):
                    arrays, labels, _ = cached.batch(batch_index)
                    ids = arrays['target_edge_id']
                    stamp = int(loader.builder.a['timestamp'][ids[0]])
                    if metrics['last_order_key'] is not None and [stamp, int(ids[0])] <= metrics['last_order_key']:
                        raise ValueError('Trial targets are duplicated or not chronological')
                    loss, norm = train_step(model, optimizer, loss_fn, arrays, labels, device)
                    n = len(labels)
                    count += n
                    positive += int(labels.sum())
                    block_loss += loss * n
                    clipped += int(norm > 5)
                    max_norm = max(norm, max_norm)
                    metrics['batches'] += 1
                    metrics['last_order_key'] = [stamp, int(ids[-1])]
                    if time.monotonic() - last_update > 15:
                        update(stage='training', block_batch=batch_index + 1,
                               block_batches=len(cached), processed_targets=metrics['targets'] + count,
                               last_batch_loss=loss, last_gradient_norm=norm)
                        last_update = time.monotonic()
                if count != entry['targets'] or positive != entry['positives'] or len(cached) != entry['batches']:
                    raise ValueError('Trial block target/class/batch coverage differs')
                finite_parameters(model)
                metrics['targets'] += count
                metrics['positives'] += positive
                metrics['weighted_loss_sum'] += block_loss
                metrics['clipped_batches'] += clipped
                metrics['blocks'].append(dict(block=entry['block'], targets=count, positives=positive,
                                              weighted_bce=block_loss / count, clipped_batches=clipped,
                                              max_gradient_norm=max_norm, seconds=time.perf_counter() - began))
                metrics['training_seconds'] += time.perf_counter() - clock
                payload = checkpoint_payload(model, optimizer, config_hash, index + 1, metrics)
                validate_resume(payload, blocks, config_hash)
                save_checkpoint(output, payload)
                update(stage='block_committed', next_block=index + 1,
                       committed_targets=metrics['targets'], processed_targets=metrics['targets'],
                       positives=metrics['positives'], batches=metrics['batches'],
                       online_weighted_bce=metrics['weighted_loss_sum'] / metrics['targets'])
                clock = time.perf_counter()
                del payload, cached, arrays, labels
            return _finish(output, config, config_hash, model, metrics, blocks, update)
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


def run_trial(project: Path, output: Path, params: dict, progress=None):
    return _run(project, output, params, progress)


def run_acceptance(project: Path, output: Path, params: dict, progress=None):
    return _run(project, output, params, progress, acceptance_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--trial', choices=TRIALS, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--acceptance-only', action='store_true')
    args = parser.parse_args()
    destination = args.output or args.project / 'result/gnn_tuning_v1/trials' / args.trial
    function = run_acceptance if args.acceptance_only else run_trial
    function(args.project, destination, dict(name=args.trial, **TRIALS[args.trial]))
