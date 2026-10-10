"""One chronological Train epoch on CUDA from the verified Parquet cache."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

from .loader import TemporalLoader
from .model import BidirectionalGINE, to_torch
from .model_cache import ParquetBlockCache, verify_cache
from .training_state import (ProcessLock, atomic_json, atomic_torch_save,
                             capture_rng, restore_rng, save_checkpoint, load_checkpoint)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'data/gnn_v1_train'
FEATURES = ROOT / 'result/gnn_train_v1'
CACHE = ROOT / 'data/gnn_train_cache_parquet'
EDGE_CACHE = ROOT / 'data/gnn_tensor_cache_v1'
OUTPUT = ROOT / 'result/gnn_train_gpu_v1'
TOTAL = 20318992
POSITIVES = 17794
SEED = 20260920


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. CPU fallback is prohibited.')
    torch.cuda.init()
    torch.set_num_threads(4)
    return torch.device('cuda:0')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_model(config, device):
    model = BidirectionalGINE(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config['positive_weight'], dtype=torch.float32, device=device))
    return model, optimizer, loss_fn


def train_step(model, optimizer, loss_fn, arrays, labels, device):
    batch = to_torch(arrays, device=device)
    y = torch.as_tensor(labels.copy(), dtype=torch.float32, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite model logits')
    loss = loss_fn(logits, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    # The total norm becomes nonfinite if any gradient contains NaN/Inf.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm.detach())


def finite_parameters(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite model parameters')


def build_config(verified, progress):
    if verified['targets'] != TOTAL or verified['positives'] != POSITIVES:
        raise ValueError('Verified Train class counts differ from frozen scope')
    if not (EDGE_CACHE / 'manifest.json').is_file():
        raise ValueError('The existing E tensor cache is missing; do not recreate implicitly')
    hashes = dict(verified.get('input_hashes', {}))
    paths = list((ARCHIVE / 'history_index').glob('*.npy'))
    paths += list((ARCHIVE / 'graph/adj_index').glob('*.npy'))
    paths += list((ARCHIVE / 'metadata').glob('*.json'))
    paths += [EDGE_CACHE / n for n in ['manifest.json', 'numeric.npy', 'category.npy']]
    for date in verified['dates']:
        merged = read_json(FEATURES / f'transaction_date={date}/merge_phase.json')
        paths += [Path(part['path']) for part in merged['parts']]
    last = time.monotonic()
    for i, path in enumerate(paths):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Input path outside project: ' + str(path))
        hashes[str(path.relative_to(ROOT))] = sha(path)
        if time.monotonic() - last > 10:
            progress(stage='freeze_model_inputs', files_hashed=i + 1, files_total=len(paths))
            last = time.monotonic()
    vocab = read_json(ARCHIVE / 'metadata/vocabulary.json')['values']
    code_names = ['train_model.py', 'train_model_acceptance.py', 'training_state.py',
                  'model_cache.py', 'model.py', 'loader.py', 'build.py', 'engine.py', 'contract.py']
    code = {n: sha(Path(__file__).with_name(n)) for n in code_names}
    return dict(version=1, dates=verified['dates'], targets=TOTAL, positives=POSITIVES,
                negatives=TOTAL - POSITIVES, positive_weight=(TOTAL - POSITIVES) / POSITIVES,
                epochs=1, batch_size=64, order='timestamp, then edge_id; fixed cached graph',
                optimizer=dict(name='AdamW', learning_rate=.001, weight_decay=.0001, clip_norm=5),
                seed=SEED, loss='BCEWithLogitsLoss',
                model=dict(bank_count=len(vocab['bank']) + 2,
                           edge_category_sizes=[len(vocab['payment_format']) + 2,
                                                len(vocab['currency']) + 2,
                                                len(vocab['currency']) + 2, 7],
                           width=64, dropout=.1),
                cache_config_sha256=verified['config_sha256'],
                cache_manifest_sha256=verified['manifest_sha256'],
                input_hashes=hashes, code_sha256=code,
                runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                             numpy=np.__version__, gpu=torch.cuda.get_device_name(0)),
                scope='Train only, one epoch. No independent validation/test or generalization claims.')


def fresh_metrics():
    return dict(targets=0, positives=0, batches=0, weighted_loss_sum=0.,
                training_seconds=0., last_order_key=None, blocks=[])


def validate_cursor(payload, blocks, config_sha256):
    if payload.get('config_sha256') != config_sha256:
        raise ValueError('Checkpoint configuration mismatch')
    cursor = payload.get('next_block')
    if type(cursor) is not int or not 0 <= cursor <= len(blocks):
        raise ValueError('Invalid resume cursor')
    metrics = payload['metrics']
    if metrics['targets'] != sum(b['targets'] for b in blocks[:cursor]):
        raise ValueError('Checkpoint target count differs from committed prefix')
    if metrics['positives'] != sum(b['positives'] for b in blocks[:cursor]):
        raise ValueError('Checkpoint positive count differs from committed prefix')
    if len(metrics['blocks']) != cursor or any(
            saved['block'] != expected['block'] for saved, expected in zip(metrics['blocks'], blocks)):
        raise ValueError('Checkpoint block order differs from committed prefix')
    if not np.isfinite(metrics['weighted_loss_sum']) or metrics['weighted_loss_sum'] < 0:
        raise ValueError('Invalid cumulative loss')
    return cursor


def checkpoint_payload(model, optimizer, config_hash, cursor, metrics):
    return dict(version=1, config_sha256=config_hash, next_block=cursor, metrics=metrics,
                model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng())


def finish(output, config, config_hash, model, optimizer, metrics, blocks, progress):
    if metrics['targets'] != TOTAL or metrics['positives'] != POSITIVES or len(metrics['blocks']) != len(blocks):
        raise ValueError('Incomplete epoch; final publication prohibited')
    finite_parameters(model)
    final = output / 'final_model.pt'
    atomic_torch_save(final, dict(model=model.state_dict(), config=config['model'],
                                 config_sha256=config_hash, epoch=1, targets=TOTAL,
                                 positives=POSITIVES, seed=SEED))
    loaded = torch.load(final, map_location='cpu', weights_only=True)
    clone = BidirectionalGINE(**loaded['config'])
    clone.load_state_dict(loaded['model'], strict=True)
    if loaded['targets'] != TOTAL or loaded['epoch'] != 1:
        raise ValueError('Final checkpoint metadata mismatch')
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), clone.state_dict()[name]):
            raise ValueError('Final checkpoint values differ: ' + name)
    report = dict(status='completed', epoch=1, targets=TOTAL, positives=POSITIVES,
                  negatives=TOTAL - POSITIVES, batches=metrics['batches'], blocks=len(blocks),
                  online_weighted_bce=metrics['weighted_loss_sum'] / TOTAL,
                  positive_weight=config['positive_weight'],
                  training_seconds=metrics['training_seconds'],
                  gpu=config['runtime']['gpu'], cuda=config['runtime']['cuda'],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  config_sha256=config_hash, final_model_sha256=sha(final),
                  checkpoint_load_verified=True,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  scope='Online training loss over changing model weights, not held-out evaluation.')
    atomic_json(output / 'training_result.json', report)
    (output / '全训练集初版训练结果.md').write_text(
        '# 全训练集 GNN 初版训练结果\n\n'
        f'范围：2022-09-01 至 09-10，共 {TOTAL:,} 笔，正例 {POSITIVES:,} 笔。\n\n'
        f'已在 {report["gpu"]} 上按时间顺序完成 1 个 epoch、{metrics["batches"]:,} 个批次。'
        f'复用 {len(blocks)} 个已核验 Parquet 缓存块，最终模型已读回核验。\n\n'
        f'全程按样本加权的在线训练 BCE：{report["online_weighted_bce"]:.8f}；'
        f'正例权重 {config["positive_weight"]:.6f}。'
        f'有效训练运行时长（包括数据加载，不包括停机、前置核验和检查点写入）：'
        f'{metrics["training_seconds"] / 3600:.3f} 小时。\n\n'
        '模型为双向两层 GINE，width=64、dropout=0.1；AdamW 学习率0.001、'
        'weight_decay=0.0001、梯度裁剪5，固定种子20260920。\n\n'
        '这是训练过程结果，不是独立验证效果。训练损失来自不断更新的模型，'
        '没有处理主验证集或测试集，不能据此宣称泛化性能。\n', encoding='utf-8')
    progress(status='completed', stage='complete', epoch=1, committed_targets=TOTAL,
             processed_targets=TOTAL, positives=POSITIVES, next_block=len(blocks),
             online_weighted_bce=report['online_weighted_bce'], current_block=None)
    return report


def run(output=OUTPUT, mode='train'):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'result'):
        raise ValueError('Model output must be under project result/')
    output.mkdir(parents=True, exist_ok=True)
    with ProcessLock(output):
        state = dict(status='preparing', pid=os.getpid(), mode=mode,
                     started_at=datetime.now(timezone.utc).isoformat(), expected_targets=TOTAL)
        def progress(**fields):
            state.update(fields)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / 'status.json', state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
        try:
            progress(stage='cuda_check')
            device = require_cuda()
            verified = verify_cache(CACHE, ARCHIVE, FEATURES, progress=progress)
            progress(stage='freeze_model_inputs')
            config = build_config(verified, progress)
            config_path = output / 'config.json'
            if config_path.exists():
                if read_json(config_path) != config:
                    raise ValueError('Frozen model configuration/inputs changed; impact review required')
            else:
                # Freeze the production config only after real GPU acceptance passes.
                config_path = output / '_acceptance_config.json'
                atomic_json(config_path, config)
            config_hash = sha(config_path)
            atomic_json(output / 'cache_verification.json', verified)
            if mode == 'acceptance' or not (output / 'acceptance.json').exists():
                if (output / 'checkpoint_latest.pt').exists() or (output / 'final_model.pt').exists():
                    raise ValueError('Do not run acceptance over a production model')
                from .train_model_acceptance import accept
                progress(stage='gpu_acceptance')
                report = accept(output, config, config_hash, verified, progress)
                if config_path.name == '_acceptance_config.json':
                    os.replace(config_path, output / 'config.json')
                    config_path = output / 'config.json'
                progress(status='ready', stage='acceptance_passed')
                if mode == 'acceptance':
                    return report
            accepted = read_json(output / 'acceptance.json')
            if accepted.get('status') != 'passed' or accepted.get('config_sha256') != config_hash:
                raise ValueError('GPU/input/resume acceptance for this exact version is required')
            if config_path.name == '_acceptance_config.json':
                # Recover a stop between successful acceptance and config publication.
                os.replace(config_path, output / 'config.json')
            for name in ('file', 'date', 'blocks', 'targets', 'files_hashed', 'files_total'):
                state.pop(name, None)
            blocks = verified['blocks']
            seed_all()
            model, optimizer, loss_fn = make_model(config, device)
            loaded = load_checkpoint(output, map_location='cpu')
            cursor, metrics = 0, fresh_metrics()
            if loaded is not None:
                cursor = validate_cursor(loaded, blocks, config_hash)
                model.load_state_dict(loaded['model'], strict=True)
                optimizer.load_state_dict(loaded['optimizer'])
                restore_rng(loaded['rng'])
                metrics = loaded['metrics']
                progress(stage='resumed', next_block=cursor, committed_targets=metrics['targets'],
                         restored_from=loaded.get('_loaded_from'))
                del loaded
            elif (output / 'final_model.pt').exists():
                raise ValueError('Final model exists without a recoverable checkpoint')
            finite_parameters(model)
            if cursor == len(blocks) and (output / 'training_result.json').exists():
                existing = read_json(output / 'training_result.json')
                if (existing.get('status') != 'completed' or existing.get('config_sha256') != config_hash
                        or existing.get('targets') != TOTAL or existing.get('epoch') != 1
                        or existing.get('final_model_sha256') != sha(output / 'final_model.pt')):
                    raise ValueError('Existing completed result identity mismatch')
                torch.load(output / 'final_model.pt', map_location='cpu', weights_only=True)
                progress(status='completed', stage='complete', epoch=1, committed_targets=TOTAL,
                         processed_targets=TOTAL, next_block=len(blocks), current_block=None)
                return existing
            torch.cuda.reset_peak_memory_stats()
            loader, loaded_date = None, None
            clock = time.perf_counter()
            for i in range(cursor, len(blocks)):
                entry = blocks[i]
                date = entry['block'].split('/')[0].split('=')[1]
                progress(status='running', stage='load_day' if date != loaded_date else 'load_block',
                         current_block=entry['block'], next_block=i,
                         committed_targets=metrics['targets'], processed_targets=metrics['targets'],
                         positives=metrics['positives'], epoch=0)
                if date != loaded_date:
                    if loader is not None:
                        loader.builder.clear_cache()
                        del loader
                        gc.collect()
                    loader = TemporalLoader(ARCHIVE, FEATURES, date, edge_cache=EDGE_CACHE)
                    loaded_date = date
                loader.builder.clear_cache()
                cached = ParquetBlockCache(CACHE / entry['block'], loader, verified['config_sha256'])
                count, positive, block_loss = 0, 0, 0.
                block_began, last_update = time.perf_counter(), time.monotonic()
                progress(stage='training', block_batches=len(cached), block_batch=0)
                for j in range(len(cached)):
                    arrays, labels, info = cached.batch(j)
                    ids = arrays['target_edge_id']
                    t = int(loader.builder.a['timestamp'][ids[0]])
                    key = [t, int(ids[0])]
                    if metrics['last_order_key'] is not None and key <= metrics['last_order_key']:
                        raise ValueError('Training targets are repeated or not chronological')
                    loss, norm = train_step(model, optimizer, loss_fn, arrays, labels, device)
                    n, pos = len(labels), int(labels.sum())
                    count += n
                    positive += pos
                    block_loss += loss * n
                    metrics['last_order_key'] = [t, int(ids[-1])]
                    metrics['batches'] += 1
                    if time.monotonic() - last_update > 15:
                        progress(stage='training', block_batch=j + 1,
                                 processed_targets=metrics['targets'] + count,
                                 committed_targets=metrics['targets'], last_batch_loss=loss,
                                 last_gradient_norm=norm)
                        last_update = time.monotonic()
                if count != entry['targets'] or positive != entry['positives']:
                    raise ValueError('Trained block coverage/classes mismatch')
                finite_parameters(model)
                metrics['targets'] += count
                metrics['positives'] += positive
                metrics['weighted_loss_sum'] += block_loss
                metrics['blocks'].append(dict(block=entry['block'], targets=count, positives=positive,
                                              weighted_bce=block_loss / count,
                                              seconds=time.perf_counter() - block_began))
                metrics['training_seconds'] += time.perf_counter() - clock
                payload = checkpoint_payload(model, optimizer, config_hash, i + 1, metrics)
                validate_cursor(payload, blocks, config_hash)
                progress(stage='checkpoint', block_batch=len(cached))
                save_checkpoint(output, payload)
                progress(stage='block_committed', next_block=i + 1,
                         committed_targets=metrics['targets'], processed_targets=metrics['targets'],
                         positives=metrics['positives'], batches=metrics['batches'],
                         online_weighted_bce=metrics['weighted_loss_sum'] / metrics['targets'])
                clock = time.perf_counter()
                del payload, cached, arrays, labels
            return finish(output, config, config_hash, model, optimizer, metrics, blocks, progress)
        except BaseException as exc:
            progress(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--mode', choices=['acceptance', 'train'], default='train')
    args = parser.parse_args()
    run(args.output, args.mode)
