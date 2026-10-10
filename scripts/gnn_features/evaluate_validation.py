"""Full, label-isolated Validation inference with resumable Parquet predictions."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .build import load_archive
from .cache_train import array_sha, sha
from .contract import HOUR
from .engine import FeatureEngine
from .evaluation_metrics import binary_metrics
from .loader import (ENUM, ECAT, CompactEdgeStore, TargetDay, TemporalBatchBuilder)
from .model import BidirectionalGINE, to_torch
from .model_cache import ParquetBlockCache
from .training_state import ProcessLock, atomic_json
from .validation_cache import DATES, TOTAL, POSITIVES, END, read, validation_spec


class ValidationEdgeStore:
    """Reuse Train memmaps; decode only Validation E to bounded (~188 MB) RAM."""
    def __init__(self, project, archive):
        project, archive = Path(project), Path(archive)
        train_cache = project / 'data/gnn_tensor_cache_v1'
        if not (train_cache / 'manifest.json').exists():
            raise ValueError('Existing verified Train E cache required')
        self.train = CompactEdgeStore(project / 'data/gnn_v1_train', train_cache)
        self.prefix = len(self.train.numeric)
        self.numeric = np.empty((TOTAL, len(ENUM)), dtype=np.float32)
        self.category = np.empty((TOTAL, len(ECAT)), dtype=np.int16)
        count = 0
        for date in DATES:
            for path in sorted((archive / f'graph/edges/transaction_date={date}').glob('*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=['edge_id'] + ENUM + ECAT):
                    tab = pa.Table.from_batches([batch])
                    n = len(tab)
                    if not np.array_equal(tab['edge_id'].to_numpy(), np.arange(self.prefix + count, self.prefix + count + n)):
                        raise ValueError('Validation E edge IDs no longer match archive')
                    self.numeric[count:count+n] = np.column_stack([tab[k].to_numpy() for k in ENUM])
                    cats = np.column_stack([tab[k].to_numpy() for k in ECAT])
                    if np.any(cats < 0) or np.any(cats > 32767):
                        raise ValueError('Invalid encoded categories')
                    self.category[count:count+n] = cats
                    count += n
        if count != TOTAL or not np.isfinite(self.numeric).all():
            raise ValueError('Invalid/incomplete Validation E')
        self.cache, self.reads = {}, 0

    def get(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.prefix + TOTAL):
            raise ValueError('Edge outside allowed Train/Validation history')
        numeric = np.empty((len(ids), len(ENUM)), dtype=np.float32)
        category = np.empty((len(ids), len(ECAT)), dtype=np.int64)
        old = ids < self.prefix
        numeric[old], category[old] = self.train.get(ids[old])
        pos = ids[~old] - self.prefix
        numeric[~old], category[~old] = self.numeric[pos], self.category[pos]
        return numeric, category


def prediction_table(ids, times, labels, scores):
    ids = np.asarray(ids, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not (ids.shape == times.shape == labels.shape == scores.shape) or ids.ndim != 1:
        raise ValueError('Prediction shape mismatch')
    if len(np.unique(ids)) != len(ids) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Duplicate prediction or invalid label')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Nonfinite/invalid probability')
    return pa.table(dict(edge_id=ids, cutoff_us=times, label=labels.astype(np.int8),
                         probability=scores.astype(np.float32)))


def verify_predictions(table, ids, times, labels):
    for name, expected in [('edge_id', ids), ('cutoff_us', times), ('label', labels)]:
        if not np.array_equal(table[name].to_numpy(), expected):
            raise ValueError('Prediction identity mismatch: ' + name)
    scores = table['probability'].to_numpy()
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid prediction probability')
    return scores


def run(project, model_path, output, progress=None):
    project, model_path, out = Path(project).resolve(), Path(model_path).resolve(), Path(output).resolve()
    if not out.is_relative_to(project / 'result') or not model_path.is_relative_to(project / 'result'):
        raise ValueError('Evaluation outputs/models must stay in project result')
    validation_spec(project)
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', pid=os.getpid(), expected_targets=TOTAL, processed_targets=0,
                 started_at=datetime.now(timezone.utc).isoformat())

    def update(**fields):
        state.update(fields)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out / 'status.json', state)
        if progress:
            progress(**fields)

    with ProcessLock(out):
        try:
            update(stage='verify_inputs')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required for Validation inference')
            torch.set_num_threads(4)
            archive = project / 'data/gnn_v1_validation'
            features = project / 'result/gnn_validation_v1'
            cache = project / 'data/gnn_validation_cache_parquet'
            manifest, cache_config = read(cache / 'manifest.json'), read(cache / 'config.json')
            if manifest['status'] != 'completed' or read(cache / 'status.json')['status'] != 'completed':
                raise ValueError('Full verified Validation cache required')
            cache_hash = sha(cache / 'config.json')
            if manifest['config_sha256'] != cache_hash or manifest['targets'] != TOTAL or manifest['dates'] != DATES:
                raise ValueError('Validation cache identity/scope differs')
            for rel, digest in cache_config['hashes'].items():
                if sha(project / rel) != digest:
                    raise ValueError('Frozen Validation input changed: ' + rel)
            blocks = manifest['blocks']
            if len({b['block'] for b in blocks}) != len(blocks) or sum(b['targets'] for b in blocks) != TOTAL:
                raise ValueError('Validation inventory incomplete/duplicated')
            if [b['block'] for b in blocks] != sorted(b['block'] for b in blocks):
                raise ValueError('Validation inventory is not chronological')
            config = dict(version=1, dates=DATES, expected_targets=TOTAL, expected_positives=POSITIVES,
                          model_path=str(model_path.relative_to(project)), model_sha256=sha(model_path),
                          cache_manifest_sha256=sha(cache / 'manifest.json'), cache_config_sha256=cache_hash,
                          code_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                       ('evaluate_validation.py', 'evaluation_metrics.py', 'model.py', 'loader.py', 'model_cache.py')},
                          selection='Average precision over the full natural-prevalence Validation; Test untouched')
            if (out / 'config.json').exists():
                if read(out / 'config.json') != config:
                    raise ValueError('Evaluation code, model or inputs changed')
            else:
                atomic_json(out / 'config.json', config)
            config_hash = sha(out / 'config.json')
            saved = torch.load(model_path, map_location='cpu', weights_only=True)
            model_config = saved.get('config') or saved.get('model_config')
            if model_config is None:
                raise ValueError('Final model lacks architecture configuration')
            model = BidirectionalGINE(**model_config)
            model.load_state_dict(saved['model'], strict=True)
            if not all(torch.isfinite(v).all().item() for v in model.state_dict().values()):
                raise ValueError('Nonfinite model weights')
            model.to('cuda').eval()
            del saved
            meta, arrays, bank, vocab = load_archive(archive)
            if meta['end_exclusive'] != END:
                raise ValueError('History extends outside Validation')
            adj = {d + '_' + s: np.load(archive / f'graph/adj_index/{d}_{s}.npy', mmap_mode='r')
                   for d in ('in', 'out') for s in ('ptr', 'edge_ids')}
            bank_cat = np.load(archive / 'history_index/bank_cat.npy', mmap_mode='r')
            if np.any(bank_cat < 0) or np.any(bank_cat >= model_config['bank_count']):
                raise ValueError('Bank encoding differs from frozen Train vocabulary')
            store = ValidationEdgeStore(project, archive)
            scores, labels = np.empty(TOTAL, dtype=np.float32), np.empty(TOTAL, dtype=np.int8)
            seen = np.zeros(meta['rows'], dtype=bool)
            offset, current_date, loader = 0, None, None
            summaries = []
            for bi, entry in enumerate(blocks):
                date = entry['block'].split('/')[0].split('=')[1]
                if date not in DATES:
                    raise ValueError('Non-Validation target date')
                if current_date != date:
                    if loader is not None:
                        loader.builder.clear_cache()
                        del loader
                        gc.collect()
                    engine = FeatureEngine(arrays, bank, vocab['payment_format'].index('ACH') + 2,
                                           meta['data_start_us'], int(np.datetime64(END, 'us').astype(np.int64)))
                    loader = SimpleNamespace(day=TargetDay(archive, features, date, arrays),
                                             builder=TemporalBatchBuilder(engine, adj, store, bank_cat))
                    current_date = date
                update(status='running', stage='inference', current_block=entry['block'],
                       completed_blocks=bi, processed_targets=offset)
                cached = ParquetBlockCache(cache / entry['block'], loader, cache_hash)
                ids = cached.targets
                pos = loader.day.positions(ids)
                expected_y, times = loader.day.labels[pos], arrays['timestamp'][ids]
                if len(ids) != entry['targets'] or np.any(seen[ids]):
                    raise ValueError('Repeated/incomplete Validation prediction targets')
                folder = out / 'predictions' / entry['block']
                record = folder / 'manifest.json'
                path = folder / 'predictions.parquet'
                if record.exists():
                    old = read(record)
                    if old['config_sha256'] != config_hash or old['target_ids_sha256'] != array_sha(ids) or old['file_sha256'] != sha(path):
                        raise ValueError('Committed prediction block changed')
                    values = verify_predictions(pq.ParquetFile(path).read(), ids, times, expected_y)
                else:
                    values = np.empty(len(ids), dtype=np.float32)
                    done = 0
                    with torch.inference_mode():
                        for j in range(len(cached)):
                            batch, y, _ = cached.batch(j)
                            logits = model(to_torch(batch, 'cuda'))
                            if not torch.isfinite(logits).all().item():
                                raise ValueError('Nonfinite Validation logits')
                            p = torch.sigmoid(logits).cpu().numpy()
                            if not np.array_equal(batch['target_edge_id'], ids[done:done+len(y)]):
                                raise ValueError('Batch target ordering differs')
                            values[done:done+len(y)] = p
                            done += len(y)
                            if (j+1) % 64 == 0:
                                update(processed_targets=offset+done, block_batch=j+1)
                    if done != len(ids):
                        raise ValueError('Incomplete prediction block')
                    table = prediction_table(ids, times, expected_y, values)
                    folder.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix('.parquet.tmp')
                    pq.write_table(table, temp, compression='zstd')
                    verify_predictions(pq.ParquetFile(temp).read(), ids, times, expected_y)
                    os.replace(temp, path)
                    atomic_json(record, dict(config_sha256=config_hash, targets=len(ids),
                                              target_ids_sha256=array_sha(ids), file_sha256=sha(path)))
                scores[offset:offset+len(ids)] = values
                labels[offset:offset+len(ids)] = expected_y
                seen[ids] = True
                offset += len(ids)
                summaries.append(dict(block=entry['block'], targets=len(ids), manifest_sha256=sha(record)))
                del cached
                update(processed_targets=offset, completed_blocks=bi+1)
            if offset != TOTAL or labels.sum() != POSITIVES or seen.sum() != TOTAL:
                raise ValueError('Full Validation prediction count mismatch')
            metrics = binary_metrics(labels, scores)
            per_day = {}
            for date in DATES:
                # Concatenated prediction order follows cache blocks, grouped by day.
                start = sum(x['targets'] for x in summaries if x['block'].split('/')[0] < 'transaction_date=' + date)
                count = sum(x['targets'] for x in summaries if x['block'].startswith('transaction_date=' + date + '/'))
                per_day[date] = binary_metrics(labels[start:start+count], scores[start:start+count])
            report = dict(status='completed', config_sha256=config_hash, model_sha256=config['model_sha256'],
                          metrics=metrics, per_day=per_day, blocks=summaries,
                          completed_at=datetime.now(timezone.utc).isoformat(),
                          scope='Full 2022-09-11..13 Validation. Thresholds selected here are not Test results.')
            atomic_json(out / 'metrics.json', report)
            update(status='completed', stage='complete', processed_targets=TOTAL, current_block=None)
            return report
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path.cwd())
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.project, args.model, args.output)
