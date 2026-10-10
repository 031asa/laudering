"""Independent v2 full-Validation inference and a single global P80 threshold.

The frozen v1 evaluator/loader/cache are imported read-only. No Test data is
opened, no validation resampling is performed, and equal scores stay together.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import traceback
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from .build import load_archive
from .cache_train import array_sha, sha
from .engine import FeatureEngine
from .evaluate_validation import ValidationEdgeStore, prediction_table, verify_predictions
from .evaluation_metrics import binary_metrics
from .loader import TargetDay, TemporalBatchBuilder
from .model import BidirectionalGINE, to_torch
from .model_cache import ParquetBlockCache
from .training_state import ProcessLock, atomic_json
from .validation_cache import DATES, TOTAL, POSITIVES, END, read, validation_spec

TRAIN_TOTAL = 20318992
TRAIN_POSITIVES = 17794
TRIALS = ('features_only', 'features_pruned', 'features_balanced')
BASELINE_SHA = 'b8c18e99903dd28f84c8a5d45ded34314f7b7b5a2cd5e25c23495ec5ade57d3a'


def _inputs(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    if y.ndim != 1 or y.shape != s.shape or not len(y):
        raise ValueError('Nonempty matching one-dimensional labels/scores required')
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Invalid labels or probabilities')
    return y.astype(np.int8, copy=False), s.astype(np.float64, copy=False)


def confusion(labels, predicted, threshold=None):
    y, hit = np.asarray(labels), np.asarray(predicted, dtype=bool)
    if y.ndim != 1 or y.shape != hit.shape or not np.isin(y, (0, 1)).all():
        raise ValueError('Invalid confusion inputs')
    positives, alerts = int(np.count_nonzero(y)), int(hit.sum())
    tp = int(np.count_nonzero(y[hit]))
    fp, fn = alerts - tp, positives - tp
    negatives = len(y) - positives
    return dict(threshold=None if threshold is None else float(threshold), samples=len(y),
                positives=positives, negatives=negatives, alerts=alerts,
                tp=tp, fp=fp, fn=fn, tn=negatives-fp,
                precision=tp/alerts if alerts else None,
                recall=tp/positives if positives else None,
                fpr=fp/negatives if negatives else None,
                f1=2*tp/(positives+alerts) if positives+alerts else None)


def precision80_point(labels, scores):
    """Maximize recall at P >= 4/5 over whole ties; no empty-alert solution.

    Equal recall prefers higher precision, then a higher threshold. The integer
    test 5*TP >= 4*(TP+FP) avoids tolerance accepting a sub-80% threshold.
    """
    y, s = _inputs(labels, scores)
    if not y.any():
        return None
    order = np.argsort(-s, kind='stable')
    ordered = s[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(s)-1]
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    alerts = ends+1
    eligible = np.flatnonzero((tp > 0) & (5*tp >= 4*alerts))
    if not len(eligible):
        return None
    # Threshold groups descend; first maximum TP minimizes FP at that TP.
    index = int(eligible[np.argmax(tp[eligible])])
    threshold = float(ordered[ends[index]])
    return confusion(y, s >= threshold, threshold)


def _distinct(mask, src, dst):
    return dict(transactions=int(mask.sum()), payer_accounts=int(len(np.unique(src[mask]))),
                receiver_accounts=int(len(np.unique(dst[mask]))),
                either_accounts=int(len(np.union1d(src[mask], dst[mask]))))


def paired_changes(labels, before_scores, before_point, after_scores, after_point, src, dst):
    """Transaction and unique-account changes at each model's one global point."""
    y, old = _inputs(labels, before_scores)
    _, new = _inputs(labels, after_scores)
    src, dst = np.asarray(src), np.asarray(dst)
    if src.shape != y.shape or dst.shape != y.shape:
        raise ValueError('Paired-account identity shape differs')
    if before_point is None or after_point is None:
        return dict(available=False, reason='One model has no nonempty P80 operating point')
    a, b = old >= before_point['threshold'], new >= after_point['threshold']
    positive = y == 1
    return dict(available=True, before_threshold=before_point['threshold'], after_threshold=after_point['threshold'],
                gained_tp=_distinct(positive & ~a & b, src, dst),
                lost_tp=_distinct(positive & a & ~b, src, dst),
                new_fp=_distinct(~positive & ~a & b, src, dst),
                removed_fp=_distinct(~positive & a & ~b, src, dst),
                retained_tp=_distinct(positive & a & b, src, dst))


def support_bucket(counts):
    n = np.asarray(counts)
    if n.ndim != 1 or np.any(n < -1) or not np.equal(n, np.floor(n)).all():
        raise ValueError('Invalid historical support counts')
    return np.where(n < 0, 'not_applicable', np.where(n == 0, '0', np.where(n < 5, '1-4', '>=5')))


def group_report(labels, scores, point, groups, src, dst, group_labels=None):
    """All groups use the global point, never subgroup-specific thresholds."""
    y, s = _inputs(labels, scores)
    if np.asarray(src).shape != y.shape or np.asarray(dst).shape != y.shape:
        raise ValueError('Group-account shape differs')
    hit = None if point is None else s >= point['threshold']
    result = {}
    for name, categories in groups.items():
        categories = np.asarray(categories)
        if categories.shape != y.shape:
            raise ValueError('Group shape differs: '+name)
        entries = {}
        for value in np.unique(categories):
            mask = categories == value
            local_y = y[mask]
            item = dict(samples=int(mask.sum()), positives=int(local_y.sum()),
                        payer_accounts=int(len(np.unique(src[mask]))), receiver_accounts=int(len(np.unique(dst[mask]))))
            item['operating_point'] = None if hit is None else confusion(local_y, hit[mask], point['threshold'])
            if hit is not None:
                item['detected_positive_accounts'] = _distinct(mask & (y == 1) & hit, src, dst)
                item['false_positive_accounts'] = _distinct(mask & (y == 0) & hit, src, dst)
            display = str(value) if group_labels is None or name not in group_labels else group_labels[name][value]
            entries[display] = item
        result[name] = entries
    return result


def checked_path(root, relative):
    """Only a relative, non-symlink block path contained in its declared root."""
    root, rel = Path(root).resolve(), Path(relative)
    path = root / rel
    if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Unsafe block path: '+str(relative))
    return path


def _disk(project):
    if shutil.disk_usage(project).free < 20*1024**3:
        raise RuntimeError('Less than 20 GiB free; stop new writes without deleting data')


def verified_training(project, trial):
    """Validate final training coverage/config/model and actually load weights."""
    project = Path(project).resolve()
    if trial not in (*TRIALS, 'lr_only'):
        raise ValueError('Undeclared trial')
    folder = project / ('result/gnn_tuning_v1' if trial == 'lr_only' else 'result/gnn_tuning_v2') / 'trials' / trial
    config = read(folder/'config.json')
    result = read(folder/'training_result.json')
    model_path = folder/'final_model.pt'
    digest, config_hash = sha(model_path), sha(folder/'config.json')
    if (result.get('status') != 'completed' or result.get('epoch') != 1 or result.get('targets') != TRAIN_TOTAL
            or result.get('positives') != TRAIN_POSITIVES or result.get('blocks') != 797
            or result.get('config_sha256') != config_hash or result.get('final_model_sha256') != digest):
        raise ValueError('Incomplete/changed final training artifact: '+trial)
    if trial == 'lr_only' and digest != BASELINE_SHA:
        raise ValueError('Frozen lr_only reference hash differs')
    for rel, expected_hash in {**config.get('code_sha256', {}), **config.get('v2_code_sha256', {})}.items():
        path = checked_path(project, rel) if len(Path(rel).parts) > 1 else Path(__file__).with_name(rel)
        if sha(path) != expected_hash:
            raise ValueError('Frozen training code changed: '+rel)
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    architecture = saved.get('config') or saved.get('model_config')
    if not isinstance(architecture, dict):
        raise ValueError('Missing final model architecture')
    if config.get('model') != architecture:
        raise ValueError('Saved architecture differs from training configuration')
    if saved.get('config_sha256') != config_hash:
        raise ValueError('Final model config identity differs')
    if saved.get('epoch') != 1 or saved.get('targets') != TRAIN_TOTAL or saved.get('positives') != TRAIN_POSITIVES:
        raise ValueError('Final model training counts differ')
    model = BidirectionalGINE(**architecture)
    model.load_state_dict(saved['model'], strict=True)
    if not all(torch.isfinite(v).all().item() for v in model.state_dict().values()):
        raise ValueError('Nonfinite final model weights')
    expected = (63, 110) if trial == 'lr_only' else ((63, 169) if trial == 'features_only' else (57, 165))
    if (architecture.get('node_dim', 63), architecture.get('r_dim', 110)) != expected:
        raise ValueError('Trial model input dimensions differ')
    del saved
    return dict(model=model, model_path=model_path, model_sha256=digest, config=config,
                config_sha256=config_hash, training_result_sha256=sha(folder/'training_result.json'))


def run(project, trial, output=None, progress=None):
    from .v2_features import SidecarStore
    from .v2_adapter import adapt_arrays

    project = Path(project).resolve()
    if trial not in TRIALS:
        raise ValueError('Undeclared v2 evaluation trial')
    root = project/'result/gnn_tuning_v2'
    out = Path(output).resolve() if output else root/'evaluations'/trial
    if out != root/'evaluations'/trial:
        raise ValueError('Evaluation output must be the canonical independent directory for this trial')
    _disk(project)
    state = dict(status='preparing', pid=os.getpid(), trial=trial, expected_targets=TOTAL,
                 processed_targets=0, started_at=datetime.now(timezone.utc).isoformat())

    def update(**fields):
        state.update(fields)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out/'status.json', state)
        if progress:
            progress(**fields)

    with ProcessLock(out):
        try:
            update(stage='verify_inputs')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is mandatory; no CPU inference fallback')
            torch.set_num_threads(4)
            validation_spec(project)
            trained = verified_training(project, trial)
            model = trained['model'].to('cuda').eval()
            archive, features = project/'data/gnn_v1_validation', project/'result/gnn_validation_v1'
            cache = project/'data/gnn_validation_cache_parquet'
            manifest, cache_config = read(cache/'manifest.json'), read(cache/'config.json')
            cache_hash = sha(cache/'config.json')
            if (manifest['status'] != 'completed' or read(cache/'status.json')['status'] != 'completed'
                    or manifest['config_sha256'] != cache_hash or manifest['targets'] != TOTAL or manifest['dates'] != DATES):
                raise ValueError('Full verified frozen Validation cache required')
            for rel, digest in cache_config['hashes'].items():
                if sha(checked_path(project, rel)) != digest:
                    raise ValueError('Frozen Validation input changed: '+rel)
            blocks = manifest['blocks']
            if (len(blocks) != 197 or len({b['block'] for b in blocks}) != 197
                    or sum(b['targets'] for b in blocks) != TOTAL
                    or [b['block'] for b in blocks] != sorted(b['block'] for b in blocks)):
                raise ValueError('Validation block inventory differs')
            sidecar = SidecarStore(project/'data/gnn_v2_target_features/validation', cache, expected_split='validation')
            if trained['config']['sidecar']['contract_sha256'] != sidecar.contract_sha256:
                raise ValueError('Train/Validation extra-feature contracts differ')
            config = dict(version=2, trial=trial, dates=DATES, expected_targets=TOTAL, expected_positives=POSITIVES,
                          model_path=str(trained['model_path'].relative_to(project)), model_sha256=trained['model_sha256'],
                          train_config_sha256=trained['config_sha256'], training_result_sha256=trained['training_result_sha256'],
                          cache_manifest_sha256=sha(cache/'manifest.json'), cache_config_sha256=cache_hash,
                          sidecar_manifest_sha256=sidecar.manifest_sha256, sidecar_config_sha256=sidecar.config_sha256,
                          sidecar_contract_sha256=sidecar.contract_sha256,
                          code_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                       ('v2_evaluate.py', 'v2_adapter.py', 'v2_features.py', 'evaluation_metrics.py',
                                        'evaluate_validation.py', 'model.py', 'loader.py', 'model_cache.py')},
                          selection='One global whole-tie threshold maximizing recall at precision >= 0.8; Test untouched')
            if (out/'config.json').exists():
                if read(out/'config.json') != config:
                    raise ValueError('Evaluation model, code, cache or sidecar identity changed')
            else:
                _disk(project)
                atomic_json(out/'config.json', config)
            config_hash = sha(out/'config.json')
            meta, arrays, bank, vocab = load_archive(archive)
            if meta['end_exclusive'] != END or meta['rows'] != TRAIN_TOTAL+TOTAL:
                raise ValueError('Validation history scope differs')
            adj = {d+'_'+s: np.load(archive/f'graph/adj_index/{d}_{s}.npy', mmap_mode='r')
                   for d in ('in', 'out') for s in ('ptr', 'edge_ids')}
            bank_cat = np.load(archive/'history_index/bank_cat.npy', mmap_mode='r')
            if np.any(bank_cat < 0) or np.any(bank_cat >= trained['config']['model']['bank_count']):
                raise ValueError('Bank vocabulary differs')
            store = ValidationEdgeStore(project, archive)
            scores, labels = np.empty(TOTAL, np.float32), np.empty(TOTAL, np.int8)
            seen = np.zeros(TOTAL, bool)
            offset, current_date, loader, summaries = 0, None, None, []
            for bi, entry in enumerate(blocks):
                pause = root/'pause_request.json'
                if pause.exists() and read(pause).get('enabled') is True:
                    update(status='paused', stage='paused', completed_blocks=bi, processed_targets=offset)
                    return dict(status='paused', completed_blocks=bi, processed_targets=offset)
                block = entry['block']
                date = block.split('/')[0].split('=')[1]
                if date not in DATES:
                    raise ValueError('Non-Validation target date')
                if current_date != date:
                    if loader is not None:
                        loader.builder.clear_cache()
                        del loader
                        gc.collect()
                    engine = FeatureEngine(arrays, bank, vocab['payment_format'].index('ACH')+2,
                                           meta['data_start_us'], int(np.datetime64(END, 'us').astype(np.int64)))
                    loader = SimpleNamespace(day=TargetDay(archive, features, date, arrays),
                                             builder=TemporalBatchBuilder(engine, adj, store, bank_cat))
                    current_date = date
                update(status='running', stage='inference', current_block=block, completed_blocks=bi, processed_targets=offset)
                cached = ParquetBlockCache(checked_path(cache, block), loader, cache_hash)
                ids = cached.targets
                pos = loader.day.positions(ids)
                expected_y, times = loader.day.labels[pos], arrays['timestamp'][ids]
                local_ids = ids-TRAIN_TOTAL
                if (len(ids) != entry['targets'] or np.any(local_ids < 0) or np.any(local_ids >= TOTAL)
                        or np.any(seen[local_ids])):
                    raise ValueError('Repeated/out-of-scope Validation target')
                # Verify sidecar block even when committed predictions are reused.
                sidecar.block(block, ids, times)
                folder = checked_path(out/'predictions', block)
                record, path = folder/'manifest.json', folder/'predictions.parquet'
                if record.exists():
                    old = read(record)
                    if (old['config_sha256'] != config_hash or old['target_ids_sha256'] != array_sha(ids)
                            or old['targets'] != len(ids) or old['file_sha256'] != sha(path)):
                        raise ValueError('Committed prediction block changed')
                    values = verify_predictions(pq.ParquetFile(path).read(), ids, times, expected_y)
                else:
                    values, done = np.empty(len(ids), np.float32), 0
                    with torch.inference_mode():
                        for j in range(len(cached)):
                            batch, y, _ = cached.batch(j)
                            bid = batch['target_edge_id']
                            extra = sidecar.block(block, bid, arrays['timestamp'][bid])
                            batch = adapt_arrays(batch, extra, trial)
                            logits = model(to_torch(batch, 'cuda'))
                            if not torch.isfinite(logits).all().item():
                                raise ValueError('Nonfinite Validation logits')
                            if not np.array_equal(bid, ids[done:done+len(y)]):
                                raise ValueError('Prediction batch target ordering differs')
                            values[done:done+len(y)] = torch.sigmoid(logits).cpu().numpy()
                            done += len(y)
                            if (j+1) % 64 == 0:
                                update(processed_targets=offset+done, block_batch=j+1)
                    if done != len(ids):
                        raise ValueError('Incomplete prediction block')
                    _disk(project)
                    table = prediction_table(ids, times, expected_y, values)
                    folder.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix('.parquet.tmp')
                    pq.write_table(table, temp, compression='zstd')
                    verify_predictions(pq.ParquetFile(temp).read(), ids, times, expected_y)
                    os.replace(temp, path)
                    atomic_json(record, dict(config_sha256=config_hash, targets=len(ids),
                                             target_ids_sha256=array_sha(ids), file_sha256=sha(path)))
                scores[local_ids], labels[local_ids], seen[local_ids] = values, expected_y, True
                offset += len(ids)
                summaries.append(dict(block=block, targets=len(ids), manifest_sha256=sha(record)))
                del cached
                update(processed_targets=offset, completed_blocks=bi+1)
            if offset != TOTAL or int(labels.sum()) != POSITIVES or not seen.all():
                raise ValueError('Full Validation prediction coverage mismatch')
            point = precision80_point(labels, scores)
            non_ach = arrays['payment_format_cat'][TRAIN_TOTAL:] != vocab['payment_format'].index('ACH')+2
            if int(labels[non_ach].sum()) != 715:
                raise ValueError('Unexpected non-ACH Validation positive count')
            non_ach_point = None if point is None else confusion(labels[non_ach], scores[non_ach] >= point['threshold'], point['threshold'])
            report = dict(status='completed', config_sha256=config_hash, model_sha256=trained['model_sha256'],
                          metrics=binary_metrics(labels, scores), precision80=point, non_ach_precision80=non_ach_point,
                          blocks=summaries, completed_at=datetime.now(timezone.utc).isoformat(),
                          scope='Full natural-prevalence Validation 2022-09-11..13; developer analysis and selection; Test not evaluated')
            if (out/'metrics.json').exists():
                committed = read(out/'metrics.json')
                report['completed_at'] = committed.get('completed_at')
                if not report['completed_at'] or committed != report:
                    raise ValueError('Completed evaluation report differs on verification replay')
            else:
                _disk(project)
                atomic_json(out/'metrics.json', report)
            update(status='completed', stage='complete', processed_targets=TOTAL, current_block=None)
            return report
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--trial', choices=TRIALS, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.project, args.trial, args.output)
    if result.get('status') == 'paused':
        raise SystemExit(75)
