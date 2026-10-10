"""Independent complete Validation predictions for frozen seed repeats."""
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
def verified_training(project, trial, seed):
    """Validate final training coverage/config/model and actually load weights."""
    project = Path(project).resolve()
    if trial not in (*TRIALS, 'lr_only'):
        raise ValueError('Undeclared trial')
    folder = trial_root(project, trial, seed)
    verify_plan(project)
    config = read(folder/'config.json')
    result = read(folder/'training_result.json')
    model_path = folder/'final_model.pt'
    digest, config_hash = sha(model_path), sha(folder/'config.json')
    if (result.get('status') != 'completed' or result.get('epoch') != 1 or result.get('targets') != TRAIN_TOTAL
            or result.get('positives') != TRAIN_POSITIVES or result.get('blocks') != 797
            or result.get('config_sha256') != config_hash or result.get('final_model_sha256') != digest):
        raise ValueError('Incomplete/changed final training artifact: '+trial)
    if config.get('seed') != seed or result.get('seed') != seed or config.get('trial') != trial:
        raise ValueError('Repeat seed/trial identity mismatch')
    if config.get('repeat_plan_sha256') != sha(project/OUTPUT/'config.json'):
        raise ValueError('Repeat plan binding mismatch')
    for rel, expected_hash in {**config.get('code_sha256', {}), **config.get('v2_code_sha256', {})}.items():
        path = checked_path(project, rel) if len(Path(rel).parts) > 1 else Path(__file__).with_name(rel)
        if sha(path) != expected_hash:
            raise ValueError('Frozen training code changed: '+rel)
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    if saved.get('seed') != seed or saved.get('trial') != trial:
        raise ValueError('Saved repeat seed/trial differs')
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


def run(project, trial, output=None, progress=None, seed=None):
    from .v2_features import SidecarStore
    from .v2_adapter import adapt_arrays

    project = Path(project).resolve()
    if trial not in TRIALS:
        raise ValueError('Undeclared v2 evaluation trial')
    root = project/OUTPUT
    check_seed(seed)
    out = Path(output).resolve() if output else eval_root(project, trial, seed)
    if out != eval_root(project, trial, seed):
        raise ValueError('Evaluation output must be the canonical independent directory for this trial')
    _disk(project)
    state = dict(status='preparing', pid=os.getpid(), trial=trial, seed=seed, expected_targets=TOTAL,
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
            trained = verified_training(project, trial, seed)
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
            config = dict(version='seed_repeat_1', trial=trial, seed=seed, dates=DATES, expected_targets=TOTAL, expected_positives=POSITIVES,
                          model_path=str(trained['model_path'].relative_to(project)), model_sha256=trained['model_sha256'],
                          train_config_sha256=trained['config_sha256'], training_result_sha256=trained['training_result_sha256'],
                          cache_manifest_sha256=sha(cache/'manifest.json'), cache_config_sha256=cache_hash,
                          sidecar_manifest_sha256=sidecar.manifest_sha256, sidecar_config_sha256=sidecar.config_sha256,
                          sidecar_contract_sha256=sidecar.contract_sha256,
                          code_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                       ('repeat_evaluate.py', 'repeat_common.py', 'v4_metrics.py', 'v2_adapter.py', 'v2_features.py', 'evaluation_metrics.py',
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
                            batch = adapt_arrays(batch, extra, 'features_only') if trial == 'features_only' else batch
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
            report = dict(status='completed', seed=seed, trial=trial, config_sha256=config_hash, model_sha256=trained['model_sha256'],
                          metrics=score_metrics(labels, scores), precision80=point, non_ach_precision80=non_ach_point,
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
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--trial', choices=TRIALS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = p.parse_args()
    result = run(a.project, a.trial, seed=a.seed)
    if result.get('status') == 'paused': raise SystemExit(75)
