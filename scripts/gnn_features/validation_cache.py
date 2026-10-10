"""Separate, resumable full-Validation cache using the frozen feature algorithms."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pyarrow.parquet as pq

from .build import load_archive
from .cache_train import (array_sha, chunks, compute_block, sha, validate_existing)
from .contract import HOUR
from .engine import FeatureEngine
from .hybrid_cache_trial import ENCODED_NAMES
from .loader import TargetDay, TemporalBatchBuilder
from .training_state import ProcessLock, atomic_json

DATES = [f'2022-09-{d:02d}' for d in (11, 12, 13)]
TOTAL = 4689006
POSITIVES = 6420
END = '2022-09-14T00:00:00'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validation_spec(project):
    split = read(Path(project) / 'config/dataset_split.json')
    item, = [s for s in split['splits'] if s['name'] == 'validation']
    if item['dates'] != DATES or item['rows'] != TOTAL or item['positive_rows'] != POSITIVES:
        raise ValueError('Frozen Validation scope changed')
    return item


def init_validation_worker(archive):
    # The shared numerical worker accepts an explicitly validated history archive.
    # Its Train-only launcher and frozen constants are never modified or invoked.
    from . import cache_train
    cache_train.WORKER = load_archive(Path(archive))


def publish_verified(work, final, root, config_hash, targets):
    root, work, final = Path(root).resolve(), Path(work), Path(final)
    if work.is_symlink() or final.is_symlink() or work.resolve() != root / '_work' / 'active':
        raise ValueError('Unsafe verified scratch path')
    if not final.resolve().is_relative_to(root) or final.exists():
        raise ValueError('Unsafe/existing destination')
    manifest = validate_existing(work, config_hash, targets)
    if manifest['block'] != final.relative_to(root).as_posix():
        raise ValueError('Verified scratch belongs to a different block; preserved')
    final.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        try:
            os.rename(work, final)
            return manifest
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.5)


def discard_unverified(work, root):
    root, work = Path(root).resolve(), Path(work)
    if work.is_symlink() or work.resolve() != root / '_work' / 'active':
        raise ValueError('Unsafe uncommitted scratch path')
    if (work / 'manifest.json').exists():
        raise ValueError('Scratch has a manifest; preserve it for checked publication')
    if work.exists():
        shutil.rmtree(work)


def run(project, output=None, progress=None, max_blocks=None):
    project = Path(project).resolve()
    archive = project / 'data/gnn_v1_validation'
    features = project / 'result/gnn_validation_v1'
    out = Path(output or project / 'data/gnn_validation_cache_parquet').resolve()
    if not out.is_relative_to(project / 'data'):
        raise ValueError('Validation cache must be inside project data')
    validation_spec(project)
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', pid=os.getpid(), expected_targets=TOTAL, dates=DATES,
                 started_at=datetime.now(timezone.utc).isoformat(), verified_targets=0)

    def update(**fields):
        state.update(fields)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out / 'status.json', state)
        if progress:
            progress(**fields)

    with ProcessLock(out):
        try:
            update(stage='preflight')
            paths = [project / 'config/dataset_split.json', archive / 'build.json']
            paths += list((archive / 'metadata').glob('*.json'))
            paths += list((archive / 'history_index').glob('*.npy'))
            paths += list((archive / 'graph/adj_index').glob('*.npy'))
            paths += sorted((archive / 'graph/edges').rglob('*.parquet'))
            paths += sorted((archive / 'supervision/targets').rglob('*.parquet'))
            paths += [project / 'data/gnn_tensor_cache_v1' / n
                      for n in ('manifest.json', 'numeric.npy', 'category.npy')]
            paths += [Path(__file__).with_name(n) for n in (
                'validation_cache.py', 'cache_train.py', 'hybrid_cache_trial.py',
                'build.py', 'engine.py', 'rolling.py', 'contract.py', 'loader.py')]
            expected_days = {}
            for date in DATES:
                folder = features / f'transaction_date={date}'
                daily, check = read(folder / 'daily_run.json'), read(folder / 'verification.json')
                if daily['status'] != 'verified' or check['status'] != 'passed' or daily['date'] != date:
                    raise ValueError('Unverified Validation R inputs')
                if read(folder / 'run_config.json')['archive_manifest_sha256'] != sha(archive / 'build.json'):
                    raise ValueError('Validation R archive identity differs')
                expected_days[date] = check
                paths += [folder / n for n in ('target_edge_ids.npy', 'merge_phase.json',
                                               'verification.json', 'run_config.json')]
                paths += [Path(p['path']) for p in read(folder / 'merge_phase.json')['parts']]
            if sum(c['full_day_rows'] for c in expected_days.values()) != TOTAL or sum(c['positive_rows'] for c in expected_days.values()) != POSITIVES:
                raise ValueError('Validation daily totals differ')
            config = dict(version=1, scope='Full independent Validation; no Test', dates=DATES,
                          end_exclusive=END, expected_targets=TOTAL, expected_positives=POSITIVES,
                          batch_size=64, max_batches_per_block=512, workers=4,
                          fanout=[10, 5], seed=20260920,
                          hashes={str(p.relative_to(project)): sha(p) for p in paths})
            if (out / 'config.json').exists():
                if read(out / 'config.json') != config:
                    raise ValueError('Frozen Validation cache code/inputs changed')
            else:
                atomic_json(out / 'config.json', config)
            config_hash = sha(out / 'config.json')
            meta, arrays, bank, vocab = load_archive(archive)
            if meta['rows'] != 20318992 + TOTAL or meta['end_exclusive'] != END:
                raise ValueError('Validation archive has wrong history extent')
            engine = FeatureEngine(arrays, bank, vocab['payment_format'].index('ACH') + 2,
                                   meta['data_start_us'], int(np.datetime64(END, 'us').astype(np.int64)))
            adj = {d + '_' + s: np.load(archive / f'graph/adj_index/{d}_{s}.npy', mmap_mode='r')
                   for d in ('in', 'out') for s in ('ptr', 'edge_ids')}
            builder = TemporalBatchBuilder(engine, adj, None, None)
            completed, new_blocks = [], 0
            with ProcessPoolExecutor(max_workers=4, initializer=init_validation_worker,
                                     initargs=(str(archive),)) as pool:
                for date in DATES:
                    day = TargetDay.__new__(TargetDay)
                    day.ids = np.load(features / f'transaction_date={date}/target_edge_ids.npy').astype(np.int64)
                    day.times = np.asarray(arrays['timestamp'][day.ids])
                    day.order = np.lexsort((day.ids, day.times))
                    lo = int(np.datetime64(date, 'us').astype(np.int64))
                    if np.any(np.diff(day.ids) <= 0) or not np.all((day.times >= lo) & (day.times < lo + 24 * HOUR)):
                        raise ValueError('Validation target identity/date mismatch')
                    if len(day.ids) != expected_days[date]['full_day_rows']:
                        raise ValueError('Validation target day incomplete')
                    daily_count = 0
                    for hour, index, batches in chunks(day):
                        rel = f'transaction_date={date}/hour={hour % 24:02d}/block={index:04d}'
                        final = out / rel
                        targets = np.concatenate(batches)
                        update(current_block=rel, stage='resume_check', block_targets=len(targets))
                        if final.exists():
                            m = validate_existing(final, config_hash, targets)
                        else:
                            work = out / '_work/active'
                            if (work / 'manifest.json').exists():
                                m = publish_verified(work, final, out, config_hash, targets)
                            else:
                                if shutil.disk_usage(out).free < 20 * 1024 ** 3:
                                    raise RuntimeError('Less than 20 GiB free')
                                discard_unverified(work, out)
                                work.mkdir(parents=True)
                                m = compute_block(work, batches, (hour + 1) * HOUR,
                                                  arrays, bank, meta, vocab, builder, pool, update)
                                m.update(status='verified', config_sha256=config_hash,
                                         target_ids_sha256=array_sha(targets), block=rel,
                                         verified_at=datetime.now(timezone.utc).isoformat())
                                atomic_json(work / 'manifest.json', m)
                                m = publish_verified(work, final, out, config_hash, targets)
                                new_blocks += 1
                        daily_count += m['targets']
                        completed.append({k: m[k] for k in ('block', 'targets', 'node_states', 'parquet_bytes')})
                        update(stage='block_verified', completed_blocks=len(completed),
                               verified_targets=sum(x['targets'] for x in completed),
                               parquet_bytes=sum(x['parquet_bytes'] for x in completed))
                        print(json.dumps(dict(stage='validation_cache_block', block=rel,
                                              verified_targets=state['verified_targets'])), flush=True)
                        gc.collect()
                        if max_blocks is not None and new_blocks >= max_blocks:
                            update(status='stopped_after_requested_limit')
                            return
                    if daily_count != len(day.ids):
                        raise ValueError('Missing Validation targets')
                    atomic_json(out / f'transaction_date={date}/manifest.json', dict(
                        status='verified', targets=daily_count,
                        blocks=[x for x in completed if x['block'].startswith(f'transaction_date={date}/')]))
            if sum(x['targets'] for x in completed) != TOTAL:
                raise ValueError('Full Validation coverage missing')
            manifest = dict(status='completed', targets=TOTAL, positives=POSITIVES, dates=DATES,
                            blocks=completed, config_sha256=config_hash, node_columns=ENCODED_NAMES,
                            feature_tables=str(features), edge_and_identity_tables=str(archive))
            atomic_json(out / 'manifest.json', manifest)
            update(status='completed', stage='complete', current_block=None)
            return manifest
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path.cwd())
    p.add_argument('--max-blocks', type=int)
    args = p.parse_args()
    run(args.project, max_blocks=args.max_blocks)
