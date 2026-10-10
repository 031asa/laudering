"""Train-only structural and sampled feature redundancy audit for v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cache_train import sha
from .contract import N_FIELDS, R_FIELDS
from .hybrid_cache_trial import ENCODED_NAMES
from .model_cache import _plans
from .training_state import ProcessLock, atomic_json

ROOT = Path(__file__).resolve().parents[2]
TOTAL = 20318992


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Audit input outside project')
    return path


def summarize(values, names):
    """Pairwise Pearson correlation of actual encoded values, ignoring missing."""
    frame = pd.DataFrame(values, columns=names)
    columns = []
    for name in names:
        x = frame[name].dropna()
        columns.append(dict(name=name, rows=len(frame), valid=len(x),
                            missing_rate=float(1 - len(x) / max(len(frame), 1)),
                            constant=bool(len(x) and x.min() == x.max()),
                            minimum=None if not len(x) else float(x.min()),
                            maximum=None if not len(x) else float(x.max())))
    corr = frame.corr(min_periods=50)
    pairs = []
    for i, left in enumerate(names):
        for j in range(i+1, len(names)):
            value = corr.iloc[i, j]
            if np.isfinite(value) and abs(value) >= .98:
                right = names[j]
                pairs.append(dict(left=left, right=right, correlation=float(value),
                    jointly_valid=int(np.sum(np.isfinite(values[:, i]) & np.isfinite(values[:, j])))))
    pairs.sort(key=lambda x: (-abs(x['correlation']), x['left'], x['right']))
    matrix = [[None if not np.isfinite(v) else float(v) for v in row] for row in corr.to_numpy()]
    return dict(rows=len(frame), columns=columns, correlation_names=names,
                correlation_matrix=matrix, high_correlation_pairs=pairs,
                interpretation='Descriptive Train sample only; correlation does not establish causal harm or authorize more deletions')


def decode_node_batch(table):
    values = np.column_stack([table[f['name']+'__encoded'].to_numpy() for f in N_FIELDS]).astype(np.float64)
    for i, field in enumerate(N_FIELDS):
        if field['nullable']:
            values[~table[field['name']+'__valid'].to_numpy().astype(bool), i] = np.nan
    return values


def check_ach_relationship(values):
    names = [f['name'] for f in N_FIELDS]
    results = {}
    for direction in ('in', 'out'):
        total = values[:, names.index(f'n_{direction}_all_count_24h')]
        ach = values[:, names.index(f'n_{direction}_ach_count_24h')]
        share = values[:, names.index(f'n_{direction}_ach_share_24h')]
        valid = np.isfinite(share)
        if (np.any(valid != (total > 0)) or np.any(share[valid] < 0)
                or np.any(share[valid] > 1) or np.any(ach > total + 3e-5)):
            raise ValueError('Invalid ACH count/share relation')
        predicted = np.log1p(np.expm1(total[valid]) * share[valid])
        error = np.abs(predicted - ach[valid])
        if np.any(error > 3e-5) or np.any(ach[~valid] != 0):
            raise ValueError('ACH share differs from ACH count / all count')
        results[direction] = dict(checked=int(valid.sum()), max_encoded_absolute_error=float(error.max()) if len(error) else 0.)
    return results


def run(project=ROOT):
    from .v2_features import EXTRA_FIELDS, SidecarStore
    project = Path(project).resolve()
    output = project / 'result/gnn_tuning_v2/audit'
    cache = project / 'data/gnn_train_cache_parquet'
    sidecar = project / 'data/gnn_v2_target_features/train'
    start = time.monotonic()
    state = dict(pid=os.getpid(), status='running', stage='train_feature_audit')

    def update(**fields):
        flag = project / 'result/gnn_tuning_v2/pause_request.json'
        if flag.exists() and read(flag).get('enabled') is True:
            atomic_json(output / 'status.json', dict(state, status='paused'))
            raise SystemExit(75)
        state.update(fields, seconds=time.monotonic()-start,
                     updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / 'status.json', state)

    with ProcessLock(output):
        try:
            update()
            root = read(cache / 'manifest.json')
            if root['status'] != 'completed' or root['targets'] != TOTAL:
                raise ValueError('Complete Train cache required')
            baseline_path = project / 'result/gnn_train_gpu_v1/config.json'
            frozen_inputs = read(baseline_path)['input_hashes']

            def frozen(path):
                path = contained(project, path)
                key = str(path.relative_to(project))
                expected = frozen_inputs.get(key)
                if expected is None or sha(path) != expected:
                    raise ValueError('Audit input not covered by unchanged Train freeze: ' + key)
                return expected

            identity = dict(source_sha256=sha(Path(__file__)),
                baseline_config_sha256=sha(baseline_path),
                cache_manifest_sha256=sha(cache / 'manifest.json'),
                sidecar_manifest_sha256=sha(sidecar / 'manifest.json'))
            final = output / 'feature_audit.json'
            if final.exists():
                saved = read(final)
                if saved.get('identity') != identity or saved.get('status') != 'passed':
                    raise ValueError('Existing audit identity differs')
                update(status='completed', reused=True)
                return saved
            ids = np.arange(101, TOTAL, 203, dtype=np.int64)
            values = np.full((len(ids), len(R_FIELDS)+len(EXTRA_FIELDS)), np.nan, dtype=np.float64)
            old_seen, extra_seen = np.zeros(len(ids), bool), np.zeros(len(ids), bool)
            old_inputs = {}
            for day in range(1, 11):
                folder = project / f'result/gnn_train_v1/transaction_date=2022-09-{day:02d}'
                frozen(folder / 'merge_phase.json')
                merged = read(folder / 'merge_phase.json')
                old_inputs[str((folder / 'merge_phase.json').relative_to(project))] = sha(folder / 'merge_phase.json')
                for part in merged['parts']:
                    path = contained(project, part['path'])
                    frozen(path)
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=['edge_id']+[f['name'] for f in R_FIELDS]):
                        tab = pa.Table.from_batches([batch])
                        edges = tab['edge_id'].to_numpy()
                        keep = (edges % 203) == 101
                        if not keep.any():
                            continue
                        edges = edges[keep]
                        positions = np.searchsorted(ids, edges)
                        if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], edges) or old_seen[positions].any():
                            raise ValueError('Duplicate/out-of-scope Train audit sample')
                        selected = tab.filter(pa.array(keep))
                        for i, field in enumerate(R_FIELDS):
                            x = selected[field['name']].cast(pa.float64()).to_numpy().copy()
                            if field['transform'] == 'log1p':
                                x = np.log1p(x)
                            values[positions, i] = x
                        old_seen[positions] = True
                update(stage='sample_train_R', dates_done=day, rows=int(old_seen.sum()))
            store = SidecarStore(sidecar, cache, expected_split='train')
            for index, entry in enumerate(root['blocks']):
                block = entry['block']
                plans = _plans(contained(cache, cache / block))
                targets = np.concatenate([p['targets'] for p in plans])
                times = np.concatenate([np.full(len(p['targets']), p['cutoff'], dtype=np.int64) for p in plans])
                keep = (targets % 203) == 101
                if keep.any():
                    targets, times = targets[keep], times[keep]
                    positions = np.searchsorted(ids, targets)
                    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], targets) or extra_seen[positions].any():
                        raise ValueError('Duplicate sidecar audit sample')
                    extra = store.block(block, targets, times)
                    x = extra[:, :len(EXTRA_FIELDS)].astype(np.float64)
                    x[extra[:, len(EXTRA_FIELDS):2*len(EXTRA_FIELDS)] == 0] = np.nan
                    values[positions, len(R_FIELDS):] = x
                    extra_seen[positions] = True
                if index % 25 == 0 or index+1 == len(root['blocks']):
                    update(stage='sample_extra_R', blocks=index+1, rows=int(extra_seen.sum()))
            if not old_seen.all() or not extra_seen.all():
                raise ValueError('Incomplete Train audit sample')
            node_values, seen_keys, node_inputs = [], set(), []
            for day in range(1, 11):
                candidates = [b for b in root['blocks'] if b['block'].startswith(f'transaction_date=2022-09-{day:02d}/')]
                for at in sorted(set([0, len(candidates)//2, len(candidates)-1])):
                    block = candidates[at]['block']
                    folder = contained(cache, cache / block)
                    frozen(folder / 'manifest.json')
                    manifest = read(folder / 'manifest.json')
                    path = folder / 'nodes.parquet'
                    if sha(path) != manifest['file_sha256']['nodes.parquet']:
                        raise ValueError('Frozen sampled N file changed')
                    pf = pq.ParquetFile(path)
                    step = max(1, (pf.metadata.num_rows+2999)//3000)
                    offset = 0
                    for batch in pf.iter_batches(batch_size=8192, columns=['node_id', 'cutoff_us']+ENCODED_NAMES):
                        tab = pa.Table.from_batches([batch])
                        take = ((np.arange(len(tab))+offset) % step) == 0
                        offset += len(tab)
                        selected = tab.filter(pa.array(take))
                        unique = []
                        for node, cutoff in zip(selected['node_id'].to_numpy(), selected['cutoff_us'].to_numpy()):
                            key = (int(node), int(cutoff))
                            unique.append(key not in seen_keys)
                            seen_keys.add(key)
                        if any(unique):
                            node_values.append(decode_node_batch(selected.filter(pa.array(unique))))
                    node_inputs.append(dict(block=block, sha256=manifest['file_sha256']['nodes.parquet']))
                update(stage='sample_train_N', dates_done=day, node_states=len(seen_keys))
            nodes = np.concatenate(node_values)
            relationship = check_ach_relationship(nodes)
            update(stage='correlations', target_rows=len(ids), node_states=len(nodes))
            report = dict(status='passed', identity=identity, scope='Train only; no labels used',
                target_sampling='edge_id % 203 == 101; deterministic label-independent sample across all ten days',
                node_sampling='First/middle/last cache block per day; up to 3000 evenly spaced unique (node,cutoff) states per block; not uniform accounts',
                inputs=dict(merge_manifests=old_inputs, sampled_nodes=node_inputs),
                exact_ach_relationship=relationship,
                structural_review=[
                    'N ACH count, all count and ACH share have a deterministic relationship.',
                    'R ACH amount shares are currency-conditional extra information, not exact duplicates of count shares.',
                    'Current-payment share is generic; for an ACH target its scope may overlap a node ACH share.',
                    'Overlapping time windows are intentional; high correlation alone does not justify deleting them.',
                    'Only the six approved ACH-specific fields are removed in B/C; this audit cannot expand that list.'
                ],
                target_features=summarize(values, [f['name'] for f in R_FIELDS]+[f['name'] for f in EXTRA_FIELDS]),
                node_features=summarize(nodes, [f['name'] for f in N_FIELDS]))
            atomic_json(final, report)
            update(status='completed', report=str(final))
            return report
        except SystemExit:
            raise
        except BaseException as exc:
            update(status='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=ROOT)
    run(parser.parse_args().project)
