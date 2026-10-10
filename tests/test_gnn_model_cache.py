import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_gnn_loader import setup_builder
from gnn_features.cache_train import PLAN_SCHEMA, array_sha, chunks
from gnn_features.contract import N_FIELDS, encode_numeric
from gnn_features.hybrid_cache_trial import ENCODED_NAMES
from gnn_features.loader import RDIM, TargetDay
import gnn_features.model_cache as mc


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def make_block(folder, builder, batches, config_hash='frozen', rel='fixture'):
    folder.mkdir(parents=True, exist_ok=True)
    rows, values, originals = [], {}, []
    for i, ids in enumerate(batches):
        r = np.zeros((len(ids), RDIM), dtype=np.float32)
        y = (np.asarray(ids) % 2).astype(np.float32)
        data, _, _ = builder.build(ids, r, y)
        originals.append(data)
        cutoff = int(builder.a['timestamp'][ids[0]])
        rows.append(dict(batch_id=i, cutoff_us=cutoff, targets=list(map(int, ids)), nodes=data['node_id'].tolist(), edges=data['edge_id'].tolist()))
        for node, features in zip(data['node_id'], data['x_num']):
            values[int(node), cutoff] = features.copy()
    keys = np.array(sorted(values), dtype=np.int64)
    matrix = np.stack([values[tuple(key)] for key in keys])
    pq.write_table(pa.table(dict(node_id=keys[:, 0], cutoff_us=keys[:, 1], **{name: matrix[:, i] for i, name in enumerate(ENCODED_NAMES)})), folder / 'nodes.parquet')
    pq.write_table(pa.Table.from_pylist(rows, schema=PLAN_SCHEMA), folder / 'plans.parquet')
    targets = np.concatenate(batches)
    manifest = dict(status='verified', config_sha256=config_hash, target_ids_sha256=array_sha(targets), targets=len(targets),
                    batches=len(rows), node_states=len(keys), block=rel, first_cutoff_us=int(keys[:, 1].min()), last_cutoff_us=int(keys[:, 1].max()),
                    file_sha256={name: mc.file_sha(folder / name) for name in ('nodes.parquet', 'plans.parquet')},
                    parquet_bytes=sum((folder / name).stat().st_size for name in ('nodes.parquet', 'plans.parquet')))
    save(folder / 'manifest.json', manifest)
    return manifest, originals


def loader_for(builder, ids):
    day = TargetDay.__new__(TargetDay)
    day.ids = np.array(ids, dtype=np.int64)
    day.times = builder.a['timestamp'][day.ids]
    day.order = np.lexsort((day.ids, day.times))
    day.r = np.zeros((len(ids), RDIM), dtype=np.float32)
    day.labels = (day.ids % 2).astype(np.float32)
    return SimpleNamespace(builder=builder, day=day)


def rehash(folder, name):
    manifest = mc.read_json(folder / 'manifest.json')
    manifest['file_sha256'][name] = mc.file_sha(folder / name)
    manifest['parquet_bytes'] = sum((folder / file).stat().st_size for file in manifest['file_sha256'])
    save(folder / 'manifest.json', manifest)


def test_replay_exact_tensors_and_separate_labels_without_recomputation(tmp_path, monkeypatch):
    engine, builder, _ = setup_builder(tmp_path)
    folder = tmp_path / 'transaction_date=2022-09-04/hour=10/block=0000'
    _, expected = make_block(folder, builder, [np.array([4, 5]), np.array([6])])
    loader = loader_for(builder, [4, 5, 6])
    monkeypatch.setattr(builder, 'sample', lambda *args: pytest.fail('Must replay frozen graph'))
    monkeypatch.setattr(engine, 'node', lambda *args: pytest.fail('Must load cached node state'))
    monkeypatch.setattr(engine, 'target', lambda *args: pytest.fail('Must load persisted target R'))
    cached = mc.ParquetBlockCache(folder, loader, 'frozen')
    assert not builder.node_cache
    assert len(cached) == 2
    np.testing.assert_array_equal(cached.targets, [4, 5, 6])
    for i, reference in enumerate(expected):
        actual, labels, timing = cached.batch(i)
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key])
        np.testing.assert_array_equal(labels, np.array([4, 5] if i == 0 else [6]) % 2)
        assert timing['targets'] == len(labels)
        assert 'label' not in actual
    # Same account at 10:00 and 10:30 must address different cached states.
    assert not np.array_equal(cached.node_lookup(0, cached.plans[0]['cutoff']), cached.node_lookup(0, cached.plans[1]['cutoff']))
    with pytest.raises(ValueError, match='outside'):
        cached.node_lookup(0, cached.origin - 1)
    with pytest.raises(ValueError, match='Missing'):
        cached.node_lookup(0, cached.origin + 1)
    with pytest.raises(IndexError):
        cached.batch(-1)


def test_corrupt_checksum_and_current_time_history_rejected(tmp_path):
    _, builder, _ = setup_builder(tmp_path)
    folder = tmp_path / 'block'
    make_block(folder, builder, [np.array([4, 5])])
    loader = loader_for(builder, [4, 5])
    path = folder / 'plans.parquet'
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[0]['edges'].append(4)  # A current target is never a historical neighbor.
    pq.write_table(pa.Table.from_pylist(rows, schema=PLAN_SCHEMA), path)
    with pytest.raises(ValueError, match='checksum'):
        mc.ParquetBlockCache(folder, loader, 'frozen')
    rehash(folder, 'plans.parquet')
    with pytest.raises(ValueError, match='current/future'):
        mc.ParquetBlockCache(folder, loader, 'frozen')


def test_node_key_and_nonfinite_payload_rejected(tmp_path):
    _, builder, _ = setup_builder(tmp_path)
    folder = tmp_path / 'block'
    make_block(folder, builder, [np.array([4, 5])])
    path = folder / 'nodes.parquet'
    table = pq.ParquetFile(path).read()
    column = table[ENCODED_NAMES[0]].to_numpy().copy()
    column[0] = np.nan
    table = table.set_column(table.schema.get_field_index(ENCODED_NAMES[0]), ENCODED_NAMES[0], pa.array(column))
    pq.write_table(table, path)
    rehash(folder, 'nodes.parquet')
    with pytest.raises(ValueError, match='nonfinite'):
        mc.ParquetBlockCache(folder, loader_for(builder, [4, 5]), 'frozen')


def synthetic_full_cache(tmp_path, monkeypatch):
    fixture_root = tmp_path / 'fixture'
    engine, builder, _ = setup_builder(fixture_root)
    project = tmp_path / 'project'
    archive, features, cache = project / 'archive', project / 'features', project / 'cache'
    archive.mkdir(parents=True)
    dates = ['2022-09-01', '2022-09-04']
    monkeypatch.setattr(mc, 'ROOT', project)
    monkeypatch.setattr(mc, '__file__', str(project / 'scripts/gnn_features/model_cache.py'))
    monkeypatch.setattr(mc, 'DATES', dates)
    monkeypatch.setattr(mc, 'EXPECTED_TARGETS', 7)
    monkeypatch.setattr(mc, 'EXPECTED_POSITIVES', 3)
    meta = dict(rows=7, end_exclusive='2022-09-11T00:00:00')
    save(archive / 'build.json', meta)
    monkeypatch.setattr(mc, 'load_archive', lambda *args: (meta, engine.arr, engine.bank, {}))
    save(project / 'config/dataset_split.json', dict(splits=[dict(name='discovery', dates=dates)]))
    paths = [project / 'config/dataset_split.json', archive / 'build.json']
    for name in ('cache_train.py', 'hybrid_cache_trial.py', 'build.py', 'engine.py', 'rolling.py', 'contract.py', 'loader.py'):
        path = project / 'scripts/gnn_features' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen test fixture', encoding='utf-8')
        paths.append(path)
    for date, ids in zip(dates, [np.array([0]), np.arange(1, 7)]):
        folder = features / f'transaction_date={date}'
        folder.mkdir(parents=True)
        np.save(folder / 'target_edge_ids.npy', ids)
        save(folder / 'merge_phase.json', dict(parts=[]))
        save(folder / 'verification.json', dict(status='passed', full_day_rows=len(ids)))
        save(folder / 'run_config.json', dict(archive_manifest_sha256=mc.file_sha(archive / 'build.json')))
        save(folder / 'daily_run.json', dict(status='verified'))
        paths.extend(folder / name for name in ('target_edge_ids.npy', 'merge_phase.json', 'verification.json', 'run_config.json'))
        labels = pa.table(dict(edge_id=ids, transaction_id=[f'tx{i}' for i in ids], label=ids % 2, split=['discovery'] * len(ids)))
        edges = pa.table(dict(edge_id=ids, transaction_id=[f'tx{i}' for i in ids], timestamp=pa.array(engine.arr['timestamp'][ids], type=pa.timestamp('us'))))
        for kind, table in [('supervision/targets', labels), ('graph/edges', edges)]:
            out = archive / kind / f'transaction_date={date}'
            out.mkdir(parents=True)
            pq.write_table(table, out / 'part.parquet')
    config = dict(dates=dates, expected_targets=7, batch_size=64, max_batches_per_block=512, seed=20260920, fanout=[10, 5],
                  end_exclusive=meta['end_exclusive'], hashes={str(path.relative_to(project)): mc.file_sha(path) for path in paths})
    save(cache / 'config.json', config)
    config_hash = mc.file_sha(cache / 'config.json')
    summaries = []
    for date, ids in zip(dates, [np.array([0]), np.arange(1, 7)]):
        day = loader_for(builder, ids).day
        day_blocks = []
        for hour, index, batches in chunks(day):
            rel = f'transaction_date={date}/hour={hour % 24:02d}/block={index:04d}'
            manifest, _ = make_block(cache / rel, builder, batches, config_hash, rel)
            summary = {key: manifest[key] for key in ('block', 'targets', 'node_states', 'parquet_bytes')}
            summaries.append(summary)
            day_blocks.append(summary)
        save(cache / f'transaction_date={date}/manifest.json', dict(status='verified', targets=len(ids), blocks=day_blocks))
    save(cache / 'manifest.json', dict(status='completed', targets=7, blocks=summaries, config_sha256=config_hash,
                                      feature_tables=str(features), edge_and_identity_tables=str(archive), node_columns=ENCODED_NAMES))
    save(cache / 'status.json', dict(status='completed', verified_targets=7, completed_blocks=len(summaries), dates=dates,
                                    parquet_bytes=sum(block['parquet_bytes'] for block in summaries)))
    return cache, archive, features


def test_full_verification_exact_coverage_labels_and_frozen_inputs(tmp_path, monkeypatch):
    cache, archive, features = synthetic_full_cache(tmp_path, monkeypatch)
    progress = []
    result = mc.verify_cache(cache, archive, features, lambda **kwargs: progress.append(kwargs))
    assert result['targets'] == 7 and result['positives'] == 3 and result['negatives'] == 4
    assert sum(block['positives'] for block in result['blocks']) == 3
    assert result['active_work_files'] == 0
    assert progress[-1]['stage'] == 'cache_verified'
    frozen_path = mc.ROOT / 'scripts/gnn_features/engine.py'
    frozen_path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='hash changed'):
        mc.verify_cache(cache, archive, features)


def test_full_verification_rejects_label_identity_and_preserves_active(tmp_path, monkeypatch):
    cache, archive, features = synthetic_full_cache(tmp_path, monkeypatch)
    pending = cache / '_work/active/nodes.parquet'
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='active work'):
        mc.verify_cache(cache, archive, features)
    assert pending.read_bytes() == b'preserve'
    pending.unlink()
    path = archive / 'supervision/targets/transaction_date=2022-09-04/part.parquet'
    table = pq.ParquetFile(path).read()
    table = table.set_column(1, 'transaction_id', pa.array(['mismatched'] * len(table)))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match='transaction identity'):
        mc.verify_cache(cache, archive, features)
