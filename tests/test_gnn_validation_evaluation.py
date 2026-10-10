import math

import numpy as np
import pyarrow as pa
import pytest

from gnn_features.cache_train import array_sha, sha
from gnn_features.evaluate_validation import prediction_table, verify_predictions
from gnn_features.training_state import atomic_json
from gnn_features.tune_pipeline import candidates, choose_best
from gnn_features.validation_cache import TOTAL, POSITIVES, discard_unverified, publish_verified


def test_predictions_reject_changed_identity_and_nonfinite():
    ids, times, labels = np.array([7, 8]), np.array([100, 100]), np.array([0, 1])
    table = prediction_table(ids, times, labels, [.01, .8])
    np.testing.assert_allclose(verify_predictions(table, ids, times, labels), [.01, .8])
    with pytest.raises(ValueError, match='identity'):
        verify_predictions(table, ids[::-1], times, labels)
    with pytest.raises(ValueError, match='identity'):
        verify_predictions(table, ids, times + 1, labels)
    with pytest.raises(ValueError):
        prediction_table(ids, times, labels, [float('nan'), .8])
    with pytest.raises(ValueError):
        prediction_table([7, 7], times, labels, [.1, .8])
    with pytest.raises(ValueError):
        prediction_table(ids, times, [0, 2], [.1, .8])


def test_parquet_readback_preserves_probability_and_label(tmp_path):
    import pyarrow.parquet as pq
    table = prediction_table([9, 11], [50, 50], [1, 0], [0.00001, 1.0])
    path = tmp_path / 'part.parquet'
    pq.write_table(table, path, compression='zstd')
    verify_predictions(pq.ParquetFile(path).read(), [9, 11], [50, 50], [1, 0])


def make_work(root, block):
    work = root / '_work/active'
    work.mkdir(parents=True)
    (work / 'nodes.parquet').write_bytes(b'fixture-node')
    (work / 'plans.parquet').write_bytes(b'fixture-plan')
    atomic_json(work / 'manifest.json', dict(status='verified', config_sha256='config',
                target_ids_sha256=array_sha([9, 10]), targets=2, block=block,
                file_sha256={n: sha(work / n) for n in ('nodes.parquet', 'plans.parquet')}))
    return work


def test_verified_scratch_never_deleted_and_checked_publication(tmp_path):
    block = 'transaction_date=2022-09-11/hour=00/block=0000'
    work = make_work(tmp_path, block)
    with pytest.raises(ValueError, match='preserve'):
        discard_unverified(work, tmp_path)
    with pytest.raises(ValueError):
        publish_verified(work, tmp_path / block, tmp_path, 'changed', [9, 10])
    assert work.exists()
    publish_verified(work, tmp_path / block, tmp_path, 'config', [9, 10])
    assert not work.exists() and (tmp_path / block / 'manifest.json').exists()


def test_verified_scratch_wrong_block_is_preserved(tmp_path):
    work = make_work(tmp_path, 'transaction_date=2022-09-11/hour=00/block=0000')
    with pytest.raises(ValueError, match='different block'):
        publish_verified(work, tmp_path / 'transaction_date=2022-09-11/hour=00/block=0001', tmp_path, 'config', [9, 10])
    assert work.exists()


def test_declared_budget_and_ap_selection():
    configs = candidates(1140.9013150500168)
    assert len(configs) == 2
    assert configs[0]['positive_weight'] == 1140.9013150500168
    assert configs[1]['positive_weight'] == math.sqrt(configs[0]['positive_weight'])
    rows = [dict(name=n, metrics=dict(samples=TOTAL, positives=POSITIVES, average_precision=ap, roc_auc=roc))
            for n, ap, roc in [('baseline', .2, .99), ('lr_only', .3, .9), ('sqrt_weight', .25, .999)]]
    assert choose_best(rows)['name'] == 'lr_only'
    with pytest.raises(ValueError):
        choose_best(rows[:2])
    rows[1]['metrics']['samples'] -= 1
    with pytest.raises(ValueError):
        choose_best(rows)


def test_validation_cached_batch_matches_direct_with_unknown_currency(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from test_gnn_validation_archive import fixture
    from gnn_features.validation_archive import build_validation_archive
    from gnn_features.validation_cache import init_validation_worker
    from gnn_features.build import load_archive
    from gnn_features.cache_train import compute_block
    from gnn_features.contract import HOUR, R_FIELDS, encode_numeric
    from gnn_features.engine import FeatureEngine
    from gnn_features.loader import TemporalBatchBuilder, EdgeStore
    from gnn_features.model_cache import ParquetBlockCache
    fixture(tmp_path)
    archive = tmp_path / 'data/gnn_v1_validation'
    build_validation_archive(tmp_path, archive)
    meta, a, bank, vocab = load_archive(archive)
    engine = FeatureEngine(a, bank, vocab['payment_format'].index('ACH') + 2,
                           meta['data_start_us'], int(np.datetime64(meta['end_exclusive'], 'us').astype(np.int64)))
    adj = {d + '_' + s: np.load(archive / f'graph/adj_index/{d}_{s}.npy')
           for d in ['in', 'out'] for s in ['ptr', 'edge_ids']}
    builder = TemporalBatchBuilder(engine, adj, EdgeStore(archive), np.load(archive / 'history_index/bank_cat.npy'))
    ids = np.array([4, 5], dtype=np.int64)
    t = int(a['timestamp'][ids[0]])
    r = []
    for edge in ids:
        values, _ = engine.target(int(edge))
        context = engine.context()
        r.append(encode_numeric(values, R_FIELDS) + [float(context['full_' + w]) for w in ['6h', 'day']])
    r = np.asarray(r, dtype=np.float32)
    labels = np.array([0, 0], dtype=np.float32)
    direct, _, _ = builder.build(ids, r, labels)
    builder.clear_cache()
    work = tmp_path / 'cache_block'
    work.mkdir()
    init_validation_worker(str(archive))
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = compute_block(work, [ids], (t // HOUR + 1) * HOUR, a, bank, meta, vocab,
                                 builder, pool, lambda **kwargs: None)
    manifest.update(status='verified', config_sha256='fixture', target_ids_sha256=array_sha(ids))
    atomic_json(work / 'manifest.json', manifest)
    loader = SimpleNamespace(builder=builder, day=SimpleNamespace(r=r, labels=labels,
                             positions=lambda targets: np.searchsorted(ids, targets)))
    cached = ParquetBlockCache(work, loader, 'fixture')
    actual, actual_y, _ = cached.batch(0)
    assert set(actual) == set(direct)
    for key in actual:
        np.testing.assert_allclose(actual[key], direct[key], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual_y, labels)
    assert np.all(a['timestamp'][actual['edge_id']] < t)
