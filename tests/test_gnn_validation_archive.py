import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gnn_features.build import build, load_archive
from gnn_features.contract import HOUR
from gnn_features.engine import FeatureEngine
from gnn_features.loader import EdgeStore, TemporalBatchBuilder
from gnn_features.validation_archive import build_validation_archive, DATES, file_sha
import gnn_features.validation_archive as va


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_transactions(root, date, rows):
    schema = pa.schema([('transaction_id', pa.string()), ('timestamp', pa.timestamp('us')),
                        ('from_bank', pa.string()), ('from_account', pa.string()),
                        ('to_bank', pa.string()), ('to_account', pa.string()),
                        ('amount_paid', pa.decimal128(38, 6)), ('amount_received', pa.decimal128(38, 6)),
                        ('payment_currency', pa.string()), ('receiving_currency', pa.string()),
                        ('payment_format', pa.string()), ('is_laundering', pa.int8())])
    payload = []
    for i, (time, src, dst, paid_curr, received_curr, amount, label) in enumerate(rows):
        payload.append(dict(transaction_id=date + '-' + str(i), timestamp=np.datetime64(date + 'T' + time, 'us').item(),
                            from_bank=src[0], from_account=src[1], to_bank=dst[0], to_account=dst[1],
                            amount_paid=Decimal(str(amount)), amount_received=Decimal(str(amount)),
                            payment_currency=paid_curr, receiving_currency=received_curr,
                            payment_format='ACH' if date == '2022-09-10' else 'NovelRail', is_laundering=label))
    folder = root / f'transaction_date={date}'
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payload, schema=schema), folder / 'part.parquet')


def fixture(project):
    root = project / 'data/silver/transactions'
    root.mkdir(parents=True)
    (root / '_manifest.json').write_text('{}', encoding='utf-8')
    split = dict(train_members=['train'], splits=[
        dict(name='train', dates=[f'2022-09-{d:02d}' for d in range(1, 11)], rows=2, positive_rows=0),
        dict(name='validation', dates=list(DATES), rows=7, positive_rows=2),
        dict(name='test', dates=['2022-09-14'], rows=100, positive_rows=100)])
    save_json(project / 'config/dataset_split.json', split)
    acc = project / 'data/silver/accounts'
    acc.mkdir(parents=True)
    pq.write_table(pa.table({'bank_id': ['A', 'B'], 'account_number': ['a', 'b']}), acc / 'accounts.parquet')
    write_transactions(root, '2022-09-10', [('22:00', ('A', 'a'), ('B', 'b'), 'USD', 'USD', 10, 0),
                                            ('23:00', ('B', 'b'), ('A', 'a'), 'USD', 'USD', 20, 0)])
    write_transactions(root, '2022-09-11', [('01:00', ('C', 'c'), ('A', 'a'), 'X', 'X', 100, 1),
                                            ('01:00', ('D', 'd'), ('A', 'a'), 'Y', 'Y', 200, 0),
                                            ('02:00', ('A', 'a'), ('B', 'b'), 'X', 'Y', 10, 0),
                                            ('02:00', ('B', 'b'), ('A', 'a'), 'X', 'X', 999, 0),
                                            ('03:00', ('A', 'a'), ('A', 'a'), 'X', 'X', 1, 0)])
    write_transactions(root, '2022-09-12', [('02:00', ('A', 'a'), ('D', 'd'), 'Y', 'X', 20, 1)])
    write_transactions(root, '2022-09-13', [('02:00', ('A', 'a'), ('C', 'c'), 'X', 'X', 10, 0)])
    # If any Test contents are opened, preparation fails immediately.
    test = root / 'transaction_date=2022-09-14'
    test.mkdir()
    (test / 'must-never-read.parquet').write_bytes(b'not a parquet file')
    train = project / 'data/gnn_v1_train'
    build(project, train, '2022-09-11T00:00:00')
    return train


def test_frozen_prefix_unknown_identities_and_strict_history(tmp_path):
    train = fixture(tmp_path)
    before = {p.relative_to(train): file_sha(p) for p in train.rglob('*') if p.is_file()}
    output = tmp_path / 'data/gnn_v1_validation'
    result = build_validation_archive(tmp_path, output)
    assert result['rows'] == 9 and result['validation_rows'] == 7 and result['validation_positives'] == 2
    assert result['node_count'] == 4
    assert {p.relative_to(train): file_sha(p) for p in train.rglob('*') if p.is_file()} == before
    assert file_sha(train / 'metadata/vocabulary.json') == file_sha(output / 'metadata/vocabulary.json')
    assert file_sha(train / 'graph/edges/transaction_date=2022-09-10/part.parquet') == file_sha(output / 'graph/edges/transaction_date=2022-09-10/part.parquet')
    _, original, original_bank, _ = load_archive(train)
    meta, arrays, bank, vocab = load_archive(output)
    for name in va.COLS:
        np.testing.assert_array_equal(arrays[name][:2], original[name])
    np.testing.assert_array_equal(bank[:2], original_bank)
    np.testing.assert_array_equal(np.load(output / 'history_index/bank_cat.npy'), [2, 3, 1, 1])
    edges = pq.ParquetFile(output / 'graph/edges/transaction_date=2022-09-11/part.parquet').read()
    assert 'is_laundering' not in edges.column_names
    assert edges['payment_currency_cat'].to_pylist() == [1] * 5
    assert edges['receiving_currency_cat'].to_pylist() == [1] * 5
    assert edges['is_cross_currency'].to_pylist() == [False, False, True, False, False]
    assert arrays['payment_currency_cat'][2] != arrays['payment_currency_cat'][3]
    engine = FeatureEngine(arrays, bank, vocab['payment_format'].index('ACH') + 2,
                           meta['data_start_us'], int(np.datetime64(meta['end_exclusive'], 'us').astype(np.int64)))
    # X inflows stay separate from Y; current-time incoming 999 must be excluded.
    values, _ = engine.target(4)
    assert values['r_src_ext_in_sum_24h'] == Decimal('100')
    assert values['r_src_current_currency_share_24h'] == 0
    adj = {d + '_' + s: np.load(output / f'graph/adj_index/{d}_{s}.npy') for d in ['in', 'out'] for s in ['ptr', 'edge_ids']}
    builder = TemporalBatchBuilder(engine, adj, EdgeStore(output), np.load(output / 'history_index/bank_cat.npy'))
    nodes, history = builder.sample(np.array([4, 5]))
    assert set(history) == {0, 1, 2, 3}
    assert np.all(arrays['timestamp'][history] < arrays['timestamp'][4])
    # Day 12 may use earlier Validation observations, but never a future edge.
    _, day_two_history = builder.sample(np.array([7]))
    assert 4 in day_two_history and 6 not in day_two_history and 8 not in day_two_history
    assert build_validation_archive(tmp_path, output) == result
    reused_status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    assert reused_status['status'] == 'completed'
    assert reused_status['rows'] == 9 and reused_status['validation_rows'] == 7


def test_resume_keeps_committed_append_files(tmp_path):
    fixture(tmp_path)
    output = tmp_path / 'data/gnn_v1_validation'
    def interrupt(value):
        if value.get('stage') == 'appending_validation' and value.get('completed_files') == 1:
            raise RuntimeError('simulate interruption')
    with pytest.raises(RuntimeError, match='simulate interruption'):
        build_validation_archive(tmp_path, output, progress=interrupt)
    assert not (output / 'build.json').exists()
    committed = output / 'graph/edges/transaction_date=2022-09-11/part.parquet'
    old_hash, old_time = file_sha(committed), committed.stat().st_mtime_ns
    result = build_validation_archive(tmp_path, output)
    assert result['rows'] == 9
    assert file_sha(committed) == old_hash and committed.stat().st_mtime_ns == old_time


def test_freeze_rejects_source_and_committed_output_changes(tmp_path):
    fixture(tmp_path)
    output = tmp_path / 'data/gnn_v1_validation'
    build_validation_archive(tmp_path, output)
    path = output / 'history_index/payment_currency_cat.npy'
    with path.open('r+b') as stream:
        stream.seek(-1, 2); stream.write(b'\x7f')
    with pytest.raises(ValueError, match='Verified stage output changed'):
        build_validation_archive(tmp_path, output)
    source = tmp_path / 'data/silver/transactions/transaction_date=2022-09-11/part.parquet'
    table = pq.ParquetFile(source).read()
    # Same row count, but changed source bytes must fail the frozen hash check.
    pq.write_table(table, source, compression='gzip')
    with pytest.raises(ValueError, match='configuration/input/code mismatch'):
        build_validation_archive(tmp_path, output)


def test_disk_floor_and_output_scope(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(va.shutil, 'disk_usage', lambda _: type('Disk', (), {'free': va.MIN_FREE - 1})())
    output = tmp_path / 'data/gnn_v1_validation'
    with pytest.raises(RuntimeError, match='20 GiB'):
        build_validation_archive(tmp_path, output)
    assert not (output / 'build.json').exists()
    with pytest.raises(ValueError, match='separate from Train'):
        build_validation_archive(tmp_path, tmp_path / 'data/gnn_v1_train')


@pytest.mark.parametrize('chunk_lengths', [[100_000], [16_384, 17_000, 31_616, 35_000]])
def test_large_sparse_unknown_encoding_preserves_exact_rows(chunk_lengths):
    # Arrow 21 Windows fill_null previously misplaced sparse null replacements
    # on a 100k row array. Test both sparse UNK positions and chunk boundaries.
    raw = np.full(100_000, 'A', dtype=object)
    known_b = [0, 1, 16_383, 16_385, 64_999, 99_998]
    unknown = [16_384, 33_383, 65_000, 77_777, 99_999]
    raw[known_b] = 'B'
    raw[unknown] = ['UNKNOWN_1', 'UNKNOWN_2', 'UNKNOWN_1', 'UNKNOWN_3', 'UNKNOWN_2']
    chunks, start = [], 0
    for size in chunk_lengths:
        chunks.append(pa.array(raw[start:start + size]))
        start += size
    column = pa.chunked_array(chunks)
    actual = va._codes(column, ['A', 'B'], unknown=True)
    expected = np.full(100_000, 2, dtype=np.int16)
    expected[known_b] = 3
    expected[unknown] = 1
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='Unregistered raw identity'):
        va._codes(column, ['A', 'B'], unknown=False)
    internal = va._codes(column, ['A', 'B', 'UNKNOWN_1', 'UNKNOWN_2', 'UNKNOWN_3'])
    expected[unknown] = [4, 5, 4, 6, 5]
    np.testing.assert_array_equal(internal, expected)
