"""Append Test history without refitting Train vocabularies or IDs.

The existing query engines use ``*_cat`` in history_index as equality keys.
For unseen raw currencies/payment formats these internal keys are extended;
only persisted E embedding columns use the frozen vocabulary's UNK=1.
No test labels enter graph/history feature inputs.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gnn_features.build import COLS, edge_encoding, scaled
from gnn_features.contract import DAY, VERSION

DATES = ('2022-09-14', '2022-09-15', '2022-09-16')
END = '2022-09-17T00:00:00'
TRAIN_END = '2022-09-14T00:00:00'
MIN_FREE = 20 * 1024**3
BATCH_SIZE = 100_000


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('Symlink output is not supported')
    write(temporary)
    os.replace(temporary, path)


def _json(path, value):
    _atomic(path, lambda p: p.write_text(json.dumps(value, ensure_ascii=True, indent=2,
                                                    allow_nan=False), encoding='utf-8'))


def _save_array(path, values):
    def write(p):
        with p.open('wb') as stream:
            np.save(stream, values)
    _atomic(path, write)


def _space(output, extra=0):
    if shutil.disk_usage(output).free < MIN_FREE + extra:
        raise RuntimeError('Insufficient disk space; keep at least 20 GiB free')


@contextmanager
def _lock(output):
    import msvcrt
    with (output / 'process.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Test archive is already being prepared') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _codes(column, values, unknown=False):
    if column.null_count:
        raise ValueError('Required identity is missing')
    result = pc.index_in(column, value_set=pa.array(values, type=column.type))
    if result.null_count and not unknown:
        raise ValueError('Unregistered raw identity')
    # Extended internal identities are deliberately not embedding indices.
    # On the installed Arrow 21 Windows runtime, fill_null can write a sparse
    # replacement to the wrong row of a large numeric array. Select explicitly
    # from the validity bitmap instead; never cast a null index to an integer.
    valid = result.is_valid().to_numpy()
    arr = np.ones(len(result), dtype=np.int64)
    arr[valid] = result.to_numpy()[valid].astype(np.int64) + 2
    if np.any(arr > np.iinfo(np.int16).max):
        raise ValueError('Internal categorical identity exceeds int16 capacity')
    return arr.astype(np.int16)


def _account_index(table):
    frame = table.select(['bank_id', 'account_number']).to_pandas()
    index = pd.MultiIndex.from_frame(frame)
    if not index.is_unique:
        raise ValueError('Duplicate account identity')
    return index


def _nodes(table, index, side):
    bank, account = ('from_bank', 'from_account') if side == 'src' else ('to_bank', 'to_account')
    if table[bank].null_count or table[account].null_count:
        raise ValueError('Account identity is missing')
    keys = pd.MultiIndex.from_arrays([table[bank].to_pandas(), table[account].to_pandas()])
    return index.get_indexer(keys), keys


def _source_files(project, train, split):
    selected = [s for s in split['splits'] if s['name'] == 'test']
    if len(selected) != 1 or tuple(selected[0]['dates']) != DATES:
        raise ValueError('Frozen Test dates differ')
    files = []
    # Do not enumerate tail directories, even to discover files.
    for date in DATES:
        found = sorted((project / 'data/silver/transactions' / f'transaction_date={date}').glob('*.parquet'))
        if not found:
            raise ValueError('Missing Test date ' + date)
        files.extend(found)
    train_dates = {d for s in split['splits'] if s['name'] in [*split['train_members'], 'validation'] for d in s['dates']}
    if max(train_dates) != '2022-09-13':
        raise ValueError('Train boundary differs')
    train_files = [train / 'build.json'] + sorted((train / 'metadata').glob('*.json'))
    train_files += sorted((train / 'history_index').glob('*.npy'))
    train_files += sorted((train / 'graph/adj_index').glob('*.npy'))
    train_files += sorted((train / 'graph/accounts').glob('*.parquet'))
    for branch in ['graph/edges', 'supervision/targets']:
        for date in sorted(train_dates):
            train_files.extend(sorted((train / branch / f'transaction_date={date}').glob('*.parquet')))
    return files, train_files, selected[0]


def build_test_archive(project: Path, output: Path, progress=None):
    """Build/resume a separate Train/Validation-prefix + Test archive, never tail.

    Completion is published only as build.json after frozen-input, per-stage
    output, prefix, label-count and temporal-index checks pass. Call under a
    background process; progress receives compact dicts. Returned metadata has
    the same required keys as build.load_archive's original Train archive.
    """
    project = Path(project).resolve()
    train = project / 'data/gnn_v1_validation'
    output = Path(output).absolute()
    if output != output.resolve() or not output.is_relative_to(project):
        raise ValueError('Test output must be a real directory inside project')
    if output == train or output.is_relative_to(train) or train.is_relative_to(output):
        raise ValueError('Test output must be separate from Train')
    if output == project or output == project / 'data':
        raise ValueError('A dedicated output directory is required')
    if output.exists() and any(output.iterdir()) and not (output / 'config.json').exists():
        if any(p.name not in {'process.lock', 'status.json', 'status.json.partial', 'config.json.partial'} for p in output.iterdir()):
            raise ValueError('Unowned nonempty output directory')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def emit(stage, **kw):
        value = dict(status='preparing', stage=stage, pid=os.getpid(), updated_at=time.time(), **kw)
        _json(output / 'status.json', value)
        if progress:
            progress(value)
    with _lock(output):
        try:
            _space(output)
            split_path = project / 'config/dataset_split.json'
            split = read_json(split_path)
            meta = read_json(train / 'build.json')
            if meta['version'] != VERSION or meta['end_exclusive'] != TRAIN_END:
                raise ValueError('Existing Train archive scope/version differs')
            if meta['split_sha256'] != file_sha(split_path):
                raise ValueError('Train split hash mismatch')
            source_manifest = project / 'data/silver/transactions/_manifest.json'
            if meta['source_manifest_sha256'] != file_sha(source_manifest):
                raise ValueError('Silver source manifest changed since Train')
            files, train_files, test = _source_files(project, train, split)
            rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            if rows != test['rows']:
                raise ValueError('Test row count differs from frozen split')
            expected_train = sum(s['rows'] for s in split['splits'] if s['name'] in [*split['train_members'], 'validation'])
            if meta['rows'] != expected_train:
                raise ValueError('Train archive row count differs from split')
            emit('freezing_inputs', test_rows=rows)
            sources = train_files + files + [split_path, source_manifest]
            input_hashes = {p.relative_to(project).as_posix(): file_sha(p) for p in sources}
            from gnn_features import build as build_module, contract as contract_module
            code = [Path(__file__), Path(build_module.__file__), Path(contract_module.__file__)]
            config = dict(version=1, train=str(train), dates=list(DATES), end_exclusive=END,
                          train_rows=meta['rows'], test_rows=rows, test_positives=test['positive_rows'],
                          input_hashes=input_hashes, code_hashes={p.name: file_sha(p) for p in code},
                          identity_rule='Frozen E vocabularies; append-only raw history identities; stable Train node/edge prefix',
                          history_rule='All preceding observations, strictly t_prime < t; same-time peers excluded')
            cp = output / 'config.json'
            if cp.exists():
                if read_json(cp) != config:
                    raise ValueError('Frozen Test configuration/input/code mismatch')
            else:
                _json(cp, config)
            state_path = output / 'stages.json'
            state = read_json(state_path) if state_path.exists() else {'phases': {}, 'appended_files': []}
            def checked(name):
                item = state['phases'].get(name)
                if item:
                    for relative, digest in item['outputs'].items():
                        if file_sha(output / relative) != digest:
                            raise ValueError('Verified stage output changed: ' + relative)
                return item
            def finish(name, paths, **info):
                state['phases'][name] = dict(outputs={p.relative_to(output).as_posix(): file_sha(p) for p in paths}, **info)
                _json(state_path, state)
            if (output / 'build.json').exists():
                for name in state['phases']:
                    checked(name)
                complete = read_json(output / 'build.json')
                if (complete.get('config_sha256') != file_sha(cp) or complete['rows'] != meta['rows'] + rows
                        or complete.get('stages_sha256') != file_sha(state_path)):
                    raise ValueError('Completed Test archive identity mismatch')
                _json(output / 'status.json', dict(status='completed', pid=os.getpid(), rows=complete['rows'],
                                                  test_rows=rows, updated_at=time.time()))
                return complete

            for branch in ['graph/edges', 'graph/accounts', 'graph/adj_index', 'history_index', 'metadata', 'supervision/targets']:
                (output / branch).mkdir(parents=True, exist_ok=True)
            if not checked('prefix_parquet'):
                emit('copying_train_prefix')
                copy_files = [p for p in train_files if p.suffix == '.parquet' and 'accounts' not in p.parts]
                _space(output, sum(p.stat().st_size for p in copy_files))
                paths = []
                for source in copy_files:
                    _space(output)
                    dest = output / source.relative_to(train)
                    _atomic(dest, lambda temporary, src=source: shutil.copyfile(src, temporary))
                    if file_sha(dest) != input_hashes[source.relative_to(project).as_posix()]:
                        raise ValueError('Train prefix copy mismatch')
                    paths.append(dest)
                finish('prefix_parquet', paths)

            if not checked('identities'):
                emit('discovering_raw_identities')
                accounts = pq.ParquetFile(train / 'graph/accounts/accounts.parquet').read()
                if not np.array_equal(accounts['node_id'].to_numpy(), np.arange(meta['node_count'])):
                    raise ValueError('Train node IDs are not contiguous/stable')
                index = _account_index(accounts)
                vocab = read_json(train / 'metadata/vocabulary.json')['values']
                prefix_vocab = read_json(train / 'metadata/history_identity_vocabulary.json')['values']
                unknown_nodes, raw_currencies, raw_payment = set(), set(prefix_vocab['currency']), set(prefix_vocab['payment_format'])
                columns = ['from_bank', 'from_account', 'to_bank', 'to_account', 'payment_currency', 'receiving_currency', 'payment_format']
                for f in files:
                    for batch in pq.ParquetFile(f).iter_batches(batch_size=BATCH_SIZE, columns=columns):
                        table = pa.Table.from_batches([batch])
                        for side in ['src', 'dst']:
                            positions, keys = _nodes(table, index, side)
                            unknown_nodes.update(keys[positions < 0].tolist())
                        raw_currencies.update(pc.unique(table['payment_currency']).to_pylist())
                        raw_currencies.update(pc.unique(table['receiving_currency']).to_pylist())
                        raw_payment.update(pc.unique(table['payment_format']).to_pylist())
                if None in raw_currencies or None in raw_payment:
                    raise ValueError('Required category is missing')
                identities = dict(currency=prefix_vocab['currency'] + sorted(raw_currencies - set(prefix_vocab['currency'])),
                                  payment_format=prefix_vocab['payment_format'] + sorted(raw_payment - set(prefix_vocab['payment_format'])))
                if any(len(v) + 1 > np.iinfo(np.int16).max for v in identities.values()):
                    raise ValueError('Raw category identity exceeds supported int16 range')
                if unknown_nodes:
                    missing = sorted(unknown_nodes)
                    bank_lookup = {value: i + 2 for i, value in enumerate(vocab['bank'])}
                    extra = pa.table({'bank_id': [x[0] for x in missing], 'account_number': [x[1] for x in missing],
                                      'node_id': pa.array(np.arange(len(accounts), len(accounts) + len(missing)), type=pa.uint32()),
                                      'bank_cat': pa.array([bank_lookup.get(x[0], 1) for x in missing], type=pa.int32())})
                    accounts = pa.concat_tables([accounts, extra.cast(accounts.schema)])
                if len(accounts) > np.iinfo(np.uint32).max:
                    raise ValueError('Too many nodes for archive')
                old_bank = read_json(train / 'metadata/bank_identity.json')
                bank_identity = old_bank + sorted(set(pc.unique(accounts['bank_id']).to_pylist()) - set(old_bank))
                bank_key = pc.index_in(accounts['bank_id'], value_set=pa.array(bank_identity)).to_numpy().astype(np.int32)
                original_bank_cat = np.load(train / 'history_index/bank_cat.npy', mmap_mode='r')
                stored_bank_cat = accounts['bank_cat'].to_numpy()
                if not np.array_equal(stored_bank_cat[:meta['node_count']], original_bank_cat):
                    raise ValueError('Train account table bank category differs from frozen array')
                bank_cat = np.empty(len(accounts), dtype=np.int32)
                bank_cat[:meta['node_count']] = original_bank_cat
                # Extra account rows were encoded with the explicit Python
                # vocabulary lookup above; the frozen Train prefix is copied.
                bank_cat[meta['node_count']:] = stored_bank_cat[meta['node_count']:]
                for name, array in [('bank_key', bank_key), ('bank_cat', bank_cat)]:
                    if not np.array_equal(array[:meta['node_count']], np.load(train / f'history_index/{name}.npy')):
                        raise ValueError('Train bank identity prefix changed')
                    _save_array(output / f'history_index/{name}.npy', array)
                account_path = output / 'graph/accounts/accounts.parquet'
                _atomic(account_path, lambda p: pq.write_table(accounts, p, compression='zstd'))
                _json(output / 'metadata/bank_identity.json', bank_identity)
                _json(output / 'metadata/history_identity_vocabulary.json', dict(values=identities, offset=2,
                       note='Internal equality keys only; E embedding unknown remains 1. Train prefix unchanged.'))
                for name in ['vocabulary.json', 'feature_manifest.json']:
                    _atomic(output / 'metadata' / name, lambda p, n=name: shutil.copyfile(train / 'metadata' / n, p))
                paths = [account_path] + [output / f'history_index/{n}.npy' for n in ['bank_key', 'bank_cat']]
                paths += list((output / 'metadata').glob('*.json'))
                finish('identities', paths, node_count=len(accounts), added_nodes=len(unknown_nodes))
                del accounts, index
                gc.collect()

            total = meta['rows'] + rows
            if total > np.iinfo(np.uint32).max:
                raise ValueError('Too many edges for archive')
            if not checked('array_prefix'):
                emit('copying_history_arrays')
                _space(output, sum(np.dtype(dtype).itemsize * total for dtype in COLS.values()))
                for name, dtype in COLS.items():
                    src = np.load(train / f'history_index/{name}.npy', mmap_mode='r')
                    target = np.lib.format.open_memmap(output / f'history_index/{name}.npy', mode='w+', dtype=dtype, shape=(total,))
                    for begin in range(0, meta['rows'], BATCH_SIZE):
                        end = min(begin + BATCH_SIZE, meta['rows'])
                        target[begin:end] = src[begin:end]
                    target.flush()
                    del target, src
                # These files grow in the append phase; do not freeze full-file hashes yet.
                finish('array_prefix', [], train_rows=meta['rows'], allocated_rows=total)

            if not checked('append'):
                emit('appending_test', completed_files=len(state['appended_files']), files=len(files))
                accounts = pq.ParquetFile(output / 'graph/accounts/accounts.parquet').read()
                index = _account_index(accounts)
                vocab = read_json(output / 'metadata/vocabulary.json')['values']
                identities = read_json(output / 'metadata/history_identity_vocabulary.json')['values']
                bank_key = np.load(output / 'history_index/bank_key.npy', mmap_mode='r')
                arrays = {name: np.load(output / f'history_index/{name}.npy', mmap_mode='r+') for name in COLS}
                count, positives = meta['rows'], 0
                for file_index, source in enumerate(files):
                    date = source.parent.name.split('=', 1)[1]
                    edge_path = output / 'graph/edges' / source.parent.name / source.name
                    label_path = output / 'supervision/targets' / source.parent.name / source.name
                    if file_index < len(state['appended_files']):
                        item = state['appended_files'][file_index]
                        if item['source'] != source.relative_to(project).as_posix() or item['start'] != count:
                            raise ValueError('Append journal identity mismatch')
                        for relative, digest in item['outputs'].items():
                            if file_sha(output / relative) != digest:
                                raise ValueError('Verified append file changed')
                        for name in COLS:
                            if hashlib.sha256(np.asarray(arrays[name][count:item['end']]).tobytes()).hexdigest() != item['arrays'][name]:
                                raise ValueError('Verified append array changed')
                        count = item['end']; positives += item['positives']
                        continue
                    for p in [edge_path, label_path]:
                        p.parent.mkdir(parents=True, exist_ok=True)
                    et, lt = edge_path.with_suffix('.parquet.partial'), label_path.with_suffix('.parquet.partial')
                    ew = lw = None
                    first, n_positive = count, 0
                    try:
                        for batch in pq.ParquetFile(source).iter_batches(batch_size=BATCH_SIZE):
                            _space(output)
                            table = pa.Table.from_batches([batch])
                            ts = table['timestamp'].cast(pa.int64()).to_numpy()
                            lo = int(np.datetime64(date, 'us').astype(np.int64))
                            if not len(ts) or not np.all((ts >= lo) & (ts < lo + DAY)):
                                raise ValueError('Timestamp outside Test partition')
                            if table['transaction_id'].null_count:
                                raise ValueError('Missing transaction identity')
                            labels = table['is_laundering'].to_numpy()
                            if not np.isin(labels, [0, 1]).all():
                                raise ValueError('Invalid Test label')
                            src, _ = _nodes(table, index, 'src'); dst, _ = _nodes(table, index, 'dst')
                            if np.any(src < 0) or np.any(dst < 0):
                                raise ValueError('Unknown account escaped identity scan')
                            vals = dict(timestamp=ts, src=src.astype(np.uint32), dst=dst.astype(np.uint32),
                                        paid_scaled=scaled(table['amount_paid']), received_scaled=scaled(table['amount_received']))
                            model_vals = dict(vals)
                            for raw, family in [('payment_format', 'payment_format'), ('payment_currency', 'currency'), ('receiving_currency', 'currency')]:
                                vals[raw + '_cat'] = _codes(table[raw], identities[family])
                                model_vals[raw + '_cat'] = _codes(table[raw], vocab[family], unknown=True)
                            size = len(table)
                            edge_ids = np.arange(count, count + size, dtype=np.uint32)
                            for key, value in vals.items():
                                arrays[key][count:count + size] = value
                            edge = table.drop(['is_laundering']).append_column('edge_id', pa.array(edge_ids)).append_column('src', pa.array(vals['src'])).append_column('dst', pa.array(vals['dst']))
                            encoding = edge_encoding(model_vals, bank_key)
                            encoding['is_cross_currency'] = pc.not_equal(table['payment_currency'], table['receiving_currency']).to_numpy()
                            for key, value in encoding.items():
                                edge = edge.append_column(key, pa.array(value))
                            label = pa.table(dict(transaction_id=table['transaction_id'], edge_id=edge_ids,
                                                  label=table['is_laundering'], split=['test'] * size))
                            if ew is None:
                                ew = pq.ParquetWriter(et, edge.schema, compression='zstd')
                                lw = pq.ParquetWriter(lt, label.schema, compression='zstd')
                            ew.write_table(edge); lw.write_table(label)
                            count += size; n_positive += int(labels.sum())
                    finally:
                        if ew is not None: ew.close()
                        if lw is not None: lw.close()
                    if count - first != pq.ParquetFile(source).metadata.num_rows:
                        raise ValueError('Append target count mismatch')
                    for arr in arrays.values(): arr.flush()
                    os.replace(et, edge_path); os.replace(lt, label_path)
                    item = dict(source=source.relative_to(project).as_posix(), start=first, end=count, positives=n_positive,
                                outputs={p.relative_to(output).as_posix(): file_sha(p) for p in [edge_path, label_path]},
                                arrays={k: hashlib.sha256(np.asarray(v[first:count]).tobytes()).hexdigest() for k, v in arrays.items()})
                    state['appended_files'].append(item); _json(state_path, state)
                    positives += n_positive
                    emit('appending_test', completed_files=file_index + 1, files=len(files), appended_rows=count - meta['rows'])
                if count != total or positives != test['positive_rows']:
                    raise ValueError('Test target/positive coverage mismatch')
                for name in COLS:
                    old = np.load(train / f'history_index/{name}.npy', mmap_mode='r')
                    for begin in range(0, meta['rows'], BATCH_SIZE):
                        end = min(begin + BATCH_SIZE, meta['rows'])
                        if not np.array_equal(arrays[name][begin:end], old[begin:end]):
                            raise ValueError('Train history prefix changed')
                appended_paths = [output / f'history_index/{name}.npy' for name in COLS]
                appended_paths += [output / relative for item in state['appended_files'] for relative in item['outputs']]
                finish('append', appended_paths, rows=rows, positives=positives)
                del arrays, accounts, index
                gc.collect()

            arrays = {name: np.load(output / f'history_index/{name}.npy', mmap_mode='r') for name in COLS}
            node_count = state['phases']['identities']['node_count']
            ext = arrays['src'] != arrays['dst']
            for direction, column in [('in', 'dst'), ('out', 'src')]:
                if checked('index_' + direction): continue
                _space(output, total * 8 + (node_count + 1) * 16)
                emit('index_' + direction, rows=total, nodes=node_count)
                order = np.lexsort((np.arange(total, dtype=np.uint32), arrays['timestamp'], arrays[column])).astype(np.uint32)
                ptr = np.r_[0, np.cumsum(np.bincount(arrays[column].astype(np.int64), minlength=node_count))]
                extorder = order[ext[order]]
                extptr = np.r_[0, np.cumsum(np.bincount(arrays[column][extorder].astype(np.int64), minlength=node_count))]
                if ptr[-1] != total or extptr[-1] != int(ext.sum()):
                    raise ValueError('Temporal index coverage mismatch')
                values = {f'history_index/{direction}_order.npy': order, f'history_index/{direction}_ptr.npy': ptr,
                          f'graph/adj_index/{direction}_edge_ids.npy': extorder, f'graph/adj_index/{direction}_ptr.npy': extptr}
                for relative, array in values.items(): _save_array(output / relative, array)
                finish('index_' + direction, [output / p for p in values])
                del order, ptr, extorder, extptr, values
                gc.collect()
            # Recheck source immutability before publishing the archive identity.
            emit('final_verification', rows=total)
            if any(file_sha(project / relative) != digest for relative, digest in input_hashes.items()):
                raise ValueError('Source changed during Test build')
            for name in state['phases']: checked(name)
            metadata = dict(version=VERSION, rows=total, allocated_rows=total, node_count=node_count,
                            external_rows=int(ext.sum()), data_start_us=meta['data_start_us'], end_exclusive=END,
                            source_manifest_sha256=meta['source_manifest_sha256'], split_sha256=meta['split_sha256'],
                            train_rows=meta['rows'], test_rows=rows, test_positives=test['positive_rows'],
                            train_archive_sha256=file_sha(train / 'build.json'), config_sha256=file_sha(cp),
                            stages_sha256=file_sha(state_path), wall_seconds=time.perf_counter() - started,
                            scope='Train/Validation prefix and Test only; Train vocab fixed; all features query strictly earlier observations',
                            status='built; feature acceptance is separate')
            _json(output / 'build.json', metadata)
            _json(output / 'status.json', dict(status='completed', pid=os.getpid(), rows=total,
                                              test_rows=rows, updated_at=time.time()))
            if progress: progress({**metadata, 'status': 'completed'})
            return metadata
        except Exception as exc:
            _json(output / 'status.json', dict(status='failed', pid=os.getpid(), error=repr(exc), updated_at=time.time()))
            raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_test_archive(args.project, args.output, progress=lambda item: print(json.dumps(item), flush=True))
