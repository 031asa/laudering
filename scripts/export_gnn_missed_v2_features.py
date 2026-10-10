"""Read-only, identity-checked export of committed Validation v2 features.

Public interface::

    frame, audit = export_v2_validation_features(
        project, edge_ids, cutoff_us, batch_size=65536, progress=None)

The frame retains request order. Raw float64 values keep the 28 contract names;
their float32 model inputs have ``__encoded`` appended. The 28 ``__valid``
columns preserve committed float32 masks. The three coverage flags are named
``v2_full_24h``, ``v2_full_72h``, and ``v2_full_7d``. Exact int64 support columns
are ``src_count_7d``/``dst_count_7d`` and
``v2_src_count_24h_reused``/``v2_dst_count_24h_reused``. A support value of -1 is
not applicable (self transfer), not zero history. Raw missing values are NaN.

No output file is written. No feature is recomputed from transaction histories.
The raw values come exclusively from SHA-verified committed _parts Parquet
files, never from raw_work.npy or other scratch files. Every published
Validation sidecar block is verified; requested raw values must reproduce the
committed 59-dimensional input bit for bit after the frozen encoding function.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gnn_features.v2_features import (
    EXTRA_COLUMNS, EXTRA_FIELDS, EXTRA_NAMES, ROLES, SPECS, WINDOWS,
    SidecarStore, _base_plan, _partitions, array_sha, encode_extra, file_sha,
    inside, read,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int_vector(value, name):
    array = np.asarray(value)
    _require(array.ndim == 1 and array.dtype.kind in 'iu', name + ' must be a one-dimensional integer array')
    if array.dtype.kind == 'u' and len(array):
        _require(int(array.max()) <= np.iinfo(np.int64).max, name + ' exceeds int64')
    return array.astype(np.int64, copy=False)


def export_v2_validation_features(
    project: str | Path,
    edge_ids,
    cutoff_us,
    *,
    batch_size: int = 65536,
    progress: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return selected raw/model features and a JSON-serializable audit.

    ``progress``, when supplied, receives one dictionary at each verified
    partition and sidecar block. Only Validation IDs are accepted. All hashes,
    requested cutoffs, complete raw external-transaction coverage, and selected
    encoding equality are checked before anything is returned to the caller.
    Reused 24h counts are verified partition exports for the caller to compare
    against original R; their prior full-population crosscheck is hash-bound.
    """
    project = Path(project).resolve()
    spec = SPECS['validation']
    root = inside(project, 'data/gnn_v2_target_features/validation')
    cache = inside(project, spec['cache'])
    archive = inside(project, spec['archive'])
    ids = _int_vector(edge_ids, 'edge_ids')
    times = _int_vector(cutoff_us, 'cutoff_us')
    _require(ids.shape == times.shape and len(np.unique(ids)) == len(ids), 'Requests must have unique IDs and matching cutoffs')
    _require(isinstance(batch_size, int) and not isinstance(batch_size, bool) and 1 <= batch_size <= 262144,
             'batch_size must be an integer in [1, 262144]')
    offset, total = spec['offset'], spec['targets']
    _require(np.all((ids >= offset) & (ids < offset + total)), 'Request contains non-Validation IDs')

    def emit(**detail):
        if progress is not None:
            progress(detail)

    store = SidecarStore(root, cache, expected_split='validation')
    _require(store.config['offset'] == offset and store.config['targets'] == total,
             'Unexpected sidecar Validation extent')
    _require(Path(store.config['archive_root']).resolve() == archive, 'Unexpected archive root')
    _require(file_sha(cache / 'config.json') == store.config['base_config_sha256'], 'Frozen cache config changed')
    _require(len(store.blocks) == 197 and store.manifest.get('contract_sha256') == store.contract_sha256,
             'Unexpected block inventory or feature contract')
    comparison_path = inside(project, 'result/gnn_tuning_v2/comparison.json')
    best_path = comparison_path.with_name('best_model.json')
    comparison_hash = file_sha(comparison_path)
    comparison, best = read(comparison_path), read(best_path)
    _require(comparison.get('status') == 'completed' and best.get('comparison_sha256') == comparison_hash,
             'Canonical completed comparison/best-model identity mismatch')
    for key, actual in {'config_sha256': store.config_sha256, 'manifest_sha256': store.manifest_sha256,
                        'contract_sha256': store.contract_sha256}.items():
        _require(comparison['sidecar_identity'][key] == actual, 'Comparison sidecar identity differs: ' + key)

    source_hashes = {
        str(comparison_path.relative_to(project)): comparison_hash,
        str(best_path.relative_to(project)): file_sha(best_path),
        str((root / 'config.json').relative_to(project)): store.config_sha256,
        str((root / 'manifest.json').relative_to(project)): store.manifest_sha256,
        str((cache / 'config.json').relative_to(project)): store.config['base_config_sha256'],
        str((cache / 'manifest.json').relative_to(project)): store.config['base_manifest_sha256'],
    }
    implementation = Path(encode_extra.__code__.co_filename).resolve()
    _require(implementation == inside(project, 'scripts/gnn_features/v2_features.py'),
             'Imported feature implementation is not from this project')
    source_hashes[str(implementation.relative_to(project))] = store.config['code_sha256']
    for name in ('pilot.json', 'count_crosscheck.json', 'currency_verification.json'):
        source_hashes[str((root / name).relative_to(project))] = file_sha(root / name)
    expected_inputs = {str(Path(k)): v for k, v in store.config['input_hashes'].items()}

    def verified_input(relative):
        path = inside(project, relative)
        rel = str(path.relative_to(project))
        actual = file_sha(path)
        _require(expected_inputs.get(rel) == actual, 'Frozen source hash mismatch: ' + rel)
        source_hashes[rel] = actual
        return path

    meta_path = verified_input(str((archive / 'build.json').relative_to(project)))
    meta = read(meta_path)
    _require(meta['rows'] == offset + total, 'Unexpected archive rows')
    arrays = {}
    for field in ('timestamp', 'src', 'dst'):
        path = verified_input(str((archive / 'history_index' / (field + '.npy')).relative_to(project)))
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        _require(array.ndim == 1 and array.dtype.kind in 'iu' and len(array) >= offset + total,
                 'Unexpected history array: ' + field)
        arrays[field] = array[:offset + total]
    _require(np.array_equal(arrays['timestamp'][ids], times), 'Requested cutoff differs from frozen history')
    lower = int(np.datetime64('2022-09-11T00:00:00', 'us').astype(np.int64))
    upper = int(np.datetime64('2022-09-14T00:00:00', 'us').astype(np.int64))
    _require(np.all((times >= lower) & (times < upper)), 'Requested time lies outside Validation')
    pointers = {}
    for direction in ('in', 'out'):
        path = verified_input(str((archive / 'graph/adj_index' / (direction + '_ptr.npy')).relative_to(project)))
        pointers[direction] = np.load(path, mmap_mode='r', allow_pickle=False)
        _require(pointers[direction].ndim == 1 and pointers[direction].dtype == np.int64,
                 'Unexpected adjacency pointer schema')

    # The dense lookup is bounded by the frozen Validation size (~18 MiB),
    # irrespective of history/account skew or individual partition file size.
    selected_lookup = np.full(total, -1, dtype=np.int32)
    selected_lookup[ids - offset] = np.arange(len(ids), dtype=np.int32)
    raw = np.full((len(ids), len(EXTRA_NAMES)), np.nan, dtype=np.float64)
    count24 = np.full((len(ids), 2), -1, dtype=np.int64)
    partition_audit = []
    expected_external = 0
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        expected_external += int(np.count_nonzero(arrays['src'][offset+start:offset+stop] != arrays['dst'][offset+start:offset+stop]))

    for role_index, (role, (direction, node_name, *_)) in enumerate(ROLES.items()):
        own_fields = [f['name'] for f in EXTRA_FIELDS if f['role'] == role]
        own_columns = [EXTRA_NAMES.index(name) for name in own_fields]
        expected_parts = list(_partitions(pointers[direction], store.config['partition_edges']))
        role_root = inside(root, '_parts/' + role)
        expected_names = {f'{start:08d}_{end:08d}' for start, end in expected_parts}
        actual_names = {path.parent.name for path in role_root.glob('*/manifest.json')}
        _require(actual_names == expected_names, 'Committed raw partition inventory differs: ' + role)
        seen = np.zeros(total, dtype=bool)
        role_rows = 0
        for part_index, (start, end) in enumerate(expected_parts):
            folder = inside(role_root, f'{start:08d}_{end:08d}')
            manifest_path, parquet_path = folder / 'manifest.json', folder / 'features.parquet'
            part = read(manifest_path)
            signature = {'config_sha256': store.config_sha256, 'role': role,
                         'start_node': start, 'end_node': end, 'status': 'verified'}
            _require(all(part.get(k) == value for k, value in signature.items()), 'Raw partition identity differs')
            actual_sha = file_sha(parquet_path)
            _require(actual_sha == part['file_sha256'], 'Committed raw partition checksum mismatch')
            parquet = pq.ParquetFile(parquet_path)
            expected_schema = pa.schema([('edge_id', pa.int64()), ('count_24h', pa.int64()),
                                         *[(name, pa.float64()) for name in own_fields]])
            _require(parquet.schema_arrow.equals(expected_schema, check_metadata=False), 'Raw partition schema differs')
            _require(parquet.metadata.num_rows == part['targets'], 'Raw partition manifest row count differs')
            rows = 0
            for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
                part_ids = batch.column('edge_id').to_numpy(zero_copy_only=False)
                _require(part_ids.dtype == np.int64 and len(part_ids) == len(np.unique(part_ids)),
                         'Raw partition has invalid or duplicate IDs')
                local = part_ids - offset
                _require(np.all((local >= 0) & (local < total)), 'Raw partition contains non-Validation ID')
                _require(not seen[local].any(), 'Raw partition has duplicate coverage')
                _require(np.all((arrays[node_name][part_ids] >= start) & (arrays[node_name][part_ids] < end)),
                         'Raw partition role/node identity differs')
                _require(np.all(arrays['src'][part_ids] != arrays['dst'][part_ids]), 'Raw external partition contains self transfer')
                seen[local] = True
                request_positions = selected_lookup[local]
                selected = request_positions >= 0
                values = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False) for name in own_fields])
                _require(values.dtype == np.float64 and not np.isinf(values).any()
                         and np.all(values[~np.isnan(values)] >= 0), 'Raw amount/reference value is invalid')
                counts = batch.column('count_24h').to_numpy(zero_copy_only=False)
                _require(counts.dtype == np.int64 and np.all(counts >= 0), 'Raw count_24h is invalid')
                for window in ('72h', '7d'):
                    count = values[:, own_fields.index(f'v2_{role}_count_{window}')]
                    _require(np.isfinite(count).all() and np.equal(count, np.floor(count)).all(),
                             'Raw historical support is not a valid integer')
                if selected.any():
                    raw[np.ix_(request_positions[selected], own_columns)] = values[selected]
                    count24[request_positions[selected], role_index] = counts[selected]
                rows += len(part_ids)
            _require(rows == part['targets'], 'Raw partition streaming row count differs')
            role_rows += rows
            relative = str(folder.relative_to(project))
            partition_audit.append({'role': role, 'folder': relative, 'rows': rows,
                                    'manifest_sha256': file_sha(manifest_path), 'features_sha256': actual_sha})
            emit(stage='raw_partitions_verified', role=role, completed=part_index+1,
                 total=len(expected_parts), rows=role_rows)
        _require(role_rows == expected_external, 'Raw role external coverage count differs')
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            external = arrays['src'][offset+start:offset+stop] != arrays['dst'][offset+start:offset+stop]
            _require(np.array_equal(seen[start:stop], external), 'Raw role exact external coverage differs')

    self_transfer = arrays['src'][ids] == arrays['dst'][ids]
    _require(np.isnan(raw[self_transfer]).all() and np.all(count24[self_transfer] == -1),
             'Self-transfer raw applicability differs')
    _require(np.all(count24[~self_transfer] >= 0), 'Requested external count24 missing')
    encoded = np.empty((len(ids), len(EXTRA_COLUMNS)), dtype=np.float32)
    support = {role + '_count_7d': np.empty(len(ids), dtype=np.int64) for role in ROLES}
    all_seen = np.zeros(total, dtype=bool)
    selected_seen = np.zeros(len(ids), dtype=bool)
    block_audit = []
    for index, entry in enumerate(store.manifest['blocks']):
        block = entry['block']
        _require(block.split('/')[0] in {'transaction_date=' + date for date in spec['dates']}, 'Non-Validation sidecar block')
        _, block_ids, block_times = _base_plan(cache, block, store.base['config_sha256'])
        local = block_ids - offset
        _require(np.all((local >= 0) & (local < total)) and not all_seen[local].any(), 'Sidecar block coverage differs')
        _require(np.array_equal(arrays['timestamp'][block_ids], block_times), 'Sidecar block time/history differs')
        all_seen[local] = True
        matrix = store.block(block, block_ids, block_times)
        block_support = store.support(block, block_ids, block_times)
        block_folder = inside(root, block)
        block_manifest = read(block_folder / 'manifest.json')
        _require(block_manifest.get('contract_sha256') == store.contract_sha256, 'Block feature contract differs')
        coverage = np.column_stack([block_times - int(meta['data_start_us']) >= window for window in WINDOWS.values()])
        _require(np.array_equal(matrix[:, -3:], coverage.astype(np.float32)), 'Window coverage flag differs')
        request_positions = selected_lookup[local]
        selected = request_positions >= 0
        positions = request_positions[selected]
        _require(not selected_seen[positions].any(), 'Requested sidecar target duplicated')
        selected_seen[positions] = True
        encoded[positions] = matrix[selected]
        for name, values in block_support.items():
            support[name][positions] = values[selected]
        block_audit.append({'block': block, 'targets': len(block_ids),
                            'manifest_sha256': entry['manifest_sha256'],
                            'features_sha256': block_manifest['file_sha256'],
                            'selected': int(selected.sum())})
        emit(stage='sidecar_blocks_verified', completed=index+1, total=len(store.blocks),
             requested_found=int(selected_seen.sum()))
    _require(all_seen.all() and selected_seen.all(), 'Incomplete sidecar or selected target coverage')
    full = np.column_stack([times - int(meta['data_start_us']) >= window for window in WINDOWS.values()])
    roundtrip = encode_extra(raw, full)
    _require(np.array_equal(roundtrip, encoded), 'Committed raw features do not exactly reproduce the 59 model inputs')
    for role in ROLES:
        values = raw[:, EXTRA_NAMES.index('v2_' + role + '_count_7d')]
        expected = np.where(np.isnan(values), -1, values).astype(np.int64)
        _require(np.array_equal(expected, support[role + '_count_7d']), 'Raw vs exact int64 7d support differs')

    data = {'edge_id': ids.copy(), 'cutoff_us': times.copy()}
    data.update({name: raw[:, i] for i, name in enumerate(EXTRA_NAMES)})
    encoded_export_columns = ([name + '__encoded' for name in EXTRA_NAMES]
                              + [name + '__valid' for name in EXTRA_NAMES]
                              + ['v2_full_24h', 'v2_full_72h', 'v2_full_7d'])
    data.update({name: encoded[:, i] for i, name in enumerate(encoded_export_columns)})
    data.update(support)
    data.update({f'v2_{role}_count_24h_reused': count24[:, i] for i, role in enumerate(ROLES)})
    frame = pd.DataFrame(data)
    # Detect source identity changes during a long read; canonical/config and
    # publication manifests are the immutable entry points, not scratch files.
    for relative, digest in source_hashes.items():
        _require(file_sha(inside(project, relative)) == digest, 'Source changed during export: ' + relative)
    audit = {
        'status': 'passed', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'helper_sha256': file_sha(__file__), 'scope': 'Read-only committed Validation feature export; no Test, training, inference, or new feature computation',
        'requested_targets': len(ids), 'requested_edge_ids_sha256': array_sha(ids),
        'requested_cutoff_us_sha256': array_sha(times), 'request_order_preserved': True,
        'validation_targets_verified': total, 'external_targets_per_role_verified': expected_external,
        'requested_self_transfers': int(self_transfer.sum()),
        'raw_partitions_verified': partition_audit, 'sidecar_blocks_verified': block_audit,
        'source_sha256': source_hashes, 'contract_sha256': store.contract_sha256,
        'raw_value_source': 'Committed _parts/{src,dst}/*/features.parquet only; raw_work.npy/count24_work.npy never read',
        'raw_fields': EXTRA_FIELDS, 'raw_float64_preserved': True,
        'encoded_column_mapping': dict(zip(EXTRA_COLUMNS, encoded_export_columns)),
        'encoded_dtype': 'float32', 'selected_raw_encode_matches_all_59_columns_exactly': True,
        'support_columns': {'src_count_7d': 'committed int64 source support', 'dst_count_7d': 'committed int64 destination support',
                            'v2_src_count_24h_reused': 'committed raw source partition count_24h',
                            'v2_dst_count_24h_reused': 'committed raw destination partition count_24h'},
        'support_not_applicable_sentinel': -1,
        'count24_crosscheck': 'Prior full-population crosscheck bound by SidecarStore and its recorded SHA; exported counts available for caller comparison against original R',
        'raw_missing_encoding': 'NaN; validity mask zero and encoded value zero',
        'output_columns': list(frame.columns),
    }
    return frame, audit
