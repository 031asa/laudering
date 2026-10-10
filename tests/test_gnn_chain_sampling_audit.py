"""Small explicit graphs test retrospective sampling-audit semantics."""
import json
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import gnn_chain_sampling_audit as audit


D = audit.DAY
T = 10 * D


def history(edges):
    """Build the actual sorted adjacency arrays used by FullHistory.incident."""
    h = audit.FullHistory.__new__(audit.FullHistory)
    h.hashes = {}
    src, dst, ts = map(np.asarray, zip(*edges))
    h.a = dict(src=src.astype(np.uint32), dst=dst.astype(np.uint32),
               timestamp=ts.astype(np.int64))
    n = max(int(src.max()), int(dst.max())) + 1
    for direction, node in [('out', src), ('in', dst)]:
        h.a[direction + '_order'] = np.lexsort((np.arange(len(edges)), ts, node))
        h.a[direction + '_ptr'] = np.r_[0, np.cumsum(np.bincount(node, minlength=n))]
    return h


def positives(h, ids):
    ids = np.array(ids, dtype=np.int64)
    return pd.DataFrame(dict(edge_id=ids, src=h.a['src'][ids], dst=h.a['dst'][ids],
                             timestamp_us=h.a['timestamp'][ids]))


def target(h, eid):
    return pd.DataFrame([dict(edge_id=eid, src=int(h.a['src'][eid]),
                              dst=int(h.a['dst'][eid]), cutoff_us=int(h.a['timestamp'][eid]))])


def full(monkeypatch, edges, positive_ids):
    # The last edge is always the target, and must not enter its own history.
    h = history(edges)
    monkeypatch.setattr(audit, 'FullHistory', lambda _: h)
    frame, manifest = audit.audit_full_neighbors('.', positives(h, positive_ids),
                                                  target(h, len(edges) - 1))
    assert manifest['status'] == 'passed'
    return h, frame, frame.iloc[0]


def test_real_incident_window_lower_inclusive_ties_and_self_excluded():
    h = history([(0, 2, T - 7*D - 1), (0, 3, T - 7*D),
                 (4, 0, T - 3*D), (0, 5, T - 1),
                 (0, 6, T), (0, 7, T), (0, 0, T - 2)])
    assert h.incident(0, 'out', T, days=7).tolist() == [1, 3]
    assert h.incident(0, 'out', T, days=3).tolist() == [3]
    assert h.incident(0, 'in', T, days=3).tolist() == [2]


def test_mixed_second_hop_uses_unlabeled_first_edge_and_excludes_direct_self(monkeypatch):
    edges = [(0, 2, T-D),        # unlabeled connecting edge
             (2, 3, T-2*D),     # positive can precede connection: topology only
             (0, 4, T-2),       # direct positive, never counted as strict second hop
             (2, 0, T-3),       # both touches a neighbor AND direct: still direct only
             (2, 2, T-4),       # positive self edge excluded
             (6, 7, T-5),       # disconnected positive
             (0, 8, T),         # tied positive excluded
             (0, 1, T)]         # target excluded even if its label is positive
    _, _, row = full(monkeypatch, edges, [1, 2, 3, 4, 5, 6, 7])
    assert row.endpoint_all_incident_unique_7d == 3
    assert row.endpoint_positive_incident_unique_7d == 2
    assert row.mixed_2hop_positive_count_7d == 1
    assert row.mixed_2hop_positive_count_72h == 1
    assert json.loads(row.full_positive_direct_ids_72h) == [3, 2]
    assert json.loads(row.full_positive_indirect_ids_72h) == [1]
    assert not row.src_positive_walk_ge2_72h


def test_seventy_two_hours_requires_both_connection_and_positive_in_window(monkeypatch):
    edges = [(0, 2, T-4*D),     # first edge too old for 72h
             (2, 3, T-D),       # positive recent, but only seven-day connection exists
             (0, 4, T-D),       # recent first edge
             (4, 5, T-4*D),     # positive too old for 72h
             (0, 6, T-3*D),     # exactly lower boundary is included
             (0, 1, T)]
    _, _, row = full(monkeypatch, edges, [1, 3, 4])
    assert row.mixed_2hop_positive_count_7d == 2
    assert row.mixed_2hop_positive_count_72h == 0
    assert row.endpoint_positive_incident_unique_72h == 1
    assert row.onehop_unique_neighbors_7d == 3
    assert row.onehop_unique_neighbors_72h == 2


@pytest.mark.parametrize('first_time,second_time,expected', [
    (T-2*D, T-D, True),
    (T-D, T-D, False),           # equal-time edges cannot form an ordered walk
    (T-D, T-2*D, False),         # reversed chronology cannot form an ordered walk
    (T-3*D-1, T-D, False),       # both historical legs must lie inside 72h
    (T-3*D, T-D, True),
    (T-D, T, False),             # the second edge at target time is excluded
])
def test_positive_walk_requires_two_strictly_time_ordered_historical_edges(
        monkeypatch, first_time, second_time, expected):
    _, _, row = full(monkeypatch,
                     [(3, 2, first_time), (2, 0, second_time), (0, 1, T)], [0, 1])
    assert bool(row.src_positive_walk_ge2_72h) is expected
    assert json.loads(row.src_positive_walk2_witness_72h) == ([0, 1] if expected else [])


def test_positive_walk_cannot_use_wrong_direction(monkeypatch):
    _, _, row = full(monkeypatch,
                     [(2, 3, T-2*D), (2, 0, T-D), (0, 1, T)], [0, 1])
    assert row.mixed_2hop_positive_count_72h == 1
    assert not row.src_positive_walk_ge2_72h


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding='utf-8')


def saved_fixture(monkeypatch, tmp_path):
    edges = [(0, 2, T-D),        # sampled unlabeled connecting edge
             (2, 3, T-2*D),     # sampled positive reachable at strict second hop
             (0, 4, T-2),       # full graph direct positive omitted by sampling
             (4, 5, T-3),       # full graph indirect positive also omitted
             (10, 11, T-4),     # other target's positive, disconnected from this target
             (10, 12, T),       # another valid target sharing the saved batch
             (0, 1, T)]
    h, full_frame, _ = full(monkeypatch, edges, [1, 2, 3, 4])
    cache = tmp_path / 'data/gnn_validation_cache_parquet'
    index = tmp_path / 'data/gnn_v1_validation/history_index'
    index.mkdir(parents=True)
    hashes = {}
    for key in ['src', 'dst', 'timestamp']:
        p = index / (key + '.npy')
        np.save(p, h.a[key])
        hashes[p.relative_to(tmp_path).as_posix()] = audit.sha(p)
    write_json(cache / 'config.json', dict(
        expected_targets=2, batch_size=64, max_batches_per_block=512,
        dates=['1970-01-11'], end_exclusive='1970-01-12T00:00:00', hashes=hashes))
    config_hash = audit.sha(cache / 'config.json')
    block = cache / 'test_block'
    block.mkdir()
    schema = pa.schema([('batch_id', pa.int32()), ('cutoff_us', pa.int64())]
                       + [(k, pa.list_(pa.int64())) for k in ['targets', 'nodes', 'edges']])
    pq.write_table(pa.Table.from_pylist([dict(
        batch_id=0, cutoff_us=T, targets=[5, 6], nodes=[0, 1, 2, 3, 10, 11, 12],
        edges=[0, 1, 4])], schema=schema), block / 'plans.parquet')
    write_json(block / 'manifest.json', dict(
        status='verified', config_sha256=config_hash,
        block='test_block', targets=2, batches=1, first_cutoff_us=T, last_cutoff_us=T,
        target_ids_sha256=hashlib.sha256(np.ascontiguousarray([5, 6], dtype=np.int64).tobytes()).hexdigest(),
        file_sha256={'plans.parquet': audit.sha(block / 'plans.parquet')}))
    write_json(cache / 'manifest.json', dict(
        status='completed', config_sha256=config_hash, targets=2,
        blocks=[dict(block='test_block', targets=2, node_states=7, parquet_bytes=123)]))
    evaluation = tmp_path / 'result/gnn_tuning_v1/evaluations/lr_only'
    write_json(evaluation / 'config.json', dict(
        cache_config_sha256=config_hash, cache_manifest_sha256=audit.sha(cache / 'manifest.json'),
        expected_targets=2))
    write_json(evaluation / 'metrics.json', dict(
        status='completed', config_sha256=audit.sha(evaluation / 'config.json')))
    return h, full_frame, cache, block


def test_saved_batch_counts_only_two_hop_reachable_label_overlay(monkeypatch, tmp_path):
    h, full_frame, cache, block = saved_fixture(monkeypatch, tmp_path)
    # The actual root inventory has no per-block manifest hash.
    assert 'manifest_sha256' not in audit.read(cache / 'manifest.json')['blocks'][0]
    original_plan_sha = audit.sha(block / 'plans.parquet')
    result, evidence = audit.audit_saved_sampling(
        tmp_path, positives(h, [1, 2, 3, 4]), target(h, 6), full_frame)
    row = result.iloc[0]
    assert row.sampled_graph_positive_edges == 2
    assert row.direct_positive_edges_sampled == 0
    assert row.indirect_positive_edges_reachable2_sampled == 1
    assert row.full_direct_positive_edges_72h == 1
    assert row.full_indirect_positive_edges_72h == 2
    assert not row.src_positive_walk_ge2_sampled
    assert evidence['status'] == 'passed'
    assert evidence['complete_unique_cached_target_count'] == 2
    assert evidence['blocks'][0]['manifest_sha256'] == audit.sha(block / 'manifest.json')
    assert audit.sha(block / 'plans.parquet') == original_plan_sha


@pytest.mark.parametrize('kind', [
    'root_manifest_sha', 'cache_config_sha', 'history_array_sha', 'plans_sha',
    'block_identity', 'block_target_count', 'block_batch_count', 'target_ids_sha',
    'target_row_identity', 'positive_row_identity',
])
def test_saved_sampling_rejects_tampered_hashes_and_identities(monkeypatch, tmp_path, kind):
    h, full_frame, cache, block = saved_fixture(monkeypatch, tmp_path)
    target_frame = target(h, 6)
    positive_frame = positives(h, [1, 2, 3, 4])
    if kind == 'root_manifest_sha':
        p = cache / 'manifest.json'; value = audit.read(p)
        value['targets'] += 1; write_json(p, value)
    elif kind == 'cache_config_sha':
        p = cache / 'config.json'; value = audit.read(p)
        value['batch_size'] = 1; write_json(p, value)
    elif kind == 'history_array_sha':
        p = tmp_path / 'data/gnn_v1_validation/history_index/src.npy'
        changed = h.a['src'].copy(); changed[0] = 7; np.save(p, changed)
    elif kind == 'plans_sha':
        with (block / 'plans.parquet').open('ab') as f:
            f.write(b'corruption')
    elif kind in {'block_identity', 'block_target_count', 'block_batch_count', 'target_ids_sha'}:
        p = block / 'manifest.json'; value = audit.read(p)
        field, wrong = {'block_identity': ('block', 'another_block'),
                        'block_target_count': ('targets', 3),
                        'block_batch_count': ('batches', 2),
                        'target_ids_sha': ('target_ids_sha256', '0' * 64)}[kind]
        value[field] = wrong; write_json(p, value)
    elif kind == 'target_row_identity':
        target_frame.loc[0, 'src'] = 9
    elif kind == 'positive_row_identity':
        positive_frame.loc[0, 'dst'] = 9
    with pytest.raises(AssertionError):
        audit.audit_saved_sampling(tmp_path, positive_frame, target_frame, full_frame)
