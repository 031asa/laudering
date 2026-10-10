"""Small semantic tests for post-hoc historical positive-graph diagnostics."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gnn_chain_audit_algorithms import DAY_US, HOUR_US, TEST_START_US, audit_positive_history


T = int(np.datetime64('2022-09-11T00:00:00', 'us').astype(np.int64))


def edges(rows):
    return pd.DataFrame(rows, columns=['edge_id', 'src', 'dst', 'timestamp']).astype('int64')


def targets(rows):
    return pd.DataFrame(rows, columns=['edge_id', 'src', 'dst', 'cutoff_us']).astype('int64')


def run(history, queries=None, **kwargs):
    query = targets([(900, 2, 8, T)]) if queries is None else queries
    return audit_positive_history(history, query, observation_end_us=kwargs.pop('end', TEST_START_US),
                                  observation_start_us=T-10*DAY_US, **kwargs)


def test_history_boundaries_target_and_all_same_timestamp_excluded():
    history = edges([(1, 1, 2, T-7*DAY_US-1), (2, 1, 2, T-7*DAY_US),
                     (3, 2, 3, T-72*HOUR_US), (4, 4, 2, T-72*HOUR_US-1),
                     (5, 4, 2, T), (900, 2, 8, T), (6, 1, 2, T+1)])
    frame, _ = run(history)
    row = frame.iloc[0]
    assert row['src_history_positive_incident_unique_edges_7d'] == 3
    assert row['src_history_positive_in_edges_7d'] == 2
    assert row['src_history_positive_out_edges_7d'] == 1
    assert row['src_history_positive_incident_unique_edges_72h'] == 1
    assert row['history_positive_unique_edges_7d'] == 3
    assert json.loads(row['src_temporal_walk_witness_edge_ids_7d']) == [2]
    assert 900 not in json.loads(row['src_temporal_walk_witness_edge_ids_7d'])


def test_same_timestamp_edges_cannot_form_two_edge_temporal_walk():
    history = edges([(1, 0, 1, T-10), (2, 1, 2, T-10)])
    row = run(history)[0].iloc[0]
    assert row['src_positive_component_unique_edges_7d'] == 2
    assert row['src_temporal_walk_max_history_edges_7d'] == 1
    assert json.loads(row['src_temporal_walk_witness_edge_ids_7d']) == [2]


def test_topological_connection_is_not_a_chronological_chain():
    history = edges([(1, 0, 1, T-10), (2, 1, 2, T-20)])
    row = run(history)[0].iloc[0]
    assert row['src_positive_undirected_nodes_distance_2_7d'] == 1
    assert row['src_positive_component_nodes_7d'] == 3
    assert row['src_temporal_walk_max_history_edges_7d'] == 1
    assert json.loads(row['src_temporal_walk_witness_edge_ids_7d']) == [2]


def test_temporal_walk_allows_revisits_and_reports_distinct_nodes():
    history = edges([(1, 1, 2, T-30), (2, 2, 1, T-20), (3, 1, 2, T-10)])
    row = run(history)[0].iloc[0]
    assert row['src_temporal_walk_max_history_edges_7d'] == 3
    assert json.loads(row['src_temporal_walk_witness_edge_ids_7d']) == [1, 2, 3]
    assert json.loads(row['src_temporal_walk_witness_nodes_7d']) == [1, 2, 1, 2]
    assert row['src_temporal_walk_witness_distinct_nodes_7d'] == 2
    assert row['src_temporal_walk_witness_revisits_accounts_7d']
    assert row['src_temporal_walk_exists_at_least_3_edges_7d']
    assert not row['src_temporal_walk_exists_at_least_4plus_edges_7d']
    assert row['prior_positive_any_7d'] and row['src_temporal_walk_ge2_7d']


def test_combined_incident_ids_do_not_double_count_both_endpoints():
    history = edges([(1, 2, 8, T-30), (2, 8, 3, T-20), (3, 4, 2, T-10)])
    row = run(history)[0].iloc[0]
    assert json.loads(row['src_history_positive_incident_edge_ids_72h']) == [1, 3]
    assert json.loads(row['dst_history_positive_incident_edge_ids_72h']) == [1, 2]
    assert json.loads(row['positive_incident_ids_72h']) == [1, 2, 3]
    assert row['prior_positive_any_7d']


def test_self_loops_removed_in_history_and_future():
    history = edges([(1, 2, 2, T-20), (2, 1, 2, T-10), (3, 8, 8, T+10), (4, 8, 9, T+20)])
    frame, audit = run(history)
    row = frame.iloc[0]
    assert row['src_history_positive_incident_unique_edges_7d'] == 1
    assert row['src_temporal_walk_max_history_edges_7d'] == 1
    assert json.loads(row['future_receiver_positive_out_edge_ids_observed_72h']) == [4]
    assert audit['self_loops_excluded_from_history_and_future'] == 2


def test_distance_buckets_are_disjoint_and_component_is_global():
    history = edges([(i, i-1, i, T-100+i) for i in range(1, 7)] + [(20, 50, 51, T-5)])
    query = targets([(900, 0, 6, T)])
    row = run(history, query)[0].iloc[0]
    assert [row[f'src_positive_undirected_nodes_distance_{tag}_7d'] for tag in ['1', '2', '3', '4plus']] == [1, 1, 1, 3]
    assert [row[f'src_positive_undirected_edges_distance_{tag}_7d'] for tag in ['1', '2', '3', '4plus']] == [1, 1, 1, 3]
    assert row['src_positive_component_nodes_7d'] == 7
    assert row['src_positive_component_unique_edges_7d'] == 6
    assert row['src_dst_undirected_positive_distance_7d'] == 6
    assert row['dst_temporal_walk_max_history_edges_7d'] == 6
    assert row['dst_temporal_walk_exists_at_least_4plus_edges_7d']
    assert row['src_temporal_walk_max_history_edges_7d'] == 0


def test_triangle_edge_buckets_and_parallel_edges_count_unique_edge_ids():
    history = edges([(1, 0, 1, T-30), (2, 0, 2, T-20), (3, 1, 2, T-10), (4, 0, 1, T-5)])
    row = run(history, targets([(900, 0, 8, T)]))[0].iloc[0]
    assert row['src_positive_undirected_nodes_distance_1_7d'] == 2
    assert row['src_positive_undirected_edges_distance_1_7d'] == 3
    assert row['src_positive_undirected_edges_distance_2_7d'] == 1
    assert row['src_positive_component_unique_edges_7d'] == 4


def test_future_censoring_and_future_labels_do_not_change_history():
    history = edges([(1, 1, 2, T-10), (2, 8, 9, T), (3, 8, 10, T+1), (4, 8, 11, T+71*HOUR_US)])
    full = run(history)[0].iloc[0]
    no_future = run(history.iloc[:1])[0].iloc[0]
    assert json.loads(full['future_receiver_positive_out_edge_ids_observed_72h']) == [3, 4]
    assert full['future_window_complete_72h']
    for key in full.index:
        if not key.startswith('future_'):
            a, b = full[key], no_future[key]
            assert (pd.isna(a) and pd.isna(b)) if not isinstance(a, list) and pd.isna(a) else a == b
    partial = run(history.iloc[:3], end=T+24*HOUR_US)[0].iloc[0]
    assert not partial['future_window_complete_72h']
    assert partial['future_receiver_positive_out_edges_observed_72h'] == 1
    assert partial['future_observed_duration_us'] == 24*HOUR_US


def test_future_upper_boundary_and_empty_history():
    when = T-DAY_US
    history = edges([(1, 8, 9, when+72*HOUR_US)])
    row = run(history, targets([(900, 2, 8, when)]))[0].iloc[0]
    assert row['history_positive_unique_edges_7d'] == 0
    assert row['src_positive_component_nodes_7d'] == 0
    assert json.loads(row['src_temporal_walk_witness_edge_ids_7d']) == []
    assert row['future_receiver_positive_out_edges_observed_72h'] == 0


def test_cutoff_reuse_order_and_currency_context():
    history = edges([(1, 1, 2, T-20), (2, 2, 3, T-10)])
    history['currency'] = ['USD', 'EUR']
    query = targets([(902, 3, 8, T+1), (900, 2, 8, T), (901, 3, 8, T)])
    frame, audit = run(history, query)
    assert frame['edge_id'].tolist() == [902, 900, 901]
    assert audit['distinct_cutoffs_built'] == 2
    assert json.loads(frame.iloc[2]['src_temporal_walk_witness_currency_context_7d']) == [{'currency': 'USD'}, {'currency': 'EUR'}]
    assert frame.iloc[2]['src_temporal_walk_max_history_edges_7d'] == 2


@pytest.mark.parametrize('bad', ['duplicate', 'target_identity', 'test', 'nonpositive'])
def test_invalid_identity_or_test_data_rejected(bad):
    history = edges([(1, 1, 2, T-10)])
    query = targets([(900, 2, 8, T)])
    if bad == 'duplicate':
        history = pd.concat([history, history], ignore_index=True)
    elif bad == 'target_identity':
        query = targets([(1, 1, 2, T)])
    elif bad == 'test':
        history.loc[0, 'timestamp'] = TEST_START_US
    else:
        history['label'] = 0
    with pytest.raises(ValueError):
        run(history, query)


def test_bruteforce_temporal_walk_oracle_on_small_random_graphs():
    rng = np.random.default_rng(20261003)
    for _ in range(20):
        triples = [(int(rng.integers(0, 4)), int(rng.integers(0, 4)), T-10+int(rng.integers(0, 4))) for _ in range(7)]
        history = edges([(i+1, u, v, timestamp) for i, (u, v, timestamp) in enumerate(triples)])
        query = targets([(900+i, i, 9, T) for i in range(4)])
        frame, _ = run(history, query)
        longest = [0] * 4

        def extend(node, previous_time, length):
            longest[node] = max(longest[node], length)
            for u, v, timestamp in triples:
                if u == node and u != v and timestamp > previous_time:
                    extend(v, timestamp, length+1)

        for node in range(4):
            extend(node, T-100, 0)
        assert frame['src_temporal_walk_max_history_edges_7d'].tolist() == longest
