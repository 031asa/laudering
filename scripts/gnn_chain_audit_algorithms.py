"""Post-hoc positive-edge graph diagnostics, never training features.

The caller supplies in-memory Train/Validation positive edges and targets.
Nothing is read from disk, written, trained, or inferred. Label-derived graph
connectivity and temporal walks are diagnostic associations, not evidence that
the same funds moved along a chain. Currencies do not constrain connectivity;
when supplied, they are retained as witness context only.

Public API: ``audit_positive_history(positive_edges, targets,
observation_end_us=..., observation_start_us=None, progress=None)`` returns
``(results_frame, audit_dict)``. Input columns are edge_id/src/dst/timestamp for
positive edges and edge_id/src/dst/cutoff_us for targets. All are integer IDs or
integer microseconds. An optional positive-edge ``label`` must be uniformly 1.
All list-valued witness, incident-ID and future-observation fields are returned
as compact JSON strings for stable Parquet/CSV serialization.

For each target cutoff t, historical windows are exactly [t-W, t). Every edge
at t, all self-loops, and the target itself are excluded. The whole 7-day
positive-edge graph supplies undirected components and distance buckets. Nodes
use ordinary shortest-path distance from an endpoint. An edge's distance is
min(distance(endpoint,u), distance(endpoint,v)) + 1, so incident edges are in
bucket 1; all 1/2/3/4+ buckets are mutually exclusive. The anchor itself is not
in a node-distance bucket, but is included in its component if present there.

The longest directed temporal *walk* must finish at the current src or dst.
Historical timestamps increase strictly. Revisiting accounts is allowed and
distinct-node counts are reported separately; this is not a simple-path or
same-funds claim. Equal-timestamp edges are processed simultaneously.

Future receiver observations use t < timestamp < t+72h and the caller's
exclusive observation_end_us. A truncated window is flagged and its zero
count must not be interpreted as absence over the entire 72 hours. Observation
may not extend into the sealed Test period beginning 2022-09-14 00:00:00.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
from typing import Callable

import numpy as np
import pandas as pd


HOUR_US = 3_600_000_000
DAY_US = 24 * HOUR_US
TEST_START_US = int(np.datetime64('2022-09-14T00:00:00', 'us').astype(np.int64))
CURRENCY_COLUMNS = ('currency', 'payment_currency', 'receiving_currency',
                    'payment_currency_cat', 'receiving_currency_cat')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_column(frame, name):
    _require(name in frame, 'Missing column: ' + name)
    values = frame[name].to_numpy()
    _require(values.ndim == 1 and values.dtype.kind in 'iu', name + ' must contain integer values')
    if values.dtype.kind == 'u' and len(values):
        _require(int(values.max()) <= np.iinfo(np.int64).max, name + ' exceeds int64')
    return values.astype(np.int64, copy=False)


def _integer_scalar(value, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name + ' must be an integer')
    number = int(value)
    _require(np.iinfo(np.int64).min <= number <= np.iinfo(np.int64).max, name + ' exceeds int64')
    return number


@dataclass(frozen=True, slots=True)
class _WalkLink:
    edge_position: int
    parent: int
    length: int


class _HistoricalGraph:
    """One cutoff at a time; repeated target cutoffs reuse this graph."""

    def __init__(self, edge_ids, src, dst, timestamps, cutoff):
        self.edge_ids, self.src, self.dst, self.timestamps = edge_ids, src, dst, timestamps
        self.cutoff = cutoff
        self.adj = defaultdict(set)
        self.incident = defaultdict(list)
        self.incoming = Counter()
        self.outgoing = Counter()
        self.recent_incoming = Counter()
        self.recent_outgoing = Counter()
        for i, (u, v, when) in enumerate(zip(src, dst, timestamps)):
            u, v = int(u), int(v)
            self.adj[u].add(v)
            self.adj[v].add(u)
            self.incident[u].append(i)
            self.incident[v].append(i)
            self.outgoing[u] += 1
            self.incoming[v] += 1
            if int(when) >= cutoff - 72 * HOUR_US:
                self.recent_outgoing[u] += 1
                self.recent_incoming[v] += 1
        self.node_component = {}
        self.components = []
        for anchor in self.adj:
            if anchor in self.node_component:
                continue
            index = len(self.components)
            nodes, edges, pending = set(), set(), [anchor]
            self.node_component[anchor] = index
            while pending:
                node = pending.pop()
                nodes.add(node)
                edges.update(self.incident[node])
                for other in self.adj[node]:
                    if other not in self.node_component:
                        self.node_component[other] = index
                        pending.append(other)
            self.components.append((nodes, edges))
        self.links: list[_WalkLink] = []
        self.best_walk = {}
        self._build_walks()
        self.endpoint_cache = {}

    def _build_walks(self):
        starts = np.r_[0, np.flatnonzero(self.timestamps[1:] != self.timestamps[:-1]) + 1]
        stops = np.r_[starts[1:], len(self.timestamps)]
        for start, stop in zip(starts, stops):
            # Freeze best_walk for an entire timestamp. No edge at this time
            # may consume another edge at the same time.
            pending = {}
            for pos in range(int(start), int(stop)):
                u, v = int(self.src[pos]), int(self.dst[pos])
                parent = self.best_walk.get(u, -1)
                length = 1 if parent < 0 else self.links[parent].length + 1
                existing = self.best_walk.get(v, -1)
                previous_length = 0 if existing < 0 else self.links[existing].length
                if length <= previous_length:
                    continue
                if v not in pending or length > pending[v][0]:
                    pending[v] = (length, pos, parent)
            # Within a tie, sorted edge_id order picks a deterministic witness.
            # Existing equal-length witnesses retain their earlier ending time.
            for node, (length, pos, parent) in pending.items():
                self.links.append(_WalkLink(pos, parent, length))
                self.best_walk[node] = len(self.links) - 1

    def distances(self, anchor):
        if anchor not in self.adj:
            return {}
        result, queue = {anchor: 0}, deque([anchor])
        while queue:
            node = queue.popleft()
            for other in self.adj[node]:
                if other not in result:
                    result[other] = result[node] + 1
                    queue.append(other)
        return result

    def endpoint(self, anchor):
        anchor = int(anchor)
        if anchor in self.endpoint_cache:
            return self.endpoint_cache[anchor]
        distances = self.distances(anchor)
        node_counts = Counter(min(distance, 4) for distance in distances.values() if distance > 0)
        component = self.node_component.get(anchor)
        component_nodes, component_edges = (set(), set()) if component is None else self.components[component]
        edge_counts = Counter(min(min(distances[int(self.src[i])], distances[int(self.dst[i])]) + 1, 4)
                              for i in component_edges)
        result = {
            'history_positive_incident_unique_edges_7d': self.incoming[anchor] + self.outgoing[anchor],
            'history_positive_in_edges_7d': self.incoming[anchor],
            'history_positive_out_edges_7d': self.outgoing[anchor],
            'history_positive_incident_unique_edges_72h': self.recent_incoming[anchor] + self.recent_outgoing[anchor],
            'history_positive_in_edges_72h': self.recent_incoming[anchor],
            'history_positive_out_edges_72h': self.recent_outgoing[anchor],
            'history_positive_incident_edge_ids_72h': sorted(int(self.edge_ids[i]) for i in self.incident[anchor]
                                                           if int(self.timestamps[i]) >= self.cutoff - 72 * HOUR_US),
            'positive_component_nodes_7d': len(component_nodes),
            'positive_component_unique_edges_7d': len(component_edges),
            'present_in_positive_graph_7d': component is not None,
        }
        for level, tag in ((1, '1'), (2, '2'), (3, '3'), (4, '4plus')):
            result['positive_undirected_nodes_distance_' + tag + '_7d'] = node_counts[level]
            result['positive_undirected_edges_distance_' + tag + '_7d'] = edge_counts[level]
        positions = []
        link = self.best_walk.get(anchor, -1)
        while link >= 0:
            item = self.links[link]
            positions.append(item.edge_position)
            link = item.parent
        positions.reverse()
        edge_ids = [int(self.edge_ids[i]) for i in positions]
        times = [int(self.timestamps[i]) for i in positions]
        nodes = [] if not positions else [int(self.src[positions[0]])] + [int(self.dst[i]) for i in positions]
        if positions:
            _require(nodes[-1] == anchor and all(a < b for a, b in zip(times, times[1:])), 'Internal temporal witness time/end error')
            _require(all(int(self.dst[a]) == int(self.src[b]) for a, b in zip(positions, positions[1:])), 'Internal temporal witness continuity error')
            _require(len(set(edge_ids)) == len(edge_ids), 'Internal temporal witness repeats an edge')
        result.update({
            'temporal_walk_max_history_edges_7d': len(positions),
            'temporal_walk_witness_edge_ids_7d': edge_ids,
            'temporal_walk_witness_timestamp_us_7d': times,
            'temporal_walk_witness_nodes_7d': nodes,
            'temporal_walk_witness_distinct_nodes_7d': len(set(nodes)),
            'temporal_walk_witness_revisits_accounts_7d': len(set(nodes)) < len(nodes),
        })
        for length, tag in ((1, '1'), (2, '2'), (3, '3'), (4, '4plus')):
            result['temporal_walk_exists_at_least_' + tag + '_edges_7d'] = len(positions) >= length
        self.endpoint_cache[anchor] = result
        return result

    def same_component(self, src, dst):
        a, b = self.node_component.get(int(src)), self.node_component.get(int(dst))
        return a is not None and b is not None and a == b


def audit_positive_history(
    positive_edges: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    observation_end_us: int,
    observation_start_us: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return per-target post-hoc graph audits, preserving input target order.

    Future labels are used only in columns beginning ``future_``. Historical
    graph columns are computed entirely from [cutoff-7d, cutoff). Positive and
    target rows at/after the exclusive observation end or sealed Test start
    cause a hard failure. An input target that is also a positive edge must
    have exactly the same source, destination and timestamp.
    """
    _require(isinstance(positive_edges, pd.DataFrame) and isinstance(targets, pd.DataFrame), 'Inputs must be DataFrames')
    end = _integer_scalar(observation_end_us, 'observation_end_us')
    start = None if observation_start_us is None else _integer_scalar(observation_start_us, 'observation_start_us')
    _require(end <= TEST_START_US and (start is None or start < end), 'Observation bounds include Test or are invalid')
    edge = {key: _integer_column(positive_edges, key) for key in ('edge_id', 'src', 'dst', 'timestamp')}
    target = {key: _integer_column(targets, key) for key in ('edge_id', 'src', 'dst', 'cutoff_us')}
    _require(len(np.unique(edge['edge_id'])) == len(positive_edges), 'Positive edge IDs must be unique')
    _require(len(np.unique(target['edge_id'])) == len(targets), 'Target edge IDs must be unique')
    if 'label' in positive_edges:
        _require(np.all(positive_edges['label'].to_numpy() == 1), 'positive_edges contains non-positive labels')
    for data, time_key in ((edge, 'timestamp'), (target, 'cutoff_us')):
        _require(np.all(data[time_key] < end), 'Input contains observations at/after observation end')
        _require(start is None or np.all(data[time_key] >= start), 'Input precedes declared observation start')
        _require(np.all(data['edge_id'] >= 0) and np.all(data['src'] >= 0) and np.all(data['dst'] >= 0), 'IDs must be nonnegative')
    known = {int(edge_id): i for i, edge_id in enumerate(edge['edge_id'])}
    for i, edge_id in enumerate(target['edge_id']):
        previous = known.get(int(edge_id))
        if previous is not None:
            _require(int(edge['src'][previous]) == int(target['src'][i])
                     and int(edge['dst'][previous]) == int(target['dst'][i])
                     and int(edge['timestamp'][previous]) == int(target['cutoff_us'][i]),
                     'Target identity differs from its positive-edge record')
    currency_columns = [name for name in CURRENCY_COLUMNS if name in positive_edges]
    currency_by_edge = {}
    if currency_columns:
        for i, edge_id in enumerate(edge['edge_id']):
            currency_by_edge[int(edge_id)] = {name: _python_scalar(positive_edges[name].iloc[i]) for name in currency_columns}
    external = edge['src'] != edge['dst']
    excluded_self_loops = int((~external).sum())
    positions = np.flatnonzero(external)
    positions = positions[np.lexsort((edge['edge_id'][positions], edge['timestamp'][positions]))]
    edge = {key: values[positions] for key, values in edge.items()}
    ordered_targets = np.argsort(target['cutoff_us'], kind='stable')
    rows = [None] * len(targets)
    builds = 0
    at = 0
    while at < len(ordered_targets):
        cutoff = int(target['cutoff_us'][ordered_targets[at]])
        stop = at + 1
        while stop < len(ordered_targets) and int(target['cutoff_us'][ordered_targets[stop]]) == cutoff:
            stop += 1
        lo = int(np.searchsorted(edge['timestamp'], cutoff - 7 * DAY_US, side='left'))
        hi = int(np.searchsorted(edge['timestamp'], cutoff, side='left'))
        graph = _HistoricalGraph(edge['edge_id'][lo:hi], edge['src'][lo:hi], edge['dst'][lo:hi], edge['timestamp'][lo:hi], cutoff)
        historical_ids = set(int(v) for v in edge['edge_id'][lo:hi])
        builds += 1
        future_lo = int(np.searchsorted(edge['timestamp'], cutoff, side='right'))
        future_end = min(cutoff + 72 * HOUR_US, end)
        future_hi = int(np.searchsorted(edge['timestamp'], future_end, side='left'))
        for order_pos in range(at, stop):
            i = int(ordered_targets[order_pos])
            edge_id, src, dst = (int(target[key][i]) for key in ('edge_id', 'src', 'dst'))
            _require(edge_id not in historical_ids, 'Current target present in historical graph')
            row = {'edge_id': edge_id, 'src': src, 'dst': dst, 'cutoff_us': cutoff,
                   'history_window_start_us_7d': cutoff - 7 * DAY_US,
                   'history_window_start_us_72h': cutoff - 72 * HOUR_US,
                   'history_window_complete_7d': None if start is None else start <= cutoff - 7 * DAY_US,
                   'history_window_complete_72h': None if start is None else start <= cutoff - 72 * HOUR_US,
                   'history_positive_unique_edges_7d': hi - lo,
                   'history_positive_graph_nodes_7d': len(graph.adj),
                   'src_dst_same_positive_component_7d': graph.same_component(src, dst),
                   'src_dst_undirected_positive_distance_7d': graph.distances(src).get(dst)}
            for role, anchor in (('src', src), ('dst', dst)):
                endpoint = graph.endpoint(anchor)
                row.update({role + '_' + key: value for key, value in endpoint.items()})
                if currency_columns:
                    row[role + '_temporal_walk_witness_currency_context_7d'] = [currency_by_edge[e] for e in endpoint['temporal_walk_witness_edge_ids_7d']]
            row['prior_positive_any_7d'] = (row['src_history_positive_incident_unique_edges_7d'] > 0
                                          or row['dst_history_positive_incident_unique_edges_7d'] > 0)
            row['src_temporal_walk_ge2_7d'] = row['src_temporal_walk_max_history_edges_7d'] >= 2
            row['positive_incident_ids_72h'] = sorted(set(row['src_history_positive_incident_edge_ids_72h'])
                                                    | set(row['dst_history_positive_incident_edge_ids_72h']))
            future_positions = np.arange(future_lo, future_hi, dtype=np.int64)
            future_positions = future_positions[edge['src'][future_positions] == dst]
            future_ids = [int(value) for value in edge['edge_id'][future_positions]]
            future_times = [int(value) for value in edge['timestamp'][future_positions]]
            row.update({
                'future_receiver_positive_out_edges_observed_72h': len(future_ids),
                'future_receiver_positive_out_unique_destination_accounts_observed_72h': len(set(int(v) for v in edge['dst'][future_positions])),
                'future_receiver_positive_out_edge_ids_observed_72h': future_ids,
                'future_receiver_positive_out_timestamp_us_observed_72h': future_times,
                'future_receiver_positive_out_first_gap_us_observed_72h': None if not future_times else future_times[0] - cutoff,
                'future_window_complete_72h': end >= cutoff + 72 * HOUR_US,
                'future_observed_until_exclusive_us': future_end,
                'future_observed_duration_us': max(0, future_end - cutoff),
            })
            # JSON strings have stable Parquet/CSV schemas even when every
            # witness in a batch is empty; the temporal ordering is preserved.
            row = {key: json.dumps(value, ensure_ascii=True, separators=(',', ':')) if isinstance(value, list) else value
                   for key, value in row.items()}
            rows[i] = row
            completed = order_pos + 1
            if progress is not None and (completed % 100 == 0 or completed == len(targets)):
                progress({'stage': 'positive_history_audit', 'completed_targets': completed,
                          'total_targets': len(targets), 'distinct_cutoffs_built': builds, 'cutoff_us': cutoff})
        at = stop
    frame = pd.DataFrame(rows)
    audit = {
        'status': 'passed', 'scope': 'Post-hoc label-derived Train/Validation diagnostics only; not model inputs and not evidence of shared funds',
        'targets': len(targets), 'input_positive_edges': len(positive_edges),
        'self_loops_excluded_from_history_and_future': excluded_self_loops,
        'external_positive_edges': len(edge['edge_id']), 'distinct_cutoffs_built': builds,
        'cache_policy': 'All equal-cutoff targets share one graph; release graph before next cutoff to bound memory',
        'observation_start_us': start, 'observation_end_exclusive_us': end, 'sealed_test_start_us': TEST_START_US,
        'history_window': '[t-7d,t); 72h counts use [t-72h,t); every same-timestamp edge and current target excluded',
        'undirected_node_distance': 'Shortest undirected hop count; anchor excluded from node-distance buckets',
        'undirected_edge_distance': 'min(distance(anchor,u), distance(anchor,v))+1; incident edges have distance 1',
        'distance_buckets': ['1', '2', '3', '4plus'], 'distance_buckets_are_disjoint': True,
        'component_scope': 'Entire positive-edge graph in the 7d window; absent anchor has 0 nodes and 0 edges',
        'temporal_walk': 'Directed, strictly increasing historical timestamp, ending at current src/dst; account revisits allowed',
        'temporal_walk_exists_columns': 'At least 1,2,3,4 historical edges; these nested existence flags are not disjoint buckets',
        'summary_flags': {'prior_positive_any_7d': 'Either endpoint has at least one incident historical positive edge',
                          'src_temporal_walk_ge2_7d': 'Longest strictly increasing historical positive walk ending at current src has at least two edges'},
        'incident_ids_72h': 'positive_incident_ids_72h is the sorted union of both endpoint incident positive edge IDs; individual endpoint lists are also supplied',
        'list_columns_encoding': 'All witness, incident-ID, and future-ID/time/currency lists are compact JSON strings for stable Parquet/CSV serialization',
        'positive_graph_distance_caution': 'Positive-only higher-hop presence depends on endpoint incidence and is not an independent mixed-neighbor test; use structure for clustering diagnostics',
        'temporal_witness_ties': 'Keep earlier-ending equal-length witness; at a single timestamp traverse ascending edge_id order',
        'currency_constraint': 'None; currency columns are optional witness context, not proof of a single-currency or same-funds chain',
        'currency_context_columns': currency_columns,
        'future_observation': 'Current receiver outgoing positive edges with t < timestamp < min(t+72h, observation_end); separate retrospective columns only',
        'future_censoring': 'future_window_complete_72h=False means observed counts are partial, including when zero',
        'test_access': 'No filesystem I/O; rejects all supplied edges/targets at or after the observation bound and rejects bounds entering Test',
        'target_order_preserved': True, 'output_columns': list(frame.columns),
    }
    return frame, audit


def _python_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
