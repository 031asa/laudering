"""Standalone retrospective audit. Never trains, writes frozen inputs, or reads Test."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow as pa
from gnn_chain_audit_algorithms import audit_positive_history
from gnn_chain_audit_statistics import match_controls, run_primary_statistics, describe_ach_reference
from gnn_chain_sampling_audit import audit_full_neighbors, audit_saved_sampling

pa.set_cpu_count(1)
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'result/gnn_chain_audit_20261003'
STARTED = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write(name, x):
    assert shutil.disk_usage(OUT).free >= 20 * 1024**3
    p = OUT / name
    p.with_suffix(p.suffix + '.tmp').write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    p.with_suffix(p.suffix + '.tmp').replace(p)


def progress(detail=None, **kw):
    d = dict(detail or {}, **kw, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic()-STARTED, 1))
    write('audit_status.json', d)
    print(json.dumps(clean(d), ensure_ascii=True), flush=True)


def stage_frame(name, inputs, fn):
    p = OUT / (name + '.parquet'); a = OUT / (name + '.audit.json')
    if p.exists() and a.exists():
        audit = json.loads(a.read_text(encoding='utf-8'))
        actual = sha(p)
        assert audit['output_sha256'] == actual, name
        if audit['inputs'] != inputs:
            # Explicit, byte-bound review for this diagnostic-only parser repair.
            # The original stage provenance stays intact; arbitrary changes fail.
            review_path = OUT / 'sampling_format_recovery.json'
            review = json.loads(review_path.read_text(encoding='utf-8'))
            assert review['status'] == 'reviewed_unchanged_completed_computation'
            assert review['old_inputs'] == audit['inputs'] and review['new_inputs'] == inputs
            assert review['unchanged_stage_outputs'][name] == actual
            write(name + '.reuse_audit.json', dict(original_audit_sha256=sha(a),
                review_sha256=sha(review_path), original_inputs=audit['inputs'],
                current_inputs=inputs, output_sha256=actual))
        progress(stage=name, status='verified_reuse')
        return pd.read_parquet(p)
    assert not p.exists() and not a.exists(), f'Uncommitted stage: {name}'
    frame, detail = fn()
    frame.to_parquet(p, index=False)
    assert len(pd.read_parquet(p)) == len(frame)
    write(a.name, dict(inputs=inputs, output_sha256=sha(p), rows=len(frame), detail=detail))
    return frame


def main():
    cm = json.loads((OUT/'candidate_manifest.json').read_text(encoding='utf-8'))
    lm = json.loads((OUT/'labels_manifest.json').read_text(encoding='utf-8'))
    assert cm['candidate_sha256'] == sha(OUT/'candidates.parquet')
    assert lm['positive_edges']['sha256'] == sha(OUT/'positive_edges.parquet')
    d = pd.read_parquet(OUT/'candidates.parquet'); edges = pd.read_parquet(OUT/'positive_edges.parquet')
    assert len(d) == 106420 and int(d.label.sum()) == 6420 and len(edges) == 24214
    modules = {n: sha(PROJECT/'scripts'/n) for n in ['gnn_chain_audit_algorithms.py','gnn_chain_audit_statistics.py','gnn_chain_sampling_audit.py']}
    inputs = dict(candidates_sha256=cm['candidate_sha256'], positive_edges_sha256=lm['positive_edges']['sha256'], modules=modules)
    matches = {}; all_ids = set(d.loc[d.cohort.isin(['ACH_FN','ACH_TP','G_FN']), 'edge_id'])
    for cohort, expected in [('ACH_FN', 772), ('G_FN', 662)]:
        ids = d.loc[d.cohort == cohort, 'edge_id'].to_numpy(); assert len(ids) == expected
        m = match_controls(d, ids, cohort, max_controls=3, uniform_negative_pool_verified=True)
        matches[cohort] = m
        for k in ['pairs', 'unmatched']:
            m[k].to_csv(OUT/f'{cohort}_{k}.csv', index=False, encoding='utf-8-sig')
        write(f'{cohort}_matching.json', {k:v for k,v in m.items() if k not in ['pairs','unmatched']})
        all_ids.update(m['pairs'].control_edge_id)
        progress(stage='matching', cohort=cohort, summary=m['summary'])
    t = d[d.edge_id.isin(all_ids)].sort_values('edge_id').copy()
    assert len(t) == len(all_ids)
    for cohort, m in matches.items():
        t.loc[t.edge_id.isin(m['pairs'].control_edge_id), 'cohort'] = cohort + '_CONTROL'
    if (OUT/'targets.parquet').exists():
        pd.testing.assert_frame_equal(pd.read_parquet(OUT/'targets.parquet'),t.reset_index(drop=True))
    else:
        t.to_parquet(OUT/'targets.parquet', index=False)
    inputs['targets_sha256'] = sha(OUT/'targets.parquet')
    write('audit_inputs.json', inputs)
    progress(stage='positive_history', status='running', targets=len(t))
    positive = stage_frame('positive_history_features', inputs, lambda: audit_positive_history(
        edges.rename(columns={'timestamp_us':'timestamp'}), t,
        observation_start_us=int(np.datetime64('2022-09-01','us').astype(np.int64)),
        observation_end_us=int(np.datetime64('2022-09-14','us').astype(np.int64)),
        progress=lambda info: progress(info, stage='positive_history')))
    progress(stage='full_neighbors', status='running')
    full = stage_frame('full_history_neighbors', inputs, lambda: audit_full_neighbors(PROJECT,edges,t,progress=progress))
    check = positive.merge(full, on='edge_id', validate='one_to_one')
    assert np.array_equal(check.prior_positive_any_7d, check.endpoint_positive_incident_unique_7d > 0)
    for row in check[['positive_incident_ids_72h','full_positive_direct_ids_72h']].itertuples(index=False):
        assert set(json.loads(row[0])) == set(json.loads(row[1]))
    progress(stage='saved_plans', status='running')
    sampled = stage_frame('sampling_audit', inputs, lambda: audit_saved_sampling(PROJECT,edges,t,full,progress=progress))
    feature = check.merge(sampled, on='edge_id', validate='one_to_one')
    # Avoid overlap with the identity columns emitted by positive-history audit.
    detail = t.merge(feature.drop(columns=[c for c in feature if c in t and c != 'edge_id']), on='edge_id', validate='one_to_one')
    detail.to_parquet(OUT/'transaction_audit.parquet', index=False)
    detail.to_csv(OUT/'transaction_audit.csv', index=False, encoding='utf-8-sig')
    stats = run_primary_statistics(feature, matches)
    write('matched_statistics.json', stats)
    write('ach_detected_reference.json', describe_ach_reference(d, feature))
    summary = {}
    flags = ['prior_positive_any_7d','src_temporal_walk_ge2_7d','mixed_2hop_positive_any_7d',
        'direct_positive_any_sampled','indirect_positive_any_reachable2_sampled','src_positive_walk_ge2_sampled']
    for cohort, g in detail.groupby('cohort'):
        s = dict(transactions=len(g), payers=int(g.src.nunique()), receivers=int(g.dst.nunique()),
                 counts={c:int(g[c].sum()) for c in flags})
        for c in ['src_temporal_walk_max_history_edges_7d','dst_temporal_walk_max_history_edges_7d',
                  'src_positive_component_unique_edges_7d','dst_positive_component_unique_edges_7d',
                  'endpoint_positive_density_7d','endpoint_all_incident_unique_7d']:
            s[c] = dict(quantiles={str(q):float(g[c].quantile(q)) for q in [0,.25,.5,.75,.9,1]},
                        histogram={str(k):int(v) for k,v in g[c].value_counts().sort_index().items()} if 'walk_max' in c else {})
        s['sampling'] = {}
        for key, before, after in [
            ('direct_positive_72h',g.endpoint_positive_incident_unique_72h>0,g.direct_positive_any_sampled),
            ('mixed_2hop_72h',g.mixed_2hop_positive_any_72h,g.indirect_positive_any_reachable2_sampled),
            ('temporal_walk2_72h',g.src_positive_walk_ge2_72h,g.src_positive_walk_ge2_sampled)]:
            s['sampling'][key] = dict(eligible=int(before.sum()), retained=int((before&after).sum()), lost=int((before&~after).sum()))
        s['future_receiver_out_positive_observed'] = int((g.future_receiver_positive_out_edges_observed_72h > 0).sum())
        s['future_window_complete'] = int(g.future_window_complete_72h.sum())
        summary[cohort] = s
    write('cohort_summary.json', summary)
    for name, expected in modules.items(): assert sha(PROJECT/'scripts'/name) == expected
    write('run_manifest.json', dict(status='completed', inputs=inputs,
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['run_manifest.json','audit_status.json']},
        elapsed_seconds=time.monotonic()-STARTED,
        notes=['Retrospective final-label oracle audit, not deployable features.',
               'Prior graph strictly before target; no sealed Test read.',
               'Six exploratory matched tests; dependency clusters and Holm adjustment.',
               'Temporal walks need not be simple paths, same currency, or amount conserving.']))
    progress(stage='completed', status='completed', targets=len(t), cohort_counts=t.cohort.value_counts().to_dict())


if __name__ == '__main__':
    try: main()
    except Exception as e:
        progress(stage='failed', status='failed', error=repr(e))
        raise
