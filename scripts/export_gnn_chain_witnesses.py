"""Export the actual positive-history walk witnesses; no training or Test reads."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
OUT = P/'result/gnn_chain_audit_20261003'
t = pd.read_parquet(OUT/'targets.parquet').set_index('edge_id')
f = pd.read_parquet(OUT/'positive_history_features.parquet').set_index('edge_id')
e = pd.read_parquet(OUT/'positive_edges.parquet').set_index('edge_id')
v = json.loads((P/'data/gnn_v1_validation/metadata/vocabulary.json').read_text(encoding='utf-8'))
cases = []
for eid in t.index[t.cohort.eq('ACH_FN')]:
    row = f.loc[eid]
    if int(row.src_temporal_walk_max_history_edges_7d) < 2: continue
    ids = json.loads(row.src_temporal_walk_witness_edge_ids_7d)
    path = e.loc[ids]
    assert (path.timestamp_us.diff().dropna() > 0).all()
    assert int(path.iloc[-1].dst) == int(t.loc[eid].src)
    assert int(path.timestamp_us.max()) < int(t.loc[eid].cutoff_us)
    assert int(path.timestamp_us.min()) >= int(t.loc[eid].cutoff_us)-7*86400_000000
    assert np.array_equal(path.dst.to_numpy()[:-1], path.src.to_numpy()[1:])
    chain = path.reset_index().to_dict('records')
    for edge in chain:
        edge['timestamp'] = str(np.datetime64(int(edge['timestamp_us']), 'us'))
        edge['final_label'] = 1
    currency_ok = bool((path.receiving_currency_cat.to_numpy()[:-1] == path.payment_currency_cat.to_numpy()[1:]).all()
                       and path.iloc[-1].receiving_currency_cat == e.loc[eid].payment_currency_cat)
    cases.append(dict(target_edge_id=int(eid), score=float(t.loc[eid].score_lr_only),
        target={k:str(t.loc[eid,k]) for k in ['timestamp','from_bank','from_account','to_bank','to_account','amount_paid','amount_received','payment_currency','receiving_currency']},
        historical_edges=len(ids), witness_distinct_accounts=int(row.src_temporal_walk_witness_distinct_nodes_7d),
        witness_revisits_accounts=bool(row.src_temporal_walk_witness_revisits_accounts_7d),
        this_witness_currency_junctions_compatible=currency_ok, history=chain))
assert len(cases) == 36
result = dict(scope='All 36 of the 772 ACH misses having at least two chronological positive history edges ending at the current payer in [t-7d,t).',
    limitations=['Final labels are retrospective oracle labels, without confirmation times.',
                 'Each witness is a longest temporal walk, may revisit accounts, and does not prove amount continuity or identical funds.',
                 'A currency-incompatible longest witness does not exclude another shorter compatible walk.',
                 'No Test or post-target edges appear in historical witnesses.'],
    counts=dict(cases=36,revisited_witnesses=sum(c['witness_revisits_accounts'] for c in cases),
                currency_compatible_witnesses=sum(c['this_witness_currency_junctions_compatible'] for c in cases)),
    currency_vocabulary=v, cases=sorted(cases,key=lambda c:(-c['historical_edges'],c['target_edge_id'])))
(OUT/'ach_deep_walk_witnesses.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result['counts']))
