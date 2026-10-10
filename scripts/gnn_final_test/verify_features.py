"""Test-only daily verifier with NumPy nullable-float finite checks.

All identity, coverage, reference and audit checks are copied from verify_daily.
The frozen Train verifier remains unchanged; Arrow fill_null is not invoked.
"""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import pickle
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from gnn_features.build import load_archive,write_json
from gnn_features.engine import FeatureEngine
from gnn_features.contract import HOUR,DAY,N_FIELDS,R_FIELDS
from gnn_features.benchmark import peak_rss

TEST_DATES = ('2022-09-14', '2022-09-15', '2022-09-16')


def assert_finite_nullable(column):
    """Allow actual NULLs; reject every non-NULL NaN or positive/negative Inf."""
    if not pa.types.is_floating(column.type):
        raise TypeError('Floating-point column required')
    valid = column.is_valid().to_numpy(zero_copy_only=False)
    values = column.to_numpy(zero_copy_only=False)
    if not np.isfinite(values[valid]).all():
        raise AssertionError('Nonfinite non-NULL numeric output')


def require_test_date(run, prep):
    """Reject Train, Validation, tail or mismatched day metadata before verification."""
    date = prep.get('date')
    if date not in TEST_DATES or run.get('date') != date:
        raise ValueError('Verifier is restricted to Test dates 2022-09-14 through 2022-09-16')
    start = int(np.datetime64(date, 'us').astype(np.int64))
    if prep.get('start_us') != start or prep.get('end_us') != start + DAY:
        raise ValueError('Test date boundaries must cover one complete day')


def equal_value(a,b):
    if a is None or b is None:return a is None and b is None
    if isinstance(b,float):return math.isclose(float(a),b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def verify(archive,out,old_hour,old_nodes):
    began=time.perf_counter();meta,a,bank,vocab=load_archive(archive)
    run=json.loads((out/'daily_run.json').read_text(encoding='utf-8'))
    prep=json.loads((out/'prepare.json').read_text(encoding='utf-8'))
    require_test_date(run, prep)
    account=json.loads((out/'account_phase.json').read_text(encoding='utf-8'))
    merged=json.loads((out/'merge_phase.json').read_text(encoding='utf-8'))
    ids=np.load(out/'target_edge_ids.npy');seen=np.zeros(meta['rows'],dtype=bool)
    catalog=pq.ParquetFile(out/'catalog/catalog.parquet').read()
    assert np.array_equal(catalog['edge_id'].to_numpy(),ids)
    rng=np.random.default_rng(20260919);sample=set()
    for h in range(24):
        choices=ids[(a['timestamp'][ids]>=prep['start_us']+h*HOUR)&(a['timestamp'][ids]<prep['start_us']+(h+1)*HOUR)]
        if len(choices):
            sample.update(map(int,rng.choice(choices,min(16,len(choices)),replace=False)))
            sample.add(int(choices[np.argmin(a['timestamp'][choices])]))
            sample.add(int(choices[np.argmax(a['timestamp'][choices])]))
    sample_ids=np.array(sorted(sample),dtype=np.int64)
    engine=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],prep['end_us'])
    reference={};node_reference={}
    for i in sorted(sample,key=lambda i:(a['timestamp'][i],i)):
        r,audit=engine.target(i);reference[i]=(r,audit)
        for node in {int(a['src'][i]),int(a['dst'][i])}:node_reference[(node,int(a['timestamp'][i]))]=dict(engine.node(node))
    has_overlap=prep['date']=='2022-09-04'
    lo=prep['start_us']+12*HOUR if has_overlap else -1;hi=lo+HOUR if has_overlap else -1
    overlap=[];sample_seen=set();total=0;finite_checks=0;sample_R=0
    print(json.dumps(dict(stage='daily_verify',sample_transactions=len(sample),sample_node_times=len(node_reference))),flush=True)
    for part in merged['parts']:
        for batch in pq.ParquetFile(part['path']).iter_batches(batch_size=8192):
            tab=pa.Table.from_batches([batch]);eid=tab['edge_id'].to_numpy();ts=tab['timestamp'].cast(pa.int64()).to_numpy()
            if np.any(eid<0) or np.any(eid>=meta['rows']):raise AssertionError('Invalid edge index')
            if np.any(seen[eid]) or len(np.unique(eid))!=len(eid):raise AssertionError('Duplicate target')
            seen[eid]=True;pos=np.searchsorted(ids,eid)
            assert np.all(pos<len(ids)) and np.array_equal(ids[pos],eid)
            assert tab['transaction_id'].equals(catalog['transaction_id'].take(pa.array(pos)))
            assert np.array_equal(ts,a['timestamp'][eid]) and np.all((ts>=prep['start_us'])&(ts<prep['end_us']))
            assert np.array_equal(tab['full_6h'].to_numpy(),ts-6*HOUR>=meta['data_start_us'])
            assert np.array_equal(tab['full_day'].to_numpy(),ts-ts%DAY>=meta['data_start_us'])
            for f in R_FIELDS:
                col=tab[f['name']]
                if pa.types.is_floating(col.type):assert_finite_nullable(col)
                finite_checks+=len(col)
            mask=(ts>=lo)&(ts<hi)
            if mask.any():overlap.append(tab.filter(pa.array(mask)))
            for row in tab.filter(pa.array(np.isin(eid,sample_ids))).to_pylist():
                expected=reference[row['edge_id']][0]
                for name,value in expected.items():assert equal_value(row[name],value),(row['edge_id'],name,row[name],value)
                sample_seen.add(row['edge_id']);sample_R+=len(expected)
            total+=len(tab)
    assert total==len(ids)==prep['target_rows'] and seen.sum()==len(ids) and seen[ids].all() and sample_seen==sample
    overlap_result=None
    if has_overlap:
        actual=pa.concat_tables(overlap).sort_by([('transaction_id','ascending')])
        expected=pq.ParquetFile(old_hour/'target_features.parquet').read().sort_by([('transaction_id','ascending')])
        assert actual['transaction_id'].equals(expected['transaction_id']) and actual['timestamp'].equals(expected['timestamp'])
        for f in R_FIELDS:
            name=f['name'];x=actual[name];y=expected[name]
            if pa.types.is_floating(x.type):assert np.allclose(x.to_numpy(),y.to_numpy(),rtol=1e-12,atol=1e-12,equal_nan=True),name
            else:assert x.equals(y),name
        overlap_result=dict(start_us=lo,end_us=hi,R_rows=len(expected),R_values=len(expected)*54)
    # Archive-local edge IDs can change; comparisons above deliberately use stable transaction_id.
    old=pq.ParquetFile(old_nodes/'requested_nodes.parquet').read()
    if not has_overlap:old=old.slice(0,0)
    nf=[f['name'] for f in N_FIELDS]
    positions={(int(node),int(t)):j for j,(node,t) in enumerate(zip(old['node_id'].to_pylist(),old['cutoff_us'].to_pylist()))}
    old_matrix=np.column_stack([old[f].to_numpy() for f in nf]);node_seen=np.zeros(len(old),dtype=bool);node_sample_seen=set();sample_N=0
    partial_rows=0
    for part in account['parts']:
        file=pq.ParquetFile(part['path']);partial_rows+=file.metadata.num_rows
        for batch in file.iter_batches(batch_size=4096):
            tab=pa.Table.from_batches([batch]);t=tab['cutoff'].to_numpy();edge=tab['edge_id'].to_numpy()
            mask=((t>=lo)&(t<hi))|np.isin(edge,sample_ids)
            if not mask.any():continue
            for row in tab.filter(pa.array(mask)).to_pylist():
                n,_,_=pickle.loads(row['payload']);key=(row['node'],row['cutoff'])
                if lo<=row['cutoff']<hi:
                    p=positions[key];values=np.array([np.nan if n[f] is None else n[f] for f in nf])
                    assert np.allclose(values,old_matrix[p],rtol=1e-12,atol=1e-12,equal_nan=True),key
                    node_seen[p]=True
                if key in node_reference:
                    for f,value in node_reference[key].items():assert equal_value(n[f],value),(key,f)
                    if key not in node_sample_seen:sample_N+=len(n)
                    node_sample_seen.add(key)
    assert partial_rows==2*len(ids) and node_seen.all() and node_sample_seen==set(node_reference)
    if has_overlap:overlap_result.update(N_rows=len(old),N_values=len(old)*41,all_audits_equal=True)
    seen[:]=False
    old_audit=pq.ParquetFile(old_hour/'target_audit.parquet').read()
    if not has_overlap:old_audit=old_audit.slice(0,0)
    audit_expected={r['transaction_id']:r for r in old_audit.to_pylist()};audit_seen=set();sample_audit_seen=set();audit_rows=0
    for part in merged['parts']:
        file=pq.ParquetFile(part['audit_path']);audit_rows+=file.metadata.num_rows
        for batch in file.iter_batches(batch_size=4096):
            tab=pa.Table.from_batches([batch]);eid=tab['edge_id'].to_numpy()
            assert np.all((eid>=0)&(eid<meta['rows']))
            assert not np.any(seen[eid]) and len(np.unique(eid))==len(eid)
            seen[eid]=True;pos=np.searchsorted(ids,eid)
            assert np.all(pos<len(ids)) and np.array_equal(ids[pos],eid)
            assert tab['transaction_id'].equals(catalog['transaction_id'].take(pa.array(pos)))
            t=a['timestamp'][eid]
            mask=((t>=lo)&(t<hi))|np.isin(eid,sample_ids)
            for row in tab.filter(pa.array(mask)).to_pylist():
                if row['transaction_id'] in audit_expected:
                    e=audit_expected[row['transaction_id']]
                    for field in ['missing_reasons','ratio_support_scaled']:assert json.loads(row[field])==json.loads(e[field])
                    audit_seen.add(row['transaction_id'])
                if row['edge_id'] in reference:
                    e=reference[row['edge_id']][1]
                    for field in ['missing_reasons','ratio_support_scaled']:assert json.loads(row[field])==e[field]
                    sample_audit_seen.add(row['edge_id'])
    assert audit_rows==len(ids) and seen.sum()==len(ids) and seen[ids].all()
    assert len(audit_seen)==len(old_audit) and sample_audit_seen==sample
    # Supervision is read after all feature checks, for counts only.
    positive=label_rows=0
    for f in (archive/'supervision/targets'/f'transaction_date={prep["date"]}').glob('*.parquet'):
        labels=pq.ParquetFile(f).read(columns=['label'])['label'];positive+=pc.sum(labels).as_py();label_rows+=len(labels)
    assert label_rows==len(ids)
    result=dict(status='passed',full_day_rows=total,positive_rows=positive,self_transfer_rows=int(np.sum(a['src'][ids]==a['dst'][ids])),
        full_day_integrity='Every target exactly once; stable transaction ID and timestamp match archive; finite numeric outputs; two endpoint fragments and audit coverage',
        raw_R_values_checked_for_finiteness=finite_checks,
        overlap=overlap_result,
        stratified_reference=dict(seed=20260919,hours=24,transactions=len(sample),R_values=sample_R,node_times=len(node_reference),N_values=sample_N,all_audits_equal=True),
        verification_seconds=time.perf_counter()-began,verifier_peak_rss_bytes=peak_rss(),
        limitation='Not a full-day recomputation with the original slow algorithm; hourly stratified checks and full historical benchmark overlap on September 4 only',
        sample_edge_ids=sorted(sample))
    write_json(out/'verification.json',result);run['status']='verified';run['verification_seconds']=result['verification_seconds'];write_json(out/'daily_run.json',run)
    print(json.dumps({k:v for k,v in result.items() if k!='sample_edge_ids'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-hour',type=Path,default=Path('result/gnn_v1_dev'));p.add_argument('--old-nodes',type=Path,default=Path('result/gnn_benchmark_reference'))
    a=p.parse_args();verify(a.archive,a.output,a.old_hour,a.old_nodes)
