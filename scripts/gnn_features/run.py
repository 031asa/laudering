"""Compute the approved real-data hour and verify a sampled graph interface (CPU)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build import load_archive, edge_encoding, write_json
from .contract import HOUR, DAY, N_FIELDS, E_FIELDS, R_FIELDS, VERSION, encode_numeric
from .engine import FeatureEngine, money

def json_dump(value): return json.dumps(value,ensure_ascii=False,default=str,separators=(',',':'))

def feature_type(name):
    if '_sum_' in name and not any(s in name for s in ['paid_to','out_to']): return pa.decimal128(38,6)
    if '_count_' in name or '_unique_payers_' in name: return pa.int64()
    return pa.float64()

def sample_batch(engine,ids,seed=20260915):
    a=engine.arr; t=int(a['timestamp'][ids[0]])
    if not np.all(a['timestamp'][ids]==t): raise ValueError('Mixed cutoff batch')
    engine.at(t);rng=np.random.default_rng(seed)
    nodes=set(map(int,a['src'][ids]))|set(map(int,a['dst'][ids]));frontier=set(nodes);edges=set()
    for limit in [10,5]:
        next_nodes=set()
        for node in sorted(frontier):
            for direction in ['in','out']:
                picked=engine.neighbors(node,direction,limit,rng)
                edges.update(map(int,picked))
                next_nodes.update(map(int,a['src'][picked]));next_nodes.update(map(int,a['dst'][picked]))
        frontier=next_nodes;nodes.update(next_nodes)
    nodes=np.array(sorted(nodes),dtype=np.uint32);edges=np.array(sorted(edges),dtype=np.uint32)
    local={int(n):i for i,n in enumerate(nodes)}
    route=np.array([[local[int(s)] for s in a['src'][edges]],[local[int(d)] for d in a['dst'][edges]]],dtype=np.int64)
    target_route=np.array([[local[int(s)] for s in a['src'][ids]],[local[int(d)] for d in a['dst'][ids]]],dtype=np.int64)
    ctx=engine.context()
    x=np.array([encode_numeric(engine.node(n),N_FIELDS)+[float(ctx['full_'+w]) for w in ['1h','24h','72h']] for n in nodes],dtype=np.float32)
    past=edge_encoding({k:v[edges] for k,v in a.items() if k in ['timestamp','src','dst','paid_scaled','received_scaled','payment_format_cat','payment_currency_cat','receiving_currency_cat']},engine.bank)
    now=edge_encoding({k:v[ids] for k,v in a.items() if k in ['timestamp','src','dst','paid_scaled','received_scaled','payment_format_cat','payment_currency_cat','receiving_currency_cat']},engine.bank)
    enum=[f for f in E_FIELDS if f['transform']!='embedding']; ecat=[f for f in E_FIELDS if f['transform']=='embedding']
    edge_num=np.column_stack([past[f['name']] for f in enum]+[np.log1p((t-a['timestamp'][edges])/60_000_000)]).astype(np.float32)
    edge_cat=np.column_stack([past[f['name']] for f in ecat]).astype(np.int64)
    target_num=np.column_stack([now[f['name']] for f in enum]).astype(np.float32)
    target_cat=np.column_stack([now[f['name']] for f in ecat]).astype(np.int64)
    r=np.array([encode_numeric(engine.target(i)[0],R_FIELDS)+[float(ctx['full_'+w]) for w in ['6h','day']] for i in ids],dtype=np.float32)
    assert np.all(a['timestamp'][edges]<t) and np.all(a['timestamp'][edges]>=t-72*HOUR)
    assert np.all(a['src'][edges]!=a['dst'][edges])
    assert not set(map(int,ids))&set(map(int,edges))
    for tensor in [x,edge_num,target_num,r]: assert np.isfinite(tensor).all()
    paircount=Counter(zip(map(int,a['src'][edges]),map(int,a['dst'][edges])))
    tensors=dict(node_id=nodes,edge_id=edges,target_edge_id=ids,x_num=x,
        edge_index_in=route,edge_index_out=route[::-1].copy(),edge_attr_num=edge_num,edge_attr_cat=edge_cat,
        edge_label_index=target_route,target_base_num=target_num,target_base_cat=target_cat,target_attr=r)
    report=dict(cutoff_us=t,seed=seed,fanout_per_direction=[10,5],nodes=len(nodes),edges=len(edges),targets=len(ids),
        earliest_message_us=int(a['timestamp'][edges].min()) if len(edges) else None,
        latest_message_us=int(a['timestamp'][edges].max()) if len(edges) else None,
        parallel_account_pairs=sum(c>1 for c in paircount.values()),
        shapes={k:list(v.shape) for k,v in tensors.items()},labels_in_inputs=False,
        status='CPU graph data interface verified; no neural network forward/backward in this stage')
    return tensors,report

def run(archive,out,start,end):
    import psutil
    if out.exists() and any(out.iterdir()): raise ValueError('Output exists; choose new run directory')
    out.mkdir(parents=True,exist_ok=True); proc=psutil.Process();peak=proc.memory_info().rss;beg=time.perf_counter()
    meta,a,bank,vocab=load_archive(archive)
    lo=int(np.datetime64(start,'us').astype(np.int64));hi=int(np.datetime64(end,'us').astype(np.int64))
    if hi>int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)) or lo<meta['data_start_us']: raise ValueError('Target range outside archive coverage')
    ids=np.flatnonzero((a['timestamp']>=lo)&(a['timestamp']<hi)).astype(np.uint32)
    ids=ids[np.lexsort((ids,a['timestamp'][ids]))]
    if not len(ids): raise ValueError('No targets')
    needed=set(map(int,ids)); tx={}
    for f in sorted((archive/'graph/edges').rglob('*.parquet')):
        date=f.parent.name.split('=',1)[1]
        if not start[:10]<=date<=end[:10]:continue
        tab=pq.ParquetFile(f).read(columns=['edge_id','transaction_id','timestamp'])
        tt=tab['timestamp'].cast(pa.int64()).to_numpy();tab=tab.filter(pa.array((tt>=lo)&(tt<hi)))
        tx.update(zip(map(int,tab['edge_id'].to_pylist()),tab['transaction_id'].to_pylist()))
    if set(tx)!=needed or len(set(tx.values()))!=len(ids): raise ValueError('Target identity mismatch')
    engine=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)))
    schema=pa.schema([pa.field('transaction_id',pa.string(),False),pa.field('edge_id',pa.uint32(),False),
        pa.field('timestamp',pa.timestamp('us'),False)]+[pa.field(f['name'],feature_type(f['name']),True) for f in R_FIELDS]+
        [pa.field('full_6h',pa.bool_(),False),pa.field('full_day',pa.bool_(),False)])
    audit_schema=pa.schema([('transaction_id',pa.string()),('missing_reasons',pa.string()),('ratio_support_scaled',pa.string())])
    writer=pq.ParquetWriter(out/'target_features.parquet',schema,compression='zstd')
    aw=pq.ParquetWriter(out/'target_audit.parquet',audit_schema,compression='zstd')
    rows=[];audits=[];contexts={};samples={};missing=Counter();valid=Counter();n_unique=0;n_seconds=0;r_seconds=0
    start_ts=None;per_minute=[];last_tick=time.perf_counter()
    for position,i in enumerate(ids):
        i=int(i);t=int(a['timestamp'][i]);engine.at(t)
        if start_ts!=t:
            if start_ts is not None:
                per_minute.append(dict(cutoff_us=start_ts,wall_seconds=time.perf_counter()-minute_start,targets=minute_rows))
            start_ts=t;minute_start=time.perf_counter();minute_rows=0
        minute_rows+=1
        u,v=int(a['src'][i]),int(a['dst'][i]);q=time.perf_counter()
        for node in {u,v}:
            if node not in engine.nodes:n_unique+=1
            engine.node(node)
        n_seconds+=time.perf_counter()-q
        q=time.perf_counter();r,audit=engine.target(i);r_seconds+=time.perf_counter()-q
        ctx=audit.pop('context');contexts[t]=dict(cutoff_us=t,**ctx)
        for k,val in r.items():
            if val is None:missing[k]+=1
            else:valid[k]+=1
        rows.append(dict(transaction_id=tx[i],edge_id=i,timestamp=np.datetime64(t,'us').item(),**r,full_6h=ctx['full_6h'],full_day=ctx['full_day']))
        audits.append(dict(transaction_id=tx[i],missing_reasons=json_dump(audit['missing_reasons']),ratio_support_scaled=json_dump(audit['ratio_support_scaled'])))
        kind=('self_transfer' if u==v else 'cold_start' if engine.node(u)['n_in_all_count_72h']+engine.node(u)['n_out_all_count_72h']==0 and engine.state(u,'in').last() is None and engine.state(u,'out').last() is None else 'external_history')
        if kind not in samples:
            current={k:np.asarray([a[k][i]]) for k in ['timestamp','src','dst','paid_scaled','received_scaled','payment_format_cat','payment_currency_cat','receiving_currency_cat']}
            samples[kind]=dict(transaction_id=tx[i],edge_id=i,cutoff_us=t,src=u,dst=v,paid=money(a['paid_scaled'][i]),received=money(a['received_scaled'][i]),
                E={k:v[0].item() for k,v in edge_encoding(current,bank).items()},N_src=dict(engine.node(u)),N_dst=dict(engine.node(v)),R=r,audit=audit,history_context=ctx)
        if len(rows)>=1000:
            writer.write_table(pa.Table.from_pylist(rows,schema=schema));aw.write_table(pa.Table.from_pylist(audits,schema=audit_schema));rows.clear();audits.clear()
        peak=max(peak,proc.memory_info().rss)
        if time.perf_counter()-last_tick>15:
            print(json.dumps(dict(stage='features',done=position+1,total=len(ids),cutoff=str(np.datetime64(t,'us')),rss_mb=round(proc.memory_info().rss/1e6))),flush=True);last_tick=time.perf_counter()
    if rows:writer.write_table(pa.Table.from_pylist(rows,schema=schema));aw.write_table(pa.Table.from_pylist(audits,schema=audit_schema))
    writer.close();aw.close()
    per_minute.append(dict(cutoff_us=start_ts,wall_seconds=time.perf_counter()-minute_start,targets=minute_rows))
    pq.write_table(pa.Table.from_pylist(list(contexts.values())),out/'history_context.parquet',compression='zstd')
    feature_seconds=time.perf_counter()-beg
    # Only now read supervision; never pass it into the feature engine.
    labels={}
    for f in sorted((archive/'supervision/targets').rglob('*.parquet')):
        if not start[:10]<=f.parent.name.split('=',1)[1]<=end[:10]:continue
        tab=pq.ParquetFile(f).read()
        for row in tab.to_pylist():
            if row['edge_id'] in needed:labels[row['edge_id']]=row
    if set(labels)!=needed:raise ValueError('Supervision coverage mismatch')
    pq.write_table(pa.Table.from_pylist([labels[int(i)] for i in ids]),out/'target_supervision.parquet',compression='zstd')
    bank_cat=np.load(archive/'history_index/bank_cat.npy',mmap_mode='r')
    graph_reports=[]
    for kind,sample in samples.items():
        same=ids[a['timestamp'][ids]==sample['cutoff_us']]
        batch_ids=np.array([sample['edge_id']]+[int(i) for i in same if int(i)!=sample['edge_id']][:31],dtype=np.uint32)
        tensors,gr=sample_batch(engine,batch_ids)
        tensors['x_cat']=bank_cat[tensors['node_id']].astype(np.int64).reshape(-1,1)
        gr['shapes']['x_cat']=list(tensors['x_cat'].shape)
        np.savez_compressed(out/f'batch_{kind}.npz',**tensors)
        y=np.array([labels[int(i)]['label'] for i in batch_ids],dtype=np.float32)
        np.save(out/f'batch_{kind}_labels.npy',y)
        gr['label_counts']=dict(Counter(map(int,y)));gr['sample_kind']=kind;graph_reports.append(gr)
        sample['message_graph']=gr;sample['supervision']=labels[sample['edge_id']]
    if set(samples)!={'self_transfer','cold_start','external_history'}:raise ValueError('Real sample types missing: '+str(set(samples)))
    # Verify persisted values and row identities, not just successful writer completion.
    table=pq.ParquetFile(out/'target_features.parquet').read()
    assert len(table)==len(ids) and table['edge_id'].to_pylist()==ids.tolist()
    for kind,s in samples.items():
        row=table.filter(pa.array(table['edge_id'].to_numpy()==s['edge_id'])).to_pylist()[0]
        for f in R_FIELDS:assert row[f['name']]==s['R'][f['name']],(kind,f['name'])
    # Cache speed measured separately from uncached feature computation.
    engine.at(int(a['timestamp'][ids[-1]]));node=int(a['src'][ids[-1]]);engine.node(node)
    q=time.perf_counter()
    for _ in range(10000):engine.node(node)
    hit_seconds=time.perf_counter()-q
    peak=max(peak,proc.memory_info().rss)
    report=dict(version=VERSION,status='passed',target_start=start,target_end_exclusive=end,target_rows=len(ids),
        target_positive_rows=sum(row['label'] for row in labels.values()),self_transfer_rows=int(np.sum(a['src'][ids]==a['dst'][ids])),
        history_rows=meta['rows'],history_start_us=meta['data_start_us'],history_end_exclusive=meta['end_exclusive'],
        cutoff_count=len(contexts),unique_endpoint_time_queries=n_unique,node_query_seconds=n_seconds,r_query_seconds=r_seconds,
        feature_run_seconds=feature_seconds,total_seconds=time.perf_counter()-beg,cache_10000_hits_seconds=hit_seconds,
        peak_rss_sampled_bytes=peak,disk_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
        missing_counts=dict(missing),valid_counts=dict(valid),all_54_fields_computed=True,per_minute=per_minute,
        graph_batches=graph_reports,scope='One approved training hour; complete available prior history; all labels retained',
        limits=['Exact indexed queries are a development baseline, not a full-scale throughput claim',
                'Node features cached only within cutoff; no persistent snapshot table',
                'No GNN forward/backward or model training in this stage',
                'Train-only vocabulary uses frozen Sep 1–10; no validation/test feature discovery',
                'Memory is sampled process RSS, not OS peak or VRAM'])
    write_json(out/'samples.json',samples);write_json(out/'verification.json',report)
    print(json_dump(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',default='2022-09-04T12:00:00');p.add_argument('--end',default='2022-09-04T13:00:00');a=p.parse_args()
    run(a.archive,a.output,a.start,a.end)
