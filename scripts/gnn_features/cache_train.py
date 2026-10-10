"""Train-only, resumable Parquet cache. One bounded block, four CPU workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import msvcrt
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build import load_archive
from .contract import HOUR, N_FIELDS, encode_numeric
from .engine import FeatureEngine
from .rolling import RollingEngine
from .loader import TemporalBatchBuilder, TargetDay
from .hybrid_cache_trial import COMPLEX, ENCODED_NAMES, complex_values, pack_direction, bulk_direction, encode_columns, publish
from .train import training_spec

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'data/gnn_v1_train'
FEATURES=ROOT/'result/gnn_train_v1'
DEFAULT_OUT=ROOT/'data/gnn_train_cache_parquet'
PLAN_SCHEMA=pa.schema([('batch_id',pa.int32()),('cutoff_us',pa.int64())]+[(n,pa.list_(pa.int64())) for n in ['targets','nodes','edges']])


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,value):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.int64).tobytes()).hexdigest()


def clear_work(out):
    root=Path(out).resolve();work=root/'_work'/'active'
    # Delete only the fixed, uncommitted scratch block under this output root.
    if work.exists():
        resolved=work.resolve()
        if work.is_symlink() or resolved!=work or not resolved.is_relative_to(root):
            raise ValueError('Unsafe scratch path')
        shutil.rmtree(work)


def chunks(day,limit=512):
    """Preserve exact-timestamp batch boundaries; never cross a calendar hour."""
    if limit<1:raise ValueError('Invalid block limit')
    for hour,group in itertools.groupby(day.batches(64),lambda ids:int(day.times[np.searchsorted(day.ids,ids[0])])//HOUR):
        block=[];index=0
        for ids in group:
            block.append(ids)
            if len(block)==limit:
                yield hour,index,block;index+=1;block=[]
        if block:yield hour,index,block


def init_worker():
    global WORKER
    WORKER=load_archive(ARCHIVE)


def complex_part(work,part):
    meta,a,bank,vocab=WORKER
    table=pq.read_table(Path(work)/'keys.parquet')
    keys=np.column_stack([table[n].to_numpy() for n in ['node_id','cutoff_us']])
    positions=np.flatnonzero(keys[:,0]%4==part)
    values=np.empty((len(positions),len(COMPLEX)),dtype=np.float64)
    previous=None;started=last=time.perf_counter()
    for j,p in enumerate(positions):
        node,t=map(int,keys[p])
        if node!=previous:
            engine=RollingEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)))
            previous=node
        engine.at(t);r=complex_values(engine,node)
        values[j]=[np.nan if r[n] is None else r[n] for n in COMPLEX]
        if time.perf_counter()-last>30:
            print(json.dumps(dict(stage='complex',part=part,done=j+1,total=len(positions))),flush=True);last=time.perf_counter()
    path=Path(work)/f'complex_{part}.parquet'
    pq.write_table(pa.table(dict(position=positions,**{n:values[:,i] for i,n in enumerate(COMPLEX)})),path,compression='zstd')
    return dict(part=part,rows=len(positions),seconds=time.perf_counter()-started)


def validate_existing(folder,config_hash,targets):
    m=read(folder/'manifest.json')
    if m['status']!='verified' or m['config_sha256']!=config_hash or m['target_ids_sha256']!=array_sha(targets) or m['targets']!=len(targets):
        raise ValueError('Existing block identity/config mismatch: '+str(folder))
    for name,digest in m['file_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('Existing Parquet checksum mismatch: '+str(folder/name))
    return m


def compute_block(work,batches,keys_end,a,bank,meta,vocab,builder,pool,update):
    started=time.perf_counter();rows=[];requests=[]
    update(stage='sampling')
    for i,ids in enumerate(batches):
        nodes,edges=builder.sample(ids);t=int(a['timestamp'][ids[0]])
        if not np.all(a['timestamp'][ids]==t):raise ValueError('Mixed timestamp batch')
        if not np.all((a['timestamp'][edges]<t)&(a['timestamp'][edges]>=t-72*HOUR)) or np.any(a['src'][edges]==a['dst'][edges]):
            raise ValueError('Invalid historical graph')
        if np.any(np.diff(nodes)<=0) or np.any(np.diff(edges)<=0):raise ValueError('Duplicate graph identities')
        if not np.isin(np.r_[a['src'][ids],a['dst'][ids],a['src'][edges],a['dst'][edges]],nodes).all():
            raise ValueError('Incomplete graph endpoints')
        rows.append(dict(batch_id=i,cutoff_us=t,targets=ids.tolist(),nodes=nodes.tolist(),edges=edges.tolist()))
        requests.append(np.column_stack([nodes,np.full(len(nodes),t,dtype=np.int64)]))
        if (i+1)%100==0:update(stage='sampling',sampled_batches=i+1)
    keys=np.unique(np.concatenate(requests),axis=0);del requests
    pq.write_table(pa.Table.from_pylist(rows,schema=PLAN_SCHEMA),work/'plans.parquet',compression='zstd',compression_level=3)
    pq.write_table(pa.table(dict(node_id=keys[:,0],cutoff_us=keys[:,1])),work/'keys.parquet',compression='zstd')
    sampling=time.perf_counter()-started;began=time.perf_counter()
    update(stage='node_features',node_states=len(keys))
    pending=[pool.submit(complex_part,str(work),part) for part in range(4)]
    raw={};origin=meta['data_start_us']-72*HOUR
    for d in ['in','out']:
        pack=pack_direction(a,bank,keys,d,origin,keys_end,vocab['payment_format'].index('ACH')+2)
        raw.update(bulk_direction(pack,keys,d,origin,keys_end,'cpu'));del pack
    parts=[f.result() for f in pending]
    matrix=np.empty((len(keys),len(COMPLEX)),dtype=np.float64);seen=np.zeros(len(keys),dtype=bool)
    for part in range(4):
        table=pq.read_table(work/f'complex_{part}.parquet');positions=table['position'].to_numpy()
        if np.any(positions<0) or np.any(positions>=len(keys)) or len(np.unique(positions))!=len(positions) or seen[positions].any():
            raise ValueError('Invalid complex feature partition')
        matrix[positions]=np.column_stack([table[n].to_numpy() for n in COMPLEX]);seen[positions]=True
    if not seen.all():raise ValueError('Incomplete complex features')
    raw.update(dict(zip(COMPLEX,matrix.T)));values=encode_columns(raw,keys,meta['data_start_us'])
    del raw,matrix,table
    publish(work,keys,values);node_seconds=time.perf_counter()-began
    update(stage='verification');began=time.perf_counter()
    # Exact full readback, then independent original-engine spot checks across the block.
    table=pq.read_table(work/'nodes.parquet')
    if table.num_rows!=len(keys) or table.column_names!=['node_id','cutoff_us']+ENCODED_NAMES:raise ValueError('Invalid node schema')
    for i,n in enumerate(['node_id','cutoff_us']):np.testing.assert_array_equal(table[n].to_numpy(),keys[:,i])
    for i,n in enumerate(ENCODED_NAMES):np.testing.assert_array_equal(table[n].to_numpy(),values[:,i])
    if not np.isin(values[:,41:],[0,1]).all():raise ValueError('Invalid validity/coverage flag')
    if pq.read_table(work/'plans.parquet').to_pylist()!=rows:raise ValueError('Plan roundtrip differs')
    reference=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)))
    rng=np.random.default_rng(20260920)
    selected=np.unique(np.r_[np.linspace(0,len(keys)-1,min(128,len(keys)),dtype=int),rng.choice(len(keys),min(128,len(keys)),replace=False)])
    for p in selected:
        node,t=map(int,keys[p]);reference.at(t)
        expected=np.array(encode_numeric(reference.node(node),N_FIELDS)+[float(reference.context()['full_'+w]) for w in ['1h','24h','72h']],dtype=np.float32)
        np.testing.assert_allclose(values[p],expected,rtol=1e-6,atol=1e-6)
        reference.states.clear();reference.nodes.clear()
    # Scratch files are all Parquet, removed only after successful verification.
    for name in ['keys.parquet']+[f'complex_{p}.parquet' for p in range(4)]:
        (work/name).unlink()
    return dict(targets=sum(map(len,batches)),batches=len(batches),node_states=len(keys),
                first_cutoff_us=int(keys[:,1].min()),last_cutoff_us=int(keys[:,1].max()),
                sampling_seconds=sampling,node_seconds=node_seconds,verification_seconds=time.perf_counter()-began,
                sampled_reference_states=len(selected),parquet_bytes=sum((work/n).stat().st_size for n in ['nodes.parquet','plans.parquet']),
                file_sha256={n:sha(work/n) for n in ['nodes.parquet','plans.parquet']},parts=parts)


def run(out,max_blocks=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'process.lock').open('a+b');lock.seek(0)
    if (out/'process.lock').stat().st_size==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=time.perf_counter();state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),completed_blocks=0,verified_targets=0,parquet_bytes=0)
    def update(**fields):
        state.update(fields);state.update(updated_at=datetime.now(timezone.utc).isoformat(),invocation_seconds=time.perf_counter()-started)
        save(out/'status.json',state)
    try:
        update(stage='preflight')
        dates,end,expected,positive=training_spec(read(ROOT/'config/dataset_split.json'))
        if dates!=[f'2022-09-{i:02d}' for i in range(1,11)] or expected!=20318992:raise ValueError('Frozen Train scope changed')
        paths=[ROOT/'config/dataset_split.json',ARCHIVE/'build.json']
        paths += [Path(__file__).with_name(n) for n in ['cache_train.py','hybrid_cache_trial.py','build.py','engine.py','rolling.py','contract.py','loader.py']]
        for day in dates:
            folder=FEATURES/f'transaction_date={day}'
            if read(folder/'daily_run.json')['status']!='verified' or read(folder/'verification.json')['status']!='passed':raise ValueError('Unverified source day')
            if read(folder/'run_config.json')['archive_manifest_sha256']!=sha(ARCHIVE/'build.json'):raise ValueError('Source archive mismatch')
            paths += [folder/n for n in ['target_edge_ids.npy','merge_phase.json','verification.json','run_config.json']]
        config=dict(version=1,dates=dates,end_exclusive=end,expected_targets=expected,batch_size=64,max_batches_per_block=512,
                    workers=4,seed=20260920,fanout=[10,5],history='[t-72h,t), latest-event gaps use all prior history',
                    runtime=dict(python=sys.version,numpy=np.__version__,pyarrow=pa.__version__),
                    format='Parquet ZSTD; encoded N float32 with validity and coverage columns; fixed sampling graph plans',
                    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths})
        if (out/'config.json').exists():
            if read(out/'config.json')!=config:raise ValueError('Cache code/input configuration changed; explicit impact review required')
        else:save(out/'config.json',config)
        config_hash=sha(out/'config.json')
        meta,a,bank,vocab=load_archive(ARCHIVE)
        if meta['rows']!=expected or meta['end_exclusive']!=end:raise ValueError('Archive outside frozen Train')
        engine=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(end,'us').astype(np.int64)))
        adj={d+'_'+s:np.load(ARCHIVE/f'graph/adj_index/{d}_{s}.npy',mmap_mode='r') for d in ['in','out'] for s in ['ptr','edge_ids']}
        builder=TemporalBatchBuilder(engine,adj,None,None)
        completed=[];new_blocks=0;update(expected_targets=expected,dates=dates)
        with ProcessPoolExecutor(max_workers=4,initializer=init_worker) as pool:
            for date in dates:
                folder=FEATURES/f'transaction_date={date}';day=TargetDay.__new__(TargetDay)
                day.ids=np.load(folder/'target_edge_ids.npy').astype(np.int64)
                day.times=np.asarray(a['timestamp'][day.ids]);day.order=np.lexsort((day.ids,day.times))
                lo=int(np.datetime64(date,'us').astype(np.int64));hi=lo+24*HOUR
                if np.any(np.diff(day.ids)<=0) or not np.all((day.times>=lo)&(day.times<hi)) or len(day.ids)!=read(folder/'verification.json')['full_day_rows']:
                    raise ValueError('Invalid day targets')
                daily_count=0
                for hour,index,batches in chunks(day):
                    rel=f'transaction_date={date}/hour={hour%24:02d}/block={index:04d}'
                    final=out/rel;targets=np.concatenate(batches)
                    update(current_block=rel,stage='resume_check',block_targets=len(targets),sampled_batches=0)
                    if final.exists():m=validate_existing(final,config_hash,targets)
                    else:
                        if shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; stopping before next block')
                        clear_work(out);work=out/'_work'/'active';work.mkdir(parents=True)
                        m=compute_block(work,batches,(hour+1)*HOUR,a,bank,meta,vocab,builder,pool,update)
                        m.update(status='verified',config_sha256=config_hash,target_ids_sha256=array_sha(targets),block=rel,
                                 verified_at=datetime.now(timezone.utc).isoformat(),reference_note='Full readback and graph integrity; up to 256 original-engine node states per block, not all-state recomputation.')
                        save(work/'manifest.json',m);final.parent.mkdir(parents=True,exist_ok=True)
                        if work.resolve()!=out/'_work'/'active' or not final.resolve().is_relative_to(out):raise ValueError('Unsafe publish path')
                        os.rename(work,final);new_blocks+=1
                    daily_count+=m['targets'];completed.append({k:m[k] for k in ['block','targets','node_states','parquet_bytes']})
                    update(stage='block_verified',completed_blocks=len(completed),verified_targets=sum(x['targets'] for x in completed),parquet_bytes=sum(x['parquet_bytes'] for x in completed))
                    print(json.dumps(dict(stage='verified',block=rel,targets=m['targets'],verified_targets=state['verified_targets'],seconds=m.get('node_seconds'))),flush=True)
                    gc.collect()
                    if max_blocks is not None and new_blocks>=max_blocks:
                        update(status='stopped_after_requested_limit');return
                if daily_count!=len(day.ids):raise ValueError('Incomplete day coverage')
                save(out/f'transaction_date={date}'/'manifest.json',dict(status='verified',targets=daily_count,blocks=[x for x in completed if x['block'].startswith(f'transaction_date={date}/')]))
        if sum(x['targets'] for x in completed)!=expected:raise ValueError('Incomplete Train coverage')
        save(out/'manifest.json',dict(status='completed',targets=expected,blocks=completed,config_sha256=config_hash,
             feature_tables=str(FEATURES),edge_and_identity_tables=str(ARCHIVE),node_columns=ENCODED_NAMES))
        clear_work(out);update(status='completed',stage='complete',current_block=None)
    except BaseException as exc:
        update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUT);p.add_argument('--max-blocks',type=int)
    args=p.parse_args()
    if args.max_blocks is not None and args.max_blocks<1:p.error('--max-blocks must be positive')
    run(args.output,args.max_blocks)
