"""Bounded CPU/bulk-CPU/CUDA experiment; does not modify frozen feature engines."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build import load_archive, write_json
from .contract import HOUR, N_FIELDS
from .rolling import RollingEngine
from .prepare_hour import node_worker, signature

SIMPLE_SUFFIXES = ['all_count_1h', 'all_count_24h', 'all_count_72h',
                   'ext_count_24h', 'ext_count_72h', 'burst_1h_vs_23h',
                   'last_all_gap_min', 'cross_bank_share_24h', 'ach_count_24h',
                   'ach_share_24h', 'midday_share_72h']
SIMPLE = [f'n_{d}_{s}' for d in ['in', 'out'] for s in SIMPLE_SUFFIXES] + ['n_in_last_ext_gap_min']
COMPLEX = [f['name'] for f in N_FIELDS if f['name'] not in SIMPLE]
ENCODED_NAMES = [f['name']+'__encoded' for f in N_FIELDS] + [f['name']+'__valid' for f in N_FIELDS if f['nullable']] + ['full_1h','full_24h','full_72h']


def pack_direction(a, bank, keys, direction, origin, end, ach):
    span = end-origin+1
    if int(keys[:, 0].max()) > (np.iinfo(np.int64).max-span)//span:
        raise ValueError('Packed key overflow')
    own, cp = ('dst','src') if direction=='in' else ('src','dst')
    selected = np.zeros(len(bank), dtype=bool)
    selected[keys[:,0]] = True
    ids = a[direction+'_order']
    ids = ids[(a['timestamp'][ids]<end) & selected[a[own][ids]]]
    ts = a['timestamp'][ids]
    codes = a[own][ids].astype(np.int64)*span + (ts-origin)
    if np.any(np.diff(codes)<0):
        raise ValueError('History must be ordered by account and timestamp')
    ext = a[own][ids] != a[cp][ids]
    return dict(codes=codes, ext=ext,
                cross=ext & (bank[a[own][ids]] != bank[a[cp][ids]]),
                ach=a['payment_format_cat'][ids]==ach,
                midday=((ts//HOUR)%24>=11) & ((ts//HOUR)%24<17))


def bulk_direction(pack, keys, direction, origin, end, device):
    """Int64 boundary/prefix counts; float64 ratios, then CPU contract encoding."""
    span=end-origin+1
    codes=pack['codes']; q=keys[:,0]*span+(keys[:,1]-origin)
    lower=np.column_stack([keys[:,0]*span + np.maximum(keys[:,1]-w*HOUR-origin, 0) for w in [1,24,72]])
    raw={}
    if device=='cuda':
        import torch
        c=torch.as_tensor(codes, device='cuda'); query=torch.as_tensor(q,device='cuda')
        low=torch.as_tensor(lower,device='cuda')
        hi=torch.searchsorted(c,query,right=False); lo=torch.searchsorted(c,low.reshape(-1),right=False).reshape(-1,3)
        count=(hi[:,None]-lo).to(torch.float64)
        def interval(flag, col):
            p=torch.cat([torch.zeros(1,dtype=torch.int64,device='cuda'),torch.cumsum(torch.as_tensor(flag,device='cuda'),0,dtype=torch.int64)])
            return (p[hi]-p[lo[:,col]]).to(torch.float64)
        def divide(x,y):
            return torch.where(y>0,x/y,torch.full_like(x,float('nan')))
        def gap(history):
            if not len(history):return torch.full((len(q),),float('nan'),device='cuda',dtype=torch.float64)
            ix=torch.searchsorted(history,query,right=False)-1
            last=history[ix.clamp_min(0)]
            valid=(ix>=0) & (last//span==torch.as_tensor(keys[:,0],device='cuda'))
            return torch.where(valid,(query-last).to(torch.float64)/60_000_000,float('nan'))
        ext24=interval(pack['ext'],1);ext72=interval(pack['ext'],2)
        cross=interval(pack['cross'],1);ach=interval(pack['ach'],1);midday=interval(pack['midday'],2)
        vals=[count[:,0],count[:,1],count[:,2],ext24,ext72,(count[:,0]+1)/((count[:,1]-count[:,0])/23+1),
              gap(c),divide(cross,ext24),ach,divide(ach,count[:,1]),divide(midday,count[:,2])]
        if direction=='in':vals.append(gap(c[torch.as_tensor(pack['ext'],device='cuda')]))
        matrix=torch.stack(vals,1).cpu().numpy()
    else:
        hi=np.searchsorted(codes,q,side='left');lo=np.searchsorted(codes,lower,side='left')
        count=(hi[:,None]-lo).astype(np.float64)
        def interval(flag,col):
            p=np.r_[np.int64(0),np.cumsum(flag,dtype=np.int64)]
            return (p[hi]-p[lo[:,col]]).astype(np.float64)
        def divide(x,y):
            return np.divide(x,y,out=np.full_like(x,np.nan),where=y>0)
        def gap(history):
            if not len(history):return np.full(len(q),np.nan)
            ix=np.searchsorted(history,q,side='left')-1; last=history[np.maximum(ix,0)]
            return np.where((ix>=0)&(last//span==keys[:,0]),(q-last)/60_000_000,np.nan)
        ext24=interval(pack['ext'],1);ext72=interval(pack['ext'],2)
        cross=interval(pack['cross'],1);ach=interval(pack['ach'],1);midday=interval(pack['midday'],2)
        vals=[count[:,0],count[:,1],count[:,2],ext24,ext72,(count[:,0]+1)/((count[:,1]-count[:,0])/23+1),
              gap(codes),divide(cross,ext24),ach,divide(ach,count[:,1]),divide(midday,count[:,2])]
        if direction=='in':vals.append(gap(codes[pack['ext']]))
        matrix=np.column_stack(vals)
    names=[f'n_{direction}_{s}' for s in SIMPLE_SUFFIXES]
    if direction=='in':names.append('n_in_last_ext_gap_min')
    return dict(zip(names,matrix.T))


def complex_values(engine,node):
    r={}
    for d in ['in','out']:
        s=engine.state(node,d);p='n_'+d+'_'
        cp=s.counts('cp','24h');banks=s.counts('bank','24h')
        clock=s.counts('hour','72h');total=clock.total
        vals=dict(unique_cp_24h=len(cp),cp_hhi_24h=cp.hhi(),
                  active_hours_24h=len(s.counts('actual_hour','24h')),
                  active_days_72h=len(s.counts('actual_day','72h')),
                  bank_unique_24h=len(banks),bank_hhi_24h=banks.hhi(),
                  hour_entropy_72h=-sum((v/total)*math.log(v/total) for v in clock.values()) if total else None)
        for name in ['hour','minute']:
            freq=s.counts('actual_'+name,'72h')
            vals['peak_'+name+'_share_72h']=max(freq.values(),default=0)/total if total else None
        r.update({p+k:v for k,v in vals.items()})
    assert set(r)==set(COMPLEX)
    return r


def complex_worker(args):
    archive,source,out,part=args
    keys=np.load(Path(source)/'node_keys.npy',mmap_mode='r')
    positions=np.flatnonzero(keys[:,0]%4==part)
    meta,a,bank,vocab=load_archive(Path(archive))
    values=np.lib.format.open_memmap(Path(out)/f'complex_{part}.npy',mode='w+',dtype=np.float64,shape=(len(positions),len(COMPLEX)))
    np.save(Path(out)/f'positions_{part}.npy',positions)
    prev=None;started=last=time.perf_counter()
    for j,p in enumerate(positions):
        node,t=map(int,keys[p])
        if node!=prev:
            engine=RollingEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)))
            prev=node
        engine.at(t);r=complex_values(engine,node)
        values[j]=[np.nan if r[n] is None else r[n] for n in COMPLEX]
        if time.perf_counter()-last>30:
            print(json.dumps(dict(stage='complex',part=part,done=j+1,total=len(positions))),flush=True);last=time.perf_counter()
    values.flush()
    return dict(partition=part,rows=len(positions),seconds=time.perf_counter()-started)


def encode_columns(raw,keys,start):
    cols=[];masks=[]
    for f in N_FIELDS:
        v=np.asarray(raw[f['name']],dtype=np.float64);valid=~np.isnan(v)
        if not f['nullable'] and not valid.all():raise ValueError('Unexpected null '+f['name'])
        x=np.where(valid,v,0.)
        if f['transform']=='log1p':x=np.log1p(x)
        elif f['transform']=='entropy_ln24':x=x/math.log(24)
        cols.append(x)
        if f['nullable']:masks.append(valid)
    result=np.column_stack(cols+masks+[(keys[:,1]-start)>=w*HOUR for w in [1,24,72]]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite model values')
    return result


def publish(out,keys,values):
    table=pa.table(dict(node_id=keys[:,0],cutoff_us=keys[:,1],**{n:values[:,i] for i,n in enumerate(ENCODED_NAMES)}))
    pq.write_table(table,out/'nodes.parquet',compression='zstd',compression_level=3,row_group_size=65536)


def shared_graph_stage(out,source,m):
    """Measure the unchanged graph stage once; add it explicitly to each mode."""
    from .loader import TemporalLoader
    started=time.perf_counter()
    loader=TemporalLoader(m['archive'],m['result_root'],m['start'][:10],edge_cache='data/gnn_tensor_cache_v1')
    load=time.perf_counter()-started
    lo=int(np.datetime64(m['start'],'us').astype(np.int64));hi=int(np.datetime64(m['end'],'us').astype(np.int64))
    rows=[];requests=[];began=time.perf_counter()
    for ids in loader.day.batches(64):
        t=int(loader.builder.a['timestamp'][ids[0]])
        if not lo<=t<hi:continue
        nodes,edges=loader.builder.sample(ids)
        rows.append(dict(batch_id=len(rows),targets=ids.tolist(),nodes=nodes.tolist(),edges=edges.tolist()))
        requests.append(np.column_stack([nodes,np.full(len(nodes),t,dtype=np.int64)]))
        if len(rows)%100==0:print(json.dumps(dict(stage='shared_graph',batches=len(rows))),flush=True)
    keys=np.unique(np.concatenate(requests),axis=0)
    schema=pa.schema([('batch_id',pa.int32())]+[(k,pa.list_(pa.int64())) for k in ['targets','nodes','edges']])
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),out/'shared_plans.parquet',compression='zstd',compression_level=3)
    plan_seconds=time.perf_counter()-began;total=time.perf_counter()-started
    assert len(rows)==m['batches']
    np.testing.assert_array_equal(keys,np.load(source/'node_keys.npy'))
    for i,row in enumerate(rows):
        with np.load(source/'plans'/('%05d.npz'%i),allow_pickle=False) as old:
            for k in ['targets','nodes','edges']:np.testing.assert_array_equal(row[k],old[k])
    return dict(loader_initialization_seconds=load,sampling_dedup_and_parquet_seconds=plan_seconds,
                total_seconds=total,batches=len(rows),exact_plans_and_node_keys=True,
                parquet_bytes=(out/'shared_plans.parquet').stat().st_size)


def run(out):
    source=Path('result/gnn_hour_cache_trial');m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if m['status']!='passed' or m['signature']!=signature(m['archive'],m['result_root'],m['start'],m['end'],m['batch_size']):
        raise ValueError('Source must be verified and current')
    out.mkdir(exist_ok=False,parents=True)
    keys=np.load(source/'node_keys.npy',mmap_mode='r');reference=np.load(source/'nodes.npy',mmap_mode='r')
    meta,a,bank,vocab=load_archive(Path(m['archive']))
    end=int(np.datetime64(m['end'],'us').astype(np.int64));origin=meta['data_start_us']-72*HOUR
    # CUDA initialized once; report separately rather than hiding the cold start.
    tick=time.perf_counter();import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);torch.cuda.init();torch.cuda.synchronize()
    startup=time.perf_counter()-tick
    reports=[]
    for mode in ['original_cpu','bulk_cpu','hybrid_gpu']:
        folder=out/mode;folder.mkdir();tick=time.perf_counter();timings={}
        if mode=='original_cpu':
            np.save(folder/'node_keys.npy',keys)
            with ProcessPoolExecutor(max_workers=4) as pool:
                parts=list(pool.map(node_worker,[(m['archive'],str(folder),p,4) for p in range(4)]))
            values=np.empty(reference.shape,dtype=np.float32)
            for p in range(4):
                ix=np.load(folder/f'positions_{p}.npy');values[ix]=np.load(folder/f'nodes_{p}.npy',mmap_mode='r')
        else:
            raw={};torch.cuda.reset_peak_memory_stats()
            with ProcessPoolExecutor(max_workers=4) as pool:
                pending=[pool.submit(complex_worker,(m['archive'],str(source),str(folder),p)) for p in range(4)]
                for d in ['in','out']:
                    began=time.perf_counter();pack=pack_direction(a,bank,keys,d,origin,end,vocab['payment_format'].index('ACH')+2)
                    timings[d+'_pack_seconds']=time.perf_counter()-began
                    began=time.perf_counter();raw.update(bulk_direction(pack,keys,d,origin,end,'cuda' if mode=='hybrid_gpu' else 'cpu'))
                    timings[d+'_bulk_including_transfer_seconds']=time.perf_counter()-began
                    del pack;gc.collect()
                parts=[f.result() for f in pending]
            complex_matrix=np.empty((len(keys),len(COMPLEX)),dtype=np.float64)
            for p in range(4):
                ix=np.load(folder/f'positions_{p}.npy');complex_matrix[ix]=np.load(folder/f'complex_{p}.npy',mmap_mode='r')
            raw.update(dict(zip(COMPLEX,complex_matrix.T)))
            values=encode_columns(raw,keys,meta['data_start_us'])
            timings['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated() if mode=='hybrid_gpu' else None
            del raw,complex_matrix
        prepared=time.perf_counter()-tick
        began=time.perf_counter();publish(folder,keys,values);write_seconds=time.perf_counter()-began
        total=time.perf_counter()-tick
        began=time.perf_counter()
        np.testing.assert_array_equal(np.isnan(values),np.isnan(reference))
        np.testing.assert_allclose(values,reference,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(values[:,41:],reference[:,41:])
        decoded=pq.read_table(folder/'nodes.parquet')
        np.testing.assert_array_equal(decoded['node_id'].to_numpy(),keys[:,0])
        np.testing.assert_array_equal(decoded['cutoff_us'].to_numpy(),keys[:,1])
        for i,n in enumerate(ENCODED_NAMES):np.testing.assert_array_equal(decoded[n].to_numpy(),values[:,i])
        report=dict(mode=mode,prepare_seconds=prepared,parquet_write_seconds=write_seconds,total_seconds=total,
                    verification_seconds=time.perf_counter()-began,parquet_bytes=(folder/'nodes.parquet').stat().st_size,
                    max_abs_difference=float(np.max(np.abs(values-reference))),timings=timings,parts=parts)
        write_json(folder/'benchmark.json',report);reports.append(report)
        print(json.dumps(report),flush=True)
        del values,decoded;gc.collect()
    result=dict(status='passed',start=m['start'],end=m['end'],targets=m['targets'],node_states=len(keys),
                simple_fields=SIMPLE,complex_fields=COMPLEX,cuda_cold_start_seconds=startup,
                python_torch=torch.__version__,gpu=torch.cuda.get_device_name(),runs=reports,
                source_signature=m['signature'],
                scope='Same full-hour verified graph requests. Timed from request keys through N computation, worker startup, transfers, merging and Parquet write. Shared archive open and graph sampling excluded; source sampling time not a new measurement. Four CPU workers each mode; one run per mode in fixed order; OS caches not cleared. No full-Train preprocessing or model training.')
    write_json(out/'benchmark.json',result)
    result['shared_graph_stage']=shared_graph_stage(out,source,m)
    for report in reports:
        report['combined_stage_seconds']=result['shared_graph_stage']['total_seconds']+report['total_seconds']+(startup if report['mode']=='hybrid_gpu' else 0)
    result['combined_timing_note']='Sum of separately measured shared graph preparation and each N pipeline, not three monolithic cold runs. Verification excluded. CUDA cold startup included only in hybrid combined time. One run each; small timing differences are inconclusive.'
    write_json(out/'benchmark.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
