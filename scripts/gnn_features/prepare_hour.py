"""Fixed one-hour graph plans plus deduplicated, account-ordered N cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .build import load_archive,write_json
from .contract import N_FIELDS,encode_numeric
from .loader import TemporalLoader
from .rolling import RollingEngine


def signature(archive,result_root,start,end,batch_size):
    archive=Path(archive);folder=Path(result_root)/('transaction_date='+start[:10])
    paths=[archive/'build.json',folder/'merge_phase.json',archive/'metadata/feature_manifest.json']
    paths += [Path(__file__).with_name(n) for n in ['loader.py','engine.py','rolling.py','contract.py','prepare_hour.py']]
    return dict(start=start,end=end,batch_size=batch_size,seed=20260920,fanout=[10,5],
        hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def node_worker(args):
    archive,out,part,parts=args;out=Path(out)
    meta,a,bank,vocab=load_archive(Path(archive));keys=np.load(out/'node_keys.npy',mmap_mode='r')
    positions=np.flatnonzero(keys[:,0]%parts==part);np.save(out/('positions_%d.npy'%part),positions)
    values=np.lib.format.open_memmap(out/('nodes_%d.npy'%part),mode='w+',dtype=np.float32,shape=(len(positions),63))
    started=time.perf_counter();last=started;previous=None;engine=None
    for j,p in enumerate(positions):
        node,t=map(int,keys[p])
        if node!=previous:
            engine=RollingEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)))
            previous=node
        engine.at(t);ctx=engine.context()
        values[j]=encode_numeric(engine.node(node),N_FIELDS)+[float(ctx['full_'+w]) for w in ['1h','24h','72h']]
        if time.perf_counter()-last>30:
            print(json.dumps(dict(stage='N_cache',partition=part,done=j+1,total=len(positions))),flush=True);last=time.perf_counter()
    if not np.isfinite(values).all():raise ValueError('Nonfinite node cache')
    values.flush()
    return dict(partition=part,rows=len(positions),seconds=time.perf_counter()-started)


class HourCache:
    def __init__(self,out,loader):
        self.out=Path(out);manifest=json.loads((self.out/'manifest.json').read_text(encoding='utf-8'))
        expected=signature(manifest['archive'],manifest['result_root'],manifest['start'],manifest['end'],manifest['batch_size'])
        if manifest['signature']!=expected:raise ValueError('Stale graph/N cache')
        self.manifest=manifest;self.loader=loader
        self.keys=np.load(self.out/'node_keys.npy',mmap_mode='r');self.values=np.load(self.out/'nodes.npy',mmap_mode='r')
        self.lo=int(np.datetime64(manifest['start'],'us').astype(np.int64));self.hi=int(np.datetime64(manifest['end'],'us').astype(np.int64));self.span=self.hi-self.lo
        if int(self.keys[:,0].max())>(np.iinfo(np.int64).max-self.span)//self.span:raise ValueError('Key encoding overflow')
        self.codes=self.keys[:,0]*self.span+(self.keys[:,1]-self.lo)
        if np.any(np.diff(self.codes)<=0):raise ValueError('Unsorted or duplicate node keys')

    def lookup(self,node,t):
        if not self.lo<=t<self.hi:raise KeyError('Cutoff outside prepared interval')
        key=node*self.span+(t-self.lo);p=int(np.searchsorted(self.codes,key))
        if p==len(self.codes) or self.codes[p]!=key:raise KeyError((node,t))
        return self.values[p]

    def batch(self,i):
        with np.load(self.out/'plans'/('%05d.npz'%i),allow_pickle=False) as f:
            ids=f['targets'];graph=(f['nodes'],f['edges'])
        p=self.loader.day.positions(ids)
        return self.loader.builder.build(ids,self.loader.day.r[p],self.loader.day.labels[p],prepared=graph,node_lookup=self.lookup)


def run(archive,result_root,out,start,end,batch_size=64):
    started=time.perf_counter()
    if out.exists():raise ValueError('Choose a new output directory')
    out.mkdir(parents=True);(out/'plans').mkdir()
    lo=int(np.datetime64(start,'us').astype(np.int64));hi=int(np.datetime64(end,'us').astype(np.int64))
    if hi-lo!=3_600_000_000 or start[:10]!=end[:10]:raise ValueError('This experiment is exactly one hour within a day')
    loader=TemporalLoader(archive,result_root,start[:10],edge_cache=Path('data/gnn_tensor_cache_v1'))
    startup=time.perf_counter()-started;began=time.perf_counter();requests=[];batches=0;targets=0;nodes_total=0
    for ids in loader.day.batches(batch_size):
        t=int(loader.builder.a['timestamp'][ids[0]])
        if not lo<=t<hi:continue
        nodes,edges=loader.builder.sample(ids)
        np.savez(out/'plans'/('%05d.npz'%batches),targets=ids,nodes=nodes,edges=edges)
        requests.append(np.column_stack([nodes,np.full(len(nodes),t,dtype=np.int64)]))
        batches+=1;targets+=len(ids);nodes_total+=len(nodes)
        if batches%100==0:print(json.dumps(dict(stage='plans',batches=batches,targets=targets)),flush=True)
    expected=int(np.sum((loader.day.times>=lo)&(loader.day.times<hi)))
    if targets!=expected or not targets:raise ValueError('Target coverage failed')
    keys=np.unique(np.concatenate(requests),axis=0);np.save(out/'node_keys.npy',keys);del requests
    plan_seconds=time.perf_counter()-began;began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(node_worker,[(str(archive.resolve()),str(out.resolve()),p,4) for p in range(4)]))
    values=np.lib.format.open_memmap(out/'nodes.npy',mode='w+',dtype=np.float32,shape=(len(keys),63))
    seen=np.zeros(len(keys),dtype=bool)
    for part in parts:
        p=np.load(out/('positions_%d.npy'%part['partition']));v=np.load(out/('nodes_%d.npy'%part['partition']),mmap_mode='r')
        if seen[p].any():raise ValueError('Duplicate node partition')
        values[p]=v;seen[p]=True
    if not seen.all():raise ValueError('Incomplete node cache')
    values.flush();del values;node_seconds=time.perf_counter()-began
    manifest=dict(status='prepared_pending_acceptance',archive=str(archive.resolve()),result_root=str(result_root.resolve()),start=start,end=end,batch_size=batch_size,
        signature=signature(archive,result_root,start,end,batch_size),targets=targets,batches=batches,node_occurrences=nodes_total,unique_node_times=len(keys),
        startup_seconds=startup,plan_seconds=plan_seconds,node_prepare_seconds=node_seconds,parts=parts)
    write_json(out/'manifest.json',manifest);cache=HourCache(out,loader)
    # Evenly spaced real batches across the full hour: independent direct-query comparison.
    selected=np.unique(np.linspace(0,batches-1,12,dtype=int));comparisons=[];began=time.perf_counter()
    for i in selected:
        with np.load(out/'plans'/('%05d.npz'%int(i)),allow_pickle=False) as f:ids=f['targets']
        loader.builder.clear_cache();old,y,old_time=loader.batch(ids)
        loader.builder.clear_cache();tick=time.perf_counter();new,z,_=cache.batch(int(i));elapsed=time.perf_counter()-tick
        for name in old:
            if name=='x_num':np.testing.assert_allclose(new[name],old[name],rtol=1e-6,atol=1e-6)
            else:np.testing.assert_array_equal(new[name],old[name])
        np.testing.assert_array_equal(y,z)
        comparisons.append(dict(batch=int(i),targets=len(ids),nodes=len(new['node_id']),baseline_seconds=old_time['seconds'],cached_seconds=elapsed))
        print(json.dumps(dict(stage='comparison',**comparisons[-1])),flush=True)
    verify_seconds=time.perf_counter()-began
    # Sequential full-hour reads; only one batch retained at a time.
    loader.builder.clear_cache();began=time.perf_counter();seen_targets=[]
    for i in range(batches):
        b,y,r=cache.batch(i);seen_targets.append(b['target_edge_id'])
    read_seconds=time.perf_counter()-began
    ids=np.concatenate(seen_targets)
    expected_ids=loader.day.ids[(loader.day.times>=lo)&(loader.day.times<hi)]
    np.testing.assert_array_equal(np.sort(ids),expected_ids)
    disk=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    manifest.update(status='passed',verification_seconds=verify_seconds,full_hour_cached_read_seconds=read_seconds,comparisons=comparisons,
        disk_bytes_including_partition_intermediates=disk,total_seconds=time.perf_counter()-started,
        limitation='Direct-query numeric comparison on 12 evenly spaced batches only; cached target coverage checked for every batch. No model training.')
    write_json(out/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['signature','parts','comparisons']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=Path('data/gnn_v1_train'));p.add_argument('--result-root',type=Path,default=Path('result/gnn_train_v1'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',default='2022-09-04T12:00:00');p.add_argument('--end',default='2022-09-04T13:00:00')
    args=p.parse_args();run(args.archive,args.result_root,args.output,args.start,args.end)
