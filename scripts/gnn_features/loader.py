"""CPU temporal minibatches using persisted E/R and bounded as-of N caching."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build import load_archive
from .contract import HOUR,N_FIELDS,E_FIELDS,R_FIELDS,encode_numeric
from .engine import FeatureEngine

ENUM=[f['name'] for f in E_FIELDS if f['transform']!='embedding']
ECAT=[f['name'] for f in E_FIELDS if f['transform']=='embedding']
RDIM=len(R_FIELDS)+sum(f['nullable'] for f in R_FIELDS)+2


def encode_r(table):
    """Apply the existing model transform to stored values, never query R again."""
    values=[];masks=[]
    for f in R_FIELDS:
        column=table[f['name']]
        valid=np.asarray(column.is_valid());x=column.cast(pa.float64()).to_numpy().copy()
        x[~valid]=0
        if f['transform']=='log1p':x=np.log1p(x)
        elif f['transform']=='entropy_ln24':x/=np.log(24)
        values.append(x)
        if f['nullable']:masks.append(valid)
    result=np.column_stack(values+masks+[table[k].to_numpy() for k in ['full_6h','full_day']]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite stored R')
    return result


class EdgeStore:
    """Index persisted contiguous edge-ID row groups; cache at most 16 groups."""
    def __init__(self,archive,max_groups=16):
        self.groups=[];self.cache=OrderedDict();self.max_groups=max_groups;self.reads=0
        for path in sorted((Path(archive)/'graph/edges').rglob('*.parquet')):
            pf=pq.ParquetFile(path);col=pf.schema_arrow.get_field_index('edge_id')
            for rg in range(pf.num_row_groups):
                metadata=pf.metadata.row_group(rg);s=metadata.column(col).statistics
                if s is None or not s.has_min_max:raise ValueError('Missing edge-ID row-group statistics')
                lo,hi=int(s.min),int(s.max)
                if hi-lo+1!=metadata.num_rows:raise ValueError('Noncontiguous edge IDs')
                self.groups.append((lo,hi,str(path),rg))
        self.groups.sort();self.starts=np.array([g[0] for g in self.groups],dtype=np.int64)
        if not self.groups or self.groups[0][0]!=0 or any(a[1]+1!=b[0] for a,b in zip(self.groups,self.groups[1:])):
            raise ValueError('Incomplete archive edge-ID index')

    def get(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        numeric=np.empty((len(ids),len(ENUM)),dtype=np.float32);category=np.empty((len(ids),len(ECAT)),dtype=np.int64)
        if not len(ids):return numeric,category
        if np.any(ids<0) or np.any(ids>self.groups[-1][1]):raise ValueError('Edge outside archive')
        groups=np.searchsorted(self.starts,ids,side='right')-1
        for group in np.unique(groups):
            group=int(group);lo,hi,path,rg=self.groups[group]
            if group not in self.cache:
                table=pq.ParquetFile(path).read_row_group(rg,columns=['edge_id']+ENUM+ECAT)
                if not np.array_equal(table['edge_id'].to_numpy(),np.arange(lo,hi+1)):raise ValueError('Edge row-group ordering changed')
                self.cache[group]=(np.column_stack([table[k].to_numpy() for k in ENUM]).astype(np.float32),
                    np.column_stack([table[k].to_numpy() for k in ECAT]).astype(np.int64))
                self.reads+=1
                while len(self.cache)>self.max_groups:self.cache.popitem(last=False)
            self.cache.move_to_end(group);num,cat=self.cache[group];pos=np.flatnonzero(groups==group);offset=ids[pos]-lo
            numeric[pos]=num[offset];category[pos]=cat[offset]
        return numeric,category


class CompactEdgeStore:
    """Derived float32/int16 memory maps copied from persisted E, not recomputed."""
    def __init__(self,archive,cache):
        archive=Path(archive);cache=Path(cache);started=time.perf_counter()
        meta=json.loads((archive/'build.json').read_text(encoding='utf-8'));n=meta['rows']
        signature=dict(version=1,archive_sha256=hashlib.sha256((archive/'build.json').read_bytes()).hexdigest(),
            rows=n,numeric_columns=ENUM,category_columns=ECAT)
        if (cache/'manifest.json').exists():
            saved=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
            if any(saved[k]!=v for k,v in signature.items()):raise ValueError('Incompatible E tensor cache')
        else:
            if cache.exists() and any(cache.iterdir()):raise ValueError('Incomplete E cache; choose new directory')
            cache.mkdir(parents=True,exist_ok=True)
            num=np.lib.format.open_memmap(cache/'numeric.npy',mode='w+',dtype=np.float32,shape=(n,len(ENUM)))
            cat=np.lib.format.open_memmap(cache/'category.npy',mode='w+',dtype=np.int16,shape=(n,len(ECAT)))
            count=0
            for path in sorted((archive/'graph/edges').rglob('*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32768,columns=['edge_id']+ENUM+ECAT):
                    table=pa.Table.from_batches([batch]);ids=table['edge_id'].to_numpy()
                    if not np.array_equal(ids,np.arange(count,count+len(ids))):raise ValueError('Archive ordering differs')
                    numeric=np.column_stack([table[k].to_numpy() for k in ENUM]).astype(np.float32)
                    category=np.column_stack([table[k].to_numpy() for k in ECAT])
                    if not np.isfinite(numeric).all() or np.any(category<0) or np.any(category>32767):raise ValueError('Invalid E cache values')
                    num[count:count+len(ids)]=numeric;cat[count:count+len(ids)]=category;count+=len(ids)
            if count!=n:raise ValueError('Incomplete E cache')
            num.flush();cat.flush();del num,cat
            saved=dict(**signature,build_seconds=time.perf_counter()-started)
            (cache/'manifest.json').write_text(json.dumps(saved,indent=2),encoding='utf-8')
        self.numeric=np.load(cache/'numeric.npy',mmap_mode='r');self.category=np.load(cache/'category.npy',mmap_mode='r')
        if self.numeric.shape!=(n,len(ENUM)) or self.category.shape!=(n,len(ECAT)):raise ValueError('Invalid E cache shape')
        self.cache={};self.reads=0;self.build_seconds=saved['build_seconds']

    def get(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        if np.any(ids<0) or np.any(ids>=len(self.numeric)):raise ValueError('Edge outside archive')
        return np.asarray(self.numeric[ids]),np.asarray(self.category[ids],dtype=np.int64)


class TargetDay:
    """Load one verified day of R and separate supervision, preserving edge identity."""
    def __init__(self,archive,result_root,date,arrays):
        started=time.perf_counter();folder=Path(result_root)/f'transaction_date={date}'
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))
        run=read(folder/'daily_run.json');config=read(folder/'run_config.json')
        if run['status']!='verified':raise ValueError('Targets must have passed verification')
        if config['archive_manifest_sha256']!=hashlib.sha256((Path(archive)/'build.json').read_bytes()).hexdigest():
            raise ValueError('Target R belongs to a different archive')
        self.ids=np.load(folder/'target_edge_ids.npy').astype(np.int64)
        if np.any(np.diff(self.ids)<=0):raise ValueError('Target IDs must be unique ascending')
        self.r=np.empty((len(self.ids),RDIM),dtype=np.float32);seen=np.zeros(len(self.ids),dtype=bool)
        for part in read(folder/'merge_phase.json')['parts']:
            for batch in pq.ParquetFile(part['path']).iter_batches(batch_size=8192,columns=['edge_id','timestamp']+[f['name'] for f in R_FIELDS]+['full_6h','full_day']):
                table=pa.Table.from_batches([batch]);ids=table['edge_id'].to_numpy();p=self.positions(ids)
                if seen[p].any() or len(np.unique(p))!=len(p):raise ValueError('Duplicate R')
                if not np.array_equal(table['timestamp'].cast(pa.int64()).to_numpy(),arrays['timestamp'][ids]):raise ValueError('R cutoff mismatch')
                self.r[p]=encode_r(table);seen[p]=True
        if not seen.all():raise ValueError('Missing R')
        self.labels=np.empty(len(self.ids),dtype=np.float32);seen[:]=False
        for path in sorted((Path(archive)/'supervision/targets'/f'transaction_date={date}').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32768,columns=['edge_id','label']):
                p=self.positions(batch.column('edge_id').to_numpy());y=batch.column('label').to_numpy()
                if seen[p].any() or len(np.unique(p))!=len(p) or not np.isin(y,[0,1]).all():raise ValueError('Invalid supervision')
                self.labels[p]=y;seen[p]=True
        if not seen.all():raise ValueError('Missing supervision')
        self.times=np.asarray(arrays['timestamp'][self.ids]);self.order=np.lexsort((self.ids,self.times))
        self.load_seconds=time.perf_counter()-started

    def positions(self,ids):
        p=np.searchsorted(self.ids,ids)
        if np.any(p>=len(self.ids)) or not np.array_equal(self.ids[p],ids):raise ValueError('Target is not in loaded day')
        return p

    def batches(self,batch_size=64):
        if batch_size<1:raise ValueError('Positive batch size required')
        ordered=self.times[self.order];bounds=np.r_[0,np.flatnonzero(np.diff(ordered))+1,len(ordered)]
        for begin,end in zip(bounds[:-1],bounds[1:]):
            for start in range(begin,end,batch_size):yield self.ids[self.order[start:min(end,start+batch_size)]]


class TemporalBatchBuilder:
    def __init__(self,engine,adj,edge_store,bank_cat,max_nodes_cached=32768):
        self.engine=engine;self.a=engine.arr;self.adj=adj;self.edges=edge_store;self.bank_cat=bank_cat
        self.node_cache=OrderedDict();self.max_nodes_cached=max_nodes_cached

    def clear_cache(self):
        self.node_cache.clear();self.edges.cache.clear();self.engine.states.clear();self.engine.nodes.clear()

    def sample(self,ids,seed=20260920):
        ids=np.asarray(ids,dtype=np.int64);a=self.a
        if not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Empty or duplicate targets')
        if np.any(ids<0) or np.any(ids>=len(a['timestamp'])):raise ValueError('Invalid target ID')
        t=int(a['timestamp'][ids[0]])
        if not np.all(a['timestamp'][ids]==t):raise ValueError('Mixed cutoff batch')
        self.engine.at(t);rng=np.random.default_rng(seed)
        nodes=set(map(int,a['src'][ids]))|set(map(int,a['dst'][ids]));frontier=set(nodes);edge_ids=set()
        for fanout in [10,5]:
            following=set()
            for node in sorted(frontier):
                for d in ['in','out']:
                    ptr=self.adj[d+'_ptr'];ordered=self.adj[d+'_edge_ids'][int(ptr[node]):int(ptr[node+1])]
                    ts=a['timestamp'][ordered];lo,hi=np.searchsorted(ts,[t-72*HOUR,t],side='left')
                    choices=ordered[lo:hi]
                    picked=choices if len(choices)<=fanout else rng.choice(choices,size=fanout,replace=False)
                    edge_ids.update(map(int,picked));following.update(map(int,a['src'][picked]));following.update(map(int,a['dst'][picked]))
            frontier=following;nodes.update(following)
        nodes=np.array(sorted(nodes),dtype=np.int64);edge_ids=np.array(sorted(edge_ids),dtype=np.int64)
        return nodes,edge_ids

    def build(self,ids,r,labels,seed=20260920,prepared=None,node_lookup=None):
        began=time.perf_counter();ids=np.asarray(ids,dtype=np.int64);a=self.a
        if not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Empty or duplicate targets')
        if np.any(ids<0) or np.any(ids>=len(a['timestamp'])):raise ValueError('Invalid target ID')
        t=int(a['timestamp'][ids[0]])
        if not np.all(a['timestamp'][ids]==t):raise ValueError('Mixed cutoff batch')
        if np.shape(r)!=(len(ids),RDIM) or np.shape(labels)!=(len(ids),):raise ValueError('Target feature/label shape mismatch')
        self.engine.at(t);ctx=self.engine.context()
        nodes,edge_ids=self.sample(ids,seed) if prepared is None else prepared
        if np.any(np.diff(nodes)<=0) or np.any(np.diff(edge_ids)<=0):raise ValueError('Prepared IDs must be sorted unique')
        if not np.isin(np.r_[a['src'][ids],a['dst'][ids],a['src'][edge_ids],a['dst'][edge_ids]],nodes).all():raise ValueError('Incomplete prepared nodes')
        sampling=time.perf_counter()-began;started=time.perf_counter();hits=0;encoded=[]
        for node in nodes:
            key=(t,int(node))
            if key in self.node_cache:hits+=1;self.node_cache.move_to_end(key)
            else:
                if node_lookup is None:
                    value=encode_numeric(self.engine.node(node),N_FIELDS)+[float(ctx['full_'+w]) for w in ['1h','24h','72h']]
                else:value=node_lookup(int(node),t)
                self.node_cache[key]=np.asarray(value,dtype=np.float32)
                # Retain compact vectors only; raw per-account histories are transient.
                self.engine.states.clear();self.engine.nodes.clear()
                while len(self.node_cache)>self.max_nodes_cached:self.node_cache.popitem(last=False)
            encoded.append(self.node_cache[key])
        x=np.stack(encoded);node_seconds=time.perf_counter()-started;started=time.perf_counter();reads=self.edges.reads
        historical_num,historical_cat=self.edges.get(edge_ids);current_num,current_cat=self.edges.get(ids)
        age=np.log1p((t-a['timestamp'][edge_ids])/60_000_000).astype(np.float32)
        route=np.vstack([np.searchsorted(nodes,a[c][edge_ids]) for c in ['src','dst']])
        targets=np.vstack([np.searchsorted(nodes,a[c][ids]) for c in ['src','dst']])
        tensors=dict(node_id=nodes,edge_id=edge_ids,target_edge_id=ids,x_num=x,x_cat=np.asarray(self.bank_cat[nodes],dtype=np.int64).reshape(-1,1),
            edge_index_in=route,edge_index_out=route[::-1].copy(),edge_attr_num=np.column_stack([historical_num,age]),edge_attr_cat=historical_cat,
            edge_label_index=targets,target_base_num=current_num,target_base_cat=current_cat,target_attr=np.asarray(r,dtype=np.float32))
        assert np.all(a['timestamp'][edge_ids]<t) and np.all(a['timestamp'][edge_ids]>=t-72*HOUR)
        assert np.all(a['src'][edge_ids]!=a['dst'][edge_ids]) and not np.intersect1d(edge_ids,ids).size
        for name,value in tensors.items():
            if value.dtype.kind=='f' and not np.isfinite(value).all():raise ValueError('Nonfinite tensor '+name)
        report=dict(cutoff_us=t,targets=len(ids),nodes=len(nodes),history_edges=len(edge_ids),node_cache_hits=hits,node_cache_misses=len(nodes)-hits,
            sampling_seconds=sampling,node_seconds=node_seconds,edge_read_and_pack_seconds=time.perf_counter()-started,
            seconds=time.perf_counter()-began,edge_row_groups_read=self.edges.reads-reads,
            tensor_bytes=sum(v.nbytes for v in tensors.values()),shapes={k:list(v.shape) for k,v in tensors.items()})
        return tensors,np.asarray(labels,dtype=np.float32).copy(),report


class TemporalLoader:
    def __init__(self,archive,result_root,date,edge_cache=None):
        archive=Path(archive);meta,a,bank,vocab=load_archive(archive)
        self.day=TargetDay(archive,result_root,date,a)
        engine=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64)))
        adj={d+'_'+s:np.load(archive/f'graph/adj_index/{d}_{s}.npy',mmap_mode='r') for d in ['in','out'] for s in ['ptr','edge_ids']}
        store=CompactEdgeStore(archive,edge_cache) if edge_cache is not None else EdgeStore(archive)
        self.builder=TemporalBatchBuilder(engine,adj,store,np.load(archive/'history_index/bank_cat.npy',mmap_mode='r'))

    def batch(self,ids,seed=20260920):
        p=self.day.positions(ids)
        return self.builder.build(ids,self.day.r[p],self.day.labels[p],seed)

    def __iter__(self):
        for ids in self.day.batches():yield self.batch(ids)
