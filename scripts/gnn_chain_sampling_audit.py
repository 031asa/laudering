"""Exact unlabeled-neighbor label audit and replay of saved batched plans."""
from pathlib import Path
import hashlib,json
from functools import lru_cache
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DAY=86400_000000
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class FullHistory:
    def __init__(self,project):
        self.project=Path(project).resolve()
        self.cache=self.project/'data/gnn_validation_cache_parquet'
        frozen={str(Path(k)):v for k,v in read(self.cache/'config.json')['hashes'].items()}
        self.a={};self.hashes={}
        for key in ['src','dst','timestamp','in_ptr','out_ptr','in_order','out_order']:
            p=self.project/'data/gnn_v1_validation/history_index'/f'{key}.npy'
            rel=str(p.relative_to(self.project));digest=sha(p)
            assert frozen[rel]==digest
            self.hashes[rel]=digest;self.a[key]=np.load(p,mmap_mode='r')
    def bounds(self,order,lo,hi,t):
        ts=self.a['timestamp']
        while lo<hi:
            m=(lo+hi)//2
            if ts[order[m]]<t:lo=m+1
            else:hi=m
        return lo
    def incident(self,node,direction,t,days=7):
        ptr=self.a[direction+'_ptr'];order=self.a[direction+'_order']
        lo,hi=int(ptr[node]),int(ptr[node+1])
        hi=self.bounds(order,lo,hi,t);lo=self.bounds(order,lo,hi,t-days*DAY)
        ids=np.asarray(order[lo:hi],dtype=np.int64)
        return ids[self.a['src'][ids]!=self.a['dst'][ids]]


def audit_full_neighbors(project,positive_edges,targets,progress=None):
    h=FullHistory(project);a=h.a
    p=positive_edges.sort_values(['timestamp_us','edge_id'])
    p=p[p.src!=p.dst]
    pe=p.edge_id.to_numpy(np.int64);ps=p.src.to_numpy();pdst=p.dst.to_numpy();pt=p.timestamp_us.to_numpy(np.int64)
    assert np.array_equal(a['src'][pe],ps) and np.array_equal(a['dst'][pe],pdst) and np.array_equal(a['timestamp'][pe],pt)
    result=[]
    for i,row in enumerate(targets.itertuples(index=False)):
        u,v,t=int(row.src),int(row.dst),int(row.cutoff_us)
        lo,hi=np.searchsorted(pt,[t-7*DAY,t],side='left')
        e,s,d,ts=pe[lo:hi],ps[lo:hi],pdst[lo:hi],pt[lo:hi]
        direct=(s==u)|(d==u)|(s==v)|(d==v)
        record={'edge_id':int(row.edge_id)}
        directions=[]
        for role,node in [('src',u),('dst',v)]:
            for direction in ['in','out']:
                ids=h.incident(node,direction,t)
                record[f'{role}_all_{direction}_7d']=len(ids)
                directions.append(ids)
                ids72=ids[a['timestamp'][ids]>=t-3*DAY]
                record[f'{role}_all_{direction}_72h']=len(ids72)
        edges=np.unique(np.concatenate(directions))
        neighbors=np.unique(np.r_[a['src'][edges],a['dst'][edges]])
        neighbors=neighbors[(neighbors!=u)&(neighbors!=v)]
        # Positive edge at graph distance exactly 2 via any historical first edge.
        indirect=(np.isin(s,neighbors)|np.isin(d,neighbors))&~direct
        recent=ts>=t-3*DAY
        edges72=edges[a['timestamp'][edges]>=t-3*DAY]
        neighbors72=np.unique(np.r_[a['src'][edges72],a['dst'][edges72]])
        neighbors72=neighbors72[(neighbors72!=u)&(neighbors72!=v)]
        indirect72=(np.isin(s,neighbors72)|np.isin(d,neighbors72))&~direct&recent
        record.update(
            endpoint_all_incident_unique_7d=len(edges),endpoint_all_incident_unique_72h=len(edges72),
            endpoint_positive_incident_unique_7d=int(direct.sum()),endpoint_positive_incident_unique_72h=int((direct&recent).sum()),
            endpoint_positive_density_7d=float(direct.sum()/len(edges)) if len(edges) else np.nan,
            onehop_unique_neighbors_7d=len(neighbors),onehop_unique_neighbors_72h=len(neighbors72),
            mixed_2hop_positive_count_7d=int(indirect.sum()),mixed_2hop_positive_any_7d=bool(indirect.any()),
            mixed_2hop_positive_count_72h=int(indirect72.sum()),mixed_2hop_positive_any_72h=bool(indirect72.any()),
            full_positive_direct_ids_72h=json.dumps(e[direct&recent].tolist()),
            full_positive_indirect_ids_72h=json.dumps(e[indirect72].tolist()))
        # Exact existence of two time-ordered positive historical edges ending at payer.
        incoming=e[(d==u)&recent];two=[]
        for second in incoming:
            x=int(a['src'][second]);t2=int(a['timestamp'][second])
            first=e[(d==x)&(ts<t2)&recent]
            if len(first):two=[int(first[-1]),int(second)];break
        record['src_positive_walk_ge2_72h']=bool(two)
        record['src_positive_walk2_witness_72h']=json.dumps(two)
        result.append(record)
        if progress and ((i+1)%200==0 or i+1==len(targets)):progress(stage='full_neighbors',completed=i+1,total=len(targets))
    return pd.DataFrame(result),dict(status='passed',source_hashes=h.hashes,
            history='Exact [t-7d,t) and [t-72h,t), self edges excluded, no degree cap.',
            mixed_2hop='Positive edge incident to a node reached by ANY unlabeled historical edge from either target endpoint, excluding positive edges directly incident to the endpoints. Time ordering within the two-edge topological connection is NOT asserted.',
            density='Positive unique incident edges / all unique incident external edges at target endpoints; no claimed full 2-hop edge-density denominator.')


def audit_saved_sampling(project,positive_edges,targets,full_features,progress=None):
    project=Path(project).resolve();cache=project/'data/gnn_validation_cache_parquet'
    # Root cache inventory has no block-manifest hashes. Anchor its real format
    # to the completed frozen evaluation, then verify block identities below.
    evaluation=project/'result/gnn_tuning_v1/evaluations/lr_only'
    evaluation_config=read(evaluation/'config.json');evaluation_metrics=read(evaluation/'metrics.json')
    evaluation_config_hash=sha(evaluation/'config.json')
    assert evaluation_metrics['status']=='completed'
    assert evaluation_metrics['config_sha256']==evaluation_config_hash
    config_hash=sha(cache/'config.json');inventory_hash=sha(cache/'manifest.json')
    assert config_hash==evaluation_config['cache_config_sha256']
    assert inventory_hash==evaluation_config['cache_manifest_sha256']
    config=read(cache/'config.json');inventory=read(cache/'manifest.json')
    assert inventory['status']=='completed' and inventory['config_sha256']==config_hash
    assert inventory['targets']==config['expected_targets']==evaluation_config['expected_targets']
    frozen={str(k).replace('\\','/'):v for k,v in config['hashes'].items()}
    a={};source_hashes={}
    for n in ['src','dst','timestamp']:
        p=project/'data/gnn_v1_validation/history_index'/f'{n}.npy'
        key=p.relative_to(project).as_posix();digest=sha(p)
        assert digest==frozen[key]
        source_hashes[key]=digest;a[n]=np.load(p,mmap_mode='r',allow_pickle=False)
    positive_ids=positive_edges.edge_id.to_numpy(dtype=np.int64)
    assert len(np.unique(positive_ids))==len(positive_ids)
    assert np.all((positive_ids>=0)&(positive_ids<len(a['timestamp'])))
    assert np.array_equal(a['src'][positive_ids],positive_edges.src.to_numpy())
    assert np.array_equal(a['dst'][positive_ids],positive_edges.dst.to_numpy())
    assert np.array_equal(a['timestamp'][positive_ids],positive_edges.timestamp_us.to_numpy())
    positive_set=set(positive_edges.edge_id.astype(int))
    assert targets.edge_id.is_unique and full_features.edge_id.is_unique
    target_rows=targets.set_index('edge_id').to_dict('index')
    details=full_features.set_index('edge_id').to_dict('index')
    assert set(target_rows)==set(details)
    for eid,target in target_rows.items():
        assert 0<=eid<len(a['timestamp'])
        assert (int(target['src']),int(target['dst']),int(target['cutoff_us']))==(
            int(a['src'][eid]),int(a['dst'][eid]),int(a['timestamp'][eid]))
    needed=set(target_rows);seen=set();rows=[];audit=[]
    covered=np.zeros(len(a['timestamp']),dtype=np.bool_);total_targets=0;block_names=set()
    first_allowed=int(np.datetime64(min(config['dates']),'us').astype(np.int64))
    end_allowed=int(np.datetime64(config['end_exclusive'],'us').astype(np.int64))
    for block_i,entry in enumerate(inventory['blocks']):
        assert entry['block'] not in block_names;block_names.add(entry['block'])
        folder=(cache/entry['block']).resolve();assert folder.is_relative_to(cache.resolve())
        manifest=read(folder/'manifest.json')
        block_manifest_hash=sha(folder/'manifest.json')
        assert manifest['status']=='verified' and manifest['config_sha256']==config_hash
        assert manifest['block']==entry['block'] and manifest['targets']==entry['targets']
        path=folder/'plans.parquet';assert sha(path)==manifest['file_sha256']['plans.parquet']
        pf=pq.ParquetFile(path)
        assert pf.metadata.num_rows==manifest['batches'] and 1<=manifest['batches']<=config['max_batches_per_block']
        selected_count=0;block_targets=[];batch_count=0;previous=None;first_cutoff=None
        for batch in pf.iter_batches(columns=['batch_id','cutoff_us','targets','edges'],batch_size=64,use_threads=False):
            for plan in batch.to_pylist():
                assert plan['batch_id']==batch_count and plan['cutoff_us'] is not None
                batch_count+=1;t=int(plan['cutoff_us'])
                assert first_allowed<=t<end_allowed
                assert plan['targets'] is not None and all(x is not None for x in plan['targets'])
                target_ids=np.asarray(plan['targets'],dtype=np.int64)
                assert 1<=len(target_ids)<=config['batch_size']
                assert np.all((target_ids>=0)&(target_ids<len(a['timestamp'])))
                assert np.all(np.diff(target_ids)>0) and np.all(a['timestamp'][target_ids]==t)
                if previous is not None:assert (t,int(target_ids[0]))>previous
                previous=(t,int(target_ids[-1]));first_cutoff=t if first_cutoff is None else first_cutoff
                block_targets.append(target_ids)
                hits=needed.intersection(plan['targets'])
                if not hits:continue
                assert plan['edges'] is not None and all(x is not None for x in plan['edges'])
                edge_ids=np.asarray(plan['edges'],np.int64)
                assert np.all((edge_ids>=0)&(edge_ids<len(a['timestamp']))) and np.all(np.diff(edge_ids)>0)
                assert np.all((a['timestamp'][edge_ids]>=t-3*DAY)&(a['timestamp'][edge_ids]<t))
                assert np.all(a['src'][edge_ids]!=a['dst'][edge_ids])
                selected_positive=np.array([e for e in plan['edges'] if e in positive_set],dtype=np.int64)
                s,d=a['src'][edge_ids],a['dst'][edge_ids]
                ps,pd_,pt=a['src'][selected_positive],a['dst'][selected_positive],a['timestamp'][selected_positive]
                for eid in hits:
                    assert eid not in seen;seen.add(eid);selected_count+=1
                    target=target_rows[eid];assert int(target['cutoff_us'])==t
                    u,v=int(target['src']),int(target['dst'])
                    touch=(s==u)|(d==u)|(s==v)|(d==v)
                    neighbors=np.unique(np.r_[s[touch],d[touch]])
                    direct=(ps==u)|(pd_==u)|(ps==v)|(pd_==v)
                    indirect=(np.isin(ps,neighbors)|np.isin(pd_,neighbors))&~direct
                    full_direct=set(json.loads(details[eid]['full_positive_direct_ids_72h']))
                    full_indirect=set(json.loads(details[eid]['full_positive_indirect_ids_72h']))
                    assert set(selected_positive[direct]).issubset(full_direct)
                    assert set(selected_positive[indirect]).issubset(full_indirect)
                    temporal2=False;witness=[]
                    for second in selected_positive[pd_==u]:
                        x=int(a['src'][second]);t2=int(a['timestamp'][second])
                        first=selected_positive[(pd_==x)&(pt<t2)]
                        if len(first):temporal2=True;witness=[int(first[-1]),int(second)];break
                    assert not temporal2 or details[eid]['src_positive_walk_ge2_72h']
                    rows.append(dict(edge_id=eid,block=entry['block'],batch_id=plan['batch_id'],batch_targets=len(plan['targets']),
                        sampled_graph_edges=len(edge_ids),sampled_graph_positive_edges=len(selected_positive),
                        direct_positive_edges_sampled=int(direct.sum()),indirect_positive_edges_reachable2_sampled=int(indirect.sum()),
                        direct_positive_any_sampled=bool(direct.any()),indirect_positive_any_reachable2_sampled=bool(indirect.any()),
                        src_positive_walk_ge2_sampled=temporal2,sampled_walk2_witness=json.dumps(witness),
                        full_direct_positive_edges_72h=len(full_direct),full_indirect_positive_edges_72h=len(full_indirect),
                        full_src_positive_walk_ge2_72h=bool(details[eid]['src_positive_walk_ge2_72h'])))
        all_ids=np.concatenate(block_targets)
        # Same byte contract as cache_train.array_sha used by model_cache.
        target_ids_hash=hashlib.sha256(np.ascontiguousarray(all_ids,dtype=np.int64).tobytes()).hexdigest()
        assert len(all_ids)==manifest['targets'] and batch_count==manifest['batches']
        assert target_ids_hash==manifest['target_ids_sha256']
        assert first_cutoff==manifest['first_cutoff_us'] and previous[0]==manifest['last_cutoff_us']
        assert first_cutoff//(DAY//24)==previous[0]//(DAY//24)
        assert len(np.unique(all_ids))==len(all_ids) and not covered[all_ids].any()
        covered[all_ids]=True;total_targets+=len(all_ids)
        assert sha(folder/'manifest.json')==block_manifest_hash
        audit.append(dict(block=entry['block'],manifest_sha256=block_manifest_hash,
                          plans_sha256=manifest['file_sha256']['plans.parquet'],
                          target_ids_sha256=target_ids_hash,targets=len(all_ids),batches=batch_count,
                          selected=selected_count))
        if progress and ((block_i+1)%25==0 or block_i+1==len(inventory['blocks'])):progress(stage='saved_plans',completed=block_i+1,total=len(inventory['blocks']),targets_seen=len(seen))
    assert seen==needed and total_targets==inventory['targets']
    assert sha(cache/'config.json')==config_hash and sha(cache/'manifest.json')==inventory_hash
    return pd.DataFrame(rows),dict(status='passed',cache_config_sha256=config_hash,cache_manifest_sha256=inventory_hash,
        evaluation_config_sha256=evaluation_config_hash,source_hashes=source_hashes,
        complete_unique_cached_target_count=total_targets,blocks=audit,
        note='Saved real batched graph, never resampled. Labels are overlaid only for retrospective inspection; they were NOT passed into graph messages. Count only edges reachable within two hops for indirect evidence. Comparing within 72h avoids falsely attributing window exclusion to sampling.')
