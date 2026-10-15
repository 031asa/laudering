"""Authorized post-hoc tail sensitivity analysis with frozen models/thresholds.

Small-scope direct feature calculation uses the same numerical production
functions. A real original-Test batch must reproduce the published cached
inputs and predictions before any tail predictions are accepted.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import traceback
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from gnn_features.build import load_archive
from gnn_features.cache_train import sha, array_sha
from gnn_features.contract import HOUR, DAY
from gnn_features.engine import FeatureEngine
from gnn_features.loader import TemporalBatchBuilder, TargetDay, encode_r
from gnn_features.model import to_torch
from gnn_features.model_cache import ParquetBlockCache
from gnn_features.training_state import ProcessLock, atomic_json
from gnn_features.v2_adapter import adapt_arrays
from gnn_final_test import common as original
from gnn_final_test.a_features import (ROLES, WINDOWS, encode_extra, role_features,
    _role_input, compute_selected, SidecarStore)
from gnn_final_test.acceptance import verify_inference
from .archive import build_test_archive, DATES, END
from .edge_store import TestEdgeStore

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = 'result/gnn_tail_test'
PREFIX, TOTAL, POSITIVES = 31891231, 6987, 4084
SEEDS, TRIALS = original.SEEDS, original.TRIALS

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sources(project):
    paths=sorted((project/'scripts/gnn_tail_test').glob('*.py'))
    paths += [project/'tests/test_gnn_tail_test.py',project/'scripts/start_gnn_tail_test.ps1']
    return {str(p.relative_to(project)): sha(p) for p in paths}

def raw_r(engine, ids):
    rows=[]
    for edge in ids:
        r,_=engine.target(int(edge));t=int(engine.arr['timestamp'][edge])
        r.update(full_6h=t-engine.data_start>=6*HOUR,full_day=t-t%DAY>=engine.data_start)
        rows.append(r)
        engine.states.clear();engine.nodes.clear()
    return encode_r(pa.Table.from_pylist(rows))

def extra_targets(arrays, start, ids):
    """Evaluate unchanged production rolling functions only for target nodes."""
    ids=np.asarray(ids,dtype=np.int64)
    raw=np.full((len(ids),28),np.nan)
    for ri,(role,(_,node_key,_,_)) in enumerate(ROLES.items()):
        external=arrays['src'][ids]!=arrays['dst'][ids]
        for node in np.unique(arrays[node_key][ids[external]]):
            hist,t,a,g=_role_input(arrays,role,int(node),int(node)+1)
            values,_=role_features(t,a,g)
            order=np.argsort(hist);positions=np.flatnonzero(external & (arrays[node_key][ids]==node))
            where=np.searchsorted(hist[order],ids[positions])
            if np.any(where>=len(hist)) or not np.array_equal(hist[order][where],ids[positions]):
                raise ValueError('A target missing from external role history')
            raw[positions,ri*14:(ri+1)*14]=values[order[where]]
    times=arrays['timestamp'][ids]
    result=encode_extra(raw,np.column_stack([times-start>=w for w in WINDOWS.values()]))
    # Independent slow exact-sort reference on EVERY tail target.
    for i in range(0,len(ids),256):
        ref=compute_selected(arrays,start,ids[i:i+256])
        np.testing.assert_allclose(result[i:i+256],ref,rtol=2e-6,atol=2e-6)
    return result

def run(project=ROOT):
    project=Path(project).resolve();out=project/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    def update(**kw):
        state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());atomic_json(out/'status.json',state)
        print(json.dumps(kw,ensure_ascii=True),flush=True)
    def guard():
        if shutil.disk_usage(project).free<20*1024**3:raise RuntimeError('Disk below 20 GiB')
        if (out/'pause_request.json').exists() and read(out/'pause_request.json').get('enabled'):raise RuntimeError('Explicit user pause')
    with ProcessLock(out):
        try:
            guard();original.protocol(project)
            if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
            torch.set_num_threads(1)
            models={}
            for seed in SEEDS:
                for trial in TRIALS:
                    trained=original.bound_training(project,trial,seed)
                    trained['model']=trained['model'].to('cuda').eval();models[seed,trial]=trained
            config=dict(version=1,authorization_date='2026-10-12',scope='Post-hoc sensitivity analysis after observing original Test; not a new confirmatory Test',
                dates=list(DATES),targets=TOTAL,positives=POSITIVES,prefix=PREFIX,source_sha256=sources(project),
                original_test_protocol_sha256=sha(project/'result/gnn_final_test/protocol.json'),
                original_comparison_sha256=sha(project/'result/gnn_final_test/comparison.json'),
                models=[dict(seed=s,trial=t,model_sha256=m['model_sha256'],binding_sha256=m['binding_sha256'],thresholds=m['binding']['thresholds']) for (s,t),m in models.items()],
                graph_seed=20260920,batch_size=64,no_training=True,no_threshold_selection=True,no_model_selection=True)
            if (out/'config.json').exists() and read(out/'config.json')!=config:raise ValueError('Tail config/source/model binding changed')
            atomic_json(out/'config.json',config);config_hash=sha(out/'config.json')
            engineering=read(out/'engineering_tests.json')
            if engineering['status']!='passed' or engineering['source_sha256']!=sources(project):raise ValueError('Tail engineering acceptance missing/changed')
            update(stage='archive')
            archive=project/'data/gnn_tail_test_archive'
            build_test_archive(project,archive,progress=lambda x:update(stage='archive',archive_stage=x.get('stage')))
            meta,a,bank,vocab=load_archive(archive)
            if meta['rows']!=PREFIX+TOTAL or meta['end_exclusive']!=END:raise ValueError('Tail extent mismatch')
            adj={d+'_'+k:np.load(archive/f'graph/adj_index/{d}_{k}.npy',mmap_mode='r') for d in ('in','out') for k in ('ptr','edge_ids')}
            extra_arrays=dict(a)
            for d in ('in','out'):extra_arrays.pop(d+'_order',None)
            extra_arrays.update(adj)
            bank_cat=np.load(archive/'history_index/bank_cat.npy',mmap_mode='r')
            store=TestEdgeStore(project,archive)
            engine=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],int(np.datetime64(END,'us').astype(np.int64)))
            builder=TemporalBatchBuilder(engine,adj,store,bank_cat)
            update(stage='original_test_input_parity')
            # The copied prefix plus appended FUTURE tail must reproduce old Test.
            oldarchive=project/'data/gnn_final_test_archive';oldcache=project/'data/gnn_final_test_cache'
            rel=read(oldcache/'manifest.json')['blocks'][0]['block'];date=rel.split('/')[0].split('=')[1]
            day=TargetDay(oldarchive,project/'result/gnn_final_test/features',date,a)
            cached=ParquetBlockCache(oldcache/rel,SimpleNamespace(day=day,builder=builder),sha(oldcache/'config.json'))
            oldbatch,y,_=cached.batch(0);ids=oldbatch['target_edge_id'];r=raw_r(engine,ids)
            fresh,_,_=builder.build(ids,r,y)
            for key in oldbatch:np.testing.assert_allclose(fresh[key],oldbatch[key],rtol=2e-6,atol=2e-6,err_msg=key)
            oldside=SidecarStore(project/'data/gnn_final_test_A_features/test',oldcache,'test')
            ext=extra_targets(extra_arrays,meta['data_start_us'],ids)
            np.testing.assert_allclose(ext,oldside.block(rel,ids,a['timestamp'][ids]),rtol=2e-6,atol=2e-6)
            with torch.inference_mode():
                for (seed,trial),m in models.items():
                    batch=adapt_arrays(fresh,ext,'features_only') if trial=='features_only' else fresh
                    score=torch.sigmoid(m['model'](to_torch(batch,'cuda'))).cpu().numpy()
                    saved=pq.ParquetFile(project/f'result/gnn_final_test/evaluations/{seed}/{trial}/predictions'/rel/'predictions.parquet').read(columns=['edge_id','probability'])
                    order=np.argsort(saved['edge_id'].to_numpy());pos=np.searchsorted(saved['edge_id'].to_numpy()[order],ids)
                    np.testing.assert_allclose(score,saved['probability'].to_numpy()[order[pos]],rtol=2e-6,atol=2e-6)
            atomic_json(out/'original_test_parity.json',dict(status='passed',config_sha256=config_hash,targets=len(ids),models=8,R_N_E_graph_and_A_match=True,published_predictions_match=True))
            del day,cached,oldside;builder.clear_cache();gc.collect()
            ids=np.arange(PREFIX,PREFIX+TOTAL,dtype=np.int64);times=a['timestamp'][ids];labels=np.empty(TOTAL,np.int8);seen=np.zeros(TOTAL,bool)
            for date in DATES:
                for f in sorted((archive/'supervision/targets'/f'transaction_date={date}').glob('*.parquet')):
                    tab=pq.ParquetFile(f).read(columns=['edge_id','label']);pos=tab['edge_id'].to_numpy()-PREFIX
                    if np.any(pos<0) or np.any(pos>=TOTAL) or seen[pos].any():raise ValueError('Tail label identity mismatch')
                    labels[pos]=tab['label'].to_numpy();seen[pos]=True
            if not seen.all() or labels.sum()!=POSITIVES:raise ValueError('Tail label count mismatch')
            update(stage='tail_A_features',targets=TOTAL)
            extra=extra_targets(extra_arrays,meta['data_start_us'],ids)
            order=np.lexsort((ids,times));bounds=np.r_[0,np.flatnonzero(np.diff(times[order]))+1,TOTAL]
            done=0;bi=0;accepted=set()
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                for start in range(lo,hi,64):
                    guard();pos=order[start:min(hi,start+64)];bid=ids[pos]
                    r=raw_r(engine,bid);batch,_,_=builder.build(bid,r,labels[pos]);values={}
                    for (seed,trial),m in models.items():
                        name=f'{seed}_{trial}';folder=out/'predictions'/name;folder.mkdir(parents=True,exist_ok=True)
                        path=folder/f'{bi:06d}.parquet';mp=folder/f'{bi:06d}.json'
                        current=adapt_arrays(batch,extra[pos],'features_only') if trial=='features_only' else batch
                        if mp.exists():
                            record=read(mp)
                            if record['config_sha256']!=config_hash or sha(path)!=record['file_sha256']:raise ValueError('Tail committed prediction changed')
                            tab=pq.ParquetFile(path).read()
                            if not np.array_equal(tab['edge_id'].to_numpy(),bid):raise ValueError('Tail prediction IDs changed')
                        else:
                            with torch.inference_mode():logits=m['model'](to_torch(current,'cuda'))
                            if name not in accepted:
                                acceptance=verify_inference(m['model'],current,logits,m['model_path'],config_hash)
                                atomic_json(out/f'acceptance_{name}.json',acceptance);accepted.add(name)
                            score=torch.sigmoid(logits).detach().cpu().numpy()
                            if not np.isfinite(score).all():raise ValueError('Nonfinite tail score')
                            tab=pa.table(dict(edge_id=bid,timestamp_us=times[pos],label=labels[pos],probability=score))
                            tmp=path.with_suffix('.tmp');pq.write_table(tab,tmp,compression='zstd');os.replace(tmp,path)
                            atomic_json(mp,dict(config_sha256=config_hash,targets=len(bid),target_ids_sha256=array_sha(bid),file_sha256=sha(path)))
                    done+=len(bid);bi+=1
                    if bi%25==0:update(stage='inference',processed_targets=done,batches=bi,total_models=8)
            update(stage='comparison',processed_targets=done)
            from .compare import compare
            compare(project,config_hash,a,bank,vocab)
            original.protocol(project)
            if sha(project/'result/gnn_final_test/comparison.json')!=config['original_comparison_sha256']:raise ValueError('Original Test changed')
            update(status='completed',stage='complete',processed_targets=TOTAL,completed_models=8)
        except BaseException as exc:
            update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=ROOT)
    run(p.parse_args().project)
