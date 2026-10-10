"""Small real Train benchmark, no fitting, no validation/test exposure."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import psutil
from .loader import TemporalLoader
from .build import write_json


def run(archive,result_root,out,date,edge_cache=None):
    if out.exists():raise ValueError('Choose a new benchmark output directory')
    out.mkdir(parents=True);began=time.perf_counter()
    loader=TemporalLoader(archive,result_root,date,edge_cache);startup=time.perf_counter()-began
    print(json.dumps(dict(stage='loader_ready',date=date,rows=len(loader.day.ids),startup_seconds=startup,R_load_seconds=loader.day.load_seconds)),flush=True)
    reports=[];peak=psutil.Process().memory_info().rss;checked=0
    # Verify iterator identity coverage and exact timestamp grouping without building all graphs.
    for ids in loader.day.batches(64):
        assert len(ids)<=64 and np.all(loader.builder.a['timestamp'][ids]==loader.builder.a['timestamp'][ids[0]])
        checked+=len(ids)
    assert checked==len(loader.day.ids)
    for clock in ['06:00','12:00','19:00']:
        t=int(np.datetime64(date+'T'+clock,'us').astype(np.int64));positions=np.flatnonzero(loader.day.times==t)
        if not len(positions):raise ValueError('No targets for chosen benchmark cutoff')
        ids=loader.day.ids[positions[:32]];loader.builder.clear_cache()
        first=None
        for mode in ['cold','repeat']:
            tensors,y,report=loader.batch(ids);report.update(mode=mode,time=clock,positive_targets=int(y.sum()))
            if first is None:
                first=tensors
                np.savez_compressed(out/f'batch_{clock.replace(":","")}.npz',**tensors)
                np.save(out/f'labels_{clock.replace(":","")}.npy',y)
            else:
                for name in first:np.testing.assert_array_equal(first[name],tensors[name])
            reports.append(report);peak=max(peak,psutil.Process().memory_info().rss)
            print(json.dumps(report),flush=True)
        next_ids=loader.day.ids[positions[32:64]]
        if len(next_ids):
            _,_,report=loader.batch(next_ids);report.update(mode='next_batch_same_cutoff',time=clock)
            reports.append(report);peak=max(peak,psutil.Process().memory_info().rss)
            print(json.dumps(report),flush=True)
    result=dict(status='passed',date=date,day_rows=len(loader.day.ids),all_day_iterator_rows_checked=checked,
        startup_seconds=startup,target_day_load_seconds=loader.day.load_seconds,wall_seconds=time.perf_counter()-began,
        sampled_process_rss_bytes=peak,reports=reports,edge_cache=str(edge_cache) if edge_cache else None,
        scope='Persisted E/R; exact cutoff historical graph; on-demand N; CPU only; no training')
    write_json(out/'benchmark.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=Path('data/gnn_v1_train'))
    p.add_argument('--result-root',type=Path,default=Path('result/gnn_train_v1'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--date',default='2022-09-04');p.add_argument('--edge-cache',type=Path)
    args=p.parse_args();run(args.archive,args.result_root,args.output,args.date,args.edge_cache)
