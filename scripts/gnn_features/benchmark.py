"""Account-partition benchmark; Spark runs requests, not a full-history shuffle.

All variants reuse the same immutable archive. Partition outputs contain only
actually requested endpoints, retained to verify every N and R value.
"""
import argparse
import ctypes
import itertools
import json
import os
import platform
from pathlib import Path
import pickle
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build import load_archive, write_json
from .engine import FeatureEngine
from .rolling import RollingEngine
from .partitioned import target_side
from .contract import N_FIELDS,R_FIELDS


def peak_rss():
    if os.name!='nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in ['peak','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile']]
    c=Counters();c.cb=ctypes.sizeof(c)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(c.peak)


def partition_worker(partition,rows,archive,output,mode):
    began=time.perf_counter();meta,a,bank,vocab=load_archive(Path(archive))
    cls=RollingEngine if mode=='rolling' else FeatureEngine
    end=int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64))
    try:
        from pyspark import TaskContext
        context=TaskContext.get();attempt=context.taskAttemptId() if context else 0
    except ImportError:attempt=0
    path=Path(output)/f'part-{partition:04d}-attempt-{attempt}.parquet'
    schema=pa.schema([('node',pa.int64()),('cutoff',pa.int64()),('edge_id',pa.int64()),('side',pa.string()),('payload',pa.binary())])
    writer=pq.ParquetWriter(path,schema,compression='zstd')
    batch=[];accounts=0;requests=0;unique=0;previous=None;max_account_seconds=0;max_account_node=None;last_progress=began
    for node,group in itertools.groupby(rows,key=lambda x:int(x[0])):
        if previous is not None and node<=previous:raise AssertionError('Account must be contiguous and sorted within partition')
        previous=node;accounts+=1;started=time.perf_counter()
        engine=cls(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'],end)
        prev_t=None
        for row in group:
            _,t,i,side=row;t=int(t);i=int(i);side=str(side)
            if t!=int(a['timestamp'][i]) or node!=int(a['src' if side=='src' else 'dst'][i]):
                raise AssertionError('Request identity/time does not match archived transaction')
            if prev_t is not None and t<prev_t:raise AssertionError('Cutoff order violated')
            if t!=prev_t:unique+=1
            prev_t=t;engine.at(t);n=engine.node(node);r,audit=target_side(engine,i,side,n)
            payload=pickle.dumps((n,r,audit),protocol=5)
            batch.append(dict(node=node,cutoff=t,edge_id=i,side=side,payload=payload));requests+=1
            if len(batch)>=256:
                writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch.clear()
        took=time.perf_counter()-started
        if took>max_account_seconds:max_account_seconds=took;max_account_node=node
        del engine
        if time.perf_counter()-last_progress>20:
            print(json.dumps(dict(stage='partition_progress',partition=partition,accounts=accounts,requests=requests)),flush=True)
            last_progress=time.perf_counter()
    if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    writer.close()
    return iter([dict(partition=partition,path=str(path),requests=requests,accounts=accounts,node_time_queries=unique,
        seconds=time.perf_counter()-began,worker_peak_rss_bytes=peak_rss(),max_account_seconds=max_account_seconds,max_account_node=max_account_node)])


def collect(parts,request_rows,ids,a,baseline,out):
    start=time.perf_counter();nf=[f['name'] for f in N_FIELDS];rf=[f['name'] for f in R_FIELDS]
    keys=sorted({(int(row[0]),int(row[1])) for row in request_rows});keypos={key:i for i,key in enumerate(keys)}
    nvals=np.full((len(keys),len(nf)),np.nan);n_seen=np.zeros(len(keys),dtype=bool)
    idpos={int(i):p for p,i in enumerate(ids)};rcols={f:[None]*len(ids) for f in rf}
    merged=[dict(missing_reasons={},ratio_support_scaled={}) for _ in ids];seen=np.zeros((len(ids),2),dtype=bool)
    for part in parts:
        for batch in pq.ParquetFile(part['path']).iter_batches(batch_size=256):
            for row in pa.Table.from_batches([batch]).to_pylist():
                # Payloads are generated by this local job, never external input.
                n,r,audit=pickle.loads(row['payload']);ni=keypos[(row['node'],row['cutoff'])];ri=idpos[row['edge_id']];side=0 if row['side']=='src' else 1
                assert not seen[ri,side];seen[ri,side]=True
                nv=np.array([np.nan if n[f] is None else n[f] for f in nf],dtype=np.float64)
                if n_seen[ni]:assert np.allclose(nvals[ni],nv,rtol=1e-12,atol=1e-12,equal_nan=True)
                else:nvals[ni]=nv;n_seen[ni]=True
                for f,v in r.items():rcols[f][ri]=v
                for k in merged[ri]:merged[ri][k].update(audit[k])
    assert seen.all() and n_seen.all()
    ntable=pa.table({'node_id':[k[0] for k in keys],'cutoff_us':[k[1] for k in keys],**{f:pa.array(nvals[:,j],mask=np.isnan(nvals[:,j])) for j,f in enumerate(nf)}})
    pq.write_table(ntable,out/'requested_nodes.parquet',compression='zstd')
    # Identity and expected values are read only during acceptance, after calculation.
    expected=pq.ParquetFile(baseline/'target_features.parquet').read()
    assert expected['edge_id'].to_pylist()==ids.tolist()
    columns={f.name:expected[f.name] for f in expected.schema if f.name not in rf}
    for f in rf:columns[f]=pa.array(rcols[f],type=expected.schema.field(f).type)
    actual=pa.table(columns).select(expected.column_names)
    differences=[]
    for f in rf:
        x=expected[f];y=actual[f]
        if pa.types.is_floating(x.type):
            equal=np.allclose(x.to_numpy(),y.to_numpy(),rtol=1e-12,atol=1e-12,equal_nan=True)
        else:equal=x.equals(y)
        if not equal:differences.append(f)
    if differences:raise AssertionError('R columns differ: '+str(differences))
    audit_expected=pq.ParquetFile(baseline/'target_audit.parquet').read().to_pylist()
    for pos,row in enumerate(audit_expected):
        for key in ['missing_reasons','ratio_support_scaled']:
            assert json.loads(row[key])==merged[pos][key],(pos,key)
    pq.write_table(actual,out/'target_features.parquet',compression='zstd')
    audit_table=pa.table({'transaction_id':expected['transaction_id'],**{key:[json.dumps(v[key],sort_keys=True,separators=(',',':')) for v in merged] for key in ['missing_reasons','ratio_support_scaled']}})
    pq.write_table(audit_table,out/'target_audit.parquet',compression='zstd')
    return dict(seconds=time.perf_counter()-start,R_rows=len(ids),R_values_compared=len(ids)*len(rf),N_rows=len(keys),N_values=len(keys)*len(nf),
        R_matches_original=True,all_missing_reasons_and_ratio_support_match=True,float_tolerance='rtol=atol=1e-12; money/counts/NULL exact')


def compare_nodes(reference,out):
    expected=pq.ParquetFile(Path(reference)/'requested_nodes.parquet').read()
    actual=pq.ParquetFile(Path(out)/'requested_nodes.parquet').read()
    for key in ['node_id','cutoff_us']:assert expected[key].equals(actual[key]),key
    mismatches=[]
    for f in N_FIELDS:
        name=f['name'];x=expected[name];y=actual[name]
        if any(token in name for token in ['_count_','_unique_','_active_']):equal=x.equals(y)
        else:equal=np.allclose(x.to_numpy(),y.to_numpy(),rtol=1e-12,atol=1e-12,equal_nan=True)
        if not equal:mismatches.append(name)
    if mismatches:raise AssertionError('N columns differ: '+str(mismatches))
    return dict(status='passed',node_time_rows=len(expected),values_compared=len(expected)*len(N_FIELDS),reference=str(reference))


def run(archive,baseline,out,mode,partitions,cores,node_reference=None,warm_repeat=False):
    began=time.perf_counter()
    if out.exists() and any(out.iterdir()):raise ValueError('Refusing to overwrite results')
    (out/'parts').mkdir(parents=True)
    meta,a,bank,vocab=load_archive(archive)
    lo=int(np.datetime64('2022-09-04T12:00:00','us').astype(np.int64));hi=int(np.datetime64('2022-09-04T13:00:00','us').astype(np.int64))
    ids=np.flatnonzero((a['timestamp']>=lo)&(a['timestamp']<hi)).astype(np.uint32);ids=ids[np.lexsort((ids,a['timestamp'][ids]))]
    requests=[(int(a[col][i]),int(a['timestamp'][i]),int(i),side) for i in ids for col,side in [('src','src'),('dst','dst')]]
    requests.sort(key=lambda row:(row[0],row[1],row[2],row[3]))
    prepare=time.perf_counter()-began;started=time.perf_counter();startup=0;spark=None
    warm_seconds=0;warm_parts=None;shutdown=0;first_job=None;spark_version=None
    if mode=='spark':
        from pyspark.sql import SparkSession
        scratch=out/'spark_scratch';scratch.mkdir()
        spark=(SparkSession.builder.master(f'local[{cores}]').appName('AML-account-partition-rolling-benchmark')
            .config('spark.ui.enabled','false').config('spark.ui.showConsoleProgress','false')
            .config('spark.sql.shuffle.partitions',str(partitions)).config('spark.sql.adaptive.enabled','false')
            .config('spark.sql.session.timeZone','UTC').config('spark.local.dir',str(scratch))
            .config('spark.hadoop.fs.file.impl','com.globalmentor.apache.hadoop.fs.BareLocalFileSystem')
            .config('spark.python.worker.reuse','true').getOrCreate())
        spark.sparkContext.setLogLevel('ERROR');startup=time.perf_counter()-started
        spark_version=spark.version;job_start=time.perf_counter()
        archive_path=str(archive.resolve());parts_path=str((out/'parts').resolve())
        df=spark.createDataFrame(requests,'node long, cutoff long, edge_id long, side string')
        grouped=df.repartition(partitions,'node').sortWithinPartitions('node','cutoff','edge_id','side')
        print(json.dumps(dict(stage='spark',cores=cores,partitions=partitions,requests=len(requests))),flush=True)
        try:
            parts=grouped.rdd.mapPartitionsWithIndex(lambda p,rs:partition_worker(p,rs,archive_path,parts_path,'rolling')).collect()
            first_job=time.perf_counter()-job_start
            print(json.dumps(dict(stage='spark_first_job_finished',seconds=first_job)),flush=True)
            if warm_repeat:
                warm_start=time.perf_counter();warm_dir=out/'warm'/'parts';warm_dir.mkdir(parents=True)
                warm_path=str(warm_dir.resolve())
                # New DataFrame/RDD reruns partitioning and calculation; no feature/shuffle result cache.
                fresh=spark.createDataFrame(requests,'node long, cutoff long, edge_id long, side string')
                fresh=fresh.repartition(partitions,'node').sortWithinPartitions('node','cutoff','edge_id','side')
                warm_parts=fresh.rdd.mapPartitionsWithIndex(lambda p,rs:partition_worker(p,rs,archive_path,warm_path,'rolling')).collect()
                warm_seconds=time.perf_counter()-warm_start
                print(json.dumps(dict(stage='spark_warm_job_finished',seconds=warm_seconds)),flush=True)
        finally:
            closing=time.perf_counter();spark.stop();shutdown=time.perf_counter()-closing
    else:
        parts=list(partition_worker(0,iter(requests),str(archive),str(out/'parts'),mode))
    calculation=time.perf_counter()-started-warm_seconds
    assert sum(p['accounts'] for p in parts)==len({r[0] for r in requests}), 'An account crossed partitions'
    print(json.dumps(dict(stage='calculated',mode=mode,seconds=calculation,partitions=parts)),flush=True)
    verification=collect(parts,requests,ids,a,baseline,out)
    if node_reference is not None:verification['N_comparison']=compare_nodes(node_reference,out)
    single_run=prepare+calculation+verification['seconds']
    warm_report=None
    if warm_parts is not None:
        assert sum(p['accounts'] for p in warm_parts)==len({r[0] for r in requests})
        warm_verified=collect(warm_parts,requests,ids,a,baseline,out/'warm')
        if node_reference is not None:warm_verified['N_comparison']=compare_nodes(node_reference,out/'warm')
        warm_report=dict(status='passed',job_seconds=warm_seconds,job_plus_merge_verify_seconds=warm_seconds+warm_verified['seconds'],
            verification=warm_verified,partitions=warm_parts,protocol='Same live Spark session, new DataFrame/shuffle; all features recomputed; existing archive and OS cache reused')
        write_json(out/'warm'/'benchmark.json',warm_report)
    report=dict(mode=mode,status='passed',target_rows=len(ids),history_rows=meta['rows'],prepare_seconds=prepare,
        calculation_and_parts_seconds=calculation,spark_startup_seconds=startup,
        spark_first_job_seconds=first_job,spark_shutdown_seconds=shutdown,
        single_run_equivalent_seconds=single_run,warm_repeat=warm_report,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pyarrow=pa.__version__,spark=spark_version),
        total_seconds=time.perf_counter()-began,verification=verification,partitions=parts,
        driver_peak_rss_bytes=peak_rss(),cores=cores if mode=='spark' else 1,
        bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
        protocol='Same 32010 targets, full available history, all 41 N/54 R; account repartition of requests, immutable mmap archive reused; no neural network training',
        timing='Includes worker startup and partial result writes; total also includes comparison and final merge; existing archive/index build excluded for every variant')
    write_json(out/'benchmark.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=Path('data/gnn_v1_dev'));p.add_argument('--baseline',type=Path,default=Path('result/gnn_v1_dev'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['reference','rolling','spark'],required=True);p.add_argument('--partitions',type=int,default=8);p.add_argument('--cores',type=int,default=2)
    p.add_argument('--node-reference',type=Path)
    p.add_argument('--warm-repeat',action='store_true')
    a=p.parse_args();run(a.archive,a.baseline,a.output,a.mode,a.partitions,a.cores,a.node_reference,a.warm_repeat)
