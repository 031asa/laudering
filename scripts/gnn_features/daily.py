"""Full-day, bounded-memory Spark run. Only partition summaries return to driver."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import time
import uuid
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .build import load_archive,write_json
from .benchmark import partition_worker,peak_rss
from .contract import DAY,R_FIELDS,N_FIELDS
from .run import feature_type


def merge_parquet_file(partition,path,output,data_start):
    """Bound JVM/Python transfer: exchange file paths, stream payloads locally."""
    def rows():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():yield SimpleNamespace(**row)
    return merge_partition(partition,rows(),output,data_start)


def merge_partition(partition,rows,output,data_start):
    """Join two endpoint fragments with a streaming, edge-local buffer."""
    from pyspark import TaskContext
    began=time.perf_counter();attempt=TaskContext.get().taskAttemptId()
    path=Path(output)/f'part-{partition:04d}-{attempt}.parquet'
    audit_path=Path(output).parent/'audit'/path.name
    schema=pa.schema([('transaction_id',pa.string()),('edge_id',pa.int64()),('timestamp',pa.timestamp('us'))]+
        [(f['name'],feature_type(f['name'])) for f in R_FIELDS]+[('full_6h',pa.bool_()),('full_day',pa.bool_())])
    audit_schema=pa.schema([('transaction_id',pa.string()),('edge_id',pa.int64()),('missing_reasons',pa.string()),('ratio_support_scaled',pa.string())])
    fw=pq.ParquetWriter(path,schema,compression='zstd');aw=pq.ParquetWriter(audit_path,audit_schema,compression='zstd')
    features=[];audits=[];count=0;nulls={f['name']:0 for f in R_FIELDS};last_progress=began
    names={f['name'] for f in R_FIELDS}
    for edge_id,group in itertools.groupby(rows,key=lambda r:int(r.edge_id)):
        pair=list(group)
        if len(pair)!=2 or {r.side for r in pair}!={'src','dst'}:raise AssertionError('Target must have exactly two distinct sides')
        r={};missing={};support={};t=int(pair[0].cutoff);tx=pair[0].transaction_id
        for row in pair:
            if row.cutoff!=t or row.transaction_id!=tx:raise AssertionError('Mismatched target identity')
            n,fragment,audit=pickle.loads(row.payload)
            if set(r)&set(fragment):raise AssertionError('Overlapping R fragments')
            r.update(fragment);missing.update(audit['missing_reasons']);support.update(audit['ratio_support_scaled'])
        if set(r)!=names or {k for k,v in r.items() if v is None}!=set(missing):raise AssertionError('Incomplete R or missing audit')
        for k in missing:nulls[k]+=1
        features.append(dict(transaction_id=tx,edge_id=edge_id,timestamp=np.datetime64(t,'us').item(),**r,
            full_6h=t-6*3_600_000_000>=data_start,full_day=t-t%DAY>=data_start))
        audits.append(dict(transaction_id=tx,edge_id=edge_id,missing_reasons=json.dumps(missing,separators=(',',':')),
            ratio_support_scaled=json.dumps(support,separators=(',',':'))))
        count+=1
        if len(features)>=256:
            fw.write_table(pa.Table.from_pylist(features,schema=schema));aw.write_table(pa.Table.from_pylist(audits,schema=audit_schema))
            features.clear();audits.clear()
        if time.perf_counter()-last_progress>30:
            print(json.dumps(dict(stage='merge_progress',partition=partition,targets=count)),flush=True);last_progress=time.perf_counter()
    if features:
        fw.write_table(pa.Table.from_pylist(features,schema=schema));aw.write_table(pa.Table.from_pylist(audits,schema=audit_schema))
    fw.close();aw.close()
    return iter([dict(partition=partition,path=str(path),audit_path=str(audit_path),rows=count,null_counts=nulls,
        seconds=time.perf_counter()-began,worker_peak_rss_bytes=peak_rss())])


def prepare(archive,out,date,meta,a):
    began=time.perf_counter();lo=int(np.datetime64(date,'us').astype(np.int64));hi=lo+DAY
    coverage_end=int(np.datetime64(meta['end_exclusive'],'us').astype(np.int64))
    if lo<meta['data_start_us'] or hi>coverage_end:raise ValueError('Archive does not cover complete requested day')
    ids=np.flatnonzero((a['timestamp']>=lo)&(a['timestamp']<hi)).astype(np.uint32)
    if not len(ids):raise ValueError('Empty target date')
    (out/'requests').mkdir();(out/'catalog').mkdir()
    schema=pa.schema([('node',pa.int64()),('cutoff',pa.int64()),('edge_id',pa.int64()),('side',pa.string())])
    cat_schema=pa.schema([('edge_id',pa.int64()),('transaction_id',pa.string())])
    rw=pq.ParquetWriter(out/'requests/requests.parquet',schema,compression='zstd')
    cw=pq.ParquetWriter(out/'catalog/catalog.parquet',cat_schema,compression='zstd')
    count=0
    for f in sorted((archive/'graph/edges'/f'transaction_date={date}').glob('*.parquet')):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32768,columns=['edge_id','transaction_id','timestamp','src','dst']):
            tab=pa.Table.from_batches([batch]);edge=tab['edge_id'].cast(pa.int64());ts=tab['timestamp'].cast(pa.int64())
            tt=ts.to_numpy()
            if not np.all((tt>=lo)&(tt<hi)):raise AssertionError('Wrong date in archive partition')
            for col,side in [('src','src'),('dst','dst')]:
                rw.write_table(pa.Table.from_arrays([tab[col].cast(pa.int64()),ts,edge,pa.array([side]*len(tab))],schema=schema))
            cw.write_table(pa.Table.from_arrays([edge,tab['transaction_id']],schema=cat_schema));count+=len(tab)
    rw.close();cw.close()
    if count!=len(ids):raise AssertionError('Catalog/archive target counts differ')
    np.save(out/'target_edge_ids.npy',ids)
    result=dict(date=date,start_us=lo,end_us=hi,target_rows=count,requests=2*count,
        accounts=len(np.unique(np.r_[a['src'][ids],a['dst'][ids]])),seconds=time.perf_counter()-began)
    write_json(out/'prepare.json',result);return result


def run(archive,out,date,cores,partitions,resume):
    began=time.perf_counter();archive=archive.resolve();out=out.resolve()
    meta,a,bank,vocab=load_archive(archive)
    config=dict(archive=str(archive),archive_manifest_sha256=hashlib.sha256((archive/'build.json').read_bytes()).hexdigest(),
        date=date,cores=cores,partitions=partitions,algorithm_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['rolling.py','partitioned.py']})
    if out.exists() and any(out.iterdir()):
        if not resume or not (out/'run_config.json').exists():raise ValueError('Existing run requires explicit --resume')
        if json.loads((out/'run_config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Resume configuration differs')
    else:out.mkdir(parents=True,exist_ok=True);write_json(out/'run_config.json',config)
    prep=json.loads((out/'prepare.json').read_text(encoding='utf-8')) if (out/'prepare.json').exists() else prepare(archive,out,date,meta,a)
    print(json.dumps(dict(stage='prepared',**prep)),flush=True)
    from pyspark.sql import SparkSession
    startup=time.perf_counter()
    spark=(SparkSession.builder.master(f'local[{cores}]').appName('AML-full-day-rolling-features')
        .config('spark.ui.enabled','false').config('spark.ui.showConsoleProgress','false')
        .config('spark.sql.shuffle.partitions',str(partitions)).config('spark.sql.adaptive.enabled','false')
        .config('spark.sql.session.timeZone','UTC').config('spark.local.dir',str(out/'spark_scratch'))
        .config('spark.hadoop.fs.file.impl','com.globalmentor.apache.hadoop.fs.BareLocalFileSystem')
        .config('spark.python.worker.reuse','true').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR');startup=time.perf_counter()-startup
    try:
        checkpoint=out/'account_phase.json'
        if checkpoint.exists():phase=json.loads(checkpoint.read_text(encoding='utf-8'));parts=phase['parts']
        else:
            started=time.perf_counter();(out/'parts').mkdir(exist_ok=True)
            df=spark.read.parquet((out/'requests').as_posix()).repartition(partitions,'node').sortWithinPartitions('node','cutoff','edge_id','side')
            archive_path=str(archive);part_path=str(out/'parts')
            parts=df.rdd.mapPartitionsWithIndex(lambda p,rs:partition_worker(p,rs,archive_path,part_path,'rolling')).collect()
            if sum(p['requests'] for p in parts)!=prep['requests'] or sum(p['accounts'] for p in parts)!=prep['accounts']:raise AssertionError('Account partition coverage failed')
            phase=dict(seconds=time.perf_counter()-started,parts=parts);write_json(checkpoint,phase)
        print(json.dumps(dict(stage='account_phase_complete',seconds=phase['seconds'],requests=sum(p['requests'] for p in parts))),flush=True)
        checkpoint=out/'merge_phase.json'
        if checkpoint.exists():merge=json.loads(checkpoint.read_text(encoding='utf-8'))
        else:
            started=time.perf_counter();(out/'features').mkdir(exist_ok=True);(out/'audit').mkdir(exist_ok=True)
            partial=spark.read.parquet(*[Path(p['path']).as_posix() for p in parts])
            catalog=spark.read.parquet((out/'catalog').as_posix())
            paired=partial.join(catalog,'edge_id').select('edge_id','side','cutoff','payload','transaction_id').repartition(partitions,'edge_id').sortWithinPartitions('edge_id','side')
            dest=str(out/'features');data_start=meta['data_start_us']
            # Large binary rows previously crossed the JVM/Python socket here.
            # Materialize sorted partitions natively, then pass only file paths.
            # Each edge stays in one file; no max-records splitting is permitted.
            materialized=out/('merge_input_'+uuid.uuid4().hex)
            paired.write.option('maxRecordsPerFile',0).parquet(materialized.as_posix())
            files=sorted(materialized.glob('part-*.parquet'))
            if not files:raise AssertionError('No materialized merge partitions')
            jobs=[(i,str(path)) for i,path in enumerate(files)]
            merged=spark.sparkContext.parallelize(jobs,min(partitions,len(jobs))).flatMap(lambda item:merge_parquet_file(item[0],item[1],dest,data_start)).collect()
            if sum(p['rows'] for p in merged)!=prep['target_rows']:raise AssertionError('Merged output coverage failed')
            merge=dict(seconds=time.perf_counter()-started,parts=merged);write_json(checkpoint,merge)
        print(json.dumps(dict(stage='merge_phase_complete',seconds=merge['seconds'],rows=sum(p['rows'] for p in merge['parts']))),flush=True)
    finally:spark.stop()
    result=dict(status='computed_pending_verification',date=date,target_rows=prep['target_rows'],history_rows=meta['rows'],
        history_start_us=meta['data_start_us'],history_end=meta['end_exclusive'],cores=cores,partitions=partitions,
        prepare_seconds=prep['seconds'],spark_startup_seconds=startup,account_phase_seconds=phase['seconds'],merge_phase_seconds=merge['seconds'],
        invocation_seconds=time.perf_counter()-began,completed_stage_seconds=prep['seconds']+startup+phase['seconds']+merge['seconds'],
        node_time_queries=sum(p['node_time_queries'] for p in phase['parts']),driver_peak_rss_bytes=peak_rss(),
        disk_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
        scope='All transactions of one complete day; all prior history; no labels or GNN training; only partition summaries collected on driver')
    write_json(out/'daily_run.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--date',default='2022-09-04');p.add_argument('--cores',type=int,default=4);p.add_argument('--partitions',type=int,default=8);p.add_argument('--resume',action='store_true')
    args=p.parse_args();run(args.archive,args.output,args.date,args.cores,args.partitions,args.resume)
