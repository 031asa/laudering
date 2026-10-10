"""Build reusable label-free archive and memory-mapped history indexes from silver."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .contract import VERSION, HOUR, DAY, manifest

COLS={'timestamp':'int64','src':'uint32','dst':'uint32','paid_scaled':'int64','received_scaled':'int64',
      'payment_format_cat':'int16','payment_currency_cat':'int16','receiving_currency_cat':'int16'}

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def scaled(column):
    column=column.combine_chunks() if isinstance(column,pa.ChunkedArray) else column
    if column.type!=pa.decimal128(38,6) or column.null_count: raise ValueError('Amount schema/null contract failed')
    raw=np.frombuffer(column.buffers()[1],dtype='<i8',offset=column.offset*16,count=len(column)*2).reshape(-1,2)
    if np.any(raw[:,1]!=0) or np.any(raw[:,0]<0): raise ValueError('Amount exceeds supported nonnegative scaled-int64 range')
    return raw[:,0].copy()

def encode(col,vocab):
    if col.null_count: raise ValueError('Required category missing')
    index=pc.index_in(col,value_set=pa.array(vocab))
    if index.null_count: raise ValueError('Category missing from build vocabulary')
    return index.to_numpy().astype(np.int16)+2

def edge_encoding(a,bank_key):
    ts=a['timestamp']; hour=ts//HOUR%24
    return {'paid_log1p':np.log1p(a['paid_scaled']/1e6),'received_log1p':np.log1p(a['received_scaled']/1e6),
        'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
        'is_midday':(hour>=11)&(hour<17),'is_cross_bank':bank_key[a['src']]!=bank_key[a['dst']],
        'is_cross_currency':a['payment_currency_cat']!=a['receiving_currency_cat'],
        'is_self_transfer':a['src']==a['dst'],'weekday_cat':((ts//DAY+3)%7).astype(np.int8),
        **{c:a[c] for c in ['payment_format_cat','payment_currency_cat','receiving_currency_cat']}}

def build(project,output,end):
    import psutil
    start=time.perf_counter(); proc=psutil.Process(); peak=proc.memory_info().rss
    if (output/'build.json').exists():
        existing=json.loads((output/'build.json').read_text(encoding='utf-8'))
        if existing['version']!=VERSION or existing['end_exclusive']!=end: raise ValueError('Existing archive differs; choose a new output directory')
        print('Reusing verified archive',output,flush=True); return
    if output.exists() and any(output.iterdir()): raise ValueError('Partial/existing output; choose a new directory, no silent overwrite')
    output.mkdir(parents=True,exist_ok=True)
    for name in ['graph/edges','graph/accounts','graph/adj_index','history_index','metadata','supervision/targets']:
        (output/name).mkdir(parents=True,exist_ok=True)
    split_path=project/'config/dataset_split.json'; split=json.loads(split_path.read_text(encoding='utf-8'))
    train_dates={d for s in split['splits'] if s['name'] in split['train_members'] for d in s['dates']}
    end_us=int(np.datetime64(end,'us').astype(np.int64))
    # Exclusive midnight after the final training day still contains Train only.
    train_start=int(np.datetime64(min(train_dates),'us').astype(np.int64))
    train_end=int(np.datetime64(max(train_dates),'us').astype(np.int64))+DAY
    if not train_start<end_us<=train_end: raise ValueError('Archive must end within frozen Train, including its exclusive upper bound')
    root=project/'data/silver/transactions'; files=sorted(root.glob('transaction_date=*/*.parquet'))
    vocab_sets={c:set() for c in ['payment_format','payment_currency','receiving_currency','bank']}
    print('Fitting train-only categorical vocabularies',flush=True)
    for f in files:
        date=f.parent.name.split('=',1)[1]
        if date not in train_dates: continue
        tab=pq.ParquetFile(f).read(columns=['payment_format','payment_currency','receiving_currency','from_bank','to_bank'])
        for c in ['payment_format','payment_currency','receiving_currency']:
            vocab_sets[c].update(pc.unique(tab[c]).to_pylist())
        for c in ['from_bank','to_bank']: vocab_sets['bank'].update(pc.unique(tab[c]).to_pylist())
    currencies=sorted(vocab_sets['payment_currency']|vocab_sets['receiving_currency'])
    vocab={'payment_format':sorted(vocab_sets['payment_format']),'currency':currencies,'bank':sorted(vocab_sets['bank'])}
    if any(None in v for v in vocab.values()): raise ValueError('NULL vocabulary')
    write_json(output/'metadata/vocabulary.json',dict(values=vocab,offset=2,missing=0,UNK=1,fit_dates=sorted(train_dates)))
    accounts=pa.concat_tables([pq.ParquetFile(f).read(columns=['bank_id','account_number']) for f in sorted((project/'data/silver/accounts').glob('*.parquet'))]).sort_by([('bank_id','ascending'),('account_number','ascending')])
    frame=accounts.to_pandas(); index=pd.MultiIndex.from_frame(frame)
    if not index.is_unique: raise ValueError('Duplicate account keys')
    node_count=len(accounts)
    allbanks=sorted(pc.unique(accounts['bank_id']).to_pylist())
    bank_key=pc.index_in(accounts['bank_id'],value_set=pa.array(allbanks)).to_numpy().astype(np.int32)
    bank_cat=pc.fill_null(pc.add(pc.index_in(accounts['bank_id'],value_set=pa.array(vocab['bank'])),2),1).to_numpy().astype(np.int32)
    accounts=accounts.append_column('node_id',pa.array(np.arange(node_count,dtype=np.uint32))).append_column('bank_cat',pa.array(bank_cat))
    pq.write_table(accounts,output/'graph/accounts/accounts.parquet',compression='zstd')
    np.save(output/'history_index/bank_key.npy',bank_key); np.save(output/'history_index/bank_cat.npy',bank_cat)
    write_json(output/'metadata/bank_identity.json',allbanks)
    chosen=[f for f in files if int(np.datetime64(f.parent.name.split('=',1)[1],'us').astype(np.int64))<end_us]
    capacity=sum(pq.ParquetFile(f).metadata.num_rows for f in chosen)
    arrays={name:np.lib.format.open_memmap(output/f'history_index/{name}.npy',mode='w+',dtype=dtype,shape=(capacity,)) for name,dtype in COLS.items()}
    count=0; t0=None; id_hash=hashlib.sha256(); file_info=[]
    date_split={d:s['name'] for s in split['splits'] for d in s['dates']}
    for fi,f in enumerate(chosen):
        edge_path=output/'graph/edges'/f.parent.name/f.name
        edge_path.parent.mkdir(parents=True,exist_ok=True)
        label_path=output/'supervision/targets'/f.parent.name/f.name
        label_path.parent.mkdir(parents=True,exist_ok=True)
        ew=lw=None; written=0
        for batch in pq.ParquetFile(f).iter_batches(batch_size=100_000):
            tab=pa.Table.from_batches([batch]); ts=tab['timestamp'].cast(pa.int64()).to_numpy()
            tab=tab.filter(pa.array(ts<end_us))
            if not len(tab): continue
            ts=tab['timestamp'].cast(pa.int64()).to_numpy()
            src=index.get_indexer(pd.MultiIndex.from_arrays([tab['from_bank'].to_pandas(),tab['from_account'].to_pandas()]))
            dst=index.get_indexer(pd.MultiIndex.from_arrays([tab['to_bank'].to_pandas(),tab['to_account'].to_pandas()]))
            if min(src.min(),dst.min())<0: raise ValueError('Unmapped account')
            vals={'timestamp':ts,'src':src.astype(np.uint32),'dst':dst.astype(np.uint32),
                  'paid_scaled':scaled(tab['amount_paid']),'received_scaled':scaled(tab['amount_received']),
                  'payment_format_cat':encode(tab['payment_format'],vocab['payment_format']),
                  'payment_currency_cat':encode(tab['payment_currency'],vocab['currency']),
                  'receiving_currency_cat':encode(tab['receiving_currency'],vocab['currency'])}
            size=len(tab); edge_ids=np.arange(count,count+size,dtype=np.uint32)
            for k,v in vals.items(): arrays[k][count:count+size]=v
            edge=tab.drop(['is_laundering']).append_column('edge_id',pa.array(edge_ids)).append_column('src',pa.array(vals['src'])).append_column('dst',pa.array(vals['dst']))
            for k,v in edge_encoding(vals,bank_key).items(): edge=edge.append_column(k,pa.array(v))
            labels=pa.table({'transaction_id':tab['transaction_id'],'edge_id':edge_ids,'label':tab['is_laundering'],'split':[date_split[f.parent.name.split('=',1)[1]]]*size})
            if ew is None:
                ew=pq.ParquetWriter(edge_path,edge.schema,compression='zstd'); lw=pq.ParquetWriter(label_path,labels.schema,compression='zstd')
            ew.write_table(edge); lw.write_table(labels)
            t0=min(t0,int(ts.min())) if t0 is not None else int(ts.min())
            id_hash.update(tab['transaction_id'].combine_chunks().buffers()[2].to_pybytes())
            count+=size; written+=size; peak=max(peak,proc.memory_info().rss)
        if ew: ew.close(); lw.close()
        file_info.append(dict(path=str(f.relative_to(project)),rows=written))
        print(json.dumps({'stage':'archive','file':fi+1,'files':len(chosen),'rows':count,'rss_mb':round(proc.memory_info().rss/1e6)}),flush=True)
    del index,frame,accounts,tab; gc.collect()
    for arr in arrays.values(): arr.flush()
    arrays={k:v[:count] for k,v in arrays.items()}
    ext=arrays['src']!=arrays['dst']
    for d,col in [('in','dst'),('out','src')]:
        print('Building exact '+d+' time index',flush=True)
        order=np.lexsort((np.arange(count,dtype=np.uint32),arrays['timestamp'],arrays[col])).astype(np.uint32)
        ptr=np.r_[0,np.cumsum(np.bincount(arrays[col].astype(np.int64),minlength=node_count))]
        np.save(output/f'history_index/{d}_order.npy',order); np.save(output/f'history_index/{d}_ptr.npy',ptr)
        extorder=order[ext[order]]
        extptr=np.r_[0,np.cumsum(np.bincount(arrays[col][extorder].astype(np.int64),minlength=node_count))]
        np.save(output/f'graph/adj_index/{d}_edge_ids.npy',extorder); np.save(output/f'graph/adj_index/{d}_ptr.npy',extptr)
        peak=max(peak,proc.memory_info().rss); del order,ptr,extorder,extptr; gc.collect()
    write_json(output/'metadata/feature_manifest.json',manifest())
    metadata=dict(version=VERSION,rows=count,allocated_rows=capacity,node_count=node_count,external_rows=int(ext.sum()),
        data_start_us=t0,end_exclusive=end,source_manifest_sha256=hashlib.sha256((root/'_manifest.json').read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),edge_identity_digest=id_hash.hexdigest(),
        files=file_info,wall_seconds=time.perf_counter()-start,peak_rss_sampled_bytes=peak,
        disk_bytes=sum(f.stat().st_size for f in output.rglob('*') if f.is_file()),
        scope='All silver history from data start to end_exclusive; no account filtering; train vocabulary through frozen train end',
        status='built; feature acceptance is separate')
    write_json(output/'build.json',metadata)
    print(json.dumps(metadata,ensure_ascii=True),flush=True)

def load_archive(output):
    meta=json.loads((output/'build.json').read_text(encoding='utf-8'))
    arrays={name:np.load(output/f'history_index/{name}.npy',mmap_mode='r')[:meta['rows']] for name in COLS}
    for d in ['in','out']:
        for suffix in ['order','ptr']: arrays[d+'_'+suffix]=np.load(output/f'history_index/{d}_{suffix}.npy',mmap_mode='r')
    bank=np.load(output/'history_index/bank_key.npy',mmap_mode='r')
    vocab=json.loads((output/'metadata/vocabulary.json').read_text(encoding='utf-8'))['values']
    return meta,arrays,bank,vocab

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);p.add_argument('--end',default='2022-09-04T13:00:00'); args=p.parse_args()
    build(args.project,args.output,args.end)
