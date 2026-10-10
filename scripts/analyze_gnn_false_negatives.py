"""Retrospective Validation-only error analysis. Never trains or alters inputs."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gnn_features.contract import R_FIELDS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/gnn_error_analysis_v1'
ARCH=ROOT/'data/gnn_v1_validation'
PREFIX=20318992
TOTAL=4689006
DATES=['2022-09-11','2022-09-12','2022-09-13']
SEED=20261002

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def progress(stage,**kw):
    print(json.dumps(dict(stage=stage,**kw),ensure_ascii=True),flush=True)
def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(x) for x in obj]
    if isinstance(obj,np.integer):return int(obj)
    if isinstance(obj,np.floating):return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj

def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    if (OUT/'basic_analysis.json').exists():
        raise FileExistsError('Analysis already exists; inspect before replacing')
    meta=read(ARCH/'build.json')
    assert meta['end_exclusive']=='2022-09-14T00:00:00' and meta['rows']==PREFIX+TOTAL
    arr={p.stem:np.load(p,mmap_mode='r') for p in (ARCH/'history_index').glob('*.npy')}
    vocab=read(ARCH/'metadata/vocabulary.json')['values']
    best=read(ROOT/'result/gnn_tuning_v1/best_model.json')
    threshold=best['validation_threshold'];assert best['name']=='lr_only'
    labels=np.full(TOTAL,-1,dtype=np.int8);scores={};provenance={}
    expected_times=arr['timestamp'][PREFIX:]
    for name in ['lr_only','baseline','sqrt_weight']:
        folder=ROOT/f'result/gnn_tuning_v1/evaluations/{name}'
        metrics=read(folder/'metrics.json');config_sha=sha(folder/'config.json')
        score=np.full(TOTAL,np.nan,dtype=np.float32);seen=np.zeros(TOTAL,bool)
        for entry in metrics['blocks']:
            assert entry['block'].split('/')[0].split('=')[1] in DATES
            block=folder/'predictions'/entry['block'];manifest=read(block/'manifest.json')
            assert sha(block/'manifest.json')==entry['manifest_sha256']
            assert manifest['config_sha256']==config_sha
            assert manifest['file_sha256']==sha(block/'predictions.parquet')
            table=pq.ParquetFile(block/'predictions.parquet').read()
            ids=table['edge_id'].to_numpy();p=ids-PREFIX
            assert len(p)==entry['targets'] and (p>=0).all() and (p<TOTAL).all()
            assert len(np.unique(p))==len(p) and not seen[p].any()
            assert hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()==manifest['target_ids_sha256']
            assert np.array_equal(table['cutoff_us'].to_numpy(),expected_times[p])
            y=table['label'].to_numpy()
            if name=='lr_only':labels[p]=y
            else: assert np.array_equal(labels[p],y)
            score[p]=table['probability'].to_numpy();seen[p]=True
        assert seen.all() and np.isfinite(score).all() and labels.sum()==6420
        scores[name]=score
        provenance[name]={'metrics_sha256':sha(folder/'metrics.json'),'config_sha256':config_sha,'prediction_blocks':len(metrics['blocks'])}
        progress('predictions_verified',model=name,rows=TOTAL)
    positive=labels==1;detected=scores['lr_only']>=threshold
    fn=positive&~detected;tp=positive&detected;fp=~positive&detected
    assert (fn.sum(),tp.sum(),fp.sum())==(2905,3515,810)
    src=np.asarray(arr['src'][PREFIX:]);dst=np.asarray(arr['dst'][PREFIX:])
    fmt=np.asarray(arr['payment_format_cat'][PREFIX:]);cur=np.asarray(arr['payment_currency_cat'][PREFIX:])
    paid=np.asarray(arr['paid_scaled'][PREFIX:])/1e6
    seconds=np.asarray(expected_times)//1000000
    date=(seconds//86400).astype(np.int32)
    daylabels={int(np.datetime64(d,'D').astype(int)):d for d in DATES}
    bank=arr['bank_key']
    basic=pd.DataFrame({'date':date,'hour':seconds//3600%24,'payment_format':fmt,'currency':cur,
                        'from_bank_key':bank[src],'to_bank_key':bank[dst],
                        'self_transfer':src==dst,'cross_bank':bank[src]!=bank[dst],
                        'cross_currency':cur!=arr['receiving_currency_cat'][PREFIX:],
                        'amount_band':np.searchsorted([1,10,100,1000,10000,100000,1000000],paid,side='right'),
                        'positive':positive,'tp':tp,'fn':fn,'fp':fp})
    amountlabels=['<1','1-10','10-100','100-1k','1k-10k','10k-100k','100k-1m','>=1m']
    grouping={}
    def category(k,v):
        if k=='date':return daylabels[int(v)]
        if k=='payment_format':return vocab['payment_format'][int(v)-2] if int(v)>=2 else 'UNK'
        if k=='currency':return vocab['currency'][int(v)-2] if int(v)>=2 else 'UNK'
        if k=='amount_band':return amountlabels[int(v)]
        return int(v) if isinstance(v,(int,np.integer)) else v
    for keys in [['date'],['payment_format'],['currency'],['hour'],['self_transfer'],['cross_bank'],['cross_currency'],
                 ['from_bank_key'],['to_bank_key'],['date','payment_format'],['currency','amount_band']]:
        g=basic.groupby(keys,observed=True,sort=True).agg(total=('positive','size'),positives=('positive','sum'),tp=('tp','sum'),fn=('fn','sum'),fp=('fp','sum')).reset_index()
        g['recall']=g['tp']/g['positives'].replace(0,np.nan)
        g['precision']=g['tp']/(g['tp']+g['fp']).replace(0,np.nan)
        g['fn_share']=g['fn']/2905
        for k in keys:g[k]=g[k].map(lambda v:category(k,v))
        if keys in [['from_bank_key'],['to_bank_key']]:
            g=g[(g.positives>0)|(g.fp>0)].sort_values('fn',ascending=False)
        grouping['+'.join(keys)]=clean(g.to_dict('records'))
    q=[0,.1,.25,.5,.75,.9,1]
    score_summary={name:dict(zip(map(str,q),np.quantile(scores['lr_only'][mask],q).tolist()))
                   for name,mask in [('FN',fn),('TP',tp),('FP',fp),('TN',~positive&~detected)]}
    bins=[0,.01,.1,.3,.5,.7,.9,threshold,1.000001]
    scorebands=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(scores['lr_only']>=lo)&(scores['lr_only']<hi)
        scorebands.append(dict(lo=lo,hi=hi,total=int(mask.sum()),positive=int((mask&positive).sum()),negative=int((mask&~positive).sum())))
    model_overlap={}
    for name in ['baseline','sqrt_weight']:
        th=read(ROOT/f'result/gnn_tuning_v1/evaluations/{name}/metrics.json')['metrics']['best_f1']['threshold']
        hit=scores[name]>=th
        model_overlap[name]=dict(threshold=th,lr_fn_other_tp=int((fn&hit).sum()),lr_tp_other_fn=int((tp&~hit).sum()),
                                 both_fn=int((fn&~hit).sum()),union_tp=int((positive&(detected|hit)).sum()),union_fp=int((~positive&(detected|hit)).sum()))
    # Sample controls only for richer feature inspection. Full-population grouping above is exact.
    rng=np.random.default_rng(SEED)
    negative_sample=np.sort(rng.choice(np.flatnonzero(~positive),size=100000,replace=False))
    selected=np.unique(np.r_[np.flatnonzero(positive|fp),negative_sample])
    is_selected=np.zeros(TOTAL,bool);is_selected[selected]=True
    sample=basic.iloc[selected].copy()
    sample.insert(0,'edge_id',selected+PREFIX)
    sample['src']=src[selected];sample['dst']=dst[selected]
    sample['timestamp']=np.asarray(expected_times[selected],dtype='datetime64[us]')
    sample['amount_paid_float']=paid[selected]
    for name,score in scores.items():sample['score_'+name]=score[selected]
    sample['group']=np.where(positive[selected],np.where(detected[selected],'TP','FN'),np.where(detected[selected],'FP','TN_sample'))
    sample['uniform_negative_control']=np.isin(selected,negative_sample)
    index=np.full(TOTAL,-1,np.int32);index[selected]=np.arange(len(selected))
    # Exact raw IDs/decimal amounts and metadata, aligned by edge_id.
    raw=[];raw_seen=np.zeros(len(selected),bool)
    cols=['edge_id','transaction_id','from_bank','from_account','to_bank','to_account','amount_paid','amount_received','payment_currency','receiving_currency','payment_format','source_record_number']
    for d in DATES:
        for p in sorted((ARCH/f'graph/edges/transaction_date={d}').glob('*.parquet')):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=65536,columns=cols):
                table=pa.Table.from_batches([batch]);pos=table['edge_id'].to_numpy()-PREFIX
                keep=is_selected[pos]
                if keep.any():
                    tab=table.filter(pa.array(keep));positions=index[tab['edge_id'].to_numpy()-PREFIX]
                    assert not raw_seen[positions].any();raw_seen[positions]=True
                    frame=tab.to_pandas()
                    frame['amount_paid']=frame['amount_paid'].map(str);frame['amount_received']=frame['amount_received'].map(str)
                    raw.append(frame)
        progress('raw_join',date=d)
    assert raw_seen.all()
    raw=pd.concat(raw).sort_values('edge_id').reset_index(drop=True)
    sample=sample.drop(columns=['payment_format']).reset_index(drop=True)
    sample=sample.merge(raw,on='edge_id',validate='one_to_one',sort=False)
    assert np.array_equal(sample.edge_id.to_numpy(),selected+PREFIX)
    rmatrix=np.full((len(selected),len(R_FIELDS)),np.nan,np.float64);r_seen=np.zeros(len(selected),bool)
    for d in DATES:
        folder=ROOT/f'result/gnn_validation_v1/transaction_date={d}'
        assert read(folder/'daily_run.json')['status']=='verified'
        for entry in read(folder/'merge_phase.json')['parts']:
            p=Path(entry['path']).resolve();assert p.is_relative_to(folder.resolve())
            for batch in pq.ParquetFile(p).iter_batches(batch_size=16384,columns=['edge_id']+[f['name'] for f in R_FIELDS]):
                tab=pa.Table.from_batches([batch]);pos=tab['edge_id'].to_numpy()-PREFIX;keep=is_selected[pos]
                if keep.any():
                    tab=tab.filter(pa.array(keep));indices=index[tab['edge_id'].to_numpy()-PREFIX]
                    assert not r_seen[indices].any();r_seen[indices]=True
                    for j,f in enumerate(R_FIELDS):
                        # to_numpy represents nullable float entries with NaN; no Arrow fill_null.
                        rmatrix[indices,j]=tab[f['name']].cast(pa.float64()).to_numpy()
        progress('stored_R_join',date=d)
    assert r_seen.all()
    sample=pd.concat([sample,pd.DataFrame(rmatrix,columns=[f['name'] for f in R_FIELDS])],axis=1)
    sample.to_parquet(OUT/'inspection_sample.parquet',index=False)
    sample[sample.positive].to_csv(OUT/'all_validation_positives.csv',index=False,encoding='utf-8-sig')
    sample[sample.group=='FN'].to_csv(OUT/'false_negatives.csv',index=False,encoding='utf-8-sig')
    feature_summary={}
    masks={'FN':sample.group.eq('FN'),'TP':sample.group.eq('TP'),'negative_uniform':sample.uniform_negative_control,'FP':sample.group.eq('FP')}
    for f in R_FIELDS:
        name=f['name'];feature_summary[name]={}
        for key,mask in masks.items():
            values=sample.loc[mask,name].dropna().to_numpy()
            feature_summary[name][key]={'count':int(mask.sum()),'valid':len(values),'missing_rate':float(sample.loc[mask,name].isna().mean()),
                                      'quantiles':dict(zip(['q10','q25','q50','q75','q90'],np.quantile(values,[.1,.25,.5,.75,.9]).tolist())) if len(values) else {}}
    # Threshold tradeoff on saved predictions, with whole ties included.
    order=np.argsort(-scores['lr_only'],kind='stable');s=scores['lr_only'][order];yy=labels[order]
    cum=np.cumsum(yy,dtype=np.int64);ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tps=cum[ends];fps=ends+1-tps;rec=tps/6420;prec=tps/(ends+1)
    tradeoffs=[]
    for kind,targets in [('precision',[.7,.8,.9]),('recall',[.6,.7,.8,.9])]:
        for target in targets:
            valid=np.flatnonzero((prec>=target) if kind=='precision' else (rec>=target))
            i=int(valid[np.argmax(rec[valid])]) if kind=='precision' else int(valid[0])
            tradeoffs.append(dict(constraint=kind,target=target,threshold=float(s[ends[i]]),precision=float(prec[i]),recall=float(rec[i]),tp=int(tps[i]),fp=int(fps[i]),alerts=int(ends[i]+1)))
    summary=dict(status='completed',scope='Retrospective Validation diagnosis; labels for error grouping only. No Test, no retraining.',
                 selected_model='lr_only',threshold=threshold,samples=TOTAL,positives=6420,tp=int(tp.sum()),fn=int(fn.sum()),fp=int(fp.sum()),
                 fn_below_05=int((fn&(scores['lr_only']<.5)).sum()),fn_at_least_05=int((fn&(scores['lr_only']>=.5)).sum()),
                 scores=score_summary,scorebands=scorebands,groups=grouping,other_model_overlap=model_overlap,threshold_tradeoffs=tradeoffs,
                 R_features=feature_summary,negative_control={'method':'simple random sample without replacement from all Validation negatives','seed':SEED,'count':100000},
                 selected_rows=len(selected),source_hashes=provenance,wall_seconds=time.time()-start)
    save('basic_analysis.json',clean(summary))
    progress('complete',seconds=summary['wall_seconds'],selected_rows=len(selected))

if __name__=='__main__':main()
