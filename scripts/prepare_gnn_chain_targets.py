"""Prepare verified fixed Validation cohorts and uniform negative candidates."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gnn_features.v2_compare import load_predictions
from export_gnn_missed_v2_features import export_v2_validation_features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/gnn_chain_audit_20261003'
PREFIX,TOTAL=20318992,4689006

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(**detail):print(json.dumps(detail,ensure_ascii=True),flush=True)

def main():
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'candidates.parquet').exists()
    archive=ROOT/'data/gnn_v1_validation'
    frozen={str(Path(k)):v for k,v in read(ROOT/'data/gnn_validation_cache_parquet/config.json')['hashes'].items()}
    sc=read(ROOT/'data/gnn_v2_target_features/validation/config.json')
    r_hash={str(Path(k)):v for k,v in sc['input_hashes'].items()}
    used={}
    def verify(p,hashes=frozen):
        p=Path(p).resolve();rel=str(p.relative_to(ROOT));digest=sha(p)
        assert hashes[rel]==digest,rel
        used[rel]=digest
        return p
    arrays={n:np.load(verify(archive/'history_index'/f'{n}.npy'),mmap_mode='r') for n in ['timestamp','src','dst']}
    y,scores,pred_audit=load_predictions(ROOT,'lr_only',arrays['timestamp'][PREFIX:])
    source=ROOT/'result/gnn_error_analysis_v1/inspection_sample.parquet'
    base=pd.read_parquet(source)
    base=base[base.positive|base.uniform_negative_control].copy().sort_values('edge_id').reset_index(drop=True)
    ids=base.edge_id.to_numpy(np.int64);pos=ids-PREFIX
    assert len(base)==106420 and base.uniform_negative_control.sum()==100000
    assert np.array_equal(base.positive.to_numpy(),y[pos]==1)
    assert np.array_equal(base.score_lr_only.to_numpy(),scores[pos])
    expected=np.sort(np.random.default_rng(20261002).choice(np.flatnonzero(y==0),100000,replace=False))+PREFIX
    assert np.array_equal(np.sort(base.loc[base.uniform_negative_control,'edge_id'].to_numpy()),expected)
    select=np.zeros(TOTAL,bool);select[pos]=True
    raw_cols=['edge_id','timestamp','src','dst','from_bank','from_account','to_bank','to_account','amount_paid','amount_received','payment_format','payment_currency','receiving_currency']
    raw=[];r=[]
    for date in ['2022-09-11','2022-09-12','2022-09-13']:
        for p in sorted((archive/f'graph/edges/transaction_date={date}').glob('*.parquet')):
            verify(p)
            for b in pq.ParquetFile(p).iter_batches(columns=raw_cols,batch_size=65536,use_threads=False):
                tab=pa.Table.from_batches([b]);local=tab['edge_id'].to_numpy().astype(np.int64)-PREFIX
                keep=select[local]
                if keep.any():raw.append(tab.filter(pa.array(keep)).to_pandas())
        m=verify(ROOT/f'result/gnn_validation_v1/transaction_date={date}/merge_phase.json')
        for part in read(m)['parts']:
            p=verify(part['path'],r_hash)
            for b in pq.ParquetFile(p).iter_batches(columns=['edge_id','timestamp','r_pair_fwd_count_72h','r_src_ext_out_count_24h','r_dst_ext_in_count_24h'],batch_size=65536,use_threads=False):
                tab=pa.Table.from_batches([b]);local=tab['edge_id'].to_numpy().astype(np.int64)-PREFIX;keep=select[local]
                if keep.any():r.append(tab.filter(pa.array(keep)).to_pandas())
        emit(stage='raw_and_R_verified',date=date)
    raw=pd.concat(raw).sort_values('edge_id').reset_index(drop=True)
    r=pd.concat(r).sort_values('edge_id').reset_index(drop=True)
    assert np.array_equal(raw.edge_id,ids) and np.array_equal(r.edge_id,ids)
    for col in raw_cols:
        if col in ['amount_paid','amount_received']:
            values=raw[col].map(lambda v:format(v,'.6f'))
            assert np.array_equal(values,base[col]);raw[col]=values
        else:assert np.array_equal(raw[col],base[col]),col
    for col in ['r_pair_fwd_count_72h','r_src_ext_out_count_24h','r_dst_ext_in_count_24h']:
        assert np.array_equal(r[col].to_numpy(),base[col].to_numpy(),equal_nan=True)
    def progress(d):
        if d.get('completed',0)%25==0 or d.get('completed')==d.get('total'):emit(**d)
    extra,feature_audit=export_v2_validation_features(ROOT,ids,arrays['timestamp'][ids],progress=progress)
    raw['cutoff_us']=arrays['timestamp'][ids]
    raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d')
    raw['label']=y[pos]
    raw['uniform_negative_control']=base.uniform_negative_control.to_numpy()
    raw['score_lr_only']=scores[pos]
    threshold=read(ROOT/'result/gnn_tuning_v2/best_model.json')['validation_threshold'] if 'validation_threshold' in read(ROOT/'result/gnn_tuning_v2/best_model.json') else .945133626461029
    comparison=read(ROOT/'result/gnn_tuning_v2/comparison.json')
    threshold=next(x for x in comparison['results'] if x['name']=='lr_only')['precision80']['threshold']
    raw['outcome_lr_only']=np.where(raw.label==1,np.where(raw.score_lr_only>=threshold,'TP','FN'),np.where(raw.score_lr_only>=threshold,'FP','TN'))
    result=pd.concat([raw,r.drop(columns=['edge_id','timestamp']),extra.drop(columns=['edge_id','cutoff_us'])],axis=1)
    for role,col in [('src','r_src_ext_out_count_24h'),('dst','r_dst_ext_in_count_24h')]:
        assert np.array_equal(result[col].fillna(-1),result[f'v2_{role}_count_24h_reused'])
    rule=(result.payment_format=='ACH')&(result.r_pair_fwd_count_72h==0)&(result.v2_dst_current_to_q90_7d>1)
    result['cohort']=np.where((result.label==1)&rule,np.where(result.outcome_lr_only=='FN','ACH_FN','ACH_TP'),np.where((result.label==1)&(result.from_bank=='70')&(result.payment_format!='ACH'),'G_FN','candidate'))
    assert result.cohort.eq('ACH_FN').sum()==772
    assert result.cohort.eq('ACH_TP').sum()==1508
    assert result.cohort.eq('G_FN').sum()==662
    output=OUT/'candidates.parquet'
    result.to_parquet(output,index=False)
    manifest=dict(status='passed',rows=len(result),cohort_counts={str(k):int(v) for k,v in result.cohort.value_counts().items()},
                  normal_same_ACH_rule=int(((result.label==0)&rule).sum()),source_sample_sha256=sha(source),
                  candidate_sha256=sha(output),source_hashes=used,prediction_audit=pred_audit,feature_audit=feature_audit,
                  script_sha256=sha(__file__),scope='Retrospective Validation cohorts only; final labels for audit; no Test or training.')
    (OUT/'candidate_manifest.json').write_text(json.dumps(manifest,ensure_ascii=True,indent=2),encoding='utf-8')
    emit(stage='completed',rows=len(result),cohorts=manifest['cohort_counts'],normal_same_ACH_rule=manifest['normal_same_ACH_rule'])

if __name__=='__main__':main()
