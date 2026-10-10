"""Independently reconcile the completed analysis export before delivery."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gnn_missed_data_20261003'
NAMES=('lr_only','features_only','features_pruned','features_balanced')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    destination=OUT/'12_导出复核.json'
    assert not destination.exists()
    audit=read(OUT/'11_审计清单.json')
    summary=read(OUT/'08_模型阈值与全量指标.json')
    groups=read(OUT/'09_分组计数与抽样说明.json')
    result=ROOT/'result/gnn_tuning_v2'
    correction=read(result/'recovery/20261003_bank70_correction/verification.json')
    expected=correction['output_sha256']['comparison.corrected.json']
    assert sha(result/'comparison.corrected.json')==expected==audit['source_comparison_sha256']
    comparison=read(result/'comparison.corrected.json')
    rows={x['name']:x for x in comparison['results']}
    for n in NAMES:
        p=ROOT/('result/gnn_tuning_v1' if n=='lr_only' else 'result/gnn_tuning_v2')/'evaluations'/n
        metrics=read(p/'metrics.json')
        provenance=audit['prediction_provenance'][n]
        assert metrics['metrics']==provenance['metrics']==rows[n]['metrics']==summary['models'][n]['metrics']
        assert sha(p/'metrics.json')==provenance['metrics_sha256']==rows[n]['metrics_sha256']
        assert sha(p/'config.json')==provenance['config_sha256']
        assert summary['models'][n]['global_P80']==rows[n]['precision80']
    tables={}
    csv_frames={}
    for name,record in audit['tables'].items():
        path=OUT/name
        assert sha(path)==record['sha256']
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(stream)
            header=next(reader)
            count=0
            for row in reader:
                assert len(row)==len(header),(name,count)
                count+=1
        assert count==record['rows'] and len(header)==record['columns']
        tables[name]=dict(rows=count,columns=len(header))
        if name.startswith(('01_','02_','03_','04_','05_','06_')):
            use=['edge_id','label']+['score_'+n for n in NAMES]
            if not name.startswith('06_'):use+=['outcome_'+n for n in NAMES]+['from_bank','is_non_ACH','is_uniform_negative_control','is_false_positive_any_model','missed_by_all_four']
            df=pd.read_csv(path,usecols=use,dtype={'from_bank':str},float_precision='round_trip')
            assert df.edge_id.is_unique
            assert df.edge_id.between(20318992,25007997).all()
            for n in NAMES:
                s=df['score_'+n].to_numpy()
                assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
                if not name.startswith('06_'):
                    predicted=s>=summary['models'][n]['global_P80']['threshold']
                    expected_outcome=np.where(df.label==1,np.where(predicted,'TP','FN'),np.where(predicted,'FP','TN'))
                    assert np.array_equal(expected_outcome,df['outcome_'+n])
            csv_frames[name[:2]]=df
    fn,non,pos,control,fp,g=(csv_frames[k] for k in ['01','02','03','04','05','06'])
    assert len(fn)==2887 and (fn.outcome_lr_only=='FN').all()
    assert len(non)==715 and (non.label==1).all() and non.is_non_ACH.all()
    assert set(non.edge_id).issubset(set(fn.edge_id))
    assert (non.from_bank=='70').sum()==662
    assert len(pos)==6420 and (pos.label==1).all()
    assert (control.label==0).all() and control.is_uniform_negative_control.all()
    assert len(control)==sum(x['exported_controls'] for x in groups['negative_control_strata'].values())
    assert len(fp)==groups['false_positive_union'] and (fp.label==0).all() and fp.is_false_positive_any_model.all()
    assert len(g)==477760 and int(g.label.sum())==662
    for n in NAMES:
        assert (pos['outcome_'+n]=='TP').sum()==summary['models'][n]['global_P80']['tp']
        assert (fp['outcome_'+n]=='FP').sum()==summary['models'][n]['global_P80']['fp']
    by_id=pos.set_index('edge_id')
    for subset in [fn,non]:
        assert np.array_equal(subset[['score_'+n for n in NAMES]].to_numpy(),by_id.loc[subset.edge_id,['score_'+n for n in NAMES]].to_numpy())
    for subset in [pos,control,fp]:
        local=subset[(subset.from_bank=='70')&subset.is_non_ACH].set_index('edge_id')
        actual=g.set_index('edge_id').loc[local.index]
        assert np.array_equal(local[['score_'+n for n in NAMES]].to_numpy(),actual[['score_'+n for n in NAMES]].to_numpy())
    # Every exported column must have a dictionary definition, exactly once.
    dictionary=pd.read_csv(OUT/'07_字段字典.csv')
    header=pd.read_csv(OUT/'01_当前模型漏检_2887.csv',nrows=0).columns.tolist()
    assert dictionary.field.is_unique and dictionary.field.tolist()==header
    assert dictionary.description.notna().all()
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()}
    payload=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
                 comparison_correction_sha_chain=True,complete_metrics_and_config_hashes=True,
                 full_csv_shape_and_scope_check=True,all_csv_outcomes_recomputed=True,
                 positive_and_FP_totals_match_full_evaluation=True,joined_G_scores_match=True,
                 dictionary_complete=True,files=files,table_counts=tables,test_accessed=False,
                 verifier_script_sha256=sha(Path(__file__)))
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    zpath=OUT.parent/'GNN漏检交易数据包_20261003.zip'
    with zipfile.ZipFile(zpath,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        assert destination.name not in z.namelist()
        z.write(destination,destination.name)
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(files)+1
        assert z.testzip() is None
        for name,item in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
        assert hashlib.sha256(z.read(destination.name)).hexdigest()==sha(destination)
    print(json.dumps(dict(status='passed',zip=str(zpath),zip_sha256=sha(zpath),zip_bytes=zpath.stat().st_size,files=len(files)+1,counts=tables),ensure_ascii=True))


if __name__=='__main__':main()
