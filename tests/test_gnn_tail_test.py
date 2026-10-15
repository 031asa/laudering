import ast
import inspect
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gnn_features.build import build,load_archive,COLS
from gnn_features.validation_archive import build_validation_archive
from gnn_final_test.archive import build_test_archive as build_original, file_sha
from gnn_final_test import a_features
from gnn_tail_test import archive
from gnn_tail_test.pipeline import extra_targets,raw_r
from gnn_tail_test.compare import summarize,paired
from gnn_features.engine import FeatureEngine
from test_gnn_validation_archive import write_transactions,save_json

def fixture(p,flip=False,future=99):
    silver=p/'data/silver/transactions';silver.mkdir(parents=True);(silver/'_manifest.json').write_text('{}')
    splits=[dict(name='train',dates=[f'2022-09-{i:02d}' for i in range(1,11)],rows=2,positive_rows=0),
        dict(name='validation',dates=[f'2022-09-{i:02d}' for i in range(11,14)],rows=3,positive_rows=0),
        dict(name='test',dates=[f'2022-09-{i:02d}' for i in range(14,17)],rows=3,positive_rows=0),
        dict(name='tail',dates=list(archive.DATES),rows=13,positive_rows=1 if flip else 12)]
    save_json(p/'config/dataset_split.json',dict(train_members=['train'],splits=splits))
    acc=p/'data/silver/accounts';acc.mkdir(parents=True)
    pq.write_table(pa.table(dict(bank_id=['A','B'],account_number=['a','b'])),acc/'accounts.parquet')
    write_transactions(silver,'2022-09-10',[('22:00',('A','a'),('B','b'),'USD','USD',10,0),('23:00',('B','b'),('A','a'),'USD','USD',20,0)])
    for i in range(11,17):write_transactions(silver,f'2022-09-{i:02d}',[('01:00',('A','a'),('B','b'),'Z','Z',30,0)])
    write_transactions(silver,'2022-09-17',[('02:00',('B','b'),('A','a'),'Z','Z',40,int(flip)),('02:00',('C','c'),('A','a'),'A-new','A-new',900,int(not flip))])
    for i in range(18,29):write_transactions(silver,f'2022-09-{i:02d}',[('02:00',('A','a'),('B','b'),'Z','Z',future,int(not flip))])
    build(p,p/'data/gnn_v1_train','2022-09-11T00:00:00');build_validation_archive(p,p/'data/gnn_v1_validation');build_original(p,p/'data/gnn_final_test_archive')
    return p/'data/gnn_tail_test_archive'

def test_tail_append_preserves_full_test_prefix_and_resume(tmp_path):
    out=fixture(tmp_path);old=tmp_path/'data/gnn_final_test_archive'
    before={p.relative_to(old):file_sha(p) for p in old.rglob('*') if p.is_file()}
    result=archive.build_test_archive(tmp_path,out)
    assert result['rows']==21 and result['test_rows']==13
    previous=load_archive(old)[1];actual=load_archive(out)[1]
    for name in COLS:np.testing.assert_array_equal(actual[name][:8],previous[name])
    assert before=={p.relative_to(old):file_sha(p) for p in old.rglob('*') if p.is_file()}
    assert archive.build_test_archive(tmp_path,out)==result

def test_tail_labels_and_future_do_not_change_current_inputs(tmp_path):
    results=[]
    for name,flip,future in [('a',False,99),('b',True,999999)]:
        out=fixture(tmp_path/name,flip,future);archive.build_test_archive(tmp_path/name,out)
        meta,a,bank,vocab=load_archive(out)
        engine=FeatureEngine(a,bank,vocab['payment_format'].index('ACH')+2,meta['data_start_us'])
        r=raw_r(engine,np.array([8,9]));extra=dict(a)
        for d in ('in','out'):
            extra.pop(d+'_order',None)
            for k in ('ptr','edge_ids'):extra[d+'_'+k]=np.load(out/f'graph/adj_index/{d}_{k}.npy')
        ext=extra_targets(extra,meta['data_start_us'],np.array([8,9]))
        results.append((r,ext))
    for a,b in zip(*results):np.testing.assert_array_equal(a,b)

def test_paired_accounts_are_set_differences():
    y=np.array([1,1,0]);lr=np.array([.9,.1,.9]);a=np.array([.1,.9,.1]);thresholds={'x':.5}
    m=paired(y,lr,a,thresholds,thresholds,np.array([1,2,3]),np.array([2,4,5]))['x']
    assert m['added_tp']==m['lost_tp']==1 and m['lost_fp']==1
    assert m['new_detected_accounts']==m['lost_detected_accounts']==1
