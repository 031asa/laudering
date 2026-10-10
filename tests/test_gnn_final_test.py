import ast
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from gnn_features import v2_features as old
from gnn_features.build import build,load_archive
from gnn_features.validation_archive import build_validation_archive,file_sha,COLS
from gnn_final_test import a_features as new,archive as final
from gnn_final_test.common import test_metrics as fixed_metrics
from test_gnn_validation_archive import write_transactions,save_json

def fixture(project,flip=False,future_amount=99):
    silver=project/'data/silver/transactions';silver.mkdir(parents=True)
    (silver/'_manifest.json').write_text('{}')
    split=dict(train_members=['train'],splits=[
        dict(name='train',dates=[f'2022-09-{d:02d}' for d in range(1,11)],rows=2,positive_rows=0),
        dict(name='validation',dates=[f'2022-09-{d:02d}' for d in range(11,14)],rows=3,positive_rows=0),
        dict(name='test',dates=list(final.DATES),rows=4,positive_rows=2)])
    save_json(project/'config/dataset_split.json',split)
    import pyarrow as pa,pyarrow.parquet as pq
    acc=project/'data/silver/accounts';acc.mkdir(parents=True)
    pq.write_table(pa.table({'bank_id':['A','B'],'account_number':['a','b']}),acc/'accounts.parquet')
    write_transactions(silver,'2022-09-10',[('22:00',('A','a'),('B','b'),'USD','USD',10,0),('23:00',('B','b'),('A','a'),'USD','USD',20,0)])
    for date in ('2022-09-11','2022-09-12','2022-09-13'):
        write_transactions(silver,date,[('01:00',('A','a'),('B','b'),'Z','Z',30,0)])
    write_transactions(silver,'2022-09-14',[('02:00',('B','b'),('A','a'),'Z','Z',10,int(flip)),('02:00',('C','c'),('A','a'),'A-new','A-new',999,int(not flip))])
    write_transactions(silver,'2022-09-15',[('02:00',('A','a'),('B','b'),'Z','Z',40,int(flip))])
    write_transactions(silver,'2022-09-16',[('02:00',('A','a'),('A','a'),'Z','Z',future_amount,int(not flip))])
    tail=silver/'transaction_date=2022-09-17';tail.mkdir();(tail/'never_read.parquet').write_bytes(b'invalid; tail forbidden')
    build(project,project/'data/gnn_v1_train','2022-09-11T00:00:00')
    build_validation_archive(project,project/'data/gnn_v1_validation')
    return project/'data/gnn_final_test_archive'

def test_test_archive_keeps_validation_identity_prefix_and_resume(tmp_path):
    out=fixture(tmp_path);prefix=tmp_path/'data/gnn_v1_validation'
    before={p.relative_to(prefix):file_sha(p) for p in prefix.rglob('*') if p.is_file()}
    result=final.build_test_archive(tmp_path,out)
    assert result['rows']==9 and result['test_rows']==4
    original=load_archive(prefix)[1];actual=load_archive(out)[1]
    for name in COLS: np.testing.assert_array_equal(actual[name][:5],original[name])
    # Newly encountered A-new sorts before Z, but must never renumber the Validation Z identity.
    assert actual['payment_currency_cat'][5]==original['payment_currency_cat'][4]
    assert actual['payment_currency_cat'][6]>actual['payment_currency_cat'][5]
    assert before=={p.relative_to(prefix):file_sha(p) for p in prefix.rglob('*') if p.is_file()}
    assert final.build_test_archive(tmp_path,out)==result

def test_labels_and_future_values_do_not_change_current_features(tmp_path):
    outputs=[]
    for name,flip,future in [('a',False,99),('b',True,'9007199254.740993')]:
        out=fixture(tmp_path/name,flip,future);final.build_test_archive(tmp_path/name,out)
        meta,a,bank,vocab=load_archive(out)
        # compute_selected uses external adjacency rather than all-history pointers.
        extra=dict(a)
        for direction in ('in','out'):
            extra[direction+'_ptr']=np.load(out/f'graph/adj_index/{direction}_ptr.npy')
            extra[direction+'_edge_ids']=np.load(out/f'graph/adj_index/{direction}_edge_ids.npy')
        outputs.append(new.compute_selected(extra,meta['data_start_us'],np.array([5,6,7])))
    np.testing.assert_array_equal(outputs[0],outputs[1])

@pytest.mark.parametrize('name',['role_features','compute_selected','encode_extra','contract','_role_input','_check_reference'])
def test_A_numerical_algorithms_are_identical_to_frozen_implementation(name):
    a=ast.dump(ast.parse(inspect.getsource(getattr(old,name))),include_attributes=False)
    b=ast.dump(ast.parse(inspect.getsource(getattr(new,name))),include_attributes=False)
    assert a==b
    assert old.contract()==new.contract()

def test_frozen_thresholds_not_reoptimized_and_whole_ties():
    y=np.array([1,0,1,0]);s=np.array([.7,.7,.6,.1])
    m=fixed_metrics(y,s,{'validation_maximum_f1':.7,'validation_precision80':.9})
    assert m['fixed_thresholds']['validation_maximum_f1']['tp']==1
    assert m['fixed_thresholds']['validation_maximum_f1']['fp']==1
    assert m['fixed_thresholds']['validation_precision80']['alerts']==0
    assert m['average_precision']==pytest.approx((.5+2/3)/2)
    assert 'best_f1' not in m

def test_no_null_threshold_search_fallback():
    m=fixed_metrics(np.array([1,0]),np.array([.9,.1]),{'validation_precision80':None})
    assert m['fixed_thresholds']['validation_precision80'] is None
    with pytest.raises(ValueError): fixed_metrics(np.array([1,0]),np.array([float('nan'),.1]),{'x':.5})

