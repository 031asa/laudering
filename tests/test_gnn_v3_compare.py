"""Small-data final comparison, bank identity, support coverage and immutable replay."""
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gnn_features import v3_compare as compare
from gnn_features.v3_metrics import score_metrics


def save(path,value):
    compare.atomic_json(path,value)


def result(name,f1=.6,ap=.5,recall=.4):
    return dict(name=name,metrics=dict(best_f1=dict(f1=f1),average_precision=ap,
                precision80=None if recall is None else dict(recall=recall)))


def test_selection_uses_f1_then_ap_then_p80_without_business_gate():
    results=[result(n) for n in compare.NAMES]
    results[2]=result(compare.NAMES[2],f1=.61,ap=.1,recall=None)
    assert compare.choose_best(results)['name']==compare.NAMES[2]
    results[3]=result(compare.NAMES[3],f1=.61,ap=.2,recall=.1)
    assert compare.choose_best(results)['name']==compare.NAMES[3]
    results[2]=result(compare.NAMES[2],f1=.61,ap=.2,recall=.2)
    assert compare.choose_best(results)['name']==compare.NAMES[2]


def test_complete_tie_preserves_reference_and_invalid_inventory_rejected():
    results=[result(n) for n in compare.NAMES]
    assert compare.choose_best(results)['name']=='lr_only'
    with pytest.raises(ValueError):compare.choose_best(results[:-1])
    with pytest.raises(ValueError):compare.choose_best(results[:-1]+[results[0]])


@pytest.fixture
def predictions(tmp_path,monkeypatch):
    n=197;monkeypatch.setattr(compare,'TOTAL',n);monkeypatch.setattr(compare,'TRAIN_TOTAL',0)
    folder=tmp_path/'evaluation';folder.mkdir()
    save(folder/'config.json',dict(code_sha256={},model_sha256='model'))
    config_hash=compare.sha(folder/'config.json')
    y=(np.arange(n)%3==0).astype(np.int8);times=np.arange(n,dtype=np.int64)+1662940800000000
    s=((np.arange(n)%11)/11).astype(np.float32)
    blocks=[]
    for i in range(n):
        block=f'block={i:04d}';directory=folder/'predictions'/block;directory.mkdir(parents=True)
        ids=np.array([i],np.int64)
        pq.write_table(pa.table(dict(edge_id=ids,cutoff_us=times[i:i+1],label=y[i:i+1],probability=s[i:i+1])),directory/'predictions.parquet')
        save(directory/'manifest.json',dict(config_sha256=config_hash,targets=1,target_ids_sha256=compare.array_sha(ids),
            file_sha256=compare.sha(directory/'predictions.parquet')))
        blocks.append(dict(block=block,targets=1,manifest_sha256=compare.sha(directory/'manifest.json')))
    save(tmp_path/'data/gnn_validation_cache_parquet/manifest.json',dict(blocks=blocks))
    report=dict(status='completed',config_sha256=config_hash,model_sha256='model',metrics=score_metrics(y,s),blocks=blocks)
    save(folder/'metrics.json',report)
    return tmp_path,folder,times,y,s,report


def test_predictions_hash_labels_times_and_whole_tie_metrics_replay(predictions):
    project,folder,times,y,s,_=predictions
    actual,provenance=compare.load_new_predictions(project,folder,times,y,'model')
    np.testing.assert_array_equal(actual,s)
    assert provenance['model_sha256']=='model'


@pytest.mark.parametrize('corrupt',['times','labels','model','metrics','coverage','block_sha'])
def test_prediction_mismatch_rejected(predictions,corrupt):
    project,folder,times,y,_,report=predictions
    digest='model'
    if corrupt=='times':times=times.copy();times[0]+=1
    elif corrupt=='labels':y=y.copy();y[0]=1-y[0]
    elif corrupt=='model':digest='other'
    elif corrupt=='metrics':report['metrics']['average_precision']=0.
    elif corrupt=='coverage':report['blocks']=report['blocks'][:-1]
    elif corrupt=='block_sha':report['blocks'][0]['manifest_sha256']='changed'
    save(folder/'metrics.json',report)
    with pytest.raises(ValueError):compare.load_new_predictions(project,folder,times,y,digest)


def test_gnn_evaluation_configuration_must_bind_selected_model(predictions):
    project,folder,times,y,_,report=predictions
    config=compare.read(folder/'config.json')
    config['model_sha256']='different_checkpoint'
    save(folder/'config.json',config)
    report['config_sha256']=compare.sha(folder/'config.json')
    save(folder/'metrics.json',report)
    with pytest.raises(ValueError,match='[Mm]odel'):
        compare.load_new_predictions(project,folder,times,y,'model')


@pytest.fixture
def small_run(tmp_path,monkeypatch):
    from gnn_features import v3_features, v2_evaluate, v3_evaluate
    n=720;monkeypatch.setattr(compare,'TOTAL',n);monkeypatch.setattr(compare,'TRAIN_TOTAL',0)
    monkeypatch.setattr(compare,'POSITIVES',715)
    monkeypatch.setattr(compare,'verify_freeze',lambda p:{'freeze':'good'})
    monkeypatch.setattr(compare,'source_hashes',lambda p:{'source':'good'})
    monkeypatch.setattr(compare,'qualification',lambda p,n:dict(passed=False))
    monkeypatch.setattr(compare,'_disk',lambda p:None)
    root=tmp_path/'result/gnn_tuning_v3'
    save(root/'config.json',dict(frozen_anchors={'freeze':'good'},source_sha256={'source':'good'}))
    archive=tmp_path/'data/gnn_v1_validation';cache=tmp_path/'data/gnn_validation_cache_parquet'
    (archive/'history_index').mkdir(parents=True)
    arrays=dict(timestamp=np.arange(n,dtype=np.int64)+1662940800000000,
        src=np.arange(n,dtype=np.int64)%2,dst=np.full(n,2,np.int64),
        payment_format_cat=np.full(n,3,np.int64),bank_key=np.array([3,2,2],np.int64))
    for key,value in arrays.items():np.save(archive/'history_index'/f'{key}.npy',value)
    save(archive/'metadata/bank_identity.json',['0','1','2','70'])
    save(archive/'metadata/vocabulary.json',dict(values=dict(payment_format=['ACH','WIRE'])))
    files=[archive/'history_index'/f'{k}.npy' for k in arrays]+[archive/'metadata/bank_identity.json',archive/'metadata/vocabulary.json']
    save(cache/'config.json',dict(hashes={str(p.relative_to(tmp_path)):compare.sha(p) for p in files}))
    block='block=0000';(cache/block).mkdir(parents=True)
    pq.write_table(pa.table(dict(targets=[list(range(n))])),cache/block/'plans.parquet')
    save(cache/'manifest.json',dict(blocks=[dict(block=block,targets=n)]))
    class Store:
        manifest_sha256='m';config_sha256='c';contract_sha256='k'
        def __init__(self,*a,**kw):pass
        def support(self,block,ids,times):
            assert len(ids)==n
            return {r+'_count_24h':np.arange(n,dtype=np.int64)%7 for r in ('src_in','src_out','dst_in','dst_out')}
        def block(self,*args):pass
    monkeypatch.setattr(v3_features,'SidecarStore',Store)
    y=np.r_[np.ones(715,np.int8),np.zeros(5,np.int8)]
    values={name:np.where(y,.8,.2).astype(np.float32) for name in compare.NAMES}
    def trained(project,name):
        return dict(model_path=tmp_path/(name+'.pt'),model_sha256='model-'+name)
    monkeypatch.setattr(v2_evaluate,'verified_training',trained)
    monkeypatch.setattr(v3_evaluate,'verified_training',trained)
    monkeypatch.setattr(compare,'old_predictions',lambda p,name,t,labels:(y.copy(),values[name],{}))
    cfg=dict(distribution_manifest_sha256='m',distribution_config_sha256='c',distribution_contract_sha256='k')
    def predicted(project,folder,times,labels,digest):
        name=folder.name
        return values[name],dict(config=cfg)
    monkeypatch.setattr(compare,'load_new_predictions',predicted)
    return tmp_path,root,values


def test_comparison_uses_four_gnns_encoded_bank70_and_support_roles(small_run):
    project,root,_=small_run
    report=compare.run(project)
    assert report['bank70_identity']['encoded']==3
    assert report['best']['name']=='lr_only'
    assert report['deployment_unchanged'] and report['test_evaluated'] is False
    assert len(report['results']) == 4
    assert {r['name'] for r in report['results']} == set(compare.NAMES)
    groups=report['results'][0]['groups']['best_f1']
    assert groups['payer_bank70_non_ach']['bank70_non_ACH']['positives']==358
    for role in ('src_in','src_out','dst_in','dst_out'):
        assert set(groups[role+'_support_24h'])=={'0','1-4','>=5'}
    assert report==compare.run(project)
    selected=compare.read(root/'best_model.json')
    assert selected['research_only'] and not selected['deployment_replaced']
    assert selected['comparison_sha256']==compare.sha(root/'comparison.json')


def test_changed_completed_comparison_is_not_overwritten(small_run):
    project,root,values=small_run
    compare.run(project);original=(root/'comparison.json').read_bytes()
    values[compare.TRIALS[0]][0]=.1
    with pytest.raises(ValueError,match='replay'):compare.run(project)
    assert (root/'comparison.json').read_bytes()==original
