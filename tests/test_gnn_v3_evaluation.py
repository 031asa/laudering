"""Final-model count, source, column contract and strict-load verification."""
import json
from pathlib import Path
import pytest
import torch
from gnn_features import v3_evaluate as evaluator, v3_adapter as adapter
from gnn_features.model import BidirectionalGINE


def artifacts(tmp_path,trial='amount_distribution'):
    folder=tmp_path/'result/gnn_tuning_v3/trials'/trial
    folder.mkdir(parents=True)
    model_config=dict(bank_count=5,edge_category_sizes=[5]*4,node_dim=63,r_dim=221,width=64,dropout=.1)
    sources=('v3_train.py','v3_adapter.py','v3_features.py','v2_train.py','v2_adapter.py',
             'v2_features.py','v2_evaluate.py','tune_model.py','training_state.py')
    config=dict(version=3,trial=trial,epochs=1,seed=20260920,positive_weight=1140.9013150500168,
        optimizer=dict(learning_rate=3e-4,weight_decay=1e-4,clip_norm=5),
        group_loss=dict(applied=False,factors=dict(ACH=1.,non_ACH=1.)),
        projection=adapter.projection_contract(trial),model=model_config,
        v3_code_sha256={name:evaluator.sha(Path(evaluator.__file__).with_name(name)) for name in sources})
    evaluator.atomic_json(folder/'config.json',config)
    model=BidirectionalGINE(**model_config)
    torch.save(dict(model=model.state_dict(),config=model_config,config_sha256=evaluator.sha(folder/'config.json'),
        trial=trial,seed=20260920,epoch=1,targets=evaluator.TRAIN_TOTAL,positives=evaluator.TRAIN_POSITIVES),folder/'final_model.pt')
    result=dict(status='completed',epoch=1,targets=evaluator.TRAIN_TOTAL,positives=evaluator.TRAIN_POSITIVES,
        blocks=797,config_sha256=evaluator.sha(folder/'config.json'),final_model_sha256=evaluator.sha(folder/'final_model.pt'))
    evaluator.atomic_json(folder/'training_result.json',result)
    return folder,config,result


def test_verified_training_strict_loads_full_count_model(tmp_path):
    folder,config,_=artifacts(tmp_path)
    value=evaluator.verified_training(tmp_path,'amount_distribution')
    assert value['config']==config
    assert value['model_sha256']==evaluator.sha(folder/'final_model.pt')


@pytest.mark.parametrize('field,value',[('targets',20318991),('positives',17793),('blocks',796),('epoch',2),
                                      ('status','running'),('final_model_sha256','wrong')])
def test_partial_or_changed_result_never_evaluated(tmp_path,field,value):
    folder,_,result=artifacts(tmp_path)
    result[field]=value
    evaluator.atomic_json(folder/'training_result.json',result)
    with pytest.raises(ValueError): evaluator.verified_training(tmp_path,'amount_distribution')


def test_column_order_change_rejected_before_inference(tmp_path):
    folder,config,_=artifacts(tmp_path)
    columns=config['projection']['r_columns']
    columns[-1],columns[-2]=columns[-2],columns[-1]
    evaluator.atomic_json(folder/'config.json',config)
    with pytest.raises(ValueError,match='configuration'): evaluator.verified_training(tmp_path,'amount_distribution')


def test_missing_source_binding_rejected(tmp_path):
    folder,config,result=artifacts(tmp_path)
    del config['v3_code_sha256']['v3_adapter.py']
    evaluator.atomic_json(folder/'config.json',config)
    result['config_sha256']=evaluator.sha(folder/'config.json')
    evaluator.atomic_json(folder/'training_result.json',result)
    with pytest.raises(ValueError,match='source identities'): evaluator.verified_training(tmp_path,'amount_distribution')


def test_nonfinite_final_weights_rejected_even_with_matching_hash(tmp_path):
    folder,_,result=artifacts(tmp_path)
    saved=torch.load(folder/'final_model.pt',weights_only=True)
    saved['model']['r_encoder.0.weight'][0,0]=float('nan')
    torch.save(saved,folder/'final_model.pt')
    result['final_model_sha256']=evaluator.sha(folder/'final_model.pt')
    evaluator.atomic_json(folder/'training_result.json',result)
    with pytest.raises(ValueError,match='Nonfinite'): evaluator.verified_training(tmp_path,'amount_distribution')


def test_strict_saved_model_identity_rejects_wrong_trial(tmp_path):
    folder,_,result=artifacts(tmp_path)
    saved=torch.load(folder/'final_model.pt',weights_only=True)
    saved['trial']='amount_time_distribution'
    torch.save(saved,folder/'final_model.pt')
    result['final_model_sha256']=evaluator.sha(folder/'final_model.pt')
    evaluator.atomic_json(folder/'training_result.json',result)
    with pytest.raises(ValueError): evaluator.verified_training(tmp_path,'amount_distribution')


def test_old_trial_cannot_write_through_v3_evaluator(tmp_path):
    with pytest.raises(ValueError): evaluator.run(tmp_path,'features_only')
    assert not (tmp_path/'result').exists()
