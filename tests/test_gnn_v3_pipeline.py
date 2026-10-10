"""Fail-closed engineering gates and bounded handoff updates for v3."""
import copy
import json
from pathlib import Path
import pytest
from gnn_features import v3_pipeline as pipeline


GNN_PROOFS=('A_initial_output_equal','shared_tensors_exact','new_columns_zero',
            'new_columns_receive_gradient','identical_training_rng','checkpoint_resume_verified')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def accepted(tmp_path,monkeypatch):
    source=tmp_path/'scripts/gnn_features';source.mkdir(parents=True)
    names=('v3_train.py','v3_adapter.py','v3_features.py','v2_train.py','v2_adapter.py',
           'v2_features.py','v2_evaluate.py','tune_model.py','training_state.py')
    for name in names+('v3_metrics.py',):
        (source/name).write_text('fixture:'+name,encoding='utf-8')
    from gnn_features import v3_train
    monkeypatch.setattr(v3_train,'__file__',str(source/'v3_train.py'))
    inventory={str((source/n).relative_to(tmp_path)):pipeline.sha(source/n) for n in names}
    out=tmp_path/'result/gnn_tuning_v3';reports={}
    for trial in pipeline.TRIALS:
        folder=out/'early_gpu_acceptance'/trial
        save(folder/'config.json',dict(trial=trial,v3_code_sha256=inventory))
        report=dict(status='passed',trial=trial,config_sha256=pipeline.sha(folder/'config.json'),
            positive_groups=['ACH','non_ACH'],compared_batches=2,**{key:True for key in GNN_PROOFS})
        save(folder/'acceptance.json',report);reports[trial]=report
    save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    return tmp_path,out,reports,None


def test_complete_acceptance_passes(accepted):
    project,_,_,_=accepted
    pipeline.verify_acceptance(project)


@pytest.mark.parametrize('proof',GNN_PROOFS)
def test_cuda_proof_failure_is_not_hidden_by_passed_status(accepted,proof):
    project,out,reports,_=accepted
    trial=pipeline.TRIALS[0];reports[trial][proof]=False
    save(out/'early_gpu_acceptance'/trial/'acceptance.json',reports[trial])
    save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_cuda_requires_complete_source_inventory(accepted):
    project,out,reports,_=accepted;trial=pipeline.TRIALS[0]
    folder=out/'early_gpu_acceptance'/trial
    config=pipeline.read(folder/'config.json')
    del config['v3_code_sha256'][str(Path('scripts/gnn_features/v3_adapter.py'))]
    save(folder/'config.json',config)
    reports[trial]['config_sha256']=pipeline.sha(folder/'config.json')
    save(folder/'acceptance.json',reports[trial])
    save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_source_change_invalidates_acceptance(accepted):
    project,_,_,_=accepted
    (project/'scripts/gnn_features/v3_features.py').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_positive_group_acceptance_is_required(accepted):
    project,out,reports,_=accepted;trial=pipeline.TRIALS[0]
    reports[trial]['positive_groups']=['ACH']
    save(out/'early_gpu_acceptance'/trial/'acceptance.json',reports[trial])
    save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_trial_identity_is_required(accepted):
    project,out,reports,_=accepted;trial=pipeline.TRIALS[0]
    folder=out/'early_gpu_acceptance'/trial;config=pipeline.read(folder/'config.json')
    config['trial']=pipeline.TRIALS[1];save(folder/'config.json',config)
    reports[trial]['config_sha256']=pipeline.sha(folder/'config.json')
    save(folder/'acceptance.json',reports[trial]);save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_handoff_replaces_only_v3_section(tmp_path):
    path=tmp_path/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    path.write_text('Original v1/v2 evidence\n',encoding='utf-8')
    pipeline.write_handoff(tmp_path,'first')
    once=path.read_text(encoding='utf-8')
    path.write_text(once+'\nUser later note\n',encoding='utf-8')
    pipeline.write_handoff(tmp_path,'second')
    text=path.read_text(encoding='utf-8')
    assert text.startswith('Original v1/v2 evidence\n') and text.endswith('User later note\n')
    assert text.count('<!-- GNN_V3_STATUS_BEGIN -->')==1
    assert 'first' not in text and 'second' in text


def test_handoff_ambiguous_markers_preserves_original(tmp_path):
    path=tmp_path/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    text='text <!-- GNN_V3_STATUS_BEGIN --> missing end'
    path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError):pipeline.write_handoff(tmp_path,'new')
    assert path.read_text(encoding='utf-8')==text
