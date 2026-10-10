"""Fail-closed engineering gates and bounded handoff updates for v4."""
import copy
import json
from pathlib import Path
import pytest
from gnn_features import v4_pipeline as pipeline


GNN_PROOFS=('A_initial_output_equal','shared_tensors_exact','new_columns_zero',
            'new_columns_receive_gradient','identical_training_rng','checkpoint_resume_verified','all_six_columns_updated')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def accepted(tmp_path,monkeypatch):
    source=tmp_path/'scripts/gnn_features';source.mkdir(parents=True)
    names=('v4_train.py','v4_adapter.py','v4_features.py','v2_train.py','v2_adapter.py',
           'v2_features.py','v2_evaluate.py','tune_model.py','training_state.py')
    for name in names+('v4_metrics.py',):
        (source/name).write_text('fixture:'+name,encoding='utf-8')
    reference=tmp_path/'scripts/gnn_causal_structures.py'
    reference.write_text('fixture reference',encoding='utf-8')
    from gnn_features import v4_train
    monkeypatch.setattr(v4_train,'__file__',str(source/'v4_train.py'))
    inventory={str((source/n).relative_to(tmp_path)):pipeline.sha(source/n) for n in names}
    inventory[str(reference.relative_to(tmp_path))]=pipeline.sha(reference)
    out=tmp_path/'result/gnn_tuning_v4';reports={}
    for trial in pipeline.TRIALS:
        folder=out/'early_gpu_acceptance'/trial
        save(folder/'config.json',dict(trial=trial,v4_code_sha256=inventory))
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
    del config['v4_code_sha256'][str(Path('scripts/gnn_features/v4_adapter.py'))]
    save(folder/'config.json',config)
    reports[trial]['config_sha256']=pipeline.sha(folder/'config.json')
    save(folder/'acceptance.json',reports[trial])
    save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_source_change_invalidates_acceptance(accepted):
    project,_,_,_=accepted
    (project/'scripts/gnn_features/v4_features.py').write_text('changed',encoding='utf-8')
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
    config['trial']='undeclared_trial';save(folder/'config.json',config)
    reports[trial]['config_sha256']=pipeline.sha(folder/'config.json')
    save(folder/'acceptance.json',reports[trial]);save(out/'early_gpu_acceptance/acceptance.json',dict(status='passed',trials=reports))
    with pytest.raises(ValueError):pipeline.verify_acceptance(project)


def test_handoff_replaces_only_v4_section(tmp_path):
    path=tmp_path/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    path.write_text('Original v1/v2 evidence\n',encoding='utf-8')
    pipeline.write_handoff(tmp_path,'first')
    once=path.read_text(encoding='utf-8')
    path.write_text(once+'\nUser later note\n',encoding='utf-8')
    pipeline.write_handoff(tmp_path,'second')
    text=path.read_text(encoding='utf-8')
    assert text.startswith('Original v1/v2 evidence\n') and text.endswith('User later note\n')
    assert text.count('<!-- GNN_V4_STATUS_BEGIN -->')==1
    assert 'first' not in text and 'second' in text


def test_handoff_ambiguous_markers_preserves_original(tmp_path):
    path=tmp_path/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    text='text <!-- GNN_V4_STATUS_BEGIN --> missing end'
    path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError):pipeline.write_handoff(tmp_path,'new')
    assert path.read_text(encoding='utf-8')==text
