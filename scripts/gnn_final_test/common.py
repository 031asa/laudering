"""Predeclared Test scope and immutable Validation-selected model bindings."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from gnn_features.repeat_common import atomic_json, verify_plan, source_hashes
from gnn_features.cache_train import sha
from gnn_features.v2_evaluate import confusion, verified_training as original_training
from gnn_features.repeat_evaluate import verified_training as repeat_training

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = 'result/gnn_final_test'
SEEDS = (20260920, 20261009, 20261010, 20261011)
TRIALS = ('lr_only', 'features_only')
TOTAL, POSITIVES, PREFIX = 6883233, 6932, 25007998
DATES = [f'2022-09-{d:02d}' for d in range(14,17)]
END = '2022-09-17T00:00:00'
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()
def eval_root(project, trial, seed):
    if trial not in TRIALS or seed not in SEEDS: raise ValueError('Undeclared Test run')
    return Path(project)/OUTPUT/'evaluations'/str(seed)/trial
def sources(project):
    paths = sorted((Path(project)/'scripts/gnn_final_test').glob('*.py'))
    paths += [Path(project)/'scripts/start_gnn_final_test.ps1', Path(project)/'tests/test_gnn_final_test.py']
    return {str(p.relative_to(project)):sha(p) for p in paths}
def protocol(project):
    project=Path(project); path=project/OUTPUT/'protocol.json'
    plan=verify_plan(project)
    value=dict(version=1, seeds=list(SEEDS), trials=list(TRIALS), total_models=8,
        dates=DATES, targets=TOTAL, positives=POSITIVES, history_prefix=PREFIX,
        selection='All eight predeclared models; each threshold selected ONLY on full Validation before its Test inference',
        thresholds=['validation_maximum_f1','validation_precision80'],
        graph_sampling_seed=20260920, concurrent_cpu_workers=2,
        tail_sealed=True, train_unchanged=True, deployment_replaced=False,
        test_tuning=False, seed_selection=False, recipe_selection=False,
        repeat_plan_sha256=sha(project/'result/gnn_seed_repeats/config.json'),
        split_sha256=sha(project/'config/dataset_split.json'),
        frozen_source_sha256=source_hashes(project), test_source_sha256=sources(project))
    if path.exists():
        if read(path)!=value: raise ValueError('Final Test protocol/source identity changed')
    else: atomic_json(path,value)
    return value
def validation_source(project,trial,seed):
    if seed==20260920: return Path(project)/'result/gnn_tuning_v4/comparison.json'
    return Path(project)/'result/gnn_seed_repeats/evaluations'/str(seed)/trial/'metrics.json'
def bind(project,trial,seed):
    project=Path(project); eval_root(project,trial,seed)
    source=validation_source(project,trial,seed)
    if not source.exists(): return None
    report=read(source)
    if report.get('status')!='completed': return None
    if seed==20260920:
        selected,=[r for r in report['results'] if r['name']==trial]
        trained=original_training(project,trial)
    else:
        selected=report
        if report.get('seed')!=seed or report.get('trial')!=trial: raise ValueError('Validation run identity mismatch')
        trained=repeat_training(project,trial,seed)
    metrics=selected['metrics']
    if metrics['samples']!=4689006 or metrics['positives']!=6420: raise ValueError('Validation must be complete')
    if selected['model_sha256']!=trained['model_sha256']: raise ValueError('Validation/model SHA mismatch')
    value=dict(seed=seed,trial=trial,model_sha256=trained['model_sha256'],
        model_path=str(trained['model_path'].relative_to(project)),
        training_config_sha256=trained['config_sha256'], training_result_sha256=trained['training_result_sha256'],
        validation_source=str(source.relative_to(project)),validation_source_sha256=sha(source),
        validation_metrics=metrics, thresholds={
            'validation_maximum_f1':metrics['best_f1']['threshold'],
            'validation_precision80':None if metrics['precision80'] is None else metrics['precision80']['threshold']})
    path=project/OUTPUT/'bindings'/str(seed)/(trial+'.json')
    if path.exists():
        if read(path)!=value: raise ValueError('Frozen Validation threshold/model binding changed')
    else: atomic_json(path,value)
    trained['binding']=value; trained['binding_sha256']=sha(path)
    return trained
def bound_training(project,trial,seed):
    trained=bind(project,trial,seed)
    if trained is None: raise ValueError('Model has not completed full Validation')
    return trained
def test_metrics(y,s,thresholds):
    y,s=np.asarray(y),np.asarray(s)
    if y.ndim!=1 or y.shape!=s.shape or not np.isin(y,[0,1]).all() or not np.isfinite(s).all() or np.any((s<0)|(s>1)):
        raise ValueError('Invalid Test labels or scores')
    n,p=len(y),int(y.sum()); ap=None
    if p:
        order=np.argsort(-s,kind='stable'); ordered=s[order]
        ends=np.r_[np.flatnonzero(ordered[:-1]!=ordered[1:]),n-1]
        tp=np.cumsum(y[order],dtype=np.int64)[ends]
        ap=float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/p)
    return dict(samples=n,positives=p,average_precision=ap,
        fixed_thresholds={name:None if t is None else confusion(y,s>=t,t) for name,t in thresholds.items()},
        threshold_policy='Copied from Validation, never optimized on Test; P80 may fail to reach 80% precision on Test')
def slices(arrays,bank,bank_identity,vocab):
    ids=np.arange(PREFIX,PREFIX+TOTAL)
    formats=arrays['payment_format_cat'][ids]
    nonach=formats!=vocab['payment_format'].index('ACH')+2
    raw70=[i for i,v in enumerate(bank_identity) if str(v)=='70']
    if len(raw70)!=1: raise ValueError('Raw bank 70 identity missing/ambiguous')
    groups={'non_ACH':nonach,'bank70_non_ACH':nonach & (bank[arrays['src'][ids]]==raw70[0])}
    for date in DATES:
        lo=int(np.datetime64(date,'us').astype(np.int64))
        groups['date_'+date]=(arrays['timestamp'][ids]>=lo)&(arrays['timestamp'][ids]<lo+86400000000)
    for code in np.unique(formats): groups['payment_identity_'+str(int(code))]=formats==code
    return groups
