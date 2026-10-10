"""Pairwise seed summaries, with independent prediction replay and no seed picking."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .repeat_common import (ROOT, OUTPUT, SEEDS, OLD_SEED, TRIALS, read, sha,
    atomic_json, verify_plan, eval_root)
from .training_state import ProcessLock
from .v4_metrics import score_metrics
from .v4_compare import load_new_predictions
from .v2_compare import load_predictions, qualification
from .v2_evaluate import group_report, paired_changes, confusion, verified_training as old_training, _disk
from .repeat_evaluate import verified_training

TOTAL, POSITIVES, TRAIN_TOTAL = 4689006, 6420, 20318992

def values(row):
    m=row['metrics']; p=m['precision80']
    return dict(f1=m['best_f1']['f1'], ap=m['average_precision'],
                p80_recall=0. if p is None else p['recall'])

def statistics(xs):
    x=np.asarray(xs,dtype=np.float64)
    if len(x)<2 or not np.isfinite(x).all(): raise ValueError('At least two finite paired runs required')
    return dict(n=len(x),mean=float(x.mean()),sample_sd=float(x.std(ddof=1)),
                minimum=float(x.min()),maximum=float(x.max()))

def summarize(rows, seeds):
    keys=[(r['seed'],r['trial']) for r in rows]
    expected={(s,t) for s in seeds for t in TRIALS}
    if len(keys)!=len(expected) or set(keys)!=expected: raise ValueError('Missing/duplicate paired seed')
    lookup={(r['seed'],r['trial']):values(r) for r in rows}
    models={t:{m:statistics([lookup[s,t][m] for s in seeds]) for m in ('f1','ap','p80_recall')} for t in TRIALS}
    differences={m:[lookup[s,'features_only'][m]-lookup[s,'lr_only'][m] for s in seeds] for m in ('f1','ap','p80_recall')}
    delta={m:dict(statistics(xs),values=xs,a_wins=sum(x>0 for x in xs),ties=sum(x==0 for x in xs),
                  lr_wins=sum(x<0 for x in xs)) for m,xs in differences.items()}
    rank=lambda t:tuple(models[t][m]['mean'] for m in ('f1','ap','p80_recall'))
    preferred=None if rank(TRIALS[0])==rank(TRIALS[1]) else max(TRIALS,key=rank)
    return dict(seeds=list(seeds),models=models,a_minus_lr=delta,preferred_recipe=preferred,
        scope='Descriptive initialization/training-RNG variability on fixed Validation and fixed graph; not a significance test',
        no_p80_solution_recall=0.,checkpoint_selected=False)

def markdown(report):
    rows=report['runs']
    lines=['# lr_only与A：三组新增种子重复实验','',
        '每次完整Train 20,318,992笔／17,794正例、1 epoch；完整Validation 4,689,006笔／6,420正例。',
        '主分析为三个新种子的配对结果；20260920曾用于开发，加入后的四种子汇总另列。', '',
        '| 种子 | 模型 | 最大F1 | AP | P80召回 |', '| --- | --- | ---: | ---: | ---: |']
    for r in rows:
        m=values(r); lines.append(f"| {r['seed']} | {r['trial']} | {m['f1']:.4%} | {m['ap']:.6f} | {m['p80_recall']:.4%} |")
    for key,title in [('primary_summary','三个新种子'),('including_development_seed','含原开发种子的四次结果')]:
        s=report[key]
        lines += ['', '## '+title,'', '| 模型 | F1均值±样本SD | AP均值±样本SD | P80召回均值±样本SD |',
                  '| --- | ---: | ---: | ---: |']
        for t in TRIALS:
            m=s['models'][t]
            cells=[f"{m[k]['mean']:.4%} ± {m[k]['sample_sd']:.4%}" for k in ('f1','ap','p80_recall')]
            lines.append('| '+t+' | '+' | '.join(cells)+' |')
        d=s['a_minus_lr']['f1']
        lines += ['', f"A−lr_only的配对F1差：平均{100*d['mean']:+.4f}个百分点，样本SD {100*d['sample_sd']:.4f}个百分点；"
                  f"A胜{d['a_wins']}次，lr_only胜{d['lr_wins']}次，同分{d['ties']}次。",
                  '按预先固定的平均F1→AP→P80顺序，研究上优先的配方：'+str(s['preferred_recipe'])+'。']
    lines += ['', '## 分组与检出变化', '',
        'comparison.json逐种子记录A−lr_only新增／损失检出、误报及独立账户覆盖；非ACH、付款银行70且非ACH、日期和支付方式均复用各模型的全局阈值。',
        '银行70由冻结bank_identity解析；业务门槛单列qualification。', '',
        '所有8个模型均已严格载入、核对实际SHA和全量预测重放；新增6次训练完整性已核验。',
        '图和批次顺序固定，因此这里只测初始化和训练随机性，不包含重新采样图的波动。',
        '不挑最好种子，不替换旧模型。Validation已参与开发，3个新种子不是3份独立测试数据；不宣称统计显著或论文Test优越性。Test继续封存。',
        'P80无可行非空阈值时召回按0汇总，原始metrics仍保留null。']
    return '\n'.join(lines)+'\n'

def run(project=ROOT):
    project=Path(project).resolve(); root=project/OUTPUT
    with ProcessLock(root/'comparison_lock'):
        verify_plan(project); _disk(project)
        archive=project/'data/gnn_v1_validation'; cache=project/'data/gnn_validation_cache_parquet'
        hashes=read(cache/'config.json')['hashes']
        names=('timestamp','src','dst','payment_format_cat','bank_key')
        paths=[archive/'history_index'/f'{k}.npy' for k in names]+[archive/'metadata/bank_identity.json',archive/'metadata/vocabulary.json']
        for p in paths:
            if hashes.get(str(p.relative_to(project)))!=sha(p): raise ValueError('Frozen grouping source mismatch')
        arrays={k:np.load(archive/'history_index'/f'{k}.npy',mmap_mode='r') for k in names}
        times=arrays['timestamp'][TRAIN_TOTAL:]; src=arrays['src'][TRAIN_TOTAL:]; dst=arrays['dst'][TRAIN_TOTAL:]
        if times.shape!=(TOTAL,): raise ValueError('Validation time coverage differs')
        vocab=read(archive/'metadata/vocabulary.json')['values']; fmt=arrays['payment_format_cat'][TRAIN_TOTAL:]
        non=fmt!=vocab['payment_format'].index('ACH')+2
        identity=read(archive/'metadata/bank_identity.json')
        if identity.count('70')!=1: raise ValueError('Raw bank70 mapping ambiguous')
        bank70=(arrays['bank_key'][src]==identity.index('70')) & non
        groups=dict(ach_group=np.where(non,'non_ACH','ACH'),
            payer_bank70_non_ach=np.where(bank70,'bank70_non_ACH','other'),
            date=times.astype('datetime64[us]').astype('datetime64[D]').astype(str),
            payment_format=np.asarray(['missing','unknown']+vocab['payment_format'])[fmt])
        labels=None; runs=[]; changes=[]
        for seed in (OLD_SEED,)+SEEDS:
            pair={}; scores={}
            for trial in TRIALS:
                atomic_json(root/'comparison_status.json',dict(status='running',seed=seed,trial=trial))
                if seed==OLD_SEED:
                    trained=old_training(project,trial)
                    labels,s,provenance=load_predictions(project,trial,times,labels)
                else:
                    trained=verified_training(project,trial,seed)
                    s,provenance=load_new_predictions(project,eval_root(project,trial,seed),times,labels,trained['model_sha256'])
                if int(labels.sum())!=POSITIVES or int(labels[non].sum())!=715 or int(labels[bank70].sum())!=662:
                    raise ValueError('Validation group/class coverage differs')
                m=score_metrics(labels,s)
                if m['samples']!=TOTAL: raise ValueError('Incomplete Validation')
                p=m['precision80']; nc=None if p is None else confusion(labels[non],s[non]>=p['threshold'],p['threshold'])
                item=dict(seed=seed,trial=trial,metrics=m,model_sha256=trained['model_sha256'],
                    model_path=str(trained['model_path'].relative_to(project)),provenance=provenance,
                    qualification=qualification(p,nc),groups={point:group_report(labels,s,m[point],groups,src,dst)
                    for point in ('best_f1','precision80')})
                pair[trial]=item; scores[trial]=s; runs.append(item)
                del trained
            changes.append(dict(seed=seed,points={point:paired_changes(labels,scores['lr_only'],pair['lr_only']['metrics'][point],
                scores['features_only'],pair['features_only']['metrics'][point],src,dst) for point in ('best_f1','precision80')}))
        fresh=[r for r in runs if r['seed'] in SEEDS]
        report=dict(status='completed',runs=runs,paired_changes=changes,primary_summary=summarize(fresh,SEEDS),
            including_development_seed=summarize(runs,(OLD_SEED,)+SEEDS),plan_sha256=sha(root/'config.json'),
            research_only=True,deployment_replaced=False,test_evaluated=False,
            verified_new_trainings=6,strictly_loaded_models=8,verified_prediction_blocks=197*8,
            completed_at=datetime.now(timezone.utc).isoformat())
        verify_plan(project); _disk(project)
        atomic_json(root/'comparison.json',report)
        (root/'重复实验结果.md').write_text(markdown(report),encoding='utf-8')
        atomic_json(root/'comparison_status.json',dict(status='completed',comparison_sha256=sha(root/'comparison.json')))
        return report

if __name__=='__main__': run()
