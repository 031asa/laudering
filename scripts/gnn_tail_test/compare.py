"""Replay tail predictions and combine with immutable original Test scores."""
import json
from pathlib import Path
import statistics
import numpy as np
import pyarrow.parquet as pq
from gnn_final_test.common import test_metrics, bound_training
from gnn_final_test.compare import replay as replay_original
from gnn_features.training_state import atomic_json
from gnn_features.cache_train import sha, array_sha
from .pipeline import PREFIX, TOTAL, POSITIVES, SEEDS, TRIALS, OUTPUT, read

def paired(y,lr,a,lt,at,src,dst):
    answer={}
    for point in lt:
        if lt[point] is None or at[point] is None:
            answer[point]=None;continue
        lh,ah=lr>=lt[point],a>=at[point];positive=y==1
        new=positive&ah&~lh;lost=positive&lh&~ah
        accounts=lambda mask:set(map(int,np.r_[src[mask],dst[mask]]))
        la,aa=accounts(positive&lh),accounts(positive&ah)
        answer[point]=dict(added_tp=int(new.sum()),lost_tp=int(lost.sum()),
            added_fp=int(((y==0)&ah&~lh).sum()),lost_fp=int(((y==0)&lh&~ah).sum()),
            lr_detected_accounts=len(la),A_detected_accounts=len(aa),new_detected_accounts=len(aa-la),lost_detected_accounts=len(la-aa))
    return answer

def summarize(rows,seeds,scope):
    out={}
    for trial in TRIALS:
        selected=[r[scope] for r in rows if r['seed'] in seeds and r['trial']==trial]
        fields={'ap':[r['average_precision'] for r in selected]}
        for point in ('validation_maximum_f1','validation_precision80'):
            for metric in ('f1','precision','recall'):
                fields[point+'_'+metric]=[None if r['fixed_thresholds'][point] is None else r['fixed_thresholds'][point][metric] for r in selected]
        out[trial]={k:None if any(x is None for x in xs) else dict(mean=statistics.mean(xs),sample_sd=statistics.stdev(xs),minimum=min(xs),maximum=max(xs)) for k,xs in fields.items()}
    return out

def compare(project,config_hash,arrays,bank,vocab):
    root=Path(project)/OUTPUT;rows=[];pairs=[];original_prefix=25007998
    start=original_prefix;all_ids=np.arange(start,PREFIX+TOTAL);tail_ids=np.arange(PREFIX,PREFIX+TOTAL)
    identity=read(Path(project)/'data/gnn_tail_test_archive/metadata/bank_identity.json')
    if identity.count('70')!=1:raise ValueError('Bank70 mapping changed')
    formats=arrays['payment_format_cat'][all_ids];non=formats!=vocab['payment_format'].index('ACH')+2
    masks={'non_ACH':non,'bank70_non_ACH':non&(bank[arrays['src'][all_ids]]==identity.index('70'))}
    dates=arrays['timestamp'][all_ids].astype('datetime64[us]').astype('datetime64[D]').astype(str)
    for date in np.unique(dates):masks['date_'+date]=dates==date
    for code in np.unique(formats):masks['payment_identity_'+str(int(code))]=formats==code
    for seed in SEEDS:
        scores={};thresholds={}
        for trial in TRIALS:
            name=f'{seed}_{trial}';acceptance=read(root/f'acceptance_{name}.json')
            if acceptance['status']!='passed' or acceptance['config_sha256']!=config_hash:raise ValueError('Missing tail CUDA acceptance')
            trained=bound_training(project,trial,seed);thresholds[trial]=trained['binding']['thresholds']
            tail_y=np.empty(TOTAL,np.int8);tail_s=np.empty(TOTAL,np.float32);seen=np.zeros(TOTAL,bool);inventory=[]
            for mp in sorted((root/'predictions'/name).glob('*.json')):
                m=read(mp);path=mp.with_suffix('.parquet')
                if m['config_sha256']!=config_hash or sha(path)!=m['file_sha256']:raise ValueError('Tail prediction checksum mismatch')
                table=pq.ParquetFile(path).read();ids=table['edge_id'].to_numpy();pos=ids-PREFIX
                if array_sha(ids)!=m['target_ids_sha256'] or np.any(pos<0) or np.any(pos>=TOTAL) or seen[pos].any() or len(np.unique(pos))!=len(pos):raise ValueError('Tail coverage identity mismatch')
                if not np.array_equal(table['timestamp_us'].to_numpy(),arrays['timestamp'][ids]):raise ValueError('Tail timestamp mismatch')
                tail_y[pos]=table['label'].to_numpy();tail_s[pos]=table['probability'].to_numpy();seen[pos]=True
                inventory.append(dict(file=str(path.relative_to(root)),sha256=m['file_sha256'],targets=len(ids)))
            if not seen.all() or tail_y.sum()!=POSITIVES:raise ValueError('Tail coverage incomplete')
            report,old_y,old_s,old_t=replay_original(project,trial,seed)
            if old_t!=thresholds[trial]:raise ValueError('Original/tail threshold mismatch')
            y=np.r_[old_y,tail_y];s=np.r_[old_s,tail_s];scores[trial]=s
            combined=test_metrics(y,s,old_t);tail=test_metrics(tail_y,tail_s,old_t)
            if combined['samples']!=6890220 or combined['positives']!=11016:raise ValueError('Combined totals mismatch')
            for point,c in combined['fixed_thresholds'].items():
                if c is not None:
                    for k in ('tp','fp','fn','tn'):
                        if c[k]!=report['metrics']['fixed_thresholds'][point][k]+tail['fixed_thresholds'][point][k]:raise ValueError('Combined confusion not additive')
            row=dict(seed=seed,trial=trial,model_sha256=trained['model_sha256'],thresholds=old_t,
                original_test=report['metrics'],tail=tail,combined=combined,
                tail_groups={n:test_metrics(tail_y[m[-TOTAL:]],tail_s[m[-TOTAL:]],old_t) for n,m in masks.items() if m[-TOTAL:].any()},
                combined_groups={n:test_metrics(y[m],s[m],old_t) for n,m in masks.items()},prediction_inventory=inventory)
            rows.append(row)
            print(json.dumps(dict(stage='replayed',seed=seed,trial=trial,tail_f1=tail['fixed_thresholds']['validation_maximum_f1']['f1']),ensure_ascii=True),flush=True)
        pairs.append(dict(seed=seed,tail=paired(y[-TOTAL:],scores['lr_only'][-TOTAL:],scores['features_only'][-TOTAL:],thresholds['lr_only'],thresholds['features_only'],arrays['src'][tail_ids],arrays['dst'][tail_ids]),
            combined=paired(y,scores['lr_only'],scores['features_only'],thresholds['lr_only'],thresholds['features_only'],arrays['src'][all_ids],arrays['dst'][all_ids])))
    summary={group:{scope:summarize(rows,seeds,scope) for scope in ('original_test','tail','combined')} for group,seeds in [('three_fresh_seeds',SEEDS[1:]),('all_four_seeds',SEEDS)]}
    result=dict(status='completed',config_sha256=config_hash,results=rows,paired=pairs,summaries=summary,
        interpretation='Post-hoc tail sensitivity analysis requested after original Test was observed. Changed class mix; cannot claim improved model or replace original Test.',
        original_test_preserved=True,training_changed=False,thresholds_retuned=False,deployment_replaced=False)
    atomic_json(root/'comparison.json',result)
    lines=['# 尾部异常段追加评估（2026-10-12）','',
        '这是看过原Test结果后追加的敏感性分析。全部8模型和各自Validation阈值冻结，无重训、无阈值寻优；原Test报告保留。',
        '尾部09-17..28共6,987笔/4,084正例（58.45%）；原Test为6,883,233笔/6,932正例；合并为6,890,220笔/11,016正例。',
        '合并AP由全量预测重新排序计算；混淆矩阵逐项相加核对。不能平均两段F1或AP。','',
        '| 种子 | 配方 | 原Test F1 | 尾部F1 | 合并F1 | 合并−原Test/百分点 | 尾部TP | 尾部FP |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        a,b,c=[row[x]['fixed_thresholds']['validation_maximum_f1'] for x in ('original_test','tail','combined')]
        lines.append(f"| {row['seed']} | {row['trial']} | {a['f1']:.4%} | {b['f1']:.4%} | {c['f1']:.4%} | {100*(c['f1']-a['f1']):+.4f} | {b['tp']} | {b['fp']} |")
    for key,title in [('three_fresh_seeds','三个新种子（主分析）'),('all_four_seeds','含旧开发种子的四次汇总')]:
        lines+=['','## '+title,'','| 数据范围 | 配方 | F1均值±样本SD（范围） | AP均值±样本SD（范围） |','|---|---|---:|---:|']
        for scope,models in summary[key].items():
            for trial,m in models.items():
                cell=lambda k:f"{m[k]['mean']:.4%} ± {m[k]['sample_sd']:.4%}（{m[k]['minimum']:.4%}–{m[k]['maximum']:.4%}）"
                lines.append(f"| {scope} | {trial} | {cell('validation_maximum_f1_f1')} | {cell('ap')} |")
    lines+=['','两套固定阈值的precision/recall/F1、日期/支付方式、非ACH/bank70及配对检出/误报/独立账户差见comparison.json。',
        '所有特征只读取当前交易之前的历史，同一时刻交易互不可见；Test和尾部标签均不进入特征。原Test首批输入及8模型预测先回归核验，尾部A特征全量通过独立精确排序参考核对。',
        '尾部正例率异常高，合并指标的变化同时反映样本构成变化。只能作为追加敏感性结果，不代表模型本身提高，不作统计显著或论文优越性声明。']
    (root/'尾部追加评估结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
