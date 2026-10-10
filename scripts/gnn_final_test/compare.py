"""Replay all eight Test prediction inventories, without selecting a winner."""
import argparse
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .common import *
from gnn_features.build import load_archive
from gnn_features.cache_train import array_sha

def replay(project,trial,seed):
    folder=eval_root(project,trial,seed); report=read(folder/'metrics.json'); cfg=read(folder/'config.json')
    trained=bound_training(project,trial,seed)
    if report['model_sha256']!=trained['model_sha256'] or report['binding_sha256']!=trained['binding_sha256'] or report['config_sha256']!=sha(folder/'config.json'):
        raise ValueError('Model or Test report binding mismatch')
    acceptance=read(folder/'acceptance.json')
    if acceptance['status']!='passed' or acceptance['config_sha256']!=report['config_sha256']: raise ValueError('Missing CUDA acceptance')
    y=np.empty(TOTAL,np.int8);scores=np.empty(TOTAL,np.float32);seen=np.zeros(TOTAL,bool)
    for entry in report['blocks']:
        block=folder/'predictions'/entry['block']; record=read(block/'manifest.json'); path=block/'predictions.parquet'
        if sha(block/'manifest.json')!=entry['manifest_sha256'] or sha(path)!=record['file_sha256'] or record['config_sha256']!=report['config_sha256']:
            raise ValueError('Prediction inventory checksum mismatch')
        tab=pq.ParquetFile(path).read(); ids=tab['edge_id'].to_numpy(); pos=ids-PREFIX
        if array_sha(ids)!=record['target_ids_sha256'] or np.any(pos<0) or np.any(pos>=TOTAL) or seen[pos].any() or len(np.unique(pos))!=len(pos):
            raise ValueError('Prediction target identity mismatch')
        seen[pos]=True;y[pos]=tab['label'].to_numpy();scores[pos]=tab['probability'].to_numpy()
    if not seen.all() or int(y.sum())!=POSITIVES: raise ValueError('Test prediction coverage mismatch')
    if test_metrics(y,scores,trained['binding']['thresholds'])!=report['metrics']: raise ValueError('Metric replay differs')
    return report,y,scores,trained['binding']['thresholds']
def run(project=ROOT):
    project=Path(project);root=project/OUTPUT; protocol(project)
    atomic_json(root/'comparison_status.json',dict(status='running',pid=os.getpid(),stage='replay',updated_at=now()))
    meta,arrays,bank,vocab=load_archive(project/'data/gnn_final_test_archive')
    masks=slices(arrays,bank,read(project/'data/gnn_final_test_archive/metadata/bank_identity.json'),vocab)
    reports=[];pairs=[]
    src,dst=arrays['src'][PREFIX:],arrays['dst'][PREFIX:]
    def accounts(mask): return int(len(np.unique(np.r_[src[mask],dst[mask]])))
    for seed in SEEDS:
        lr,y,ls,lt=replay(project,'lr_only',seed);a,ay,ss,at=replay(project,'features_only',seed)
        if not np.array_equal(y,ay): raise ValueError('Paired Test labels differ')
        for r,s,t in [(lr,ls,lt),(a,ss,at)]:
            if r['groups']!={n:test_metrics(y[m],s[m],t) for n,m in masks.items()}: raise ValueError('Group replay differs')
            reports.append({k:r[k] for k in ('seed','trial','model_sha256','metrics','groups')})
        points={}
        for name in lt:
            if lt[name] is None or at[name] is None:
                points[name]=None;continue
            lh,ah=ls>=lt[name],ss>=at[name];new=(y==1)&ah&~lh;lost=(y==1)&lh&~ah
            points[name]=dict(f1_difference=a['metrics']['fixed_thresholds'][name]['f1']-lr['metrics']['fixed_thresholds'][name]['f1'],
                added_tp=int(new.sum()),lost_tp=int(lost.sum()),fp_change=int(((y==0)&ah).sum()-((y==0)&lh).sum()),
                lr_detected_accounts=accounts((y==1)&lh),A_detected_accounts=accounts((y==1)&ah),
                added_detection_accounts=accounts(new),lost_detection_accounts=accounts(lost),
                groups={n:dict(added_tp=int((new&m).sum()),lost_tp=int((lost&m).sum()),
                    fp_change=int(((y==0)&ah&m).sum()-((y==0)&lh&m).sum())) for n,m in masks.items()})
        pairs.append(dict(seed=seed,points=points))
        atomic_json(root/'comparison_status.json',dict(status='running',pid=os.getpid(),stage='replay',completed_seeds=seed,updated_at=now()))
    def summary(seeds):
        result={}
        for trial in TRIALS:
            chosen=[r for r in reports if r['seed'] in seeds and r['trial']==trial]
            fields={'average_precision':[r['metrics']['average_precision'] for r in chosen]}
            for point in ('validation_maximum_f1','validation_precision80'):
                for metric in ('precision','recall','f1'):
                    fields[point+'_'+metric]=[None if r['metrics']['fixed_thresholds'][point] is None else r['metrics']['fixed_thresholds'][point][metric] for r in chosen]
            result[trial]={k:None if any(v is None for v in values) else dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),minimum=float(min(values)),maximum=float(max(values))) for k,values in fields.items()}
        return result
    comparison=dict(status='completed',scope='All predeclared Test runs; no winner selection',results=reports,paired=pairs,
        three_fresh_seeds=summary(SEEDS[1:]),all_four_seeds=summary(SEEDS),research_only=True,deployment_replaced=False,
        warning='Single temporal Test interval. Seed variation conditional on frozen graph/batches; no significance claim. Test thresholds copied from Validation; no oracle maximum F1 or Test P80 search.',
        protocol_sha256=sha(root/'protocol.json'),completed_at=now())
    atomic_json(root/'comparison.json',comparison)
    lines=['# 最终 Test 结果','', '固定 Validation 阈值；全部八个模型，未按 Test 挑选配方、种子或阈值。','',
        '| 种子 | 模型 | 固定Val最大F1阈值的Test F1 | Test AP | Val P80阈值的Test精确率 | 召回率 |','|---|---|---:|---:|---:|---:|']
    for r in reports:
        m=r['metrics']; f=m['fixed_thresholds']['validation_maximum_f1'];p=m['fixed_thresholds']['validation_precision80']
        lines.append(f"| {r['seed']} | {r['trial']} | {f['f1']:.6f} | {m['average_precision']:.6f} | {p['precision'] if p else None} | {p['recall'] if p else None} |")
    lines+=['','完整均值、样本标准差、范围、逐种子新增/损失检出、误报、独立账户及分组见 comparison.json。',
        'Validation P80阈值不保证Test精确率达到80%。这是一个时间段的独立测试，不宣称统计显著或超越论文benchmark；旧部署不替换。']
    (root/'最终测试结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    handoff=project/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    previous=handoff.read_text(encoding='utf-8');start='<!-- GNN_FINAL_TEST_BEGIN -->';end='<!-- GNN_FINAL_TEST_END -->'
    section=start+'\n\n'+ '\n'.join(lines)+'\n\n'+end
    if start in previous:
        if previous.count(start)!=1 or previous.count(end)!=1: raise ValueError('Ambiguous handoff markers')
        updated=previous[:previous.index(start)]+section+previous[previous.index(end)+len(end):]
    else: updated=previous+'\n\n'+section+'\n'
    if handoff.read_text(encoding='utf-8')!=previous: raise ValueError('Concurrent handoff edit')
    temporary=handoff.with_suffix('.final_test.tmp');temporary.write_text(updated,encoding='utf-8');os.replace(temporary,handoff)
    atomic_json(root/'comparison_status.json',dict(status='completed',pid=os.getpid(),updated_at=now()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=ROOT);a=p.parse_args();run(a.project)
