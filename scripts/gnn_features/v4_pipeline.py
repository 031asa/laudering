"""One fresh three-structure GNN experiment with frozen GNN references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from .cache_train import sha
from .training_state import ProcessLock, atomic_json
from .v2_pipeline import verify_freeze as verify_v1, source_hashes as v2_sources

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ('structure_three',)
CODE = ('v4_features.py', 'v4_adapter.py', 'v4_train.py', 'v4_evaluate.py',
        'v4_metrics.py', 'v4_compare.py', 'v4_pipeline.py')
TRAIN_TOTAL, TRAIN_POS, VAL_TOTAL, VAL_POS = 20318992, 17794, 4689006, 6420


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes(project):
    result = {name: sha(Path(project)/'scripts/gnn_features'/name) for name in CODE}
    result['audited_structure_engine'] = sha(Path(project)/'scripts/gnn_causal_structures.py')
    return result


def verify_freeze(project):
    project = Path(project).resolve()
    anchors = verify_v1(project)
    root = project/'result/gnn_tuning_v2'
    config = read(root/'config.json')
    if config['source_sha256'] != v2_sources(project):
        raise ValueError('Frozen v2 source differs')
    if read(root/'status.json')['status'] != 'completed':
        raise ValueError('Frozen v2 pipeline must be complete')
    from .v2_evaluate import verified_training
    verified_training(project, 'features_only')
    names = ['result/gnn_tuning_v2/'+x for x in
             ('config.json', 'comparison.json', 'best_model.json', 'bank70_correction.json',
              'trials/features_only/config.json', 'trials/features_only/training_result.json',
              'trials/features_only/final_model.pt', 'evaluations/features_only/config.json',
              'evaluations/features_only/metrics.json')]
    for split in ('train', 'validation'):
        names += [f'data/gnn_v2_target_features/{split}/{n}.json' for n in ('config', 'manifest')]
    anchors.update({name: sha(project/name) for name in names})
    v3root = project/'result/gnn_tuning_v3'
    from .v3_pipeline import source_hashes as frozen_v3_sources
    if read(v3root/'config.json')['source_sha256'] != frozen_v3_sources(project):
        raise ValueError('Frozen v3 source changed')
    for name in ('config.json','comparison.json','best_model.json'):
        anchors['result/gnn_tuning_v3/'+name] = sha(v3root/name)
    for trial in ('amount_distribution','amount_time_distribution'):
        for name in ('config.json','training_result.json','final_model.pt'):
            rel = 'result/gnn_tuning_v3/trials/'+trial+'/'+name
            anchors[rel] = sha(project/rel)
    return anchors


def make_config(project):
    return dict(version=4, trials=list(TRIALS), experiment='gnn_structure_only',
                train_targets=TRAIN_TOTAL, train_positives=TRAIN_POS,
                validation_targets=VAL_TOTAL, validation_positives=VAL_POS,
                seed=20260920, epochs=1, learning_rate=.0003,
                positive_weight=1140.9013150500168, width=64, dropout=.1,
                weight_decay=.0001, clip_norm=5, node_dim=63, r_dims=[175],
                feature_window_hours=72, source_sha256=source_hashes(project),
                frozen_anchors=verify_freeze(project),
                environment_sha256=sha(Path(project)/'result/gnn_tuning_v4/environment.json'),
                ranking=['best_f1', 'average_precision', 'precision80_recall'],
                thresholds='One global maximum F1 and one global P80; indivisible score ties',
                deployment='Frozen lr_only remains unchanged; research ranking only',
                scope='Full Train/Validation development; Test 2022-09-14..16 and tail untouched')


def verify_acceptance(project):
    project = Path(project).resolve()
    out = project/'result/gnn_tuning_v4'
    report = read(out/'early_gpu_acceptance/acceptance.json')
    if report.get('status') != 'passed' or set(report.get('trials', {})) != set(TRIALS):
        raise ValueError('Source-bound real CUDA engineering acceptance required')
    for trial in TRIALS:
        folder = out/'early_gpu_acceptance'/trial
        config, accepted = read(folder/'config.json'), read(folder/'acceptance.json')
        if (accepted != report['trials'][trial] or accepted.get('status') != 'passed'
                or accepted.get('config_sha256') != sha(folder/'config.json')
                or config.get('trial') != trial or accepted.get('trial') != trial):
            raise ValueError('Early CUDA identity differs: '+trial)
        required = ('A_initial_output_equal', 'shared_tensors_exact', 'new_columns_zero',
                    'new_columns_receive_gradient', 'identical_training_rng', 'checkpoint_resume_verified', 'all_six_columns_updated')
        if any(accepted.get(key) is not True for key in required):
            raise ValueError('Missing real CUDA acceptance condition: '+trial)
        if (accepted.get('positive_groups') != ['ACH', 'non_ACH']
                or accepted.get('compared_batches', 0) < 2):
            raise ValueError('Real CUDA batches must cover both positive groups')
        inventory = config.get('v4_code_sha256', {})
        from .v4_train import _code_hashes
        if inventory != _code_hashes(project):
            raise ValueError('Incomplete or changed CUDA acceptance source binding')
        for rel, digest in inventory.items():
            path = (project/rel).resolve()
            if not path.is_relative_to(project) or sha(path) != digest:
                raise ValueError('Source changed since early CUDA acceptance: '+rel)


def write_handoff(project, status, report=None):
    path = Path(project)/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    previous = path.read_text(encoding='utf-8')
    begin, end = '<!-- GNN_V4_STATUS_BEGIN -->', '<!-- GNN_V4_STATUS_END -->'
    lines = [begin, '', '## 第四轮：A＋三项因果结构特征', '', '状态：'+status+'。',
             '- 第二轮A追加72h反向回流、时序三角回流、金额连续三边转付链计数及有效性标记；N63／R175。',
             '- 历史严格[t−72h,t)，路径时间递增、账户互异、同币种；不读取历史标签。',
             '- 全部支付方式统一计算，节点和采样图不变；不追加第三轮金额／时间分布特征。',
             '- 从未训练A复制共同参数，新列零初始化；原参数、原权重，完整Train一次1epoch。',
             '- 主指标完整Validation最大F1，同时报告AP、P80、非ACH及银行70交叉组。',
             '- 结构候选此前仅在Train匹配分析中出现关联，尚不代表模型增益。',
             '- 状态：`result/gnn_tuning_v4/status.json`；入口：`scripts/start_gnn_tuning_v4.ps1`。']
    if report:
        lines += ['', '| 模型 | AP | 最大F1 | P80召回 |', '| --- | ---: | ---: | ---: |']
        for r in report['results']:
            m = r['metrics']; p = m['precision80']
            recall = '无可行阈值' if p is None else f"{p['recall']:.4%}"
            lines.append(f"| {r['name']} | {m['average_precision']:.6f} | {m['best_f1']['f1']:.4%} | {recall} |")
        lines += ['', '研究排名第一：'+report['best']['name']+'。详见 `result/gnn_tuning_v4/调参结果.md`。']
    lines += ['', '**Validation已用于诊断和选择；单种子结果不代表统计显著。Test尚未独立检验。**', '', end]
    section = '\n'.join(lines)
    if begin in previous or end in previous:
        if previous.count(begin) != 1 or previous.count(end) != 1 or previous.index(begin) > previous.index(end):
            raise ValueError('Ambiguous v4 handoff markers')
        updated = previous[:previous.index(begin)]+section+previous[previous.index(end)+len(end):]
    else:
        updated = previous.rstrip()+'\n\n'+section+'\n'
    if path.read_text(encoding='utf-8') != previous:
        raise RuntimeError('Handoff changed concurrently')
    temp = path.with_name(path.name+'.v4.tmp')
    temp.write_text(updated, encoding='utf-8'); temp.replace(path)


class Paused(RuntimeError): pass


def run(project=ROOT, resume=False):
    project = Path(project).resolve(); out = project/'result/gnn_tuning_v4'
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', stage='preflight', pid=os.getpid(), child_pid=None,
                 child_status=None, started_at=now(), trials=list(TRIALS), max_new_trials=1)

    def update(**fields):
        state.update(fields, updated_at=now())
        atomic_json(out/'status.json', state)
        print(json.dumps(state, ensure_ascii=True), flush=True)

    def check_pause():
        flag = out/'pause_request.json'
        if flag.exists() and read(flag).get('enabled') is True: raise Paused('Explicit v4 pause enabled')

    def command(module, args, stage, child_status):
        check_pause()
        if shutil.disk_usage(project).free < 20*1024**3: raise RuntimeError('Disk reserve below 20 GiB')
        if source_hashes(project) != config['source_sha256']: raise ValueError('v4 frozen source changed')
        argv = [sys.executable, '-m', 'gnn_features.'+module, '--project', str(project)]+list(args)
        log = out/(stage+'.log')
        update(status='running', stage=stage, child_status=str(child_status), log=str(log), command=argv)
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\nRUN '+now()+'\n'); stream.flush()
            process = subprocess.Popen(argv, cwd=project,
                env={**os.environ, 'PYTHONPATH': str(project/'scripts'), 'PYTHONIOENCODING': 'utf-8'},
                stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            update(child_pid=process.pid)
            code = process.wait()
        update(child_pid=None, last_exit_code=code)
        if code == 75: raise Paused('Child paused at persistent checkpoint')
        if code: raise RuntimeError(f'{stage} failed ({code}); inspect {log}')

    with ProcessLock(out):
        try:
            flag = out/'pause_request.json'
            if resume and flag.exists() and read(flag).get('enabled') is True:
                request = read(flag)
                atomic_json(out/('pause_history_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'), request)
                request.update(enabled=False, resumed_at=now()); atomic_json(flag, request)
            update(); check_pause()
            config = make_config(project)
            tested = read(out/'engineering_tests.json')
            if tested.get('status') != 'passed' or tested.get('source_sha256') != config['source_sha256']:
                raise ValueError('Passed tests for exact v4 sources required')
            verify_acceptance(project)
            if (out/'config.json').exists():
                if read(out/'config.json') != config: raise ValueError('v4 configuration changed; audited migration required')
            else: atomic_json(out/'config.json', config)
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU GNN fallback')
            update(gpu=torch.cuda.get_device_name(0), completed_trials=[])
            write_handoff(project, '工程验收通过，全量流程已启动，尚无第四轮完整结果')
            for split in ('train', 'validation'):
                command('v4_features', ['--split', split], 'features_'+split,
                        project/'data/gnn_v4_structure_features'/split/'status.json')
            for trial in TRIALS:
                command('v4_train', ['--trial', trial, '--mode', 'accept'], 'accept_'+trial,
                        out/'trials'/trial/'status.json')
            for i, trial in enumerate(TRIALS):
                command('v4_train', ['--trial', trial, '--mode', 'train'], 'train_'+trial,
                        out/'trials'/trial/'status.json')
                command('v4_evaluate', ['--trial', trial], 'evaluate_'+trial,
                        out/'evaluations'/trial/'status.json')
                update(completed_trials=list(TRIALS[:i+1]))
            command('v4_compare', [], 'compare', out/'comparison_status.json')
            report = read(out/'comparison.json')
            if report.get('status') != 'completed': raise ValueError('Complete comparison required')
            if verify_freeze(project) != config['frozen_anchors']: raise ValueError('Frozen inputs changed')
            write_handoff(project, '一组完整GNN训练与Validation评估完成', report)
            update(status='completed', stage='complete', child_status=None, best_model=read(out/'best_model.json'))
            return 0
        except Paused as exc:
            update(status='paused', reason=str(exc), child_pid=None); return 75
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--resume', action='store_true', help='Only following explicit user resume')
    a = p.parse_args(); sys.exit(run(a.project, a.resume))
