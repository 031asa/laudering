"""Two fresh distribution-feature GNN experiments with frozen GNN references."""
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
TRIALS = ('amount_distribution', 'amount_time_distribution')
CODE = ('v3_features.py', 'v3_adapter.py', 'v3_train.py', 'v3_evaluate.py',
        'v3_metrics.py', 'v3_compare.py', 'v3_pipeline.py')
TRAIN_TOTAL, TRAIN_POS, VAL_TOTAL, VAL_POS = 20318992, 17794, 4689006, 6420


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes(project):
    return {name: sha(Path(project)/'scripts/gnn_features'/name) for name in CODE}


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
    return anchors


def make_config(project):
    return dict(version=3, trials=list(TRIALS), experiment='gnn_distribution_only',
                train_targets=TRAIN_TOTAL, train_positives=TRAIN_POS,
                validation_targets=VAL_TOTAL, validation_positives=VAL_POS,
                seed=20260920, epochs=1, learning_rate=.0003,
                positive_weight=1140.9013150500168, width=64, dropout=.1,
                weight_decay=.0001, clip_norm=5, node_dim=63, r_dims=[221, 277],
                feature_window_hours=24, source_sha256=source_hashes(project),
                frozen_anchors=verify_freeze(project),
                environment_sha256=sha(Path(project)/'result/gnn_tuning_v3/environment.json'),
                ranking=['best_f1', 'average_precision', 'precision80_recall'],
                thresholds='One global maximum F1 and one global P80; indivisible score ties',
                deployment='Frozen lr_only remains unchanged; research ranking only',
                scope='Full Train/Validation development; Test 2022-09-14..16 and tail untouched')


def verify_acceptance(project):
    project = Path(project).resolve()
    out = project/'result/gnn_tuning_v3'
    report = read(out/'early_gpu_acceptance/acceptance.json')
    if report.get('status') != 'passed' or set(report.get('trials', {})) != set(TRIALS):
        raise ValueError('Both source-bound real CUDA engineering acceptances required')
    for trial in TRIALS:
        folder = out/'early_gpu_acceptance'/trial
        config, accepted = read(folder/'config.json'), read(folder/'acceptance.json')
        if (accepted != report['trials'][trial] or accepted.get('status') != 'passed'
                or accepted.get('config_sha256') != sha(folder/'config.json')
                or config.get('trial') != trial or accepted.get('trial') != trial):
            raise ValueError('Early CUDA identity differs: '+trial)
        required = ('A_initial_output_equal', 'shared_tensors_exact', 'new_columns_zero',
                    'new_columns_receive_gradient', 'identical_training_rng', 'checkpoint_resume_verified')
        if any(accepted.get(key) is not True for key in required):
            raise ValueError('Missing real CUDA acceptance condition: '+trial)
        if (accepted.get('positive_groups') != ['ACH', 'non_ACH']
                or accepted.get('compared_batches', 0) < 2):
            raise ValueError('Real CUDA batches must cover both positive groups')
        inventory = config.get('v3_code_sha256', {})
        from .v3_train import _code_hashes
        if inventory != _code_hashes(project):
            raise ValueError('Incomplete or changed CUDA acceptance source binding')
        for rel, digest in inventory.items():
            path = (project/rel).resolve()
            if not path.is_relative_to(project) or sha(path) != digest:
                raise ValueError('Source changed since early CUDA acceptance: '+rel)


def write_handoff(project, status, report=None):
    path = Path(project)/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    previous = path.read_text(encoding='utf-8')
    begin, end = '<!-- GNN_V3_STATUS_BEGIN -->', '<!-- GNN_V3_STATUS_END -->'
    lines = [begin, '', '## 第三轮：金额与时间分布统计GNN', '',
             '状态：'+status+'。',
             '- 金额组：A＋四方向24h金额分布52维，节点63／目标关系221维。',
             '- 金额＋时间组：再加历史交易距当前分钟数的分布56维，节点63／目标关系277维。',
             '- 严格更早、同币种外部历史；不增加环、标签风险或图层，不删ACH、不调整正例权重。',
             '- 两组从A未训练参数扩展、新列权重零初始化，完整Train各1epoch；旧模型与v1/v2保持冻结。',
             '- 主指标为完整Validation最大F1；同时报告AP、P80及非ACH／银行70交叉组。',
             '- 原业务达标条件单列；研究排名不会替换冻结lr_only。',
             '- 状态：`result/gnn_tuning_v3/status.json`；入口：`scripts/start_gnn_tuning_v3.ps1`。']
    if report:
        lines += ['', '| 模型 | AP | 最大F1 | P80召回 |', '| --- | ---: | ---: | ---: |']
        for r in report['results']:
            m = r['metrics']; p = m['precision80']
            recall = '无可行阈值' if p is None else f"{p['recall']:.4%}"
            lines.append(f"| {r['name']} | {m['average_precision']:.6f} | {m['best_f1']['f1']:.4%} | {recall} |")
        lines += ['', '研究排名第一：'+report['best']['name']+'。详见 `result/gnn_tuning_v3/调参结果.md`。']
    lines += ['', '**Validation已用于诊断和选择；单种子结果不代表统计显著。Test尚未独立检验。**', '', end]
    section = '\n'.join(lines)
    if begin in previous or end in previous:
        if previous.count(begin) != 1 or previous.count(end) != 1 or previous.index(begin) > previous.index(end):
            raise ValueError('Ambiguous v3 handoff markers')
        updated = previous[:previous.index(begin)]+section+previous[previous.index(end)+len(end):]
    else:
        updated = previous.rstrip()+'\n\n'+section+'\n'
    if path.read_text(encoding='utf-8') != previous:
        raise RuntimeError('Handoff changed concurrently')
    temp = path.with_name(path.name+'.v3.tmp')
    temp.write_text(updated, encoding='utf-8'); temp.replace(path)


class Paused(RuntimeError): pass


def run(project=ROOT, resume=False):
    project = Path(project).resolve(); out = project/'result/gnn_tuning_v3'
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', stage='preflight', pid=os.getpid(), child_pid=None,
                 child_status=None, started_at=now(), trials=list(TRIALS), max_new_trials=2)

    def update(**fields):
        state.update(fields, updated_at=now())
        atomic_json(out/'status.json', state)
        print(json.dumps(state, ensure_ascii=True), flush=True)

    def check_pause():
        flag = out/'pause_request.json'
        if flag.exists() and read(flag).get('enabled') is True: raise Paused('Explicit v3 pause enabled')

    def command(module, args, stage, child_status):
        check_pause()
        if shutil.disk_usage(project).free < 20*1024**3: raise RuntimeError('Disk reserve below 20 GiB')
        if source_hashes(project) != config['source_sha256']: raise ValueError('v3 frozen source changed')
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
                raise ValueError('Passed tests for exact v3 sources required')
            verify_acceptance(project)
            if (out/'config.json').exists():
                if read(out/'config.json') != config: raise ValueError('v3 configuration changed; audited migration required')
            else: atomic_json(out/'config.json', config)
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU GNN fallback')
            update(gpu=torch.cuda.get_device_name(0), completed_trials=[])
            write_handoff(project, '工程验收通过，全量流程已启动，尚无第三轮完整结果')
            for split in ('train', 'validation'):
                command('v3_features', ['--split', split], 'features_'+split,
                        project/'data/gnn_v3_distribution_features'/split/'status.json')
            for trial in TRIALS:
                command('v3_train', ['--trial', trial, '--mode', 'accept'], 'accept_'+trial,
                        out/'trials'/trial/'status.json')
            for i, trial in enumerate(TRIALS):
                command('v3_train', ['--trial', trial, '--mode', 'train'], 'train_'+trial,
                        out/'trials'/trial/'status.json')
                command('v3_evaluate', ['--trial', trial], 'evaluate_'+trial,
                        out/'evaluations'/trial/'status.json')
                update(completed_trials=list(TRIALS[:i+1]))
            command('v3_compare', [], 'compare', out/'comparison_status.json')
            report = read(out/'comparison.json')
            if report.get('status') != 'completed': raise ValueError('Complete comparison required')
            if verify_freeze(project) != config['frozen_anchors']: raise ValueError('Frozen inputs changed')
            write_handoff(project, '两组完整GNN训练与Validation评估完成', report)
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
