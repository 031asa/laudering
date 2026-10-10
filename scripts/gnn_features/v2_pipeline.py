"""Bounded v2 experiment orchestration; frozen v1 artifacts are read-only."""
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

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ('features_only', 'features_pruned', 'features_balanced')
CODE = ('v2_features.py', 'v2_adapter.py', 'v2_train.py', 'v2_evaluate.py',
        'v2_compare.py', 'v2_audit.py', 'v2_pipeline.py')
TRAIN_TOTAL, TRAIN_POS = 20318992, 17794
VAL_TOTAL, VAL_POS = 4689006, 6420


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path outside expected root: ' + str(path))
    return path


def source_hashes(project):
    folder = Path(project) / 'scripts/gnn_features'
    return {name: sha(folder / name) for name in CODE}


def verify_freeze(project):
    """Verify completed v1 anchors and both original frozen code inventories."""
    project = Path(project).resolve()
    base = project / 'result/gnn_train_gpu_v1'
    first = project / 'result/gnn_tuning_v1'
    old_config, original = read(first / 'config.json'), read(base / 'config.json')
    for saved in (original, old_config):
        for name, digest in saved['code_sha256'].items():
            if sha(inside(project / 'scripts/gnn_features', name)) != digest:
                raise ValueError('Frozen v1 source changed: ' + name)
    winner = first / 'trials/lr_only'
    result = read(winner / 'training_result.json')
    if (result.get('status') != 'completed' or result.get('epoch') != 1
            or result.get('targets') != TRAIN_TOTAL or result.get('positives') != TRAIN_POS
            or result.get('config_sha256') != sha(winner / 'config.json')
            or result.get('final_model_sha256') != sha(winner / 'final_model.pt')):
        raise ValueError('Frozen lr_only model/result mismatch')
    complete = read(first / 'comparison.json')
    if complete.get('status') != 'completed' or len(complete.get('results', [])) != 3:
        raise ValueError('First tuning round is incomplete')
    for item in complete['results']:
        m = item['metrics']
        if m['samples'] != VAL_TOTAL or m['positives'] != VAL_POS:
            raise ValueError('First-round Validation coverage mismatch')
    split = read(project / 'config/dataset_split.json')
    for name, dates, total, positive in (
        ('train', [f'2022-09-{d:02d}' for d in range(1, 11)], TRAIN_TOTAL, TRAIN_POS),
        ('validation', [f'2022-09-{d:02d}' for d in range(11, 14)], VAL_TOTAL, VAL_POS),
    ):
        if name == 'train':
            # The frozen full Train comprises the earlier discovery and internal-validation splits.
            parts = [s for s in split['splits'] if s['name'] in ('discovery', 'internal_validation')]
            if len(parts) != 2:
                raise ValueError('Both original Train constituent splits required')
            row = dict(dates=sorted(d for part in parts for d in part['dates']),
                       rows=sum(part['rows'] for part in parts),
                       positive_rows=sum(part['positive_rows'] for part in parts))
        else:
            row, = [s for s in split['splits'] if s['name'] == name]
        if row['dates'] != dates or row['rows'] != total or row['positive_rows'] != positive:
            raise ValueError('Frozen split differs: ' + name)
        cache = project / ('data/gnn_train_cache_parquet' if name == 'train'
                           else 'data/gnn_validation_cache_parquet')
        manifest = read(cache / 'manifest.json')
        if (manifest.get('status') != 'completed' or manifest['targets'] != total
                or sum(b['targets'] for b in manifest['blocks']) != total
                or len(manifest['blocks']) != (797 if name == 'train' else 197)):
            raise ValueError('Frozen cache coverage differs: ' + name)
    anchors = ['config/dataset_split.json',
               'result/gnn_train_gpu_v1/config.json',
               'result/gnn_tuning_v1/config.json', 'result/gnn_tuning_v1/comparison.json',
               'result/gnn_tuning_v1/trials/lr_only/config.json',
               'result/gnn_tuning_v1/trials/lr_only/training_result.json',
               'result/gnn_tuning_v1/trials/lr_only/final_model.pt',
               'result/gnn_tuning_v1/evaluations/lr_only/config.json',
               'result/gnn_tuning_v1/evaluations/lr_only/metrics.json']
    for cache in ('gnn_train_cache_parquet', 'gnn_validation_cache_parquet'):
        anchors += [f'data/{cache}/config.json', f'data/{cache}/manifest.json']
    for archive in ('gnn_v1_train', 'gnn_v1_validation'):
        anchors.append(f'data/{archive}/build.json')
    return {name: sha(inside(project, name)) for name in anchors}


def make_config(project):
    return dict(version=2, trials=list(TRIALS), train_targets=TRAIN_TOTAL,
                train_positives=TRAIN_POS, validation_targets=VAL_TOTAL,
                validation_positives=VAL_POS, seed=20260920, epochs=1,
                learning_rate=.0003, positive_weight=1140.9013150500168,
                width=64, dropout=.1, weight_decay=.0001, clip_norm=5,
                operating_point='One global threshold: maximum recall at precision >= 0.8; whole score ties',
                gates=dict(min_tp=3469, min_non_ach_tp=143, min_precision=.8),
                ranking=['overall_recall', 'non_ach_recall', 'average_precision'],
                history='[t-W,t); no equal timestamps; earlier unlabeled Validation permitted',
                test='Test 2022-09-14..16 and tail untouched; Validation development evidence only',
                source_sha256=source_hashes(project), frozen_v1_anchors=verify_freeze(project))


def verify_early_acceptance(project):
    """Require source-bound real CUDA engineering checks before bulk features."""
    project = Path(project).resolve()
    folder = project / 'result/gnn_tuning_v2/early_gpu_acceptance'
    report = read(folder / 'acceptance.json')
    if report.get('status') != 'passed' or set(report.get('trials', {})) != set(TRIALS):
        raise ValueError('All three early CUDA checks are required')
    required = {str(Path('scripts/gnn_features') / name) for name in
                ('v2_train.py', 'v2_adapter.py', 'v2_features.py', 'tune_model.py')}
    for trial in TRIALS:
        config_path = folder / trial / 'config.json'
        config, accepted = read(config_path), read(folder / trial / 'acceptance.json')
        if (config.get('trial') != trial or accepted != report['trials'][trial]
                or accepted.get('status') != 'passed'
                or accepted.get('config_sha256') != sha(config_path)
                or accepted.get('positive_groups') != ['ACH', 'non_ACH']
                or accepted.get('compared_batches', 0) < 1
                or any(accepted.get(key) is not True for key in
                       ('masked_A_equals_B', 'B_C_identical_initialization',
                        'identical_training_rng', 'checkpoint_resume_verified'))):
            raise ValueError('Early CUDA acceptance is incomplete or mismatched: ' + trial)
        inventory = config.get('v2_code_sha256', {})
        if set(inventory) != required:
            raise ValueError('Early CUDA source inventory differs')
        for relative, expected in inventory.items():
            if sha(inside(project, relative)) != expected:
                raise ValueError('Source changed since early CUDA acceptance: ' + relative)


class Paused(RuntimeError):
    pass


def write_handoff(project, status, comparison=None):
    """Maintain only the v2 section, preserving the user's other handoff prose."""
    path = project / 'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    if not path.is_file():
        raise FileNotFoundError('Existing GPTPro handoff document is required')
    begin, end = '<!-- GNN_V2_STATUS_BEGIN -->', '<!-- GNN_V2_STATUS_END -->'
    previous = path.read_text(encoding='utf-8')
    lines = [begin, '', '## 第二轮：ACH专属统计删减与历史金额参照', '',
             '本节由第二轮独立流程更新；第一轮的模型、特征、缓存和代码保持冻结。', '',
             '- A（features_only）：增加59维历史金额参照；节点63维、目标关系169维。',
             '- B（features_pruned）：在A基础上删除6项ACH专属统计及对应掩码；节点57维、目标关系165维。',
             '- C（features_balanced）：与B相同输入；仅将ACH／非ACH正例损失倍率设为约0.828515／1.947473，负例不变。',
             '- 新增参照包括双方各自方向、同币种24h／72h／7d的历史笔数、P50、P90及当前金额倍数；已有24h笔数不重复加入。',
             '- 保留通用支付方式类别及当前支付方式历史占比，不强化环、不增加图层数；图采样仍为72h。',
             '- 三组均从对齐的未训练参数开始、完整Train各1epoch；种子20260920、学习率0.0003及其余结构参数固定。',
             '- 平衡达标要求整体精确率≥80%、整体TP≥3469、非ACH TP≥143；各模型一个全局阈值，同分交易整组纳入。',
             '- 在达标候选中按整体召回、非ACH召回、AP排序；无达标新候选则保留lr_only。', '',
             '状态：' + status + '。实时进度见 [第二轮状态](result/gnn_tuning_v2/status.json)。',
             '入口：`scripts/start_gnn_tuning_v2.ps1`；结果目录：`result/gnn_tuning_v2`。', '']
    if comparison is not None:
        lines += ['| 模型 | AP | P80精确率 | P80整体召回 | 同阈值非ACH召回 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for item in comparison['results']:
            p, n = item['precision80'], item['non_ach_precision80']
            details = '无非空可行阈值 | — | —' if p is None else f"{p['precision']:.2%} | {p['recall']:.2%} | {n['recall']:.2%}"
            lines.append(f"| {item['name']} | {item['metrics']['average_precision']:.6f} | {details} |")
        lines += ['', '选择结果：**' + comparison['best']['name'] + '**。',
                  '完整指标、顺序对照、分组和独立账户覆盖见 [第二轮结果](result/gnn_tuning_v2/调参结果.md) 和 [comparison.json](result/gnn_tuning_v2/comparison.json)。', '']
    lines += ['Validation已用于问题诊断、特征设计和模型选择；第二轮结果属于开发集改进证据。**Test及尾部仍未处理、未评估。**', '', end]
    section = '\n'.join(lines)
    if begin in previous or end in previous:
        if previous.count(begin) != 1 or previous.count(end) != 1 or previous.index(begin) > previous.index(end):
            raise ValueError('Ambiguous v2 handoff section; preserve document for review')
        updated = previous[:previous.index(begin)] + section + previous[previous.index(end)+len(end):]
    else:
        updated = previous.rstrip() + '\n\n' + section + '\n'
    if path.read_text(encoding='utf-8') != previous:
        raise RuntimeError('Handoff changed concurrently; preserve user edits')
    temporary = path.with_name(path.name + '.v2.tmp')
    temporary.write_text(updated, encoding='utf-8')
    temporary.replace(path)


def run(project=ROOT, resume=False):
    project = Path(project).resolve()
    out = project / 'result/gnn_tuning_v2'
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', stage='preflight', pid=os.getpid(), child_pid=None,
                 started_at=now(), trials=list(TRIALS), max_new_trials=3)

    def update(**fields):
        state.update(fields, updated_at=now())
        atomic_json(out / 'status.json', state)
        print(json.dumps(state, ensure_ascii=True), flush=True)

    def check_pause():
        flag = out / 'pause_request.json'
        if flag.exists() and read(flag).get('enabled') is True:
            raise Paused('Explicit v2 pause request enabled')

    def command(module, arguments, stage, child_status):
        check_pause()
        if shutil.disk_usage(project).free < 20 * 1024**3:
            raise RuntimeError('Less than 20 GiB free; no user data removed')
        if source_hashes(project) != config['source_sha256']:
            raise ValueError('v2 source changed after config freeze')
        args = [sys.executable, '-m', 'gnn_features.' + module, '--project', str(project)]
        args += [str(x) for x in arguments]
        log = out / (stage + '.log')
        update(status='running', stage=stage, child_status=str(child_status), log=str(log), command=args)
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\nRUN ' + now() + '\n')
            stream.flush()
            child = subprocess.Popen(args, cwd=project,
                env={**os.environ, 'PYTHONPATH': str(project / 'scripts'), 'PYTHONIOENCODING': 'utf-8'},
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            update(child_pid=child.pid)
            code = child.wait()
        update(child_pid=None, last_exit_code=code)
        if code == 75:
            raise Paused('Child paused after persistent checkpoint')
        if code:
            raise RuntimeError(f'{stage} failed ({code}); inspect {log}')

    with ProcessLock(out):
        try:
            if resume:
                flag = out / 'pause_request.json'
                if flag.exists() and read(flag).get('enabled') is True:
                    request = read(flag)
                    atomic_json(out / ('pause_history_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'), request)
                    request.update(enabled=False, resumed_at=now())
                    atomic_json(flag, request)
            update()
            check_pause()
            config = make_config(project)
            test_report = read(out / 'engineering_tests.json')
            if (test_report.get('status') != 'passed'
                    or test_report.get('source_sha256') != config['source_sha256']):
                raise ValueError('Passed tests for these exact v2 sources required')
            verify_early_acceptance(project)
            if (out / 'config.json').exists():
                if read(out / 'config.json') != config:
                    raise ValueError('v2 configuration/input/source changed; impact review required')
            else:
                atomic_json(out / 'config.json', config)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; no CPU fallback')
            update(gpu=torch.cuda.get_device_name(0), completed_trials=[])
            write_handoff(project, '第二轮流程已启动，尚无全量新模型结果')
            for split in ('train', 'validation'):
                feature_root = project / 'data/gnn_v2_target_features' / split
                # The feature command gates all bulk work on its own real-data
                # high-degree pilot, and reuses an immutable verified pilot.
                command('v2_features', ['--split', split], 'features_' + split,
                        feature_root / 'status.json')
            command('v2_audit', [], 'train_feature_audit', out / 'audit/status.json')
            for trial in TRIALS:
                folder = out / 'trials' / trial
                command('v2_train', ['--trial', trial, '--mode', 'accept'], 'accept_' + trial,
                        folder / 'status.json')
            for i, trial in enumerate(TRIALS):
                folder = out / 'trials' / trial
                command('v2_train', ['--trial', trial, '--mode', 'train'], 'train_' + trial,
                        folder / 'status.json')
                command('v2_evaluate', ['--trial', trial], 'evaluate_' + trial,
                        out / 'evaluations' / trial / 'status.json')
                update(completed_trials=list(TRIALS[:i+1]))
            command('v2_compare', [], 'compare', out / 'comparison_status.json')
            comparison = read(out / 'comparison.json')
            if comparison.get('status') != 'completed' or not (out / 'best_model.json').is_file():
                raise ValueError('Complete comparison and selected model pointer required')
            verify_freeze(project)
            write_handoff(project, '三组完整训练与Validation评估已完成', comparison)
            update(status='completed', stage='complete', child_status=None,
                   best_model=read(out / 'best_model.json'))
            return 0
        except Paused as exc:
            update(status='paused', reason=str(exc), child_pid=None)
            return 75
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--resume', action='store_true', help='Only after explicit user resume instruction')
    args = parser.parse_args()
    sys.exit(run(args.project, args.resume))
