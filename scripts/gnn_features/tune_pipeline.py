"""One bounded tuning run: prepare full Validation, baseline, two full Train trials."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from .cache_train import sha
from .training_state import ProcessLock, atomic_json
from .validation_cache import DATES, TOTAL, POSITIVES, read, validation_spec


ROOT = Path(__file__).resolve().parents[2]
PWSH = Path('C:/Users/28118/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe')
CODE = ('tune_pipeline.py', 'tune_model.py', 'validation_archive.py', 'validation_cache.py',
        'evaluate_validation.py', 'evaluation_metrics.py', 'build.py', 'daily.py', 'verify_daily.py',
        'benchmark.py', 'partitioned.py', 'rolling.py', 'engine.py', 'contract.py', 'loader.py',
        'model.py', 'model_cache.py', 'training_state.py', 'cache_train.py', 'hybrid_cache_trial.py',
        'verify_validation.py')


def candidates(positive_weight):
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError('Invalid full Train positive weight')
    return [dict(name='lr_only', learning_rate=.0003, positive_weight=positive_weight),
            dict(name='sqrt_weight', learning_rate=.0003, positive_weight=math.sqrt(positive_weight))]


def choose_best(results):
    if len(results) != 3 or {r['name'] for r in results} != {'baseline', 'lr_only', 'sqrt_weight'}:
        raise ValueError('All three predeclared candidates must finish before selection')
    for r in results:
        m = r['metrics']
        if m['samples'] != TOTAL or m['positives'] != POSITIVES or not math.isfinite(m['average_precision']):
            raise ValueError('Incomplete/nonfinite Validation metrics')
    # Stable ties keep the earlier candidate; AP is the only primary objective.
    return max(results, key=lambda r: (r['metrics']['average_precision'], r['metrics']['roc_auc']))


def run(project=ROOT):
    project = Path(project).resolve()
    out = project / 'result/gnn_tuning_v1'
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                 stage='preflight', child_pid=None, max_new_trials=2, validation_dates=DATES)

    def update(**fields):
        state.update(fields)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out / 'status.json', state)
        print(json.dumps(state, ensure_ascii=True), flush=True)

    def command(args, stage, child_status=None, name=None):
        if shutil.disk_usage(project).free < 20 * 1024 ** 3:
            raise RuntimeError('Less than 20 GiB free; no user files removed')
        log = out / f'{name or stage}.log'
        update(status='running', stage=stage, log=str(log),
               child_status=str(child_status) if child_status else None,
               command=[str(x) for x in args])
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\nRUN ' + datetime.now(timezone.utc).isoformat() + '\n')
            stream.flush()
            child = subprocess.Popen([str(x) for x in args], cwd=project,
                                     env={**os.environ, 'PYTHONPATH': str(project / 'scripts'),
                                          'PYTHONIOENCODING': 'utf-8'}, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            update(child_pid=child.pid)
            code = child.wait()
        update(child_pid=None, last_exit_code=code)
        if code:
            raise RuntimeError(f'{stage} failed ({code}); inspect {log}')

    with ProcessLock(out):
        try:
            update()
            validation_spec(project)
            base = project / 'result/gnn_train_gpu_v1'
            base_config, base_result = read(base / 'config.json'), read(base / 'training_result.json')
            if base_result['status'] != 'completed' or base_result['targets'] != 20318992 or base_result['positives'] != 17794 or base_result['epoch'] != 1:
                raise ValueError('Complete verified baseline required')
            if sha(base / 'final_model.pt') != base_result['final_model_sha256'] or sha(base / 'config.json') != base_result['config_sha256']:
                raise ValueError('Baseline model identity mismatch')
            trial_specs = candidates(base_config['positive_weight'])
            config = dict(version=1, train_dates=base_config['dates'], train_targets=20318992,
                          train_positives=17794, validation_dates=DATES, validation_targets=TOTAL,
                          validation_positives=POSITIVES, new_trials=trial_specs, epochs_per_trial=1,
                          seed=20260920, width=64, dropout=.1, weight_decay=.0001, clip_norm=5,
                          primary_metric='full Validation average_precision; ROC AUC tie break',
                          test_policy='Do not access 2022-09-14..16 or tail data; no final Test evaluation',
                          history_policy='Strictly earlier timestamps, including preceding label-free Validation events',
                          baseline_config_sha256=sha(base / 'config.json'),
                          baseline_model_sha256=sha(base / 'final_model.pt'),
                          split_sha256=sha(project / 'config/dataset_split.json'),
                          code_sha256={n: sha(Path(__file__).with_name(n)) for n in CODE})
            if (out / 'config.json').exists():
                if read(out / 'config.json') != config:
                    raise ValueError('Frozen tuning configuration/code changed; review required')
            else:
                atomic_json(out / 'config.json', config)
            for rel, digest in base_config['input_hashes'].items():
                if sha(project / rel) != digest:
                    raise ValueError('Frozen Train input changed: ' + rel)
            for rel, digest in base_config['code_sha256'].items():
                path = project / rel
                if not path.is_file():
                    path = Path(__file__).with_name(rel)
                if sha(path) != digest:
                    raise ValueError('Frozen Train source changed: ' + rel)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required; no CPU fallback')
            update(gpu=torch.cuda.get_device_name(0), completed_trials=[])
            if not PWSH.exists():
                raise FileNotFoundError(PWSH)
            archive = project / 'data/gnn_v1_validation'
            command([sys.executable, '-m', 'gnn_features.validation_archive', '--project', project,
                     '--output', archive], 'validation_archive',
                    archive / 'status.json')
            meta = read(archive / 'build.json')
            if meta['rows'] != 20318992 + TOTAL or meta['end_exclusive'] != '2022-09-14T00:00:00':
                raise ValueError('Validation history coverage mismatch')
            features = project / 'result/gnn_validation_v1'
            for date in DATES:
                folder = features / f'transaction_date={date}'
                update(current_date=date)
                if not (folder / 'verification.json').exists():
                    if not (folder / 'daily_run.json').exists():
                        command([PWSH, '-NoProfile', '-File', project / 'scripts/daily_gnn.ps1',
                                 '-Archive', archive, '-Output', folder, '-Date', date,
                                 '-Cores', '4', '-Partitions', '8', '-Resume'],
                                'validation_features', folder / 'daily_run.json', date + '_features')
                    command([sys.executable, '-m', 'gnn_features.verify_validation', '--archive', archive,
                             '--output', folder], 'verify_validation_features', folder / 'daily_run.json', date + '_verify')
                if read(folder / 'verification.json')['status'] != 'passed' or read(folder / 'daily_run.json')['status'] != 'verified':
                    raise ValueError('Validation daily verification failed')
            checks = [read(features / f'transaction_date={d}/verification.json') for d in DATES]
            if sum(c['full_day_rows'] for c in checks) != TOTAL or sum(c['positive_rows'] for c in checks) != POSITIVES:
                raise ValueError('Validation feature totals do not match split')
            cache = project / 'data/gnn_validation_cache_parquet'
            command([sys.executable, '-m', 'gnn_features.validation_cache'], 'validation_cache', cache / 'status.json')
            results = []
            for spec in [dict(name='baseline', learning_rate=.001, positive_weight=base_config['positive_weight'])] + trial_specs:
                name = spec['name']
                trial = base if name == 'baseline' else out / 'trials' / name
                if name != 'baseline':
                    command([sys.executable, '-m', 'gnn_features.tune_model', '--trial', name],
                            'train_' + name, trial / 'status.json')
                evaluation = out / 'evaluations' / name
                command([sys.executable, '-m', 'gnn_features.evaluate_validation',
                         '--model', trial / 'final_model.pt', '--output', evaluation],
                        'evaluate_' + name, evaluation / 'status.json')
                r = read(evaluation / 'metrics.json')
                if r['status'] != 'completed':
                    raise ValueError('Incomplete Validation evaluation')
                results.append(dict(**spec, model_path=str((trial / 'final_model.pt').relative_to(project)),
                                    model_sha256=r['model_sha256'], metrics=r['metrics'], per_day=r['per_day']))
                atomic_json(out / 'comparison.json', dict(status='partial', results=results))
                update(completed_trials=[r['name'] for r in results])
            best = choose_best(results)
            report = dict(status='completed', results=results, best=best,
                          config_sha256=sha(out / 'config.json'), completed_at=datetime.now(timezone.utc).isoformat(),
                          scope='Validation-selected model and threshold; independent Test untouched')
            atomic_json(out / 'comparison.json', report)
            atomic_json(out / 'best_model.json', dict(name=best['name'], path=best['model_path'],
                                                    sha256=best['model_sha256'],
                                                    validation_threshold=best['metrics']['best_f1']['threshold']))
            lines = ['# GNN 第一轮调参结果', '',
                     'Train：2022-09-01 至 10；Validation：2022-09-11 至 13，全量4,689,006笔、正例6,420笔。',
                     '两组新候选均从同一种子重新训练完整Train一轮，与已有初版共用图采样和验证输入。', '',
                     '| 版本 | 学习率 | 正例权重 | 验证AP | 验证ROC-AUC | 阈值0.5 F1 |',
                     '| --- | ---: | ---: | ---: | ---: | ---: |']
            for item in results:
                m = item['metrics']
                lines.append(f'| {item["name"]} | {item["learning_rate"]} | {item["positive_weight"]:.6f} | '
                             f'{m["average_precision"]:.6f} | {m["roc_auc"]:.6f} | {m["threshold_0_5"]["f1"]:.6f} |')
            lines += ['', '选择：' + best['name'] + '（按验证集AP，ROC-AUC仅用于同分）。',
                      '', '阈值优化和模型选择均使用验证集，结果不代表独立测试性能；尚未使用Test。',
                      '各日期指标、固定误报率召回、精确率/召回率和选定阈值详见 comparison.json。',
                      '不同正例权重的训练BCE不能直接横向排名。本轮未额外增加epoch或搜索未声明配置。']
            (out / '调参结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            update(status='completed', stage='complete', best_trial=best['name'], child_pid=None)
            return report
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=ROOT)
    run(parser.parse_args().project)
