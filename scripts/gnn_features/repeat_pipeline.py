"""Six sequential full-data seed repetitions; never resumes old experiments."""
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
from .training_state import ProcessLock
from .repeat_common import (ROOT, OUTPUT, SEEDS, TRIALS, read, atomic_json, sha,
    source_hashes, verify_plan, paused, trial_root, eval_root)

def now(): return datetime.now(timezone.utc).isoformat()

def make_plan(project):
    from .v4_pipeline import verify_freeze, source_hashes as v4_sources
    anchors = verify_freeze(project)
    v4 = project/'result/gnn_tuning_v4'
    if read(v4/'status.json')['status'] != 'completed': raise ValueError('v4 must be complete')
    if read(v4/'config.json')['source_sha256'] != v4_sources(project): raise ValueError('Frozen v4 source changed')
    for name in ('config.json', 'comparison.json', 'best_model.json', 'delivery_verification.json',
                 'trials/structure_three/final_model.pt', 'trials/structure_three/training_result.json'):
        p = v4/name; anchors[str(p.relative_to(project))] = sha(p)
    return dict(experiment='paired_original_recipe_seed_repeats', seeds=list(SEEDS), trials=list(TRIALS),
        existing_seed=20260920, new_trainings=6, train_targets=20318992, train_positives=17794,
        validation_targets=4689006, validation_positives=6420, epochs=1,
        ordering='seed ascending, lr_only then features_only; sequential CUDA',
        primary_summary='three fresh paired seeds; mean, sample SD, range and paired differences',
        secondary_summary='all four seeds including development seed20260920; reported separately',
        ranking=['mean_maximum_F1', 'mean_AP', 'mean_P80_recall'],
        selection='Recipe-level summary only; do not cherry-pick a winning seed/checkpoint',
        initialization='Each original recipe recreated with the declared seed; no warm start or new features',
        graph_sampling='Frozen cached graph/batch order; seed variability is initialization and training RNG only',
        research_only=True, deployment_replaced=False, test_evaluated=False,
        source_sha256=source_hashes(project), frozen_anchors=anchors)

def handoff(project, status):
    path = project/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
    previous = path.read_text(encoding='utf-8')
    start, end = '<!-- GNN_REPEATS_BEGIN -->', '<!-- GNN_REPEATS_END -->'
    section = '\n'.join([start, '', '## lr_only与A的三组种子重复实验', '', status,
        '- 各补跑3次：20261009、20261010、20261011；连同20260920各4次。',
        '- 各按原初始化和训练配方，完整Train一次1epoch，再完整Validation；不增加特征或重建图缓存。',
        '- 主要结论基于3个新种子的配对均值、样本标准差、范围和胜负；含旧种子的4次汇总另列。',
        '- 不挑最好种子；这是固定图/批次下初始化与训练随机性的重复，不含图采样随机性。',
        '- Test继续封存，旧模型不替换；不凭3次重复宣称统计显著。',
        '- 入口 scripts/start_gnn_seed_repeats.ps1；结果 result/gnn_seed_repeats；每小时检查。', '', end])
    if start in previous or end in previous:
        if previous.count(start) != 1 or previous.count(end) != 1: raise ValueError('Ambiguous handoff markers')
        text = previous[:previous.index(start)]+section+previous[previous.index(end)+len(end):]
    else: text = previous.rstrip()+'\n\n'+section+'\n'
    if path.read_text(encoding='utf-8') != previous: raise RuntimeError('Concurrent handoff edit')
    temp = path.with_suffix('.repeat.tmp'); temp.write_text(text, encoding='utf-8'); os.replace(temp, path)

class Paused(Exception): pass

def run(project=ROOT, resume=False, accept_only=False):
    project = Path(project).resolve(); root = project/OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', stage='preflight', pid=os.getpid(), child_pid=None,
        child_status=None, started_at=now(), completed_runs=[], new_trainings=6)
    def update(**kw):
        state.update(kw, updated_at=now()); atomic_json(root/'status.json', state)
        print(json.dumps(state, ensure_ascii=True), flush=True)
    def check_pause():
        if paused(project): raise Paused('Explicit repeat pause enabled')
    def command(module, trial, seed, mode=None):
        check_pause()
        if shutil.disk_usage(project).free < 20*1024**3: raise RuntimeError('Disk below 20 GiB')
        verify_plan(project)
        stage = f'{mode or "evaluate"}_{seed}_{trial}'
        folder = trial_root(project,trial,seed) if mode else eval_root(project,trial,seed)
        log = root/(stage+'.log')
        argv = [sys.executable,'-m','gnn_features.'+module,'--project',str(project),
                '--trial',trial,'--seed',str(seed)]+(['--mode',mode] if mode else [])
        update(status='running', stage=stage, child_status=str(folder/'status.json'), log=str(log))
        with log.open('a',encoding='utf-8') as f:
            f.write('\nRUN '+now()+'\n'); f.flush()
            child = subprocess.Popen(argv,cwd=project,env={**os.environ,'PYTHONPATH':str(project/'scripts'),
                'PYTHONIOENCODING':'utf-8'},stdout=f,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            update(child_pid=child.pid); code = child.wait()
        update(child_pid=None,last_exit_code=code)
        if code == 75: raise Paused('Child paused at checkpoint')
        if code: raise RuntimeError(f'{stage} failed ({code}); inspect {log}')
    with ProcessLock(root):
        try:
            if resume and paused(project):
                request = read(root/'pause_request.json')
                atomic_json(root/('pause_history_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'),request)
                request.update(enabled=False,resumed_at=now()); atomic_json(root/'pause_request.json',request)
            update(); check_pause()
            config = make_plan(project)
            tests = read(root/'engineering_tests.json')
            if tests.get('status') != 'passed' or tests.get('source_sha256') != config['source_sha256']:
                raise ValueError('Passed source-bound repeat tests required')
            if (root/'config.json').exists():
                if read(root/'config.json') != config: raise ValueError('Repeat plan changed; preserve and review')
            else: atomic_json(root/'config.json',config)
            handoff(project,'已启动：按固定种子依次验收、训练及验证，尚无重复实验完整结论。')
            # Acceptance precedes each run, uses real frozen Train blocks with both positive groups.
            for seed in SEEDS:
                for trial in TRIALS:
                    command('repeat_train',trial,seed,'accept' if accept_only else 'train')
                    if not accept_only:
                        command('repeat_evaluate',trial,seed)
                        state['completed_runs'].append(dict(seed=seed,trial=trial))
                        update()
            if accept_only:
                update(status='ready',stage='six_acceptances_passed',child_status=None); return 0
            check_pause(); update(stage='comparison',child_status=None)
            from .repeat_compare import run as compare
            report = compare(project)
            verify_plan(project)
            handoff(project,'全部6次训练与完整Validation评估及预测重放核验完成。详见 result/gnn_seed_repeats/重复实验结果.md。')
            update(status='completed',stage='complete',summary=str(root/'comparison.json'),
                   research_preferred_recipe=report['primary_summary']['preferred_recipe'])
            return 0
        except Paused as e:
            update(status='paused',reason=str(e),child_pid=None); return 75
        except BaseException as e:
            update(status='failed',error=repr(e),traceback=traceback.format_exc()); raise

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--resume',action='store_true'); p.add_argument('--accept-only',action='store_true')
    a=p.parse_args(); sys.exit(run(a.project,a.resume,a.accept_only))
