"""Unattended Train-only feature run: daily checkpoints, verification, fail-stop."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/gnn_train_v1'
ARCHIVE=ROOT/'data/gnn_v1_train'
PWSH=Path('C:/Users/28118/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def training_spec(split):
    members=[s for s in split['splits'] if s['name'] in split['train_members']]
    dates=sorted(d for s in members for d in s['dates'])
    if len(dates)!=len(set(dates)) or not dates:raise ValueError('Invalid training dates')
    from datetime import date,timedelta
    start=date.fromisoformat(dates[0]);end=date.fromisoformat(dates[-1])+timedelta(days=1)
    if [(start+timedelta(days=i)).isoformat() for i in range((end-start).days)]!=dates:raise ValueError('Train dates must be consecutive')
    return dates,end.isoformat()+'T00:00:00',sum(s['rows'] for s in members),sum(s['positive_rows'] for s in members)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'process.lock'
    if lock.exists():
        previous=read(lock)
        try:
            p=psutil.Process(previous['pid'])
            if abs(p.create_time()-previous['create_time'])<1:raise RuntimeError('Training feature supervisor is already running')
        except psutil.NoSuchProcess:pass
        lock.unlink()
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),create_time=psutil.Process().create_time()),f)
    began=time.perf_counter()
    state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),stage='preflight',completed_dates=[])

    def update(**fields):
        state.update(fields);state['updated_at']=datetime.now(timezone.utc).isoformat()
        state['invocation_seconds']=time.perf_counter()-began
        save(OUT/'status.json',state)

    def command(args,log):
        update(log=str(log),command=[str(a) for a in args])
        with log.open('a',encoding='utf-8') as f:
            f.write('\nRUN '+datetime.now(timezone.utc).isoformat()+'\n');f.flush()
            child=subprocess.Popen([str(a) for a in args],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'scripts'),'PYTHONIOENCODING':'utf-8'},stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            update(child_pid=child.pid)
            code=child.wait()
        update(child_pid=None,last_exit_code=code)
        if code:raise RuntimeError(f'Command failed with exit {code}; inspect {log}')

    try:
        update()
        split=read(ROOT/'config/dataset_split.json');dates,end,expected,positive=training_spec(split)
        config=dict(dates=dates,end_exclusive=end,expected_rows=expected,expected_positive_rows=positive,cores=4,partitions=8,
            split_sha256=hashlib.sha256((ROOT/'config/dataset_split.json').read_bytes()).hexdigest(),
            code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['build.py','daily.py','engine.py','rolling.py','partitioned.py','contract.py','verify_daily.py']})
        if (OUT/'config.json').exists():
            if read(OUT/'config.json')!=config:raise ValueError('Run configuration/code changed; review before resuming')
        else:save(OUT/'config.json',config)
        if shutil.disk_usage(ROOT).free<30_000_000_000:raise RuntimeError('Less than 30GB free disk space')
        if not PWSH.exists():raise FileNotFoundError(PWSH)
        update(stage='build_history',expected_rows=expected,dates=dates)
        command([sys.executable,'-m','gnn_features.build','--output',ARCHIVE,'--end',end],OUT/'build.log')
        meta=read(ARCHIVE/'build.json')
        if meta['rows']!=expected or meta['end_exclusive']!=end or meta['split_sha256']!=config['split_sha256']:raise AssertionError('Archive does not match frozen Train')
        verified=[]
        for day in dates:
            folder=OUT/f'transaction_date={day}'
            update(current_date=day)
            if not (folder/'verification.json').exists():
                if not (folder/'daily_run.json').exists():
                    update(stage='compute_day')
                    command([PWSH,'-NoProfile','-File',ROOT/'scripts/daily_gnn.ps1','-Archive',ARCHIVE,'-Output',folder,'-Date',day,'-Cores','4','-Partitions','8','-Resume'],OUT/f'{day}_compute.log')
                update(stage='verify_day')
                command([sys.executable,'-m','gnn_features.verify_daily','--archive',ARCHIVE,'--output',folder],OUT/f'{day}_verify.log')
            acceptance=read(folder/'verification.json');daily=read(folder/'daily_run.json')
            if acceptance['status']!='passed' or daily['status']!='verified' or daily['date']!=day:raise AssertionError('Daily verification failed')
            verified.append((day,daily,acceptance))
            update(completed_dates=[v[0] for v in verified],verified_rows=sum(v[2]['full_day_rows'] for v in verified))
        if sum(v[2]['full_day_rows'] for v in verified)!=expected or sum(v[2]['positive_rows'] for v in verified)!=positive:raise AssertionError('Train totals differ from frozen split')
        compute=sum(v[1]['invocation_seconds'] for v in verified);verify=sum(v[2]['verification_seconds'] for v in verified)
        report=['# 全训练集GNN特征运行结果','',f'状态：已完成并核验。训练范围{dates[0]}至{dates[-1]}，共{expected:,}笔交易。',
            '',f'历史索引构建：{meta["wall_seconds"]:.2f}秒；逐日计算落盘累计：{compute:.2f}秒；核验累计：{verify:.2f}秒。',
            '', '各日期全量检查交易身份、时间戳、唯一性和审计覆盖，并按24小时分层抽样与原查询引擎比对。9月4日额外对照原一小时全部N/R结果；未用慢版全量重算所有数值。',
            '', '每个transaction_date目录下features为54列R，audit为缺失与比率依据，merge_phase.json列出成功文件。E及历史索引位于data/gnn_v1_train；N仍按账户和截止时间查询，中间端点N仅供核验。',
            '', '不包含GNN训练、邻居采样全量验收、验证集调参或测试集处理。','', '| 日期 | 交易数 | 计算落盘秒 | 核验秒 |','| --- | ---: | ---: | ---: |']
        report.extend(f'| {d} | {a["full_day_rows"]:,} | {r["invocation_seconds"]:.2f} | {a["verification_seconds"]:.2f} |' for d,r,a in verified)
        (OUT/'全训练集运行结果.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        update(status='completed',stage='complete',current_date=None,compute_seconds=compute,verification_seconds=verify,build_seconds=meta['wall_seconds'],report=str(OUT/'全训练集运行结果.md'))
    except BaseException as exc:
        update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':run()
