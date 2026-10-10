"""Independent Test preparation/evaluation; never launches training."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from gnn_features.training_state import ProcessLock
from .common import *

class Paused(RuntimeError): pass
def run(project=ROOT,resume=False):
    project=Path(project).resolve(); root=project/OUTPUT; root.mkdir(parents=True,exist_ok=True)
    state=dict(status='preparing',pid=os.getpid(),child_pid=None,child_status=None,started_at=now())
    def update(**kw):
        state.update(kw,updated_at=now()); atomic_json(root/'status.json',state)
    def guard():
        pause=root/'pause_request.json'
        if pause.exists() and read(pause).get('enabled'): raise Paused('Explicit user pause')
        if shutil.disk_usage(project).free<20*1024**3: raise RuntimeError('Disk below 20 GiB')
    def command(argv,stage,status):
        guard(); log=root/(stage+'.log')
        update(status='running',stage=stage,child_status=str(status),log=str(log))
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([str(x) for x in argv],cwd=project,stdout=stream,stderr=stream)
            update(child_pid=child.pid)
            code=child.wait()
        update(child_pid=None,last_exit_code=code)
        if code==75: raise Paused('Child paused')
        if code: raise RuntimeError(f'{stage} exited {code}; inspect log tail')
    def py(module,*args): return [sys.executable,'-m','gnn_final_test.'+module,*args]
    with ProcessLock(root):
        try:
            if (root/'pause_request.json').exists() and read(root/'pause_request.json').get('enabled'):
                if not resume: raise Paused('Explicit user pause requires --resume')
                atomic_json(root/'pause_request.json',dict(enabled=False,resumed_at=now()))
            protocol(project)
            engineering=read(root/'engineering_tests.json')
            if engineering['status']!='passed' or engineering['source_sha256']!=sources(project):
                raise ValueError('Engineering acceptance/source binding mismatch')
            # Freeze every currently available model before reading Test inputs.
            for seed in SEEDS:
                for trial in TRIALS: bind(project,trial,seed)
            archive=project/'data/gnn_final_test_archive'; cache=project/'data/gnn_final_test_cache'
            command(py('archive','--project',project,'--output',archive),'archive',archive/'status.json')
            meta=read(archive/'build.json')
            if meta['rows']!=PREFIX+TOTAL or meta['end_exclusive']!=END: raise ValueError('Test archive extent mismatch')
            for date in DATES:
                folder=root/'features'/('transaction_date='+date)
                if not (folder/'verification.json').exists():
                    if not (folder/'daily_run.json').exists():
                        command(['powershell.exe','-NoProfile','-File',project/'scripts/daily_gnn.ps1',
                            '-Archive',archive,'-Output',folder,'-Date',date,'-Cores','2','-Partitions','8','-Resume'],
                            'features_'+date,folder/'daily_run.json')
                    command(py('verify_features','--archive',archive,'--output',folder),
                        'verify_'+date,folder/'daily_run.json')
                if read(folder/'verification.json')['status']!='passed' or read(folder/'daily_run.json')['status']!='verified':
                    raise ValueError('Daily features not verified')
            command(py('cache','--project',project),'cache',cache/'status.json')
            extra=project/'data/gnn_final_test_A_features/test'
            command(py('a_features','--project',project,'--split','test'),'A_features',extra/'status.json')
            completed=set()
            while len(completed)<8:
                guard(); protocol(project)
                for seed in SEEDS:
                    for trial in TRIALS:
                        if (seed,trial) in completed: continue
                        trained=bind(project,trial,seed)
                        if trained is None: continue
                        del trained
                        out=eval_root(project,trial,seed)
                        command(py('evaluate','--project',project,'--trial',trial,'--seed',seed),
                            f'evaluate_{seed}_{trial}',out/'status.json')
                        if read(out/'metrics.json')['status']!='completed': raise ValueError('Test evaluation incomplete')
                        completed.add((seed,trial))
                        update(completed_runs=[dict(seed=s,trial=t) for s,t in sorted(completed)])
                if len(completed)<8:
                    update(status='waiting',stage='waiting_for_predeclared_models',child_pid=None,child_status=None)
                    time.sleep(60)  # This pipeline waits for its inputs; it never starts/restarts training.
            command(py('compare','--project',project),'compare',root/'comparison_status.json')
            update(status='completed',stage='complete',completed_at=now(),child_status=None)
        except Paused as exc:
            update(status='paused',stage='paused',error=str(exc)); return 75
        except BaseException as exc:
            update(status='failed',error=repr(exc),traceback=traceback.format_exc()); raise
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--resume',action='store_true')
    a=p.parse_args();raise SystemExit(run(a.project,a.resume))
