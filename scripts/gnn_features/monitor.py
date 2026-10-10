"""Launch a local job invisibly and sample its process tree, including the JVM."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import psutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    if args.output.exists():raise ValueError('Resource report already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();stop=threading.Event();stats=dict(samples=0,peak_tree_rss_bytes=0,peak_tree_private_bytes=0,min_system_available_bytes=psutil.virtual_memory().available,process_names=[])
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def sample():
        names=set();last=began
        while not stop.is_set():
            try:processes=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
            except psutil.Error:processes=[]
            rss=private=0
            for child in processes:
                try:
                    m=child.memory_info();rss+=m.rss;private+=getattr(m,'private',m.vms);names.add(child.name())
                except psutil.Error:pass
            stats['samples']+=1;stats['peak_tree_rss_bytes']=max(stats['peak_tree_rss_bytes'],rss);stats['peak_tree_private_bytes']=max(stats['peak_tree_private_bytes'],private)
            stats['min_system_available_bytes']=min(stats['min_system_available_bytes'],psutil.virtual_memory().available)
            stats['process_names']=sorted(names)
            if time.perf_counter()-last>=30:
                print(json.dumps(dict(stage='resource_monitor',elapsed_seconds=round(time.perf_counter()-began),tree_rss_mb=round(rss/1e6),tree_private_mb=round(private/1e6))),flush=True);last=time.perf_counter()
            stop.wait(1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    with args.output.with_suffix('.log').open('w',encoding='utf-8') as log:
        for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
    code=proc.wait();stop.set();thread.join(timeout=3)
    stats.update(exit_code=code,wall_seconds=time.perf_counter()-began,
        rss_note='Sum of process working sets; shared mapped pages may be counted more than once',private_note='Sum of private committed bytes, not resident physical memory')
    args.output.write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
