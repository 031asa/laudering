"""Bind engineering acceptance to exact v4 sources and a saved pytest run."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,subprocess
import xml.etree.ElementTree as ET
from gnn_features.v4_pipeline import source_hashes
from gnn_features.training_state import atomic_json
from gnn_features.cache_train import sha

root=Path(__file__).resolve().parents[1]
out=root/'result/gnn_tuning_v4'
run=out/'engineering_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
run.mkdir(parents=True,exist_ok=False)
files=sorted((root/'tests').glob('test_gnn_v4_*.py'))+[root/'tests/test_gnn_causal_structures.py']
before=source_hashes(root)
argv=[sys.executable,'-m','pytest',*[str(p) for p in files],'-q','--tb=short','--disable-warnings',
      '--basetemp',str(run/'temp'),'--junitxml',str(run/'junit.xml')]
with (run/'pytest.log').open('w',encoding='utf-8') as log:
    completed=subprocess.run(argv,cwd=root,env={**os.environ,'PYTHONPATH':str(root/'scripts'),
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT)
after=source_hashes(root)
doc=ET.parse(run/'junit.xml').getroot()
cases=list(doc.iter('testcase'))
passed=completed.returncode==0 and before==after and all(not any(c.tag in ('failure','error','skipped') for c in t) for t in cases)
record=dict(status='passed' if passed else 'failed',exit_code=completed.returncode,tests=len(cases),
    source_sha256=before,test_sha256={str(p.relative_to(root)):sha(p) for p in files},
    junit_sha256=sha(run/'junit.xml'),log_sha256=sha(run/'pytest.log'),run=str(run.relative_to(root)),
    completed_at=datetime.now(timezone.utc).isoformat())
atomic_json(run/'result.json',record)
if passed:atomic_json(out/'engineering_tests.json',record)
print(json.dumps(record,ensure_ascii=True))
print((run/'pytest.log').read_text(encoding='utf-8',errors='replace')[-6000:])
sys.exit(0 if passed else 1)
