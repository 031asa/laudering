"""Freeze and verify complete-day dataset splits without running EDA."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANGES = [('discovery','2022-09-01','2022-09-07'),
          ('internal_validation','2022-09-08','2022-09-10'),
          ('validation','2022-09-11','2022-09-13'),
          ('test','2022-09-14','2022-09-16'),
          ('tail','2022-09-17','2022-09-28')]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n',encoding='utf-8')


def freeze():
    source = ROOT/'data/silver/transactions'
    manifest_path = source/'_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status']=='verified' and manifest['stage']=='standardized'
    entries = []
    for name,start,end in RANGES:
        dates = pd.date_range(start,end).strftime('%Y-%m-%d').tolist()
        daily = [r for r in manifest['daily_fingerprints'] if str(r['transaction_date']) in dates]
        entries.append(dict(name=name,start=start,end=end,dates=dates,
                            rows=sum(r['rows'] for r in daily),
                            positive_rows=sum(r['rows'] for r in daily if r['is_laundering']==1)))
    assigned = [d for e in entries for d in e['dates']]
    actual = sorted(p.name.split('=',1)[1] for p in source.iterdir() if p.is_dir())
    assert len(set(assigned))==len(assigned)==28 and sorted(assigned)==actual
    assert sum(e['rows'] for e in entries)==manifest['output_rows']
    split = dict(version=1,source='data/silver/transactions',
                 source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 rule='Inclusive complete calendar days, no randomization; disjoint leaf splits',
                 train_members=['discovery','internal_validation'],splits=entries,
                 history_rule='Strictly earlier timestamps only; no validation/test labels; same-minute peers excluded',
                 test_exposure='Previously inspected in global data-quality and label-count audits; no feature discovery or model selection',
                 tail_rule='Retained separately; excluded from primary train/validation/test metrics')
    path = ROOT/'config/dataset_split.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==split, 'Frozen split or source changed'
    else:
        dump(path,split)
    pd.DataFrame([dict(date=d,split=e['name'],train=e['name'] in split['train_members'],
                       partition=f'data/silver/transactions/transaction_date={d}')
                  for e in entries for d in e['dates']]).to_csv(ROOT/'config/date_assignments.csv',index=False)
    return source,split



if __name__ == "__main__":
    source, split = freeze()
    print(json.dumps(dict(status="verified", dates=28, splits=split["splits"]), indent=2))
