"""Package verified aggregate data as a single offline HTML artifact."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'result/dashboard/data.json').read_text(encoding='utf-8'))
    assert data['meta']['status']=='verified' and len(data['windows'])==28
    for name,w in data['windows'].items():
        start,end=map(int,name.split('-'))
        days=data['daily'][start:end+1]
        total=sum(d['total'] for d in days)
        positive=sum(d['positive'] for d in days)
        for field in ['frequency','outgoing','incoming','active_days','entity_registered','entity_active','entity_banks','entity_account_weighted']:
            value=w[field]
            assert all(sum(h)==(s['n'] if s else 0) for h,s in zip(value['counts'],value['summary']))
            expected=w['active_accounts'] if field in ['frequency','outgoing','incoming','active_days','entity_account_weighted'] else w['active_entities']
            assert sum(s['n'] if s else 0 for s in value['summary'])==expected
        for side in ['amount_paid','amount_received']:
            values=[v for k,v in w['amounts'].items() if k.startswith(side+'|')]
            assert sum(sum(v['n']) for v in values)==total
            assert sum(v['n'][1] for v in values)==positive
            assert all([sum(h) for h in v['hist']]==v['n'] for v in values)
        for records in data['categories'].values():
            rows=[r for r in records if start<=r['day']<=end]
            assert sum(r['total'] for r in rows)==total and sum(r['positive'] for r in rows)==positive
    raw=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    html=(ROOT/'dashboard/template.html').read_text(encoding='utf-8').replace('__DATA__',raw)
    path=ROOT/'dashboard/dist/index.html'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(html,encoding='utf-8')
    (ROOT/'dashboard/发现期数据分析看板.html').write_text(html,encoding='utf-8')
    print(f'Verified all 28 windows; wrote {path.stat().st_size:,} bytes')


if __name__=='__main__': main()
