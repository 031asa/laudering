"""Independent silver-level checks for the three delivered real examples."""
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .build import write_json

def verify(project,archive,out):
    start=time.perf_counter();samples=json.loads((out/'samples.json').read_text(encoding='utf-8'))
    accounts=pq.ParquetFile(archive/'graph/accounts/accounts.parquet').read()
    wanted={s[k] for s in samples.values() for k in ['src','dst']}
    small=accounts.filter(pc.is_in(accounts['node_id'],value_set=pa.array(sorted(wanted),type=pa.uint32())))
    nodekeys={r['node_id']:(r['bank_id'],r['account_number']) for r in small.to_pylist()}
    del accounts
    history={key:[] for key in nodekeys.values()};targets={s['transaction_id']:None for s in samples.values()}
    last_date=str(np.datetime64(max(s['cutoff_us'] for s in samples.values()),'us'))[:10]
    # Read the original silver columns, independently of generated edge IDs and indexes.
    for f in sorted((project/'data/silver/transactions').glob('transaction_date=*/*.parquet')):
        if f.parent.name.split('=',1)[1]>last_date:continue
        for b in pq.ParquetFile(f).iter_batches(batch_size=100000):
            tab=pa.Table.from_batches([b]); selected=None
            for bank,account in history:
                mask=pc.or_(pc.and_(pc.equal(tab['from_bank'],bank),pc.equal(tab['from_account'],account)),
                            pc.and_(pc.equal(tab['to_bank'],bank),pc.equal(tab['to_account'],account)))
                selected=mask if selected is None else pc.or_(selected,mask)
            for row in tab.filter(selected).to_pylist():
                src=(row['from_bank'],row['from_account']);dst=(row['to_bank'],row['to_account'])
                if row['transaction_id'] in targets:targets[row['transaction_id']]=row
                for key in {src,dst}:
                    if key in history:history[key].append(row)
    checks=0
    def eq(actual,expected):
        nonlocal checks
        if expected is None:assert actual is None,(actual,expected)
        elif isinstance(expected,float):assert math.isclose(float(actual),expected,rel_tol=1e-12,abs_tol=1e-12),(actual,expected)
        elif isinstance(expected,Decimal):assert Decimal(str(actual))==expected,(actual,expected)
        else:assert actual==expected,(actual,expected)
        checks+=1
    def timestamp(row):return int(np.datetime64(row['timestamp'],'us').astype(np.int64))
    for name,s in samples.items():
        current=targets[s['transaction_id']];assert current is not None
        t=s['cutoff_us'];day=t-t%86_400_000_000
        s['source_account']={'bank_id':nodekeys[s['src']][0],'account_number':nodekeys[s['src']][1]}
        s['destination_account']={'bank_id':nodekeys[s['dst']][0],'account_number':nodekeys[s['dst']][1]}
        eq(s['supervision']['label'],current['is_laundering'])
        for side in ['src','dst']:
            key=nodekeys[s[side]]
            for d in ['in','out']:
                same=lambda row: (row['to_bank'],row['to_account'])==key if d=='in' else (row['from_bank'],row['from_account'])==key
                prior=[row for row in history[key] if same(row) and timestamp(row)<t]
                ext=lambda row:(row['from_bank'],row['from_account'])!=(row['to_bank'],row['to_account'])
                cp=lambda row:(row['from_bank'],row['from_account']) if d=='in' else (row['to_bank'],row['to_account'])
                n=s['N_'+side];p='n_'+d+'_'
                for w in [1,24,72]:eq(n[p+f'all_count_{w}h'],len([r for r in prior if timestamp(r)>=t-w*3_600_000_000]))
                recent=[r for r in prior if timestamp(r)>=t-86_400_000_000];external=[r for r in recent if ext(r)]
                cpcounts=Counter(cp(r) for r in external);bankcounts=Counter(cp(r)[0] for r in external)
                eq(n[p+'ext_count_24h'],len(external));eq(n[p+'unique_cp_24h'],len(cpcounts));eq(n[p+'bank_unique_24h'],len(bankcounts))
                eq(n[p+'cp_hhi_24h'],sum(v*v for v in cpcounts.values())/len(external)**2 if external else None)
                eq(n[p+'bank_hhi_24h'],sum(v*v for v in bankcounts.values())/len(external)**2 if external else None)
                eq(n[p+'ach_count_24h'],sum(r['payment_format']=='ACH' for r in recent))
                eq(n[p+'last_all_gap_min'],(t-max(timestamp(r) for r in prior))/60_000_000 if prior else None)
                if s['src']==s['dst']:continue
                curr=current['payment_currency'] if side=='src' else current['receiving_currency']
                amount_field='amount_received' if d=='in' else 'amount_paid'
                currency_field='receiving_currency' if d=='in' else 'payment_currency'
                matching=[r for r in external if r[currency_field]==curr]
                with localcontext() as ctx:
                    ctx.prec=50
                    total=sum((r[amount_field] for r in matching),Decimal(0))
                eq(s['R'][f'r_{side}_ext_{d}_sum_24h'],total)
                eq(s['R'][f'r_{side}_ext_{d}_count_24h'],len(matching))
            if side=='src' and s['src']!=s['dst']:
                prior=[r for r in history[key] if (r['to_bank'],r['to_account'])==key and (r['from_bank'],r['from_account'])!=key and timestamp(r)<t and r['receiving_currency']==current['payment_currency']]
                for w,lower in [('1h',t-3_600_000_000),('6h',t-6*3_600_000_000),('24h',t-24*3_600_000_000),('72h',t-72*3_600_000_000),('day',day)]:
                    subset=[r for r in prior if timestamp(r)>=lower]
                    with localcontext() as ctx:
                        ctx.prec=50;total=sum((r['amount_received'] for r in subset),Decimal(0))
                        expected=float(current['amount_paid']/total) if total>0 else None
                    eq(s['R'][f'r_src_ext_in_sum_{w}'],total);eq(s['R'][f'r_src_ext_in_count_{w}'],len(subset));eq(s['R'][f'r_src_paid_to_ext_in_sum_{w}'],expected)
    result=dict(status='passed',checks=checks,method='Independent reread of silver; Python row predicates and Decimal sums; no generated index used',seconds=time.perf_counter()-start)
    write_json(out/'samples.json',samples)
    report=json.loads((out/'verification.json').read_text(encoding='utf-8'));report['independent_silver_verification']=result
    write_json(out/'verification.json',report)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path.cwd());p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    verify(a.project,a.archive,a.output)
