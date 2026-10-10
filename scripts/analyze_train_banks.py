"""Descriptive bank/payment analysis of the authorized training union only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/train_bank_analysis'


def add_rates(f):
    f=f.copy()
    f['positive_rate']=f.positive/f.total
    f['share_all']=f.total/f.total.sum()
    f['share_positive']=f.positive/f.positive.sum()
    f['lift']=f.positive_rate/(f.positive.sum()/f.total.sum())
    return f


def main():
    cfg=json.loads((ROOT/'config/dataset_split.json').read_text(encoding='utf-8'))
    source=ROOT/cfg['source']
    assert hashlib.sha256((source/'_manifest.json').read_bytes()).hexdigest()==cfg['source_manifest_sha256']
    splits=[s for s in cfg['splits'] if s['name'] in cfg['train_members']]
    date_split={d:s['name'] for s in splits for d in s['dates']}
    dates=sorted(date_split)
    acc=pd.concat([pq.ParquetFile(p).read(columns=['bank_id','account_number','bank_name']).to_pandas() for p in sorted((ROOT/'data/silver/accounts').glob('*.parquet'))],ignore_index=True)
    keys=pd.Index(acc.bank_id+'|'+acc.account_number)
    assert keys.is_unique
    counts={role:np.zeros(len(acc),dtype=np.int64) for role in ['from','to']}
    positives={role:np.zeros(len(acc),dtype=np.int64) for role in ['from','to']}
    groups={role:[] for role in ['from','to']}
    daily=[]
    cols=['from_bank','from_account','to_bank','to_account','payment_format','is_laundering']
    for date in dates:
        n=k=0
        for path in sorted((source/f'transaction_date={date}').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=200000,columns=cols):
                f=batch.to_pandas()
                y=f.is_laundering.to_numpy()
                assert np.isin(y,[0,1]).all()
                n+=len(f); k+=int(y.sum())
                for role in groups:
                    ix=keys.get_indexer(f[role+'_bank']+'|'+f[role+'_account'])
                    assert (ix>=0).all()
                    np.add.at(counts[role],ix,1)
                    np.add.at(positives[role],ix,y)
                    g=f.groupby([role+'_bank','payment_format']).is_laundering.agg(total='size',positive='sum').reset_index().rename(columns={role+'_bank':'bank'})
                    g['date']=date
                    groups[role].append(g)
        daily.append(dict(date=date,total=n,positive=k,split=date_split[date]))
        print(f'Analyzed {date}: {n} rows, {k} positives',flush=True)
    assert sum(x['total'] for x in daily)==sum(s['rows'] for s in splits)==20318992
    assert sum(x['positive'] for x in daily)==sum(s['positive_rows'] for s in splits)==17794
    OUT.mkdir(parents=True,exist_ok=True)
    method_tables={}
    bank_tables={}
    stats={}
    for role,pieces in groups.items():
        g=pd.concat(pieces,ignore_index=True).groupby(['date','bank','payment_format'],as_index=False)[['total','positive']].sum()
        g['split']=g.date.map(date_split)
        if role=='from':
            for period in ['train','discovery','internal_validation']:
                sub=g if period=='train' else g[g.split==period]
                m=add_rates(sub.groupby('payment_format',as_index=False)[['total','positive']].sum())
                m.to_csv(OUT/f'payment_{period}.csv',index=False,encoding='utf-8-sig')
                method_tables[period]=m.to_dict('records')
            dm=g.groupby(['date','payment_format'],as_index=False)[['total','positive']].sum()
            dm['positive_rate']=dm.positive/dm.total
            dm.to_csv(OUT/'payment_daily.csv',index=False,encoding='utf-8-sig')
        base=g.groupby(['date','payment_format'],as_index=False)[['total','positive']].sum()
        base['method_day_rate']=base.positive/base.total
        g=g.merge(base[['date','payment_format','method_day_rate']],on=['date','payment_format'],validate='many_to_one')
        g['expected_positive']=g.total*g.method_day_rate
        b=add_rates(g.groupby('bank',as_index=False)[['total','positive']].sum())
        expected=g.groupby('bank').expected_positive.sum()
        b['expected_positive_date_method']=b.bank.map(expected)
        b['observed_expected_ratio']=b.positive/b.expected_positive_date_method.replace(0,np.nan)
        a=pd.DataFrame({'bank':acc.bank_id,'active':counts[role]>0,'positive_active':positives[role]>0,'positive_count':positives[role]})
        aa=a.groupby('bank').agg(active_accounts=('active','sum'),positive_accounts=('positive_active','sum'),top_account_positive=('positive_count','max'))
        b=b.merge(aa,on='bank',validate='one_to_one')
        b['positive_account_fraction']=b.positive_accounts/b.active_accounts
        b['top_account_positive_share']=b.top_account_positive/b.positive.replace(0,np.nan)
        names=acc.groupby('bank_id').bank_name.agg(lambda x:' / '.join(sorted(set(x))))
        b['bank_name']=b.bank.map(names)
        for period in ['discovery','internal_validation']:
            sub=add_rates(g[g.split==period].groupby('bank',as_index=False)[['total','positive']].sum())
            ex=g[g.split==period].groupby('bank').expected_positive.sum()
            sub['adjusted_ratio']=sub.positive/sub.bank.map(ex).replace(0,np.nan)
            b=b.merge(sub[['bank','total','positive','lift','adjusted_ratio']].rename(columns={c:period+'_'+c for c in ['total','positive','lift','adjusted_ratio']}),on='bank',how='left',validate='one_to_one')
        bank_tables[role]=b
        b.sort_values('positive',ascending=False).to_csv(OUT/f'banks_{role}.csv',index=False,encoding='utf-8-sig')
        g.to_csv(OUT/f'bank_payment_daily_{role}.csv',index=False,encoding='utf-8-sig')
        p=b.positive.sum()/b.total.sum()
        chi=float((((b.positive-b.total*p)**2)/(b.total*p*(1-p))).sum())
        stats[role]=dict(bank_count=len(b),cramers_v=float(np.sqrt(chi/b.total.sum())),
                         top10_positive_share=float(b.nlargest(10,'positive').positive.sum()/b.positive.sum()),
                         top10_same_banks_total_share=float(b.nlargest(10,'positive').total.sum()/b.total.sum()),
                         robust_high_lift=int(((b.total>=10000)&(b.positive>=20)&(b.lift>=2)).sum()),
                         robust_adjusted_high=int(((b.total>=10000)&(b.positive>=20)&(b.observed_expected_ratio>=2)).sum()))
    report=dict(scope='train union, descriptive analysis; discovery/internal_validation also reported separately; no held-out validation/test read',
                rows=sum(x['total'] for x in daily),positive_rows=sum(x['positive'] for x in daily),daily=daily,methods=method_tables,bank_stats=stats,
                banks={r:{'top_positive':b.nlargest(12,'positive').replace({np.nan:None}).to_dict('records'),
                          'high_lift_supported':b[(b.total>=10000)&(b.positive>=20)].nlargest(12,'lift').replace({np.nan:None}).to_dict('records'),
                          'high_adjusted_supported':b[(b.total>=10000)&(b.positive>=20)].nlargest(12,'observed_expected_ratio').replace({np.nan:None}).to_dict('records')} for r,b in bank_tables.items()})
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'rows':report['rows'],'positive':report['positive_rows'],'methods':method_tables,'bank_stats':stats},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
