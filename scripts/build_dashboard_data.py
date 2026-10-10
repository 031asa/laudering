"""Full discovery-only counts with window-specific account groups; no model fitting."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp, rankdata

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/dashboard'
EDGES=np.arange(-8,16.0001,.25)
CATS=['payment_currency','receiving_currency','payment_format','hour','cross_bank','cross_currency','self_transfer']


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')


def grouped_distribution(values,risk):
    hist=[]
    qs=[]
    for mask in [~risk,risk]:
        v=values[mask]
        h=np.histogram(v,bins=[0,2,3,6,11,21,51,101,501,1001,np.inf])[0]
        hist.append(h.tolist())
        qs.append(None if not len(v) else dict(n=len(v),median=float(np.median(v)),p90=float(np.quantile(v,.9)),p99=float(np.quantile(v,.99))))
    allv,inv=np.unique(values,return_inverse=True)
    counts=[np.bincount(inv[~risk],minlength=len(allv)),np.bincount(inv[risk],minlength=len(allv))]
    n0,n1=[int(c.sum()) for c in counts]
    if n0 and n1:
        ks=float(np.max(np.abs(np.cumsum(counts[0])/n0-np.cumsum(counts[1])/n1)))
        auc=float(np.sum(counts[1]*(np.cumsum(counts[0])-.5*counts[0]))/(n0*n1))
    else: ks=auc=None
    return dict(bins=['1','2','3–5','6–10','11–20','21–50','51–100','101–500','501–1000','>1000'],counts=hist,summary=qs,ks=ks,rank_effect=None if auc is None else 2*auc-1)


def main():
    started=time.perf_counter()
    split=json.loads((ROOT/'config/dataset_split.json').read_text(encoding='utf-8'))
    source=ROOT/split['source']
    assert hashlib.sha256((source/'_manifest.json').read_bytes()).hexdigest()==split['source_manifest_sha256']
    discovery=next(s for s in split['splits'] if s['name']=='discovery')
    dates=discovery['dates']
    assert dates==pd.date_range('2022-09-01','2022-09-07').strftime('%Y-%m-%d').tolist()
    OUT.mkdir(parents=True,exist_ok=True)
    account_frames=[pq.ParquetFile(p).read(columns=['bank_id','account_number','entity_id']).to_pandas() for p in sorted((ROOT/'data/silver/accounts').glob('*.parquet'))]
    accounts=pd.concat(account_frames,ignore_index=True)
    del account_frames
    assert accounts.notna().all().all()
    key=pd.Index(accounts.bank_id+'|'+accounts.account_number)
    assert key.is_unique
    entity,_=pd.factorize(accounts.entity_id,sort=False)
    entity_size=np.bincount(entity)
    entity_banks=accounts.assign(e=entity).groupby('e').bank_id.nunique().reindex(range(len(entity_size))).to_numpy()
    n=len(accounts)
    del accounts
    counts=np.zeros((7,n),dtype=np.uint32)
    paid=np.zeros((7,n),dtype=np.uint32)
    received=np.zeros((7,n),dtype=np.uint32)
    risks=np.zeros((7,n),dtype=bool)
    catpieces={c:[] for c in CATS}
    daily=[]; sample=[]; hist={}; files=[]
    columns=['timestamp','from_bank','from_account','to_bank','to_account','amount_paid','amount_received','payment_currency','receiving_currency','payment_format','is_laundering','source_record_number']
    for d,date in enumerate(dates):
        total=pos=self_count=0
        for path in sorted((source/f'transaction_date={date}').glob('*.parquet')):
            files.append(str(path.relative_to(ROOT)))
            for b in pq.ParquetFile(path).iter_batches(batch_size=100000,columns=columns):
                f=b.to_pandas()
                assert f.timestamp.dt.strftime('%Y-%m-%d').eq(date).all()
                y=f.is_laundering.to_numpy()
                assert np.isin(y,[0,1]).all()
                src=key.get_indexer(f.from_bank+'|'+f.from_account)
                dst=key.get_indexer(f.to_bank+'|'+f.to_account)
                assert (src>=0).all() and (dst>=0).all()
                different=src!=dst
                np.add.at(paid[d],src,1); np.add.at(received[d],dst,1)
                np.add.at(counts[d],src,1); np.add.at(counts[d],dst[different],1)
                risks[d,src[y==1]]=True; risks[d,dst[y==1]]=True
                total+=len(f); pos+=int(y.sum()); self_count+=int((~different).sum())
                f['hour']=f.timestamp.dt.hour
                f['cross_bank']=f.from_bank.ne(f.to_bank)
                f['cross_currency']=f.payment_currency.ne(f.receiving_currency)
                f['self_transfer']=~different
                for c in CATS:
                    g=f.groupby(c).is_laundering.agg(total='size',positive='sum').reset_index()
                    g[c]=g[c].astype(str); g['day']=d
                    catpieces[c].append(g)
                hashes=pd.util.hash_pandas_object(f.source_record_number,index=False).to_numpy()
                keep=(y==1)|(hashes%100==0)
                for amount,currency in [('amount_paid','payment_currency'),('amount_received','receiving_currency')]:
                    f[amount]=f[amount].astype(float)
                    assert np.isfinite(f[amount]).all() and (f[amount]>0).all()
                    for (cur,label),v in f.groupby([currency,'is_laundering'])[amount]:
                        logs=np.log10(v.to_numpy())
                        assert (logs>=EDGES[0]).all() and (logs<=EDGES[-1]).all()
                        h=np.histogram(logs,bins=EDGES)[0]
                        k=(d,amount,cur,int(label))
                        hist[k]=hist.get(k,np.zeros(len(EDGES)-1,dtype=np.int64))+h
                part=f.loc[keep,['is_laundering','amount_paid','amount_received','payment_currency','receiving_currency']].copy()
                part['day']=d; sample.append(part)
        assert int(paid[d].sum())==int(received[d].sum())==total
        assert int(counts[d].sum())==2*total-self_count
        daily.append(dict(date=date,total=total,positive=pos,active_accounts=int((counts[d]>0).sum()),risk_accounts=int(risks[d].sum())))
        print(f'Aggregated {date}: {total:,} transactions; {pos:,} positives',flush=True)
    assert sum(r['total'] for r in daily)==discovery['rows']
    assert sum(r['positive'] for r in daily)==discovery['positive_rows']
    del key
    sample=pd.concat(sample,ignore_index=True)
    cats={}
    for c in CATS:
        g=pd.concat(catpieces[c],ignore_index=True).groupby(['day',c],as_index=False)[['total','positive']].sum()
        assert int(g.total.sum())==discovery['rows'] and int(g.positive.sum())==discovery['positive_rows']
        cats[c]=[dict(day=int(r.day),category=str(getattr(r,c)),total=int(r.total),positive=int(r.positive)) for r in g.itertuples()]
    del catpieces
    print('Daily counts verified. Computing all 28 date-window account/entity distributions.',flush=True)
    windows={}
    for start in range(7):
        for end in range(start,7):
            cnt=counts[start:end+1].sum(axis=0)
            active=cnt>0
            risk=risks[start:end+1].any(axis=0)[active]
            ae=entity[active]
            es=np.bincount(ae,minlength=len(entity_size))
            er=np.zeros(len(entity_size),dtype=bool); er[ae[risk]]=True
            entity_active=es>0
            w=dict(active_accounts=int(active.sum()),risk_accounts=int(risk.sum()),
                   active_entities=int(entity_active.sum()),risk_entities=int(er.sum()),
                   frequency=grouped_distribution(cnt[active],risk),
                   outgoing=grouped_distribution(paid[start:end+1].sum(axis=0)[active],risk),
                   incoming=grouped_distribution(received[start:end+1].sum(axis=0)[active],risk),
                   active_days=grouped_distribution((counts[start:end+1]>0).sum(axis=0)[active],risk),
                   entity_registered=grouped_distribution(entity_size[entity_active],er[entity_active]),
                   entity_active=grouped_distribution(es[entity_active],er[entity_active]),
                   entity_banks=grouped_distribution(entity_banks[entity_active],er[entity_active]),
                   entity_account_weighted=grouped_distribution(entity_size[ae],risk))
            # The first histogram bucket includes 0 for directional counts; label it explicitly there.
            for field in ['outgoing','incoming']:
                w[field]['bins'][0]='0–1'
            amounts={}
            s=sample[sample.day.between(start,end)]
            for amount,currency in [('amount_paid','payment_currency'),('amount_received','receiving_currency')]:
                for cur in sorted(s[currency].unique()):
                    rows=s[s[currency]==cur]
                    vals=[rows.loc[rows.is_laundering.eq(label),amount].to_numpy() for label in [0,1]]
                    full=[sum((hist.get((d,amount,cur,label),np.zeros(len(EDGES)-1,dtype=np.int64)) for d in range(start,end+1)),np.zeros(len(EDGES)-1,dtype=np.int64)) for label in [0,1]]
                    stat=dict(hist=[v.tolist() for v in full],n=[int(v.sum()) for v in full],sample_n=[len(v) for v in vals],quantiles=[None if not len(v) else np.quantile(v,[.5,.9,.99]).tolist() for v in vals],ks=None,rank_effect=None)
                    if min(map(len,vals))>=20:
                        stat['ks']=float(ks_2samp(*vals,method='asymp').statistic)
                        ranks=rankdata(np.concatenate(vals)); n0,n1=map(len,vals)
                        auc=(ranks[n0:].sum()-n1*(n1+1)/2)/(n0*n1)
                        stat['rank_effect']=float(2*auc-1)
                    amounts[amount+'|'+cur]=stat
            w['amounts']=amounts
            windows[f'{start}-{end}']=w
            print(f'Window {start+1}–{end+1} verified',flush=True)
    result=dict(meta=dict(status='verified',scope='Discovery only',dates=dates,rows=discovery['rows'],positive=discovery['positive_rows'],
                         accounts_metadata=n,seconds=round(time.perf_counter()-started,2),source_manifest_sha256=split['source_manifest_sha256'],
                         amount_method='Full-data log histograms; all positive amounts and deterministic 1% negative hash sample for quantiles/KS/rank effect. Amount float conversion is analysis only.',
                         sample_n=len(sample),sample_positive=int(sample.is_laundering.sum()),files=files,
                         limitations=['Account/entity groups are descriptive participation labels within the selected window.',
                                      'Static account metadata has no as-of date; registered entity size is descriptive only.',
                                      'No independent-counterparty count, temporal windows, p-value filtering or model fitting.',
                                      'Entity/chain dependence means IID p-values are not reported.']),daily=daily,categories=cats,amount_log_edges=EDGES.tolist(),windows=windows)
    write_json(OUT/'data.json',result)
    print(f'COMPLETE {result["meta"]["seconds"]} seconds',flush=True)


if __name__=='__main__': main()
