"""Separate bank-70 payer activity from bank/payment associations; no data deletion."""
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/train_bank_analysis'


def main():
    cfg=json.loads((ROOT/'config/dataset_split.json').read_text(encoding='utf-8'))
    dates=sorted(d for s in cfg['splits'] if s['name'] in cfg['train_members'] for d in s['dates'])
    groups=[]; accounts=[]; scanned=0
    for date in dates:
        for p in sorted((ROOT/cfg['source']/f'transaction_date={date}').glob('*.parquet')):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=250000,columns=['from_bank','from_account','to_bank','payment_format','payment_currency','is_laundering']):
                scanned+=batch.num_rows
                t=pa.Table.from_batches([batch]).filter(pc.equal(batch.column('from_bank'),'70'))
                if not t.num_rows: continue
                f=t.to_pandas()
                g=f.groupby(['to_bank','payment_format']).is_laundering.agg(total='size',positive='sum').reset_index().rename(columns={'to_bank':'bank'})
                g['date']=date; groups.append(g)
                a=f.groupby(['from_account','payment_currency','payment_format']).is_laundering.agg(total='size',positive='sum').reset_index()
                a['date']=date; accounts.append(a)
        print(f'Sensitivity scan {date}',flush=True)
    assert scanned==20318992
    special=pd.concat(groups).groupby(['date','bank','payment_format'],as_index=False)[['total','positive']].sum()
    assert special.total.sum()==1801617 and special.positive.sum()==2551
    a=pd.concat(accounts).groupby(['from_account','payment_currency','payment_format'],as_index=False)[['total','positive']].sum()
    a.to_csv(OUT/'bank70_account_currency_payment.csv',index=False,encoding='utf-8-sig')
    full=pd.read_csv(OUT/'bank_payment_daily_to.csv',dtype={'bank':str})
    joined=full.merge(special,on=['date','bank','payment_format'],how='left',suffixes=('','_70')).fillna({'total_70':0,'positive_70':0})
    joined['total']=joined.total-joined.total_70
    joined['positive']=joined.positive-joined.positive_70
    assert (joined.total>=joined.positive).all() and (joined.positive>=0).all()
    joined=joined[joined.total>0].copy()
    base=joined.groupby(['date','payment_format'],as_index=False)[['total','positive']].sum()
    base['new_rate']=base.positive/base.total
    joined=joined.merge(base[['date','payment_format','new_rate']],on=['date','payment_format'],validate='many_to_one')
    joined['new_expected']=joined.total*joined.new_rate
    b=joined.groupby('bank',as_index=False)[['total','positive','new_expected']].sum()
    b['positive_rate']=b.positive/b.total
    b['lift']=b.positive_rate/(b.positive.sum()/b.total.sum())
    b['adjusted_ratio']=b.positive/b.new_expected.replace(0,float('nan'))
    b.sort_values('positive',ascending=False).to_csv(OUT/'banks_to_excluding_from70.csv',index=False,encoding='utf-8-sig')
    m=joined.groupby('payment_format',as_index=False)[['total','positive']].sum()
    m['positive_rate']=m.positive/m.total
    m['lift']=m.positive_rate/(m.positive.sum()/m.total.sum())
    m.to_csv(OUT/'payment_excluding_from70.csv',index=False,encoding='utf-8-sig')
    keep=['11','20','1213','718','2408','1226']
    print(json.dumps({'accounts':a.groupby('from_account')[['total','positive']].sum().reset_index().to_dict('records'),'methods':m.to_dict('records'),'selected_receiving_banks':b[b.bank.isin(keep)].to_dict('records')},ensure_ascii=True),flush=True)


if __name__=='__main__': main()
