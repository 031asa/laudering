"""Retrospective strict-history diagnostics for saved Validation predictions only.

No labels are consulted in history construction; labels only select/group audit rows.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/gnn_error_analysis_v1'
ARCH=ROOT/'data/gnn_v1_validation'
PREFIX=20318992
DAY=86400*1000000
SEED=20261003

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw),ensure_ascii=True),flush=True)

class Hist:
    def __init__(self):
        self.a={p.stem:np.load(p,mmap_mode='r') for p in (ARCH/'history_index').glob('*.npy')}
    def bound(self,order,lo,hi,t):
        # Indirect binary search avoids copying a high-degree account's entire history.
        ts=self.a['timestamp']
        while lo<hi:
            m=(lo+hi)//2
            if int(ts[int(order[m])])<t:lo=m+1
            else:hi=m
        return lo
    def prior(self,node,direction,t):
        ptr=self.a[direction+'_ptr']; order=self.a[direction+'_order']
        lo=int(ptr[node]); hi=int(ptr[node+1]); end=self.bound(order,lo,hi,t)
        start=self.bound(order,lo,end,t-DAY)
        return order,start,end,lo,hi
    def window(self,node,direction,t):
        order,start,end,lo,hi=self.prior(node,direction,t)
        ids=np.asarray(order[start:end],dtype=np.int64)
        if direction=='in':ids=ids[self.a['src'][ids]!=node]
        else:ids=ids[self.a['dst'][ids]!=node]
        return ids,(order,start,end,lo,hi)
    def last_in_same(self,node,currency,t,info):
        order,start,end,lo,hi=info
        while end>lo:
            block=np.asarray(order[max(lo,end-4096):end],dtype=np.int64)
            good=(self.a['receiving_currency_cat'][block]==currency)&(self.a['src'][block]!=node)
            if good.any():
                eid=int(block[np.flatnonzero(good)[-1]])
                assert int(self.a['timestamp'][eid])<t
                return eid,(t-int(self.a['timestamp'][eid]))/60000000
            end=max(lo,end-4096)
        return -1,np.nan
    def flags(self,node,t):
        infos=[self.prior(node,d,t) for d in ('in','out')]
        prior=any(end>lo for _,_,end,lo,_ in infos)
        train=any(hi>lo and int(order[lo])<PREFIX for order,_,_,lo,hi in infos)
        return prior,train
    def cycle(self,u,v,t,u_in,v_out):
        # Exact existence in the preceding 24 h; every intermediate is examined.
        # Earliest v->x suffices to test each x->u, then choose latest valid pair.
        empty=dict(cycle_found=False,cycle_first_edge_id=-1,cycle_second_edge_id=-1,
                   cycle_x=-1,cycle_span_minutes=np.nan,cycle_recent_gap_minutes=np.nan)
        if u==v or not len(u_in) or not len(v_out):return empty
        a=self.a; incoming=u_in[(a['src'][u_in]!=u)&(a['src'][u_in]!=v)]
        outgoing=v_out[(a['dst'][v_out]!=u)&(a['dst'][v_out]!=v)]
        if not len(incoming) or not len(outgoing):return empty
        # outgoing is timestamp-ascending; equal timestamps are excluded below.
        xs,first=np.unique(a['dst'][outgoing],return_index=True)
        earliest=a['timestamp'][outgoing[first]]
        sources=a['src'][incoming];pos=np.searchsorted(xs,sources)
        valid=pos<len(xs);valid[valid]&=xs[pos[valid]]==sources[valid]
        ix=np.flatnonzero(valid)
        ix=ix[earliest[pos[ix]]<a['timestamp'][incoming[ix]]]
        if not len(ix):return empty
        second=int(incoming[ix[-1]]);x=int(a['src'][second]);t2=int(a['timestamp'][second])
        candidates=outgoing[(a['dst'][outgoing]==x)&(a['timestamp'][outgoing]<t2)]
        first_id=int(candidates[-1]);t1=int(a['timestamp'][first_id])
        assert t-DAY<=t1<t2<t and len({u,v,x})==3
        return dict(cycle_found=True,cycle_first_edge_id=first_id,cycle_second_edge_id=second,
                    cycle_x=x,cycle_span_minutes=(t-t1)/60000000,cycle_recent_gap_minutes=(t-t2)/60000000)

def main():
    begun=time.time()
    if (OUT/'history_summary.json').exists():raise FileExistsError('history analysis already exists')
    meta=json.loads((ARCH/'build.json').read_text(encoding='utf-8'))
    assert meta['end_exclusive']=='2022-09-14T00:00:00' and meta['rows']==25007998
    cols=['edge_id','src','dst','timestamp','positive','group','uniform_negative_control','payment_format','payment_currency','score_lr_only','self_transfer','from_bank','from_account','to_bank','to_account']
    sample=pq.ParquetFile(OUT/'inspection_sample.parquet').read(columns=cols).to_pandas()
    controls=sample.loc[sample.uniform_negative_control,'edge_id'].to_numpy()
    assert len(controls)==100000
    rng=np.random.default_rng(SEED); selected_controls=np.sort(rng.choice(controls,10000,replace=False))
    sample['history_negative_control']=sample.edge_id.isin(selected_controls)
    frame=sample[sample.positive|sample.group.eq('FP')|sample.history_negative_control].copy().sort_values('edge_id').reset_index(drop=True)
    assert frame.positive.sum()==6420 and frame.group.eq('FN').sum()==2905 and frame.group.eq('FP').sum()==810
    assert frame.history_negative_control.sum()==10000
    h=Hist(); a=h.a;rows=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        eid=int(row.edge_id);u=int(row.src);v=int(row.dst);t=int(a['timestamp'][eid]);cur=int(a['payment_currency_cat'][eid]);paid=int(a['paid_scaled'][eid])
        assert PREFIX<=eid<25007998 and t==row.timestamp.value//1000 and int(a['src'][eid])==u and int(a['dst'][eid])==v
        ui,ui_info=h.window(u,'in',t); vo,vo_info=h.window(v,'out',t)
        si=ui[a['receiving_currency_cat'][ui]==cur]; so=vo[a['payment_currency_cat'][vo]==cur]
        srcprior,srctrain=h.flags(u,t);dstprior,dsttrain=h.flags(v,t)
        lastid,lastgap=h.last_in_same(u,cur,t,ui_info)
        vals=dict(edge_id=eid,src_has_prior=srcprior,dst_has_prior=dstprior,src_seen_train=srctrain,dst_seen_train=dsttrain,
                  src_unique_payers_24h=len(np.unique(a['src'][ui])),dst_unique_receivers_24h=len(np.unique(a['dst'][vo])),
                  src_unique_payers_same_currency_24h=len(np.unique(a['src'][si])),dst_unique_receivers_same_currency_24h=len(np.unique(a['dst'][so])),
                  src_ext_in_count_same_currency_24h=len(si),dst_ext_out_count_same_currency_24h=len(so),
                  source_last_same_currency_in_edge_id=lastid,source_last_same_currency_in_gap_minutes=lastgap,
                  amount_match_edge_id=-1,amount_match_relative_diff=np.nan,amount_match_gap_minutes=np.nan,
                  amount_match_5pct_1h=False,amount_match_5pct_6h=False)
        if len(si):
            incoming=a['received_scaled'][si]
            diffs=np.abs(incoming.astype(np.float64)-float(paid))/paid if paid>0 else np.where(incoming==0,0.,np.inf)
            j=int(np.argmin(diffs)); best=float(diffs[j])
            # Equal best amount errors choose the most recent prior receipt.
            j=int(np.flatnonzero(diffs==best)[-1]);match=int(si[j])
            assert int(a['timestamp'][match])<t and int(a['receiving_currency_cat'][match])==cur
            vals.update(amount_match_edge_id=match,amount_match_relative_diff=best,amount_match_gap_minutes=(t-int(a['timestamp'][match]))/60000000)
            gaps=(t-a['timestamp'][si])/60000000
            vals['amount_match_5pct_1h']=bool(((diffs<=.05)&(gaps<=60)).any())
            vals['amount_match_5pct_6h']=bool(((diffs<=.05)&(gaps<=360)).any())
        for pct in (1,5,10):
            vals['amount_match_'+str(pct)+'pct_24h']=bool(np.isfinite(vals['amount_match_relative_diff']) and vals['amount_match_relative_diff']<=pct/100)
        vals['source_last_same_currency_in_over_72h']=bool(lastid>=0 and lastgap>4320)
        vals['source_no_same_currency_prior_in']=bool(lastid<0)
        vals.update(h.cycle(u,v,t,ui,vo))
        rows.append(vals)
        if (i+1)%1000==0:emit('history_rows',rows=i+1,total=len(frame),seconds=round(time.time()-begun,2))
    result=frame.merge(pd.DataFrame(rows),on='edge_id',validate='one_to_one')
    assert len(result)==len(frame)
    result.to_parquet(OUT/'history_signals.parquet',index=False)
    flags=['src_has_prior','dst_has_prior','src_seen_train','dst_seen_train','source_no_same_currency_prior_in','source_last_same_currency_in_over_72h',
           'amount_match_1pct_24h','amount_match_5pct_24h','amount_match_10pct_24h','amount_match_5pct_1h','amount_match_5pct_6h','cycle_found']
    nums=['src_unique_payers_24h','dst_unique_receivers_24h','src_unique_payers_same_currency_24h','dst_unique_receivers_same_currency_24h',
          'source_last_same_currency_in_gap_minutes','amount_match_relative_diff','amount_match_gap_minutes']
    groups={'FN':result.group.eq('FN'),'TP':result.group.eq('TP'),'FP':result.group.eq('FP'),'negative_uniform':result.history_negative_control}
    def summarize(df):
        masks={'FN':df.group.eq('FN'),'TP':df.group.eq('TP'),'FP':df.group.eq('FP'),'negative_uniform':df.history_negative_control}
        out={}
        for name,m in masks.items():
            sub=df[m];r={'rows':len(sub),'unique_src':sub.src.nunique(),'unique_dst':sub.dst.nunique(),'flags':{},'numeric':{}}
            for f in flags:r['flags'][f]={'yes':int(sub[f].sum()),'rate':float(sub[f].mean()) if len(sub) else None}
            for f in nums:
                values=sub[f].replace([np.inf,-np.inf],np.nan).dropna()
                r['numeric'][f]={'valid':len(values),'q10_q25_q50_q75_q90':np.quantile(values,[.1,.25,.5,.75,.9]).tolist() if len(values) else []}
            out[name]=r
        return out
    strata={col:{str(key):summarize(df) for key,df in result.groupby(col)} for col in ['payment_format','payment_currency']}
    concentration={}
    positives=result[result.positive]
    for col in ['src','dst']:
        g=positives.groupby(col).agg(positives=('positive','size'),fn=('group',lambda s:int(s.eq('FN').sum())),tp=('group',lambda s:int(s.eq('TP').sum()))).sort_values('fn',ascending=False)
        concentration[col]={'unique_positive_accounts':len(g),'unique_fn_accounts':int((g.fn>0).sum()),'top_20':g.head(20).reset_index().to_dict('records'),
                            'top_1_fn':int(g.fn.head(1).sum()),'top_10_fn':int(g.fn.head(10).sum()),'top_50_fn':int(g.fn.head(50).sum())}
    summary=dict(status='completed',rows=len(result),scope='Validation positives and false positives plus a uniform negative subsample. History is Train plus strictly earlier unlabeled Validation only.',
                 negative_controls={'count':10000,'seed':SEED,'parent_sample_count':100000,'method':'uniform without replacement subsample of existing uniform 100k negative sample; FP can overlap control group'},
                 definitions={'window':'[target minus 24 hours, target), excludes all equal timestamps; no truncation',
                              'amount_match':'min absolute received amount minus current paid amount, divided by current paid amount; incoming receipt currency equals current paid currency; external receipts only; zero current matches only zero receipt',
                              'cycle':'current u->v closes v->x at t1 then x->u at t2 with t-24h<=t1<t2<t and all three accounts distinct; currency unrestricted topological motif, never sum amounts; exact full coverage',
                              'last_receipt':'all available strictly prior same-currency external receipts since archive begins 2022-09-01; no future lookup',
                              'seen_train':'account has at least one edge in fixed Train prefix; absence does not establish newly opened real-world account'},
                 groups=summarize(result),strata=strata,account_concentration=concentration,wall_seconds=time.time()-begun)
    (OUT/'history_summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    emit('history_complete',rows=len(result),seconds=round(time.time()-begun,2))

if __name__=='__main__':main()
