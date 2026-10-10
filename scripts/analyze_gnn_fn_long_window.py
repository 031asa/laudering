"""Additional Validation-only history window/amount-distribution diagnostics."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from analyze_gnn_fn_history import Hist,ROOT,OUT,ARCH,DAY,clean,emit

def main():
    began=time.time()
    if (OUT/'history_long_window_summary.json').exists():raise FileExistsError('Long-window analysis already exists')
    frame=pd.read_parquet(OUT/'history_signals.parquet',columns=['edge_id','src','dst','group','positive','history_negative_control','payment_format','payment_currency','self_transfer','score_lr_only'])
    previous=pd.read_parquet(OUT/'inspection_sample.parquet',columns=['edge_id','r_src_current_to_own_ext_mean_24h','r_dst_current_to_own_ext_mean_24h'])
    frame=frame.merge(previous,on='edge_id',validate='one_to_one')
    assert len(frame)==17227 and frame.group.eq('FN').sum()==2905 and frame.history_negative_control.sum()==10000
    h=Hist();a=h.a;records=[]
    for i,r in enumerate(frame.itertuples(index=False)):
        e=int(r.edge_id);u=int(r.src);v=int(r.dst);t=int(a['timestamp'][e]);pc=int(a['payment_currency_cat'][e]);rc=int(a['receiving_currency_cat'][e]);paid=int(a['paid_scaled'][e]);received=int(a['received_scaled'][e])
        vals={'edge_id':e}
        source_info=None
        for pre,node,d,cur,amount,curkey,amtkey,opposite in [('src',u,'out',pc,paid,'payment_currency_cat','paid_scaled','dst'),('dst',v,'in',rc,received,'receiving_currency_cat','received_scaled','src')]:
            order,_,end,lo,hi=h.prior(node,d,t)
            start=h.bound(order,lo,end,t-7*DAY)
            ids=np.asarray(order[start:end],dtype=np.int64)
            ids=ids[(a[opposite][ids]!=node)&(a[curkey][ids]==cur)]
            am=np.asarray(a[amtkey][ids],dtype=np.float64)
            vals[pre+'_same_currency_ext_count_7d']=len(ids)
            vals[pre+'_same_currency_ext_count_24h']=int((a['timestamp'][ids]>=t-DAY).sum())
            med=float(np.median(am)) if len(am) else np.nan
            p90=float(np.quantile(am,.9)) if len(am) else np.nan
            vals[pre+'_amount_median_7d']=med/1000000
            vals[pre+'_amount_p90_7d']=p90/1000000
            vals[pre+'_current_to_median_7d']=amount/med if med>0 else np.nan
            vals[pre+'_current_to_p90_7d']=amount/p90 if p90>0 else np.nan
            for minimum in (1,3,5,10):vals[pre+'_amount_support_ge'+str(minimum)+'_7d']=bool(len(am)>=minimum)
            vals[pre+'_amount_above_p90_with_ge5']=bool(len(am)>=5 and amount>p90)
            vals[pre+'_amount_gt3median_with_ge5']=bool(len(am)>=5 and med>0 and amount>3*med)
            if pre=='src':source_info=(order,end,lo)
            if len(ids):assert int(a['timestamp'][ids].min())>=t-7*DAY and int(a['timestamp'][ids].max())<t
        vals.update(pair_equal_amount_count_72h=0,pair_equal_amount_last_edge_id=-1,pair_equal_amount_last_gap_minutes=np.nan,
                    pair_previous_edge_id=-1,pair_previous_gap_minutes=np.nan,pair_previous_payment_changed=False,pair_previous_exists=False,
                    pair_previous_format_count=0,pair_previous_ambiguous=False,pair_previous_change_comparable=False)
        order,end,lo=source_info
        begin72=h.bound(order,lo,end,t-3*DAY)
        ids=np.asarray(order[begin72:end],dtype=np.int64)
        same=ids[(a['dst'][ids]==v)&(a['payment_currency_cat'][ids]==pc)&(a['paid_scaled'][ids]==paid)]
        vals['pair_equal_amount_count_72h']=len(same)
        if len(same):
            last=int(same[-1]);vals['pair_equal_amount_last_edge_id']=last;vals['pair_equal_amount_last_gap_minutes']=(t-int(a['timestamp'][last]))/60000000
        while end>lo:
            ids=np.asarray(order[max(lo,end-4096):end],dtype=np.int64);same=ids[a['dst'][ids]==v]
            if len(same):
                last=int(same[-1]);last_t=int(a['timestamp'][last])
                # At the most recent timestamp a pair can have several transactions.
                # Do not invent ordering within that timestamp; mark mixed formats unknown.
                global_end=h.bound(order,lo,int(a['out_ptr'][u+1]),t)
                tie_start=h.bound(order,lo,global_end,last_t)
                tie_end=h.bound(order,tie_start,global_end,last_t+1)
                tie_ids=np.asarray(order[tie_start:tie_end],dtype=np.int64)
                tie_ids=tie_ids[a['dst'][tie_ids]==v]
                formats=np.unique(a['payment_format_cat'][tie_ids]);comparable=len(formats)==1
                vals.update(pair_previous_edge_id=last,pair_previous_gap_minutes=(t-last_t)/60000000,
                            pair_previous_payment_changed=bool(comparable and formats[0]!=a['payment_format_cat'][e]),pair_previous_exists=True,
                            pair_previous_format_count=len(formats),pair_previous_ambiguous=not comparable,pair_previous_change_comparable=comparable)
                assert int(a['timestamp'][last])<t
                break
            end=max(lo,end-4096)
        records.append(vals)
        if (i+1)%3000==0:emit('long_window_rows',rows=i+1,total=len(frame),seconds=round(time.time()-began,2))
    result=frame.merge(pd.DataFrame(records),on='edge_id',validate='one_to_one')
    result['src_original_24h_mean_missing']=result.r_src_current_to_own_ext_mean_24h.isna()
    result['dst_original_24h_mean_missing']=result.r_dst_current_to_own_ext_mean_24h.isna()
    result.to_parquet(OUT/'history_long_window.parquet',index=False)
    flags=[c for c in result.columns if '_support_ge' in c or '_with_ge5' in c]+['pair_previous_exists','pair_previous_payment_changed','pair_previous_ambiguous','pair_previous_change_comparable']
    def summ(df):
        out={}
        for name,m in [('FN',df.group.eq('FN')),('TP',df.group.eq('TP')),('FP',df.group.eq('FP')),('negative_uniform',df.history_negative_control)]:
            sub=df[m];o={'rows':len(sub),'flags':{f:{'yes':int(sub[f].sum()),'rate':float(sub[f].mean()) if len(sub) else None} for f in flags},'original_missing_support':{},'quantiles':{}}
            for pre in ['src','dst']:
                missing=sub[sub[pre+'_original_24h_mean_missing']]
                o['original_missing_support'][pre]={'missing_rows':len(missing),**{'seven_day_ge'+str(n):int(missing[pre+'_amount_support_ge'+str(n)+'_7d'].sum()) for n in [1,3,5,10]}}
                for col in [pre+'_same_currency_ext_count_7d',pre+'_current_to_median_7d',pre+'_current_to_p90_7d']:
                    v=sub[col].replace([np.inf,-np.inf],np.nan).dropna();o['quantiles'][col]={'valid':len(v),'q10_q25_q50_q75_q90':np.quantile(v,[.1,.25,.5,.75,.9]).tolist() if len(v) else []}
            o['pair_equal_amount_repeat_72h']={'yes':int((sub.pair_equal_amount_count_72h>0).sum()),'rate':float((sub.pair_equal_amount_count_72h>0).mean()) if len(sub) else None}
            out[name]=o
        return out
    summary={'status':'completed','rows':len(result),'groups':summ(result),'by_payment_format':{str(k):summ(d) for k,d in result.groupby('payment_format')},
             'definitions':{'source':'[t-7d,t) source external outgoing amounts in current payment currency; current paid amount vs median/p90',
                            'destination':'[t-7d,t) destination external incoming amounts in current receiving currency; current received amount vs median/p90',
                            'quantiles':'empirical median and linear-interpolated 90th percentile of scaled amounts; support count supplied; no currency pooling',
                            'pair_repeat':'[t-72h,t) same directed pair, same payment currency category and exact same integer scaled payment amount; self-pairs included',
                            'payment_change':'current payment format differs from the sole format at the most recent strictly prior same-pair timestamp; mixed-format timestamp ties are marked ambiguous and not changed, never arbitrarily ordered; self-pairs included',
                            'time_scope':'No Test. All same timestamp edges excluded. Available archive starts September 1, 2022. No labels used in candidate values.'},
             'wall_seconds':time.time()-began}
    (OUT/'history_long_window_summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    emit('long_window_complete',rows=len(result),seconds=round(time.time()-began,2))

if __name__=='__main__':main()
