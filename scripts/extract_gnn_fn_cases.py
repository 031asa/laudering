"""Reconstruct selected Validation errors from strict historical transactions and frozen plans."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/gnn_error_analysis_v1'
ARCH=ROOT/'data/gnn_v1_validation'
CACHE=ROOT/'data/gnn_validation_cache_parquet'
DAY=86400*1000000

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,(pd.Timestamp,np.datetime64)):return str(v)
    return v
def stamp(us):return str(np.datetime64(int(us),'us'))
def amount(n):
    n=int(n);return ('-' if n<0 else '')+str(abs(n)//1000000)+'.'+str(abs(n)%1000000).zfill(6)

def main():
    if (OUT/'case_evidence.json').exists():raise FileExistsError('Case evidence already exists')
    base=pd.read_parquet(OUT/'inspection_sample.parquet').set_index('edge_id',drop=False).rename_axis('_edge_index')
    hist=pd.read_parquet(OUT/'history_signals.parquet').set_index('edge_id',drop=False).rename_axis('_edge_index')
    arr={p.stem:np.load(p,mmap_mode='r') for p in (ARCH/'history_index').glob('*.npy')}
    vocab=read(ARCH/'metadata/vocabulary.json')['values']
    fn=base[base.group.eq('FN')]
    used=set();choices=[]
    def choose(title,pool,sort='score_lr_only',ascending=True):
        pool=pool[~pool.edge_id.isin(used)]
        if len(pool):
            row=pool.sort_values([sort,'edge_id'],ascending=[ascending,True]).iloc[0]
            used.add(int(row.edge_id));choices.append((title,int(row.edge_id)))
    choose('高频付款账户中的非ACH低分漏检',fn[(fn.from_bank=='70')&(fn.payment_currency=='US Dollar')])
    choose('其他付款来源的非ACH漏检',fn[(fn.from_bank!='70')&(fn.payment_format!='ACH')],ascending=False)
    eligible=fn[fn.edge_id.isin(hist.index)]
    h=hist.loc[eligible.index]
    choose('距上次同币种入账超过72小时',eligible[(eligible.payment_format=='ACH')&(h.source_last_same_currency_in_gap_minutes>4320)])
    choose('前序入账金额接近当前转付',eligible[(eligible.payment_format=='ACH')&(h.amount_match_relative_diff<=.01)])
    if 'cycle_found' in h:
        choose('当前交易闭合有序三边环',eligible[(eligible.payment_format=='ACH')&h.cycle_found.fillna(False)])
    choose('ACH深低分漏检',fn[(fn.payment_format=='ACH')&(fn.score_lr_only<.01)&(fn.src!=fn.dst)])
    choose('当前阈值附近的ACH漏检',fn[(fn.payment_format=='ACH')],ascending=False)
    selected={}
    control_rules=[]
    for title,e in choices:
        x=base.loc[e]; selected[e]={'title':title,'role':'FN','paired_to':None}
        for group in ['TP','negative']:
            pool=base[base.group.eq('TP') if group=='TP' else base.uniform_negative_control]
            pool=pool[(pool.date==x.date)&(pool.payment_format==x.payment_format)&(pool.payment_currency==x.payment_currency)]
            if group=='negative' and (pool.src==x.src).any():pool=pool[pool.src==x.src]
            if len(pool):
                distance=abs(np.log1p(pool.amount_paid_float)-np.log1p(x.amount_paid_float))
                distance+=.1*abs((pool.timestamp-x.timestamp).dt.total_seconds())/86400
                idx=distance.sort_values(kind='stable').index[0]
                control_rules.append({'fn_edge_id':e,'control':group,'edge_id':int(idx),'same_src':bool(base.loc[idx,'src']==x.src),
                    'criterion':'Same date, payment method and currency; negative uses same source if available; minimum log1p amount distance + 0.1 day-time distance.'})
                selected.setdefault(int(idx),{'title':title+'：对照','role':group,'paired_to':e})
            else:control_rules.append({'fn_edge_id':e,'control':group,'edge_id':None,'criterion':'No same-date/method/currency control available'})
    def adjacency(node,direction,t,window=DAY):
        a=arr[direction+'_order'][int(arr[direction+'_ptr'][node]):int(arr[direction+'_ptr'][node+1])]
        ts=arr['timestamp'][a];lo=np.searchsorted(ts,t-window,side='left');hi=np.searchsorted(ts,t,side='left')
        return a[lo:hi]
    all_edges=set();cases=[]
    for e,definition in selected.items():
        x=base.loc[e];t=int(arr['timestamp'][e]);u=int(x.src);v=int(x.dst)
        events={};counts={}
        for node,role in [(u,'source'),(v,'destination')]:
            for direction in ['in','out']:
                ids=adjacency(node,direction,t);ids=ids[arr['src'][ids]!=arr['dst'][ids]]
                counts[role+'_'+direction+'_24h']=len(ids)
                for k in ids[-5:]:events.setdefault(int(k),[]).append(role+'_'+direction+'_last5')
        witness={}
        if e in hist.index:
            hr=hist.loc[e]
            for name in ['amount_match_edge_id','source_last_same_currency_in_edge_id','cycle_first_edge_id','cycle_second_edge_id']:
                if name in hr and pd.notna(hr[name]) and int(hr[name])>=0:
                    k=int(hr[name]);assert int(arr['timestamp'][k])<t
                    witness[name]=k;events.setdefault(k,[]).append(name)
        assert all(int(arr['timestamp'][k])<t for k in events)
        all_edges.update(events);all_edges.add(e)
        rc=['r_pair_fwd_count_24h','r_pair_fwd_count_72h','r_src_current_to_own_ext_mean_24h','r_dst_current_to_own_ext_mean_24h',
            'r_src_paid_to_ext_in_sum_24h','r_src_ext_in_count_24h','r_dst_ext_out_count_24h','r_src_current_payment_share_24h','r_dst_current_payment_share_24h',
            'r_src_last_same_currency_ext_in_gap_min']
        cases.append({'edge_id':e,**definition,'target':clean(x[['transaction_id','timestamp','from_bank','from_account','to_bank','to_account','payment_format','payment_currency','amount_paid','amount_received','group','score_lr_only','score_baseline','score_sqrt_weight']].to_dict()),
            'history_24h_counts':counts,'existing_R':clean(x[rc].to_dict()),'diagnostic':clean(hist.loc[e].to_dict()) if e in hist.index else {},
            'witness_edges':witness,'history_events':events})
    # Verify and inspect the original saved batched graph, never resample a single case.
    config=read(ROOT/'result/gnn_tuning_v1/evaluations/lr_only/config.json')
    assert sha(CACHE/'manifest.json')==config['cache_manifest_sha256']
    assert sha(CACHE/'config.json')==config['cache_config_sha256']
    lookup={x['edge_id']:x for x in cases};found=set()
    for p in sorted(CACHE.glob('transaction_date=*/hour=*/block=*/plans.parquet')):
        manifest=read(p.parent/'manifest.json')
        relevant={e for e in selected if manifest['first_cutoff_us']<=int(arr['timestamp'][e])<=manifest['last_cutoff_us']}
        if not relevant:continue
        assert manifest['status']=='verified' and manifest['config_sha256']==config['cache_config_sha256']
        assert sha(p)==manifest['file_sha256']['plans.parquet']
        for b in pq.ParquetFile(p).iter_batches(columns=['batch_id','cutoff_us','targets','edges'],batch_size=128):
            for row in b.to_pylist():
                hits=relevant.intersection(row['targets'])
                if not hits:continue
                edge_set=set(row['edges'])
                assert np.all(arr['timestamp'][np.asarray(row['edges'],dtype=np.int64)]<row['cutoff_us'])
                for e in hits:
                    assert e not in found and row['cutoff_us']==int(arr['timestamp'][e]);found.add(e)
                    c=lookup[e]
                    c['cache_audit']={'block':str(p.parent.relative_to(CACHE)),'batch_id':row['batch_id'],'batch_targets':len(row['targets']),
                        'sampled_edges':len(edge_set),'plans_sha256':manifest['file_sha256']['plans.parquet'],
                        'witness_in_original_plan':{k:v in edge_set for k,v in c['witness_edges'].items()},
                        'shown_history_edges_in_plan':{str(k):k in edge_set for k in c['history_events']}}
    assert found==set(selected)
    # Resolve only account identities needed in the case tables.
    nodes={int(arr[k][e]) for e in all_edges for k in ['src','dst']};identity={}
    for b in pq.ParquetFile(ARCH/'graph/accounts/accounts.parquet').iter_batches(columns=['node_id','bank_id','account_number'],batch_size=65536):
        tab=b.to_pandas();tab=tab[tab.node_id.isin(nodes)]
        for row in tab.itertuples():identity[int(row.node_id)]={'bank':str(row.bank_id),'account':str(row.account_number)}
    assert len(identity)==len(nodes)
    def decode(key,cat):return vocab[key][int(cat)-2] if int(cat)>=2 else 'UNK'
    for c in cases:
        rows=[]
        for k,roles in sorted(c['history_events'].items(),key=lambda kv:(int(arr['timestamp'][kv[0]]),kv[0])):
            rows.append({'edge_id':k,'timestamp':stamp(arr['timestamp'][k]),'src':int(arr['src'][k]),'dst':int(arr['dst'][k]),
                'from':identity[int(arr['src'][k])],'to':identity[int(arr['dst'][k])],
                'paid':amount(arr['paid_scaled'][k]),'received':amount(arr['received_scaled'][k]),
                'payment_currency':decode('currency',arr['payment_currency_cat'][k]),'receiving_currency':decode('currency',arr['receiving_currency_cat'][k]),
                'payment_format':decode('payment_format',arr['payment_format_cat'][k]),'roles':roles,
                'in_original_plan':c['cache_audit']['shown_history_edges_in_plan'][str(k)]})
        c['history_events']=rows
    result=clean({'scope':'Post-hoc selected Validation cases; all shown historical transactions strictly precede target. History last 5 per direction is illustrative, counts cover full 24h. No Test; no model changes.',
        'selected_false_negatives':choices,'controls':control_rules,'cases':cases,'cache_config_sha256':config['cache_config_sha256']})
    (OUT/'case_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'stage':'case_evidence_complete','fn_cases':len(choices),'including_controls':len(cases),'resolved_history_edges':len(all_edges)},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
