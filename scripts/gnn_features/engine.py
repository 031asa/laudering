"""Exact indexed as-of queries. No labels accepted by this module."""
from decimal import Decimal
import math
import numpy as np
from .contract import HOUR, DAY, WINDOWS, N_FIELDS, R_FIELDS

SCALE=1_000_000
MINUTE=60_000_000

def money(x):
    # String construction does not round through Decimal's default context.
    x=int(x)
    return Decimal(f'{x//SCALE}.{x%SCALE:06d}')

def exact_sum(a):
    if not len(a): return 0
    # A bound proves vectorized accumulation cannot overflow int64.
    if int(a.max())*len(a) <= np.iinfo(np.int64).max: return int(a.sum(dtype=np.int64))
    return sum(map(int,a))

def frequencies(a):
    k,c=np.unique(a,return_counts=True)
    return dict(zip(map(int,k),map(int,c)))

def hhi(counts):
    total=sum(counts.values())
    return sum(c*c for c in counts.values())/(total*total) if total else None

def ratio(a,b): return a/b if b>0 else None

class DirectionState:
    def __init__(self, engine, node, direction, cutoff):
        self.e,self.node,self.d,self.t=engine,node,direction,cutoff
        ptr=engine.arr[direction+'_ptr']; order=engine.arr[direction+'_order']
        ids=order[int(ptr[node]):int(ptr[node+1])]
        ts=engine.arr['timestamp'][ids]
        end=int(np.searchsorted(ts,cutoff,side='left'))
        self.ids=ids[:end]; self.ts=ts[:end]
        self.cp_col='src' if direction=='in' else 'dst'
        self.curr_col='receiving_currency_cat' if direction=='in' else 'payment_currency_cat'
        self.amount_col='received_scaled' if direction=='in' else 'paid_scaled'
        self.cache={}

    def window(self,w,external=False):
        key=('window',w,external)
        if key not in self.cache:
            lower=self.t-self.t%DAY if w=='day' else self.t-WINDOWS[w]
            ids=self.ids[np.searchsorted(self.ts,lower,side='left'):]
            if external: ids=ids[self.e.arr['src'][ids]!=self.e.arr['dst'][ids]]
            self.cache[key]=ids
        return self.cache[key]

    def counts(self,kind,w):
        key=('counts',kind,w)
        if key not in self.cache:
            ids=self.window(w,kind in ['cp','bank'])
            a=self.e.arr
            vals=(a[self.cp_col][ids] if kind=='cp' else
                  self.e.bank[a[self.cp_col][ids]] if kind=='bank' else
                  a['payment_format_cat'][ids] if kind=='payment' else
                  a[self.curr_col][ids] if kind=='currency' else
                  a['timestamp'][ids]//HOUR%24)
            self.cache[key]=frequencies(vals)
        return self.cache[key]

    def last(self,kind='all',value=None):
        if kind=='all': return int(self.ts[-1]) if len(self.ts) else None
        key=('last',kind)
        if key not in self.cache:
            a=self.e.arr; ids=self.ids[a['src'][self.ids]!=a['dst'][self.ids]]
            if kind=='ext': self.cache[key]=int(a['timestamp'][ids[-1]]) if len(ids) else None
            else:
                vals=(a[self.cp_col][ids] if kind=='cp' else
                      self.e.bank[a[self.cp_col][ids]] if kind=='bank' else a[self.curr_col][ids])
                unique,revindex=np.unique(vals[::-1],return_index=True)
                self.cache[key]=dict(zip(map(int,unique),map(int,a['timestamp'][ids[::-1][revindex]])))
        return self.cache[key] if kind=='ext' else self.cache[key].get(int(value))

    def gap(self,kind='all',value=None):
        last=self.last(kind,value)
        return (self.t-last)/MINUTE if last is not None else None

    def currency(self,curr,w):
        key=('currency',int(curr),w)
        if key not in self.cache:
            a=self.e.arr; ids=self.window(w,True)
            ids=ids[a[self.curr_col][ids]==curr]
            amounts=a[self.amount_col][ids]
            self.cache[key]=dict(count=len(ids),sum=exact_sum(amounts),
                ach_sum=exact_sum(amounts[a['payment_format_cat'][ids]==self.e.ach]),
                unique=len(np.unique(a[self.cp_col][ids])))
        return self.cache[key]

    def node_values(self):
        a=self.e.arr; prefix='n_'+self.d+'_'; r={}
        def put(k,v): r[prefix+k]=v
        c={w:len(self.window(w)) for w in ['1h','24h','72h']}
        for w,v in c.items(): put('all_count_'+w,v)
        for w in ['24h','72h']: put('ext_count_'+w,len(self.window(w,True)))
        cp=self.counts('cp','24h'); banks=self.counts('bank','24h')
        put('unique_cp_24h',len(cp)); put('cp_hhi_24h',hhi(cp))
        put('burst_1h_vs_23h',(c['1h']+1)/((c['24h']-c['1h'])/23+1))
        put('active_hours_24h',len(np.unique(a['timestamp'][self.window('24h')]//HOUR)))
        put('active_days_72h',len(np.unique(a['timestamp'][self.window('72h')]//DAY)))
        put('last_all_gap_min',self.gap())
        ext=sum(banks.values())
        put('cross_bank_share_24h',ratio(ext-banks.get(int(self.e.bank[self.node]),0),ext))
        put('bank_unique_24h',len(banks)); put('bank_hhi_24h',hhi(banks))
        ach=self.counts('payment','24h').get(self.e.ach,0)
        put('ach_count_24h',ach); put('ach_share_24h',ratio(ach,c['24h']))
        hour=self.counts('hour','72h')
        put('hour_entropy_72h',-sum((x/c['72h'])*math.log(x/c['72h']) for x in hour.values()) if c['72h'] else None)
        put('midday_share_72h',ratio(sum(v for k,v in hour.items() if 11<=k<17),c['72h']))
        for name,unit in [('hour',HOUR),('minute',MINUTE)]:
            bucket=frequencies(a['timestamp'][self.window('72h')]//unit)
            put('peak_'+name+'_share_72h',ratio(max(bucket.values(),default=0),c['72h']))
        return r

class FeatureEngine:
    def __init__(self, arrays, bank_categories, ach_category, data_start, coverage_end=None):
        if any('label' in k or 'launder' in k for k in arrays): raise ValueError('Labels forbidden in feature engine')
        self.arr=arrays; self.bank=bank_categories; self.ach=ach_category; self.data_start=int(data_start)
        self.coverage_end=coverage_end
        self.cutoff=None; self.states={}; self.nodes={}; self.queries=0; self.cache_hits=0

    def at(self,t):
        t=int(t)
        if t<self.data_start or (self.coverage_end is not None and t>self.coverage_end):
            raise ValueError('Query outside known history coverage; do not imply complete windows')
        if t!=self.cutoff:
            self.cutoff=t; self.states.clear(); self.nodes.clear()
        return self

    def state(self,node,d):
        if self.cutoff is None: raise ValueError('Set cutoff before querying')
        key=(int(node),d)
        if key not in self.states: self.states[key]=DirectionState(self,int(node),d,self.cutoff)
        return self.states[key]

    def context(self):
        t=self.cutoff; r={}
        for w in ['1h','6h','24h','72h','day']:
            start=t-t%DAY if w=='day' else t-WINDOWS[w]
            r['full_'+w]=self.data_start<=start
            r['observed_hours_'+w]=max(0,t-max(start,self.data_start))/HOUR
        return r

    def node(self,node):
        node=int(node); self.queries+=1
        if node in self.nodes: self.cache_hits+=1; return self.nodes[node]
        r={**self.state(node,'in').node_values(),**self.state(node,'out').node_values(),
           'n_in_last_ext_gap_min':self.state(node,'in').gap('ext')}
        assert set(r)=={f['name'] for f in N_FIELDS}
        self.nodes[node]=r
        return r

    def target(self,edge_id):
        a=self.arr; i=int(edge_id); t=int(a['timestamp'][i]); self.at(t)
        u,v=int(a['src'][i]),int(a['dst'][i]); self_edge=u==v
        result={}; audit={}; support={}
        def put(k,value,reason=None):
            result[k]=value
            if value is None: audit[k]=reason or 'no_history'
        def frac(k,num,den,count):
            put(k,ratio(num,den),'no_history' if not count else 'zero_denominator')
            support[k]={'numerator':num,'denominator':den,'count':count}
        us=self.state(u,'out'); vi=self.state(v,'in')
        if not self_edge:
            vo=self.state(v,'out')
            fwd=us.counts('cp','24h').get(v,0)
            put('r_pair_fwd_count_24h',fwd)
            put('r_pair_fwd_count_72h',us.counts('cp','72h').get(v,0))
            put('r_pair_rev_count_24h',vo.counts('cp','24h').get(u,0))
            put('r_pair_fwd_last_gap_min',us.gap('cp',v)); put('r_pair_rev_last_gap_min',vo.gap('cp',u))
            for s,state in [('src',us),('dst',vi)]:
                den=len(state.window('24h',True))
                frac(f'r_pair_{s}_{state.d}_share_24h',fwd,den,den)
            for s,node,curr,amount,own in [('src',u,int(a['payment_currency_cat'][i]),int(a['paid_scaled'][i]),'out'),('dst',v,int(a['receiving_currency_cat'][i]),int(a['received_scaled'][i]),'in')]:
                incoming=self.state(node,'in').currency(curr,'24h'); outgoing=self.state(node,'out').currency(curr,'24h')
                p=f'r_{s}_'
                for d,stat in [('in',incoming),('out',outgoing)]:
                    put(p+f'ext_{d}_sum_24h',money(stat['sum'])); put(p+f'ext_{d}_count_24h',stat['count'])
                frac(p+'ext_out_in_ratio_24h',outgoing['sum'],incoming['sum'],incoming['count'])
                ownstat=outgoing if own=='out' else incoming
                frac(p+'current_to_own_ext_mean_24h',amount*ownstat['count'],ownstat['sum'],ownstat['count'])
                frac(p+'own_ext_ach_amount_share_24h',ownstat['ach_sum'],ownstat['sum'],ownstat['count'])
            curr=int(a['payment_currency_cat'][i]); paid=int(a['paid_scaled'][i]); ins=self.state(u,'in')
            for w in ['1h','6h','24h','72h','day']:
                stat=ins.currency(curr,w)
                if w!='24h':
                    put(f'r_src_ext_in_sum_{w}',money(stat['sum'])); put(f'r_src_ext_in_count_{w}',stat['count'])
                frac(f'r_src_paid_to_ext_in_sum_{w}',paid,stat['sum'],stat['count'])
                if w in ['1h','24h','day']: put(f'r_src_ext_unique_payers_{w}',stat['unique'])
                if w in ['24h','day']:
                    out=us.currency(curr,w)
                    frac(f'r_src_cumulative_out_to_ext_in_sum_{w}',out['sum']+paid,stat['sum'],stat['count'])
                    if w=='day':
                        put('r_src_ext_out_sum_day',money(out['sum'])); put('r_src_ext_out_count_day',out['count'])
            put('r_src_last_same_currency_ext_in_gap_min',ins.gap('currency',curr))
        for s,state,curr,bank in [('src',us,int(a['payment_currency_cat'][i]),int(self.bank[v])),('dst',vi,int(a['receiving_currency_cat'][i]),int(self.bank[u]))]:
            for kind,w,value in [('payment','24h',int(a['payment_format_cat'][i])),('currency','24h',curr),('hour','72h',t//HOUR%24),('bank','24h',bank),('bank','72h',bank)]:
                if kind=='bank' and self_edge: continue
                counts=state.counts(kind,w); den=sum(counts.values())
                frac(f'r_{s}_current_{kind}_share_{w}',counts.get(value,0),den,den)
            if not self_edge: put(f'r_{s}_current_bank_last_gap_min',state.gap('bank',bank))
        if self_edge:
            for field in R_FIELDS:
                if field['name'] not in result: put(field['name'],None,'not_applicable_self_transfer')
        if set(result)!={f['name'] for f in R_FIELDS}: raise AssertionError('Incomplete R schema')
        return result,{'missing_reasons':audit,'ratio_support_scaled':support,'context':self.context()}

    def neighbors(self,node,direction,limit,rng):
        ids=self.state(node,direction).window('72h',True)
        return ids if len(ids)<=limit else rng.choice(ids,size=limit,replace=False)

def make_orders(arrays,node_count):
    """Deterministic time order; edge ID only breaks storage ties, never visibility."""
    for d,col in [('in','dst'),('out','src')]:
        order=np.lexsort((np.arange(len(arrays['timestamp'])),arrays['timestamp'],arrays[col])).astype(np.uint32)
        count=np.bincount(arrays[col].astype(np.int64),minlength=node_count)
        arrays[d+'_order']=order; arrays[d+'_ptr']=np.r_[0,np.cumsum(count)]
    return arrays
