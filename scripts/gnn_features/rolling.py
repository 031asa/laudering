"""Incremental, exact account-local windows. Source archive is read-only."""
import math
import numpy as np
from .contract import HOUR, DAY, WINDOWS
from .engine import FeatureEngine, DirectionState, frequencies, exact_sum, ratio, MINUTE


class Frequencies(dict):
    def __init__(self):
        super().__init__(); self.total=0; self.squares=0

    def update_counts(self, values, sign):
        for key,change in frequencies(values).items():
            old=self.get(key,0);new=old+sign*change
            if new<0: raise AssertionError('Negative rolling frequency')
            self.total+=sign*change;self.squares+=new*new-old*old
            if new:self[key]=new
            elif key in self:del self[key]

    def hhi(self):return self.squares/(self.total*self.total) if self.total else None


class RollingDirectionState(DirectionState):
    def __init__(self,engine,node,direction,cutoff):
        self.e,self.node,self.d,self.t=engine,node,direction,cutoff
        ptr=engine.arr[direction+'_ptr'];order=engine.arr[direction+'_order']
        self.full_ids=order[int(ptr[node]):int(ptr[node+1])]
        self.full_ts=engine.arr['timestamp'][self.full_ids]
        self.cp_col='src' if direction=='in' else 'dst'
        self.curr_col='receiving_currency_cat' if direction=='in' else 'payment_currency_cat'
        self.amount_col='received_scaled' if direction=='in' else 'paid_scaled'
        self.freq={};self.amounts={};self.last_maps={};self.cache={};self.ranges={}
        self.advance(cutoff)

    def advance(self,t):
        if t<self.t:raise ValueError('An account state must advance chronologically')
        self.t=t;self.stop=int(np.searchsorted(self.full_ts,t,side='left'))
        self.ids=self.full_ids[:self.stop];self.ts=self.full_ts[:self.stop]
        self.cache.clear();self.ranges.clear()

    def bounds(self,w):
        if w not in self.ranges:
            lower=self.t-self.t%DAY if w=='day' else self.t-WINDOWS[w]
            self.ranges[w]=(int(np.searchsorted(self.full_ts,lower,side='left')),self.stop)
        return self.ranges[w]

    def window(self,w,external=False):
        key=(w,external)
        if key not in self.cache:
            lo,hi=self.bounds(w);ids=self.full_ids[lo:hi]
            if external:ids=ids[self.e.arr[self.cp_col][ids]!=self.node]
            self.cache[key]=ids
        return self.cache[key]

    def changes(self,previous,current):
        oldlo,oldhi=previous;lo,hi=current
        # Disjoint windows: remove old contents and add only the new window.
        if lo>=oldhi:
            return [(self.full_ids[oldlo:oldhi],-1),(self.full_ids[lo:hi],1)]
        return [(self.full_ids[oldlo:lo],-1),(self.full_ids[oldhi:hi],1)]

    def values(self,ids,kind):
        a=self.e.arr
        if kind in ['cp','bank']:
            ids=ids[a[self.cp_col][ids]!=self.node]
            cp=a[self.cp_col][ids]
            return cp if kind=='cp' else self.e.bank[cp]
        if kind=='payment':return a['payment_format_cat'][ids]
        if kind=='currency':return a[self.curr_col][ids]
        unit={'hour':HOUR,'actual_hour':HOUR,'actual_day':DAY,'actual_minute':MINUTE}[kind]
        values=a['timestamp'][ids]//unit
        return values%24 if kind=='hour' else values

    def counts(self,kind,w):
        key=(kind,w);bounds=self.bounds(w)
        if key not in self.freq:self.freq[key]=((bounds[0],bounds[0]),Frequencies())
        previous,counts=self.freq[key]
        if previous!=bounds:
            for ids,sign in self.changes(previous,bounds):
                if len(ids):counts.update_counts(self.values(ids,kind),sign)
            self.freq[key]=(bounds,counts)
        return counts

    def last(self,kind='all',value=None):
        if kind=='all':return int(self.ts[-1]) if len(self.ts) else None
        oldstop,state=self.last_maps.get(kind,(0,None if kind=='ext' else {}))
        if oldstop!=self.stop:
            a=self.e.arr;ids=self.full_ids[oldstop:self.stop]
            ids=ids[a[self.cp_col][ids]!=self.node]
            if kind=='ext':
                if len(ids):state=int(a['timestamp'][ids[-1]])
            elif len(ids):
                vals=(a[self.cp_col][ids] if kind=='cp' else self.e.bank[a[self.cp_col][ids]] if kind=='bank' else a[self.curr_col][ids])
                keys,idx=np.unique(vals[::-1],return_index=True)
                state.update(zip(map(int,keys),map(int,a['timestamp'][ids[::-1][idx]])))
            self.last_maps[kind]=(self.stop,state)
        return state if kind=='ext' else state.get(int(value))

    def currency(self,curr,w):
        key=(int(curr),w);bounds=self.bounds(w)
        if key not in self.amounts:
            self.amounts[key]=[(bounds[0],bounds[0]),0,0,0,Frequencies()]
        stat=self.amounts[key];previous=stat[0]
        if previous!=bounds:
            a=self.e.arr
            for ids,sign in self.changes(previous,bounds):
                if not len(ids):continue
                ids=ids[(a[self.cp_col][ids]!=self.node)&(a[self.curr_col][ids]==curr)]
                amounts=a[self.amount_col][ids]
                stat[1]+=sign*len(ids);stat[2]+=sign*exact_sum(amounts)
                stat[3]+=sign*exact_sum(amounts[a['payment_format_cat'][ids]==self.e.ach])
                stat[4].update_counts(a[self.cp_col][ids],sign)
            stat[0]=bounds
        return dict(count=stat[1],sum=stat[2],ach_sum=stat[3],unique=len(stat[4]))

    def node_values(self):
        p='n_'+self.d+'_';r={}
        def put(k,v):r[p+k]=v
        c={w:self.bounds(w)[1]-self.bounds(w)[0] for w in ['1h','24h','72h']}
        for w,v in c.items():put('all_count_'+w,v)
        for w in ['24h','72h']:put('ext_count_'+w,self.counts('cp',w).total)
        cp=self.counts('cp','24h');banks=self.counts('bank','24h')
        put('unique_cp_24h',len(cp));put('cp_hhi_24h',cp.hhi())
        put('burst_1h_vs_23h',(c['1h']+1)/((c['24h']-c['1h'])/23+1))
        put('active_hours_24h',len(self.counts('actual_hour','24h')))
        put('active_days_72h',len(self.counts('actual_day','72h')))
        put('last_all_gap_min',self.gap())
        put('cross_bank_share_24h',ratio(banks.total-banks.get(int(self.e.bank[self.node]),0),banks.total))
        put('bank_unique_24h',len(banks));put('bank_hhi_24h',banks.hhi())
        ach=self.counts('payment','24h').get(self.e.ach,0)
        put('ach_count_24h',ach);put('ach_share_24h',ratio(ach,c['24h']))
        clock=self.counts('hour','72h')
        put('hour_entropy_72h',-sum((v/c['72h'])*math.log(v/c['72h']) for v in clock.values()) if c['72h'] else None)
        put('midday_share_72h',ratio(sum(v for k,v in clock.items() if 11<=k<17),c['72h']))
        for name in ['hour','minute']:
            buckets=self.counts('actual_'+name,'72h')
            put('peak_'+name+'_share_72h',ratio(max(buckets.values(),default=0),c['72h']))
        return r


class RollingEngine(FeatureEngine):
    """Bound lifetime by an account partition or explicit reset; never a full graph cache."""
    def at(self,t):
        t=int(t)
        if t<self.data_start or (self.coverage_end is not None and t>self.coverage_end):raise ValueError('Query outside history coverage')
        if self.cutoff is not None and t<self.cutoff:self.states.clear()
        if t!=self.cutoff:self.nodes.clear();self.cutoff=t
        return self

    def state(self,node,d):
        if self.cutoff is None:raise ValueError('Set cutoff before querying')
        key=(int(node),d)
        if key not in self.states:self.states[key]=RollingDirectionState(self,int(node),d,self.cutoff)
        elif self.states[key].t!=self.cutoff:self.states[key].advance(self.cutoff)
        return self.states[key]
