"""Account-local R fragments. Each node belongs to one partition for a run."""
from .engine import money, ratio
from .contract import R_FIELDS, HOUR


def target_side(engine,i,side,n=None):
    """Everything in one fragment depends only on this endpoint's complete history."""
    a=engine.arr;i=int(i);t=int(a['timestamp'][i]);engine.at(t)
    u,v=int(a['src'][i]),int(a['dst'][i]);self_edge=u==v
    node,other,own=(u,v,'out') if side=='src' else (v,u,'in')
    curr=int(a['payment_currency_cat' if side=='src' else 'receiving_currency_cat'][i])
    amount=int(a['paid_scaled' if side=='src' else 'received_scaled'][i])
    if n is None:n=engine.node(node)
    ins=engine.state(node,'in');outs=engine.state(node,'out');state=outs if own=='out' else ins
    r={};reasons={};support={}
    def put(k,val,reason='no_history'):
        r[k]=val
        if val is None:reasons[k]=reason
    def frac(k,num,den,count):
        put(k,ratio(num,den),'no_history' if not count else 'zero_denominator')
        support[k]=dict(numerator=num,denominator=den,count=count)
    # Reserve each output field to exactly one endpoint, including self-transaction NULLs.
    owned=[f['name'] for f in R_FIELDS if (f['name'].startswith('r_dst_') or f['name'].startswith('r_pair_rev_') or f['name']=='r_pair_dst_in_share_24h')==(side=='dst')]
    if not self_edge:
        fwd=state.counts('cp','24h').get(other,0)
        if side=='src':
            put('r_pair_fwd_count_24h',fwd)
            put('r_pair_fwd_count_72h',outs.counts('cp','72h').get(v,0))
            put('r_pair_fwd_last_gap_min',outs.gap('cp',v))
        else:
            put('r_pair_rev_count_24h',outs.counts('cp','24h').get(u,0))
            put('r_pair_rev_last_gap_min',outs.gap('cp',u))
        den=n[f'n_{own}_ext_count_24h']
        frac(f'r_pair_{side}_{own}_share_24h',fwd,den,den)
        incoming=ins.currency(curr,'24h');outgoing=outs.currency(curr,'24h')
        p=f'r_{side}_'
        for d,stat in [('in',incoming),('out',outgoing)]:
            put(p+f'ext_{d}_sum_24h',money(stat['sum']));put(p+f'ext_{d}_count_24h',stat['count'])
        frac(p+'ext_out_in_ratio_24h',outgoing['sum'],incoming['sum'],incoming['count'])
        ownstat=outgoing if side=='src' else incoming
        frac(p+'current_to_own_ext_mean_24h',amount*ownstat['count'],ownstat['sum'],ownstat['count'])
        frac(p+'own_ext_ach_amount_share_24h',ownstat['ach_sum'],ownstat['sum'],ownstat['count'])
        if side=='src':
            for w in ['1h','6h','24h','72h','day']:
                stat=ins.currency(curr,w)
                if w!='24h':
                    put(f'r_src_ext_in_sum_{w}',money(stat['sum']));put(f'r_src_ext_in_count_{w}',stat['count'])
                frac(f'r_src_paid_to_ext_in_sum_{w}',amount,stat['sum'],stat['count'])
                if w in ['1h','24h','day']:put(f'r_src_ext_unique_payers_{w}',stat['unique'])
                if w in ['24h','day']:
                    out=outs.currency(curr,w)
                    frac(f'r_src_cumulative_out_to_ext_in_sum_{w}',out['sum']+amount,stat['sum'],stat['count'])
                    if w=='day':put('r_src_ext_out_sum_day',money(out['sum']));put('r_src_ext_out_count_day',out['count'])
            put('r_src_last_same_currency_ext_in_gap_min',ins.gap('currency',curr))
    for kind,w,value in [('payment','24h',int(a['payment_format_cat'][i])),('currency','24h',curr),('hour','72h',t//HOUR%24),('bank','24h',int(engine.bank[other])),('bank','72h',int(engine.bank[other]))]:
        if kind=='bank' and self_edge:continue
        counts=state.counts(kind,w);den=n[f'n_{own}_{"ext" if kind=="bank" else "all"}_count_{w}']
        frac(f'r_{side}_current_{kind}_share_{w}',counts.get(value,0),den,den)
    if not self_edge:put(f'r_{side}_current_bank_last_gap_min',state.gap('bank',int(engine.bank[other])))
    if self_edge:
        for name in owned:
            if name not in r:put(name,None,'not_applicable_self_transfer')
    if set(r)!=set(owned):raise AssertionError('Incomplete or overlapping side fields')
    return r,dict(missing_reasons=reasons,ratio_support_scaled=support)
