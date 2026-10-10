"""Versioned feature contract; raw values and tensor transformations are separate."""
import math

VERSION = 'gnn-v1-query-1'
HOUR = 3_600_000_000
DAY = 24 * HOUR
WINDOWS = {'1h': HOUR, '6h': 6*HOUR, '24h': DAY, '72h': 3*DAY}
N_FIELDS = []
E_FIELDS = []
R_FIELDS = []

def add(dest, name, source, formula, nullable=False, scope='all', currency=None,
        transform=None, direction=None, window=None, dependencies=None):
    if transform is None:
        transform = ('entropy_ln24' if 'entropy' in name else
                     'identity' if any(s in name for s in ['share', 'hhi', 'sin', 'cos', 'is_']) else 'log1p')
    dest.append(dict(name=name, source=source, formula=formula, nullable=nullable,
                     scope=scope, currency=currency, transform=transform,
                     direction=direction, window=window, dependencies=dependencies or ['label-free account history'],
                     missing_rule='NULL with audit reason; model fill 0 plus valid mask' if nullable else 'observed zero is valid'))

for d in ['in', 'out']:
    def n(s, f, formula, null=False, scope='all', window=None):
        add(N_FIELDS, f'n_{d}_{s}', f, formula, null, scope, direction=d, window=window)
    for w in ['1h','24h','72h']: n(f'all_count_{w}', 'F01', 'count(all)', window=w)
    for w in ['24h','72h']: n(f'ext_count_{w}', 'F01', 'count(external)', scope='ext', window=w)
    n('unique_cp_24h','F02','count distinct external counterparties', scope='ext',window='24h')
    n('cp_hhi_24h','F03','sum(counterparty_count^2)/external_count^2',True,'ext','24h')
    n('burst_1h_vs_23h','F05','(C1+1)/((C24-C1)/23+1)',window='1h/24h')
    n('active_hours_24h','F17','count distinct date-hour buckets',window='24h')
    n('active_days_72h','F17','count distinct dates',window='72h')
    n('last_all_gap_min','F18','(cutoff-last preceding event)/minute',True,window='all prior')
    n('cross_bank_share_24h','F11','external different-bank count/external count',True,'ext','24h')
    n('bank_unique_24h','F11/F52','count distinct external counterparty banks',scope='ext',window='24h')
    n('bank_hhi_24h','F52','sum(bank_count^2)/external_count^2',True,'ext','24h')
    n('ach_count_24h','F49','all ACH count',window='24h')
    n('ach_share_24h','F49','all ACH count/all count',True,window='24h')
    n('hour_entropy_72h','F19','-sum(p_hour*ln(p_hour)); merge dates by clock hour',True,window='72h')
    n('midday_share_72h','F19','count(clock hour >=11 and <17)/all count',True,window='72h')
    n('peak_hour_share_72h','F19','max(actual date-hour count)/all count',True,window='72h')
    n('peak_minute_share_72h','F19','max(actual date-minute count)/all count',True,window='72h')
add(N_FIELDS,'n_in_last_ext_gap_min','F07','gap to last external inflow, any currency',True,'ext',direction='in',window='all prior')

for name in ['paid_log1p','received_log1p','hour_sin','hour_cos','is_midday','is_cross_bank','is_cross_currency','is_self_transfer']:
    add(E_FIELDS,name,'B04' if 'log1p' in name else 'B03/B05/B06/B07',name+' from current raw edge',transform='identity',dependencies=['raw transaction, no label'])
for name in ['payment_format_cat','payment_currency_cat','receiving_currency_cat','weekday_cat']:
    add(E_FIELDS,name,'B01/B03/B05','train vocabulary; weekday Monday=0',transform='embedding',dependencies=['raw transaction, train-only vocabulary'])

def r(name, source, formula, window, direction=None, currency=None, scope='ext', nullable=False, transform=None):
    add(R_FIELDS,name,source,formula,nullable,scope,currency,transform,direction,window)
for orient,w in [('fwd','24h'),('fwd','72h'),('rev','24h')]:
    r(f'r_pair_{orient}_count_{w}','F06/F12/F35','directed external pair count',w,orient,nullable=True)
for orient in ['fwd','rev']:
    r(f'r_pair_{orient}_last_gap_min','F27/F35','gap to last preceding directed external pair event','all prior',orient,nullable=True)
for s,d in [('src','out'),('dst','in')]:
    r(f'r_pair_{s}_{d}_share_24h','F35','forward pair count / endpoint external direction count','24h',d,nullable=True)
for s,own in [('src','out'),('dst','in')]:
    currency='current payment_currency' if s=='src' else 'current receiving_currency'
    for name,formula in [('ext_in_sum','external inflow sum'),('ext_out_sum','external outflow sum'),('ext_in_count','external inflow count'),('ext_out_count','external outflow count'),('ext_out_in_ratio','O/I if I>0'),('current_to_own_ext_mean','current amount * own count/own sum if own sum>0'),('own_ext_ach_amount_share','own ACH amount/own sum if own sum>0')]:
        r(f'r_{s}_{name}_24h','F08/F09/F49',formula,'24h',own,currency,nullable=True)
for w in ['1h','6h','24h','72h','day']:
    if w!='24h':
        for kind in ['sum','count']:
            r(f'r_src_ext_in_{kind}_{w}','F22','same-currency external inflow '+kind,w,'in','current payment_currency',nullable=True)
    r(f'r_src_paid_to_ext_in_sum_{w}','F22','current paid / same-currency inflow sum if sum>0',w,'in','current payment_currency',nullable=True)
for w in ['1h','24h','day']:
    r(f'r_src_ext_unique_payers_{w}','F25','count distinct same-currency external payers',w,'in','current payment_currency',nullable=True)
for kind in ['sum','count']:
    r(f'r_src_ext_out_{kind}_day','F22','same-currency external outflow '+kind,'day','out','current payment_currency',nullable=True)
for w in ['24h','day']:
    r(f'r_src_cumulative_out_to_ext_in_sum_{w}','F22','(historical outflow + current paid)/historical inflow if inflow>0',w,'out','current payment_currency',nullable=True)
r('r_src_last_same_currency_ext_in_gap_min','F07','gap to last same-currency external inflow','all prior','in','current payment_currency',nullable=True)
for s,d in [('src','out'),('dst','in')]:
    for key,w in [('payment','24h'),('currency','24h'),('hour','72h'),('bank','24h'),('bank','72h')]:
        r(f'r_{s}_current_{key}_share_{w}','F37/F19/F51','matching current '+key+' count / historical direction count',w,d,'current account-side currency' if key=='currency' else None,scope='ext' if key=='bank' else 'all',nullable=True)
    r(f'r_{s}_current_bank_last_gap_min','F53','gap to last external same-counterparty-bank event','all prior',d,nullable=True)

assert len(N_FIELDS)==41 and len(E_FIELDS)==12 and len(R_FIELDS)==54
assert len({x['name'] for x in N_FIELDS+E_FIELDS+R_FIELDS})==107

def manifest():
    groups = {'N':N_FIELDS,'E':E_FIELDS,'R':R_FIELDS}
    return dict(version=VERSION, time_unit='microseconds; timezone unspecified',
                history='[t-window,t); all same-timestamp events excluded',
                amount='decimal(38,6); exact scaled integers internally; no mixed-currency sums',
                self_transfer='N/all retained; R/ext NULL not_applicable; no self message edges',
                node_category={'name':'bank_cat','source':'B08','transform':'embedding','vocabulary':'train only, 0=missing 1=UNK'},
                context={'N':['full_1h','full_24h','full_72h'],'R':['full_6h','full_day'], 'audit':'coverage durations from data start, not account first seen'},
                fields=[dict(f,group=g,position=i) for g,fs in groups.items() for i,f in enumerate(fs)],
                cache='per-batch (node_id, cutoff_us); no full snapshot materialization')

def encode_numeric(values, fields):
    """Masks are appended in manifest order, only for nullable numeric fields."""
    encoded=[]; masks=[]
    for f in fields:
        if f['transform']=='embedding': continue
        v=values[f['name']]
        if v is None and not f['nullable']: raise ValueError('Required value missing: '+f['name'])
        x=0.0 if v is None else float(v)
        if f['transform']=='log1p': x=math.log1p(x)
        elif f['transform']=='entropy_ln24': x/=math.log(24)
        if not math.isfinite(x): raise ValueError('Nonfinite value: '+f['name'])
        encoded.append(x)
        if f['nullable']: masks.append(float(v is not None))
    return encoded+masks
