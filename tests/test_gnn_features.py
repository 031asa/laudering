"""Boundary and arithmetic acceptance for exact temporal features."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from decimal import Decimal
import math
import numpy as np
import pytest
from gnn_features.contract import HOUR, DAY, N_FIELDS, R_FIELDS, E_FIELDS, encode_numeric, manifest
from gnn_features.engine import FeatureEngine, DirectionState, make_orders, money, exact_sum
from gnn_features.build import edge_encoding

def ts(s): return int(np.datetime64('2022-09-'+s,'us').astype(np.int64))

def fixture(rows,bank=None):
    # Row: timestamp, payer, receiver, scaled amount, format, paid currency, received currency, received amount.
    n=max(max(r[1],r[2]) for r in rows)+1
    arrays={k:[] for k in ['timestamp','src','dst','paid_scaled','received_scaled','payment_format_cat','payment_currency_cat','receiving_currency_cat']}
    for row in rows:
        time,u,v,paid,*rest=row; method,pc,rc,received=(rest+[2,2,2,paid])[0:4] if rest else (2,2,2,paid)
        for k,val in zip(arrays,[ts(time),u,v,paid,received,method,pc,rc]): arrays[k].append(val)
    arrays={k:np.asarray(v,dtype=np.int64) for k,v in arrays.items()}
    make_orders(arrays,n)
    return FeatureEngine(arrays,np.asarray(bank if bank is not None else range(n)),2,ts('01T00:00'))

def test_hand_calculated_turnover_same_timestamp():
    e=fixture([('04T09:00',0,1,50_000_000_000),('04T09:05',2,1,30_000_000_000),
               ('04T09:10',1,3,79_000_000_000),('04T09:10',4,1,9_000_000_000_000)])
    r,a=e.target(2); n=e.node(1)
    assert n['n_in_all_count_1h']==2
    assert n['n_in_unique_cp_24h']==2
    assert n['n_in_cp_hhi_24h']==.5
    assert n['n_in_last_ext_gap_min']==5
    assert r['r_src_ext_in_sum_1h']==Decimal('80000.000000')
    assert r['r_src_paid_to_ext_in_sum_1h']==.9875
    assert r['r_src_cumulative_out_to_ext_in_sum_day']==.9875
    assert a['missing_reasons']['r_src_current_to_own_ext_mean_24h']=='no_history'

def test_exact_lower_bound_and_day_boundary():
    e=fixture([('03T00:30',0,1,1000000),('03T23:30',2,1,2000000),('04T00:00',3,1,3000000),('04T00:30',1,4,4000000)])
    r,_=e.target(3)
    assert r['r_src_ext_in_sum_24h']==Decimal('6')
    assert r['r_src_ext_in_sum_1h']==Decimal('5')
    assert r['r_src_ext_in_sum_day']==Decimal('3')
    e.at(ts('04T00:30')+1)
    assert e.node(1)['n_in_all_count_24h']==2

def test_self_transfer_and_all_preferences():
    e=fixture([('04T08:00',0,0,1000000),('04T09:00',0,0,2000000)])
    r,a=e.target(1); n=e.node(0)
    assert n['n_in_all_count_24h']==n['n_out_all_count_24h']==1
    assert n['n_in_ext_count_24h']==n['n_out_ext_count_24h']==0
    assert r['r_src_current_payment_share_24h']==1
    assert r['r_dst_current_currency_share_24h']==1
    for f in R_FIELDS:
        if f['scope']=='ext':
            assert r[f['name']] is None
            assert a['missing_reasons'][f['name']]=='not_applicable_self_transfer'
    assert len(e.neighbors(0,'in',10,np.random.default_rng(0)))==0

def test_currency_receipt_side_and_own_mean():
    e=fixture([('04T08:00',0,1,1000000,2,2,3,4000000),
               ('04T08:05',2,1,9000000,3,3,2,5000000),
               ('04T08:10',1,0,2000000,2,2,2,2000000),
               ('04T09:00',1,3,1000000,2,2,3,3000000)])
    r,_=e.target(3)
    assert r['r_src_ext_in_sum_24h']==Decimal('5')
    assert r['r_src_ext_out_sum_24h']==Decimal('2')
    assert r['r_src_paid_to_ext_in_sum_1h']==.2
    assert r['r_src_current_to_own_ext_mean_24h']==.5
    assert r['r_src_own_ext_ach_amount_share_24h']==1
    assert r['r_src_last_same_currency_ext_in_gap_min']==55

def test_no_history_zero_amount_and_unseen_category_are_distinct():
    e=fixture([('04T08:00',0,1,0),('04T09:00',1,2,1000000,3,2,2,1000000)])
    r,a=e.target(1)
    assert r['r_src_ext_in_count_24h']==1
    assert r['r_src_paid_to_ext_in_sum_24h'] is None
    assert a['missing_reasons']['r_src_paid_to_ext_in_sum_24h']=='zero_denominator'
    assert a['missing_reasons']['r_dst_current_payment_share_24h']=='no_history'
    # Incoming history exists but the new payment method did not occur.
    e.arr['dst'][1]=1; e.arr['src'][1]=2
    make_orders(e.arr,3);e.states.clear(); e.nodes.clear()
    r,_=e.target(1)
    assert r['r_dst_current_payment_share_24h']==0

def test_hhi_and_clock_vs_actual_buckets():
    rows=[('02T11:00',0,2,1)]*4+[('03T11:00',0,2,1)]*4+[('03T17:00',1,2,1)]*2+[('04T10:00',2,3,1)]
    e=fixture(rows);e.at(ts('04T10:00'));n=e.node(2)
    assert n['n_in_cp_hhi_24h']==pytest.approx((4**2+2**2)/6**2)
    assert n['n_in_midday_share_72h']==.8
    assert n['n_in_peak_hour_share_72h']==.4
    assert n['n_in_hour_entropy_72h']==pytest.approx(-.8*math.log(.8)-.2*math.log(.2))
    e.at(ts('03T18:00'));assert e.node(2)['n_in_cp_hhi_24h']==pytest.approx(20/36)
    h=fixture([('04T08:00',0,2,1)]*8+[('04T08:10',1,2,1)]*2)
    h.at(ts('04T09:00'));assert h.node(2)['n_in_cp_hhi_24h']==.68

def test_inactive_expiration_and_last_event_beyond_72h():
    e=fixture([('01T08:00',0,1,1),('05T09:00',1,2,1)])
    e.at(ts('01T09:00'));assert e.node(1)['n_in_all_count_1h']==1
    r,_=e.target(1);n=e.node(1)
    assert n['n_in_all_count_72h']==0
    assert n['n_in_last_ext_gap_min']==97*60
    assert r['r_src_last_same_currency_ext_in_gap_min']==97*60

def test_permutation_future_invariance_and_target_independence():
    rows=[('04T08:00',0,1,1),('04T08:00',2,1,3),('04T09:00',1,3,2)]
    e=fixture(rows);r,_=e.target(2);n=e.node(1).copy()
    other=fixture([rows[1],rows[0],rows[2],('04T09:00',4,1,999),('05T08:00',4,1,999)])
    assert other.target(2)[0]==r and other.node(1)==n
    e.arr['paid_scaled'][2]=99;e.arr['payment_currency_cat'][2]=9
    changed,_=e.target(2)
    assert e.node(1)==n and changed!=r

def test_parallel_edges_graph_cutoffs_and_isolated_nodes():
    e=fixture([('01T09:00',0,1,1),('04T08:00',0,1,1),('04T08:00',0,1,1),('04T09:00',1,2,1)])
    e.at(ts('04T09:00')); ids=e.neighbors(1,'in',10,np.random.default_rng(1))
    assert ids.tolist()==[0,1,2]  # Exact 72h lower bound retained; parallel IDs retained.
    assert all(e.arr['timestamp'][ids]<e.cutoff)
    assert e.node(2)['n_in_all_count_72h']==0
    e.at(ts('04T09:00')+1);assert e.neighbors(1,'in',10,np.random.default_rng(1)).tolist()==[1,2]

def test_contract_encoding_context_and_exact_money():
    e=fixture([('01T00:00',0,1,0)]);r,_=e.target(0);n=e.node(0)
    assert len(n)==41 and len(r)==54 and len(E_FIELDS)==12
    assert not e.context()['full_72h'] and e.context()['full_day']
    assert all(math.isfinite(v) for v in encode_numeric(n,N_FIELDS)+encode_numeric(r,R_FIELDS))
    assert money(exact_sum(np.array([8_000_000_000_000_000_000]*3,dtype=np.int64)))==Decimal('24000000000000.000000')
    enc=edge_encoding(e.arr,e.bank)
    assert enc['weekday_cat'][0]==3 and enc['is_self_transfer'][0]==False
    with pytest.raises(ValueError):FeatureEngine({'label':np.array([0])},e.bank,2,0)
    assert len(manifest()['fields'])==107

def test_randomized_index_vs_full_scan_reference():
    rng=np.random.default_rng(31);rows=[]
    for _ in range(180):
        day=int(rng.integers(1,6)); hour=int(rng.integers(0,24));minute=int(rng.integers(0,4))*15
        rows.append((f'{day:02d}T{hour:02d}:{minute:02d}',int(rng.integers(0,9)),int(rng.integers(0,9)),int(rng.integers(0,100000)),int(rng.integers(2,5)),int(rng.integers(2,5)),int(rng.integers(2,5)),int(rng.integers(0,100000))))
    fast=fixture(rows);slow=fixture(rows)
    class ScanState(DirectionState):
        def __init__(self,engine,node,direction,cutoff):
            self.e,self.node,self.d,self.t=engine,node,direction,cutoff
            a=engine.arr;col='dst' if direction=='in' else 'src'
            # Independent full-history selection, without reading any time index.
            self.ids=np.array(sorted([i for i in range(len(a['timestamp'])) if a[col][i]==node and a['timestamp'][i]<cutoff],key=lambda i:(a['timestamp'][i],i)),dtype=np.int64)
            self.ts=a['timestamp'][self.ids];self.cp_col='src' if direction=='in' else 'dst'
            self.curr_col='receiving_currency_cat' if direction=='in' else 'payment_currency_cat'
            self.amount_col='received_scaled' if direction=='in' else 'paid_scaled';self.cache={}
        def window(self,w,external=False):
            from gnn_features.contract import WINDOWS
            lower=self.t-self.t%DAY if w=='day' else self.t-WINDOWS[w]
            a=self.e.arr
            return np.array([i for i in self.ids if a['timestamp'][i]>=lower and (not external or a['src'][i]!=a['dst'][i])],dtype=np.int64)
    def slowstate(node,d):
        key=(int(node),d)
        if key not in slow.states:slow.states[key]=ScanState(slow,int(node),d,slow.cutoff)
        return slow.states[key]
    slow.state=slowstate
    for i in rng.choice(len(rows),40,replace=False):
        assert fast.target(i)==slow.target(i)
        for node in range(9): assert fast.node(node)==slow.node(node)

def test_sample_interface_reverse_route_and_single_class_labels_not_required():
    from gnn_features.run import sample_batch
    e=fixture([('04T08:00',0,1,1),('04T08:00',0,1,1),('04T09:00',1,2,1)])
    tensors,report=sample_batch(e,np.array([2],dtype=np.uint32))
    assert report['parallel_account_pairs']==1
    assert np.array_equal(tensors['edge_index_out'],tensors['edge_index_in'][::-1])
    assert tensors['edge_id'].tolist()==[0,1]
    assert 2 in tensors['node_id']
    assert 'y' not in tensors
    assert tensors['target_attr'].shape==(1,110)
    assert tensors['edge_attr_num'].shape==(2,9)

def test_coverage_guard_and_raw_bank_identity_separate_from_model_oov():
    e=fixture([('04T08:00',0,2,1),('04T08:10',1,2,1),('04T09:00',2,3,1)],bank=[101,102,103,104])
    e.coverage_end=ts('04T09:00')
    e.at(ts('04T09:00'));assert e.node(2)['n_in_bank_unique_24h']==2
    # Bank identity codes are independent of a model vocabulary's shared UNK code.
    assert e.node(2)['n_in_bank_hhi_24h']==.5
    with pytest.raises(ValueError):e.at(ts('04T09:01'))
