import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import math
import numpy as np
import pytest
from test_gnn_features import fixture,ts
from gnn_features.rolling import RollingEngine
from gnn_features.partitioned import target_side

def equal(a,b):
    assert set(a)==set(b)
    for k,v in a.items():
        if isinstance(v,float):assert b[k]==pytest.approx(v,rel=1e-12,abs=1e-12),k
        else:assert b[k]==v,(k,v,b[k])

def test_incremental_windows_randomized_vs_original_all_fields():
    rng=np.random.default_rng(41)
    rows=[]
    for _ in range(280):
        day=int(rng.integers(1,9));h=int(rng.integers(0,24));m=int(rng.integers(0,4))*15
        rows.append((f'{day:02d}T{h:02d}:{m:02d}',int(rng.integers(0,6)),int(rng.integers(0,6)),int(rng.integers(0,100000)),int(rng.integers(2,5)),int(rng.integers(2,5)),int(rng.integers(2,5)),int(rng.integers(0,100000))))
    ref=fixture(rows);rolling=RollingEngine(ref.arr,ref.bank,ref.ach,ref.data_start)
    for i in np.argsort(ref.arr['timestamp'],kind='stable'):
        expected,ea=ref.target(i);actual,aa=rolling.target(i);equal(expected,actual);assert ea==aa
        for node in range(6):equal(ref.node(node),rolling.node(node))
        src,sa=target_side(rolling,i,'src');dst,da=target_side(rolling,i,'dst')
        assert not set(src)&set(dst);equal(expected,{**src,**dst})
        assert ea['missing_reasons']=={**sa['missing_reasons'],**da['missing_reasons']}
        assert ea['ratio_support_scaled']=={**sa['ratio_support_scaled'],**da['ratio_support_scaled']}

def test_midnight_expiration_inactivity_large_amounts_and_backwards_reset():
    ref=fixture([('01T00:00',0,1,8_000_000_000_000_000_000),('01T00:00',0,1,8_000_000_000_000_000_000),('01T23:59',0,1,1),('02T00:00',1,2,1),('08T00:00',1,2,1)])
    roll=RollingEngine(ref.arr,ref.bank,ref.ach,ref.data_start)
    for cutoff in [ts('01T00:00'),ts('01T00:00')+1,ts('01T23:59'),ts('02T00:00'),ts('02T00:00')+1,ts('08T00:00'),ts('01T23:59')]:
        ref.at(cutoff);roll.at(cutoff)
        for n in range(3):
            equal(ref.node(n),roll.node(n))
            for d in ['in','out']:
                for w in ['1h','6h','24h','72h','day']:
                    assert ref.state(n,d).currency(2,w)==roll.state(n,d).currency(2,w)

def test_tied_events_and_future_changes_do_not_leak():
    rows=[('04T08:00',0,1,2),('04T09:00',1,2,1),('04T09:00',3,1,999)]
    base=fixture(rows);changed=fixture(rows+[('04T09:00',0,1,999999),('05T09:00',0,1,999999)])
    engines=[RollingEngine(e.arr,e.bank,e.ach,e.data_start) for e in [base,changed]]
    for e in engines:e.at(ts('04T08:30'));e.node(1)
    equal(engines[0].target(1)[0],engines[1].target(1)[0]);equal(engines[0].node(1),engines[1].node(1))
