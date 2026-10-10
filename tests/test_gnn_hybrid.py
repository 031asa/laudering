import numpy as np
import pytest
from test_gnn_features import fixture, ts
from gnn_features.contract import HOUR, N_FIELDS, encode_numeric
from gnn_features.rolling import RollingEngine
from gnn_features.hybrid_cache_trial import pack_direction, bulk_direction, complex_values, encode_columns, SIMPLE, COMPLEX


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bulk_and_split_match_exact_engine_at_boundaries(device):
    if device=='cuda':
        torch=pytest.importorskip('torch')
        if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    ref=fixture([('01T00:00',0,0,1),('01T12:00',0,1,1),
                 ('03T12:00',1,0,1),('04T11:00',1,0,1),('04T11:00',1,0,1,3),
                 ('04T12:00',0,1,1),('04T12:00',1,0,1),('08T00:00',2,2,1)],bank=[0,1,1])
    cuts=[ts('01T00:00'),ts('04T12:00'),ts('04T12:00')+1,ts('07T23:00')]
    keys=np.array([(n,t) for n in range(3) for t in cuts],dtype=np.int64)
    origin=ref.data_start-72*HOUR;end=ts('08T00:00')
    raw={}
    for d in ['in','out']:
        pack=pack_direction(ref.arr,ref.bank,keys,d,origin,end,ref.ach)
        raw.update(bulk_direction(pack,keys,d,origin,end,device))
    assert len(SIMPLE)==23 and len(COMPLEX)==18 and not set(SIMPLE)&set(COMPLEX)
    for n in COMPLEX:raw[n]=np.empty(len(keys))
    expected=[]
    roll=RollingEngine(ref.arr,ref.bank,ref.ach,ref.data_start)
    for i,(node,t) in enumerate(keys):
        ref.at(t);expected_raw=ref.node(node);roll.at(t)
        complex_raw=complex_values(roll,node)
        for n in COMPLEX:raw[n][i]=np.nan if complex_raw[n] is None else complex_raw[n]
        for n in SIMPLE:
            actual=raw[n][i];v=expected_raw[n]
            if v is None:assert np.isnan(actual),n
            else:assert actual==pytest.approx(v,rel=1e-12,abs=1e-12),n
        expected.append(encode_numeric(expected_raw,N_FIELDS)+[float(ref.context()['full_'+w]) for w in ['1h','24h','72h']])
    np.testing.assert_allclose(encode_columns(raw,keys,ref.data_start),np.array(expected,dtype=np.float32),rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_entirely_empty_packed_history(device):
    if device=='cuda':
        torch=pytest.importorskip('torch')
        if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    ref=fixture([('04T12:00',0,0,1)])
    keys=np.array([[0,ts('01T00:00')]],dtype=np.int64)
    origin=ref.data_start-72*HOUR;end=ts('02T00:00')
    pack=pack_direction(ref.arr,ref.bank,keys,'in',origin,end,ref.ach)
    raw=bulk_direction(pack,keys,'in',origin,end,device)
    assert raw['n_in_all_count_24h'][0]==0
    assert np.isnan(raw['n_in_last_ext_gap_min'][0])
    assert np.isnan(raw['n_in_ach_share_24h'][0])
