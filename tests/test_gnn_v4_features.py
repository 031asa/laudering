"""Independent enumeration checks the optimized three-motif production engine."""
import numpy as np
import pytest
from test_gnn_causal_structures import arrays, e
from gnn_causal_structures import CausalStructures, amount_match
from gnn_features.v4_features import ThreeStructures, NAMES, encode_extra, contract


def brute(rows, target, window=100):
    u,v,t,paid,received,c,rc=rows[target]
    if u==v:return [-1,-1,-1]
    hist=[x for x in rows if t-window<=x[2]<t and x[0]!=x[1] and x[5]==x[6]==c]
    reverse=sum(x[0]==v and x[1]==u for x in hist) if c==rc else -1
    triangle=0 if c==rc else -1;chain=0
    for first in hist:
        for second in hist:
            if first[2]>=second[2] or first[1]!=second[0] or second[1]!=u:continue
            w=second[0]
            if w in (u,v):continue
            if c==rc and first[0]==v:triangle+=1
            if len({first[0],w,u,v})==4 and amount_match(first[4],second[3]) and amount_match(second[4],paid):chain+=1
    return [reverse,triangle,chain]


@pytest.mark.parametrize('seed',range(24))
def test_full_enumeration_ties_currencies_repeated_accounts(seed):
    rng=np.random.default_rng(seed)
    rows=[e(int(rng.integers(0,6)),int(rng.integers(0,6)),int(rng.choice([99,100,120,160,199,200,201])),
            amount=int(rng.choice([0,90,100,110,111])),received=int(rng.choice([0,90,100,110,111])),
            pc=int(rng.integers(1,3)),rc=int(rng.integers(1,3))) for _ in range(70)]
    rows += [e(2,3,200,amount=100)]
    a=arrays(rows);engine=ThreeStructures(a,window_us=100,max_cache_bytes=1024)
    for target in range(len(rows)):
        np.testing.assert_array_equal(engine.counts(target),brute(rows,target))
    assert engine.cache_bytes<=1024


@pytest.mark.parametrize('value',[0,100,8_000_000_000_000_000_000,np.iinfo(np.int64).max])
def test_amount_integer_precision_and_legacy_match(value):
    rows=[e(0,1,100,amount=value),e(1,2,150,amount=value),e(3,2,160,amount=value),e(3,1,110,amount=value),e(2,3,200,amount=value)]
    a=arrays(rows)
    raw=ThreeStructures(a,window_us=100).counts(4)
    np.testing.assert_array_equal(raw,brute(rows,4))
    legacy=CausalStructures(a,window_us=100).extract(4)['features']
    np.testing.assert_array_equal(raw,[legacy[n] for n in NAMES])


def test_empty_vs_missing_cross_currency_and_coverage_contract():
    a=arrays([e(2,3,100),e(2,3,200,pc=1,rc=2),e(2,2,200)])
    eng=ThreeStructures(a,window_us=100)
    np.testing.assert_array_equal(eng.counts(0),[0,0,0])
    np.testing.assert_array_equal(eng.counts(1),[-1,-1,0])
    np.testing.assert_array_equal(eng.counts(2),[-1,-1,-1])
    encoded=encode_extra(np.array([[0,1,-1]],np.int64))
    np.testing.assert_array_equal(encoded[:,3:],[[1,1,0]])
    assert encoded[0,2]==0 and 'data_start_us' in contract()['coverage']


def test_never_reads_labels_or_future_and_no_path_cap():
    rows=[e(0,1,110)]*90+[e(1,2,150)]*80+[e(2,3,200)]
    a=arrays(rows)
    class Poison(dict):
        def __getitem__(self,k):
            if 'label' in k:raise AssertionError('Label accessed')
            return super().__getitem__(k)
    raw=ThreeStructures(Poison(a),window_us=100).counts(len(rows)-1)
    assert raw[2]==7200
    after=arrays(rows+[e(3,2,200),e(3,2,201),e(4,5,999,amount=0)])
    np.testing.assert_array_equal(raw,ThreeStructures(after,window_us=100).counts(len(rows)-1))


@pytest.mark.parametrize('raw',[np.zeros((2,2),np.int64),np.zeros((2,3),np.float64),np.full((2,3),-2,np.int64)])
def test_bad_counts_rejected(raw):
    with pytest.raises(ValueError):encode_extra(raw)
