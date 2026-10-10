import importlib.util
from pathlib import Path
import numpy as np
import sys
import itertools

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_non_ach_repeated_payments import prior_counts
from verify_non_ach_differences import pooled_rank_moments

spec = importlib.util.spec_from_file_location('diffs', Path(__file__).resolve().parents[1]/'scripts/analyze_non_ach_differences.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_ties_count_as_half():
    np.testing.assert_allclose(m.conditional_rank([1,2,3],[1,2,2]),[1/6,2/3,1])


def test_matching_removes_between_group_difference_and_excludes_missing():
    values=np.array([100.,100.,0.,0.,999.])
    valid=np.array([1,1,1,1,0],bool)
    groups=[(1,0,np.array([0,4]),np.array([1])),(2,1,np.array([2]),np.array([3]))]
    signs=np.ones((1,15)); draws=np.ones((2,15))
    r=m.analyze(values,valid,groups,signs,draws,np.array([0,2,4]))
    assert r['periods']['all_train']['conditional_auc']==.5
    assert r['periods']['all_train']['matched_positives']==2
    assert r['valid_positive_rows']==2


def test_bh_preserves_input_order():
    np.testing.assert_allclose(m.bh([.04,.001,.03]),[.04,.003,.04])


def test_complete_separation_and_date_scope():
    values=np.array([2.,1.,0.,1.])
    groups=[(1,0,np.array([0]),np.array([1])),(9,0,np.array([2]),np.array([3]))]
    r=m.analyze(values,np.ones(4,bool),groups,np.ones((1,15)),np.ones((2,15)),np.array([0,2]))
    assert r['periods']['discovery_sep01_07']['conditional_auc']==1
    assert r['periods']['later_sep09_10']['conditional_auc']==0
    assert r['periods']['all_train']['conditional_auc']==.5


def test_prior_counts_excludes_every_same_timestamp_and_separates_keys():
    group=np.array([1,0,0,0,1,0])
    timestamp=np.array([1,1,1,2,3,3])
    label=np.array([1,0,1,0,0,1])
    n,p=prior_counts(group,timestamp,label)
    np.testing.assert_array_equal(n,[0,0,0,2,1,3])
    np.testing.assert_array_equal(p,[0,0,0,1,1,1])


def test_rank_variance_matches_complete_label_enumeration_with_ties():
    all_values=np.array([0.,1.,1.,3.,5.])
    observed,variance=pooled_rank_moments(all_values[:2],all_values[2:])
    trials=[]
    for positive in itertools.combinations(range(5),2):
        negative=[j for j in range(5) if j not in positive]
        value,v=pooled_rank_moments(all_values[list(positive)],all_values[negative])
        assert abs(v-variance)<1e-14
        trials.append(value)
    assert abs(np.mean(trials)-1.)<1e-14
    assert abs(np.var(trials)-variance)<1e-14
    assert abs(observed-m.conditional_rank(all_values[:2],all_values[2:]).sum())<1e-14
