import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gnn_causal_structure_stats import analyze, holm_adjust, _rank_and_null_variance


def dataset(payers=8, days=(1, 8, 9), shared_receiver=False):
    rows, pairs, features = [], [], []
    eid = 0
    for day in days:
        for payer in range(payers):
            case = eid
            src = payer + 100
            rows.append(dict(edge_id=eid, label=1, day=day, src=src,
                             dst=9000 if shared_receiver else payer+1000, cohort='ACH',
                             paid_scaled=100, received_scaled=100, payment_format_cat=2,
                             payment_currency_cat=2, receiving_currency_cat=2))
            features.append(dict(edge_id=eid, motif=1., zero=0., full_window_72h=1))
            eid += 1
            for j in range(2):
                rows.append(dict(edge_id=eid, label=0, day=day, src=src,
                                 dst=payer+2000+j*payers, cohort='ACH',
                                 paid_scaled=100, received_scaled=100, payment_format_cat=2,
                                 payment_currency_cat=2, receiving_currency_cat=2))
                features.append(dict(edge_id=eid, motif=0., zero=0., full_window_72h=1))
                pairs.append(dict(case_edge_id=case, control_edge_id=eid, day=day, src=src, cohort='ACH'))
                eid += 1
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(features)


def run(t, p, f):
    return analyze(t, p, f, n_boot=200, n_perm=2000)


def result(report, feature='motif', period='discovery_sep01_07'):
    return next(r for r in report['results'] if r['cohort']=='ACH' and r['feature']==feature and r['period']==period)


def test_ties_and_null_variance():
    rank, variance = _rank_and_null_variance(1, np.array([1., 0., 2.]))
    assert rank == .5 and variance > 0
    assert _rank_and_null_variance(0, np.array([0., 0.])) == (.5, 0.)


def test_all_zero_is_neutral_and_safe_json():
    t, p, f = dataset()
    r = run(t, p, f)
    z = result(r, 'zero')
    assert z['conditional_auc'] == .5
    assert z['endpoint']['conservative_p'] == 1
    assert z['endpoint']['holm_p'] == 1
    assert z['positive_nonzero_count'] == 0
    assert not any(x['feature']=='zero' and x['promising'] for x in r['screening'])
    json.dumps(r, allow_nan=False)


def test_case_equal_weight_and_missing_whole_set():
    t, p, f = dataset(payers=2, days=(1,))
    p = p.drop(index=[3])
    f.loc[f.edge_id==3, 'motif'] = 0
    f.loc[f.edge_id==4, 'motif'] = 1
    r = result(run(t,p,f))
    assert r['conditional_auc'] == .5
    assert r['ncase']==2 and r['control_references']==3
    f.loc[f.edge_id==1, 'motif'] = np.nan
    r = result(run(t,p,f))
    assert r['ncase']==1 and r['conditional_auc']==0
    assert r['excluded_any_control_missing']==1


def test_shared_receiver_collapses_endpoint_inference():
    t,p,f=dataset(shared_receiver=True)
    report=run(t,p,f)
    r=result(report)
    assert r['payer']['clusters']==8
    assert r['endpoint']['clusters']==1
    assert r['endpoint']['ci95'] is None
    assert not report['promising']


def test_later_reversal_and_coverage_are_not_selected():
    t,p,f=dataset(payers=12)
    late_ids=t.loc[t.day==9,'edge_id']
    f.loc[f.edge_id.isin(late_ids),'motif']=1-f.loc[f.edge_id.isin(late_ids),'motif']
    f.loc[f.edge_id.isin(late_ids),'full_window_72h']=0
    report=run(t,p,f)
    assert not report['promising']
    r=result(report, period='later_sep09_10')
    assert r['conditional_auc']==0
    assert r['full_window_sensitivity']['ncase']==0
    item=next(s for s in report['screening'] if s['cohort']=='ACH' and s['feature']=='motif')
    assert item['discovery_direction']=='higher'
    assert 'later_effect_below_0.03' in item['reasons_not_promising']


def test_unmatched_positive_coverage_and_input_immutability():
    t,p,f=dataset(payers=2,days=(1,))
    p=p[p.case_edge_id != 3].copy()
    before=[x.copy(deep=True) for x in (t,p,f)]
    r=result(run(t,p,f))
    assert r['total_positive_inventory']==2 and r['unmatched_cases']==1
    for a,b in zip((t,p,f),before):pd.testing.assert_frame_equal(a,b)


@pytest.mark.parametrize('violation',['duplicate','label','day','src','self','currency','amount','inf','feature_label','feature_identity'])
def test_invalid_inputs_fail_closed(violation):
    t,p,f=dataset(payers=2,days=(1,))
    if violation=='duplicate':p=pd.concat([p,p.iloc[[0]]])
    elif violation=='label':t.loc[t.edge_id==1,'label']=1
    elif violation=='day':t.loc[t.edge_id==1,'day']=2
    elif violation=='src':p.loc[0,'src']=999
    elif violation=='self':t.loc[t.edge_id==1,'dst']=100
    elif violation=='currency':t.loc[t.edge_id==1,'receiving_currency_cat']=3
    elif violation=='amount':t.loc[t.edge_id==1,'paid_scaled']=201
    elif violation=='inf':f.loc[0,'motif']=np.inf
    elif violation=='feature_label':f['label']=0
    elif violation=='feature_identity':f=f.iloc[:-1]
    with pytest.raises(ValueError):run(t,p,f)


def test_holm_includes_unavailable_slots_and_restores_order():
    assert holm_adjust([.04,.001,None]) == pytest.approx([.08,.003,1.])


def test_significant_separation_can_pass_only_after_both_periods():
    t,p,f=dataset(payers=12)
    report=run(t,p,f)
    assert any(r['cohort']=='ACH' and r['feature']=='motif' for r in report['promising'])


def test_reused_control_is_counted_once_and_does_not_add_clusters():
    t,p,f=dataset(payers=1,days=(1,))
    new=t.iloc[[0]].copy();new['edge_id']=20
    t=pd.concat([t,new],ignore_index=True)
    f=pd.concat([f,pd.DataFrame([dict(edge_id=20,motif=1.,zero=0.,full_window_72h=1)])],ignore_index=True)
    extra=p.copy();extra['case_edge_id']=20
    p=pd.concat([p,extra],ignore_index=True)
    r=result(run(t,p,f))
    assert r['ncase']==2 and r['ncontrols_unique']==2 and r['control_references']==4
    assert r['endpoint']['clusters']==1


def test_cross_payer_control_connects_other_case_components():
    t,p,f=dataset(payers=2,days=(1,))
    t.loc[t.edge_id==1,'src']=101
    p['matching_level']='payer_bank'
    r=run(t,p,f)
    assert r['cross_payer_control_references']==1
    assert result(r)['payer']['clusters']==2
    assert result(r)['endpoint']['clusters']==1
    p['matching_level']='payer'
    with pytest.raises(ValueError,match='payer_bank'):run(t,p,f)


def test_context_is_not_tested_and_fixed_count_predicates_are_tested():
    t,p,f=dataset(payers=2,days=(1,))
    f['eligible']=1
    f['not_applicable']=0
    f['same_currency_target']=1
    f['parallel2_middle_accounts_72h']=[2,1,0,3,0,0]
    f.loc[f.edge_id==3,'eligible']=0
    report=run(t,p,f)
    assert not set(['eligible','not_applicable','same_currency_target','full_window_72h']) & set(report['feature_names'])
    assert 'parallel2_middle_accounts_72h__present' in report['feature_names']
    assert 'parallel2_middle_accounts_72h__at_least_2' in report['feature_names']
    r=result(report,'parallel2_middle_accounts_72h__at_least_2')
    assert r['ncase']==1 and r['excluded_not_applicable']==1
    assert r['conditional_auc']==1
    assert report['test_family_size_per_period']==len(report['feature_names'])*3
