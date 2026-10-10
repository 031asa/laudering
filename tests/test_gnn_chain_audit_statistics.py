import copy

import numpy as np
import pandas as pd
import pytest

from gnn_chain_audit_statistics import (PRIMARY_FEATURES, dependency_clusters, _fisher_two_sided,
    describe_ach_reference, holm_adjust, match_controls, run_primary_statistics)


def row(edge, payer, label=0, **kw):
    values = dict(edge_id=edge, src=payer, label=label, date="2022-09-11",
                  payment_format="ACH", payment_currency="USD", receiving_currency="USD",
                  amount_paid="100.000000", v2_src_count_7d=6., v2_dst_count_7d=9.,
                  r_pair_fwd_count_72h=0., v2_dst_current_to_q90_7d=2., from_bank="1",
                  outcome_lr_only="FN" if label else "TN", uniform_negative_control=not bool(label))
    values.update(kw)
    return values


def test_matching_same_payer_priority_then_distance_and_three_limit():
    data = pd.DataFrame([row(1, 10, 1), row(2, 10, amount_paid="180"),
                         row(3, 20), row(4, 30, amount_paid="110"), row(5, 40, amount_paid="120")])
    result = match_controls(data, [1], "ACH_FN")
    p = result["pairs"]
    assert p.control_edge_id.tolist() == [2, 3, 4]
    assert p.same_payer.tolist() == [True, False, False]
    assert p.control_weight.tolist() == [1/3] * 3
    assert result["balance"]["exact_covariate_mismatches"] == 0
    assert result["summary"]["dependency"]["clusters"] == 1


@pytest.mark.parametrize("change", [dict(date="2022-09-12"), dict(payment_format="Cash"),
    dict(payment_currency="EUR"), dict(receiving_currency="EUR"), dict(v2_src_count_7d=3),
    dict(v2_dst_count_7d=0), dict(r_pair_fwd_count_72h=1), dict(v2_dst_current_to_q90_7d=1),
    dict(amount_paid="49.9999"), dict(amount_paid="200.0001"), dict(uniform_negative_control=False)])
def test_control_constraints_are_hard(change):
    result = match_controls(pd.DataFrame([row(1, 10, 1), row(2, 10, **change)]), [1], "ACH_FN")
    assert result["pairs"].empty
    assert result["summary"]["unmatched_cases"] == 1


def test_same_date_not_timestamp_exact_and_inclusive_caliper():
    data = pd.DataFrame([row(1, 10, 1, timestamp="2022-09-11 01:00:00"),
                         row(2, 10, amount_paid="50", timestamp="2022-09-11 23:00:00"),
                         row(3, 10, amount_paid="200")])
    result = match_controls(data, [1], "ACH_FN")
    assert set(result["pairs"].control_edge_id) == {2, 3}
    # Controls can be later in the same date; their features must use their own strict cutoff.


def test_g_requires_same_payer_and_does_not_require_ach_condition():
    data = pd.DataFrame([row(1, 70, 1, from_bank="70", payment_format="Cash", r_pair_fwd_count_72h=9),
                         row(2, 71, from_bank="70", payment_format="Cash"),
                         row(3, 70, from_bank="70", payment_format="Cash", v2_dst_current_to_q90_7d=.5)])
    result = match_controls(data, [1], "G_FN")
    assert result["pairs"].control_edge_id.tolist() == [3]
    assert "15" in result["summary"]["g_independence_warning"]


def test_zero_amount_matches_only_zero():
    data = pd.DataFrame([row(1, 10, 1, amount_paid="0"), row(2, 10, amount_paid="0"), row(3, 10, amount_paid=".01")])
    assert match_controls(data, [1], "ACH_FN")["pairs"].control_edge_id.tolist() == [2]


def test_missing_support_excludes_case_and_missing_ratio_is_not_zero_or_one():
    data = pd.DataFrame([row(1, 10, 1, v2_src_count_7d=np.nan), row(2, 10)])
    result = match_controls(data, [1], "ACH_FN")
    assert result["unmatched"].reason.tolist() == ["missing_history_support"]
    with pytest.raises(ValueError, match="predeclared"):
        match_controls(pd.DataFrame([row(1, 10, 1, v2_dst_current_to_q90_7d=np.nan)]), [1], "ACH_FN")


def test_payer_links_and_reused_controls_form_transitive_components():
    pairs = pd.DataFrame(dict(case_edge_id=[1, 2, 3, 4], control_edge_id=[101, 101, 102, 103],
                             case_payer=[10, 11, 20, 30], control_payer=[20, 20, 21, 31]))
    p, info = dependency_clusters(pairs)
    assert p.dependency_cluster.tolist()[:3] == [10, 10, 10]
    assert info["clusters"] == 2
    assert info["largest_cluster_cases"] == 3


def test_no_mutation_reuse_count_and_forbidden_covariates_ignored():
    data = pd.DataFrame([row(1, 10, 1), row(2, 10, 1), row(3, 10)])
    before = data.copy(deep=True)
    r = match_controls(data, [1, 2], "ACH_FN")
    pd.testing.assert_frame_equal(data, before)
    assert r["summary"]["unique_controls"] == 1
    assert r["summary"]["max_control_reuse"] == 2
    altered = data.assign(score_lr_only=[99, -99, 1], prior_positive_any_7d=[1, 0, 1], historical_label_count=[42, 8, 1000])
    pd.testing.assert_frame_equal(match_controls(altered, [1, 2], "ACH_FN")["pairs"], r["pairs"])


def test_uniform_pool_must_be_explicit_and_duplicate_ids_rejected():
    data = pd.DataFrame([row(1, 10, 1), row(2, 10)]).drop(columns="uniform_negative_control")
    with pytest.raises(ValueError, match="verified"):
        match_controls(data, [1], "ACH_FN")
    assert len(match_controls(data, [1], "ACH_FN", uniform_negative_pool_verified=True)["pairs"]) == 1
    with pytest.raises(ValueError, match="Duplicate edge_id"):
        match_controls(pd.concat([data, data]), [1], "ACH_FN", uniform_negative_pool_verified=True)


def test_holm_six_keeps_unavailable_slot():
    assert holm_adjust([.001, .01, .03, .04, .2, None]) == [.006, .05, .12, .12, .4, None]


def test_exploratory_fisher_known_table_and_degenerate():
    assert _fisher_two_sided(1, 9, 11, 3) == pytest.approx(.0027594561852200836)
    assert _fisher_two_sided(0, 10, 0, 10) == pytest.approx(1)


def statistics_fixture():
    rows, ach, g = [], [], []
    for i in range(3):
        ids = [10*i+1, 10*i+2]
        rows += [row(ids[0], 100+i, 1), row(ids[1], 100+i)]
        ach.append(ids[0])
        ids = [100+10*i+1, 100+10*i+2]
        rows += [row(ids[0], 200+i, 1, from_bank="70", payment_format="Cash"),
                 row(ids[1], 200+i, from_bank="70", payment_format="Cash")]
        g.append(ids[0])
    data = pd.DataFrame(rows)
    matches = {"ACH_FN": match_controls(data, ach, "ACH_FN", max_controls=1),
               "G_FN": match_controls(data, g, "G_FN", max_controls=1)}
    features = data[["edge_id"]].copy()
    for feature in PRIMARY_FEATURES:
        features[feature] = data.label
    return data, matches, features


def test_primary_rates_exact_permutation_cluster_ci_and_determinism():
    _, matches, features = statistics_fixture()
    first = run_primary_statistics(features, matches, n_perm=5000)
    assert first == run_primary_statistics(features, matches, n_perm=5000)
    assert first["primary_test_family_size"] == 6
    for r in first["tests"]:
        assert r["effect_rate_difference"] == 1
        assert r["ci95"] == [1, 1]
        assert r["permutation_p"] == .25  # 2 of 2**3 sign patterns are as extreme.
        assert r["holm6_p"] == 1
        assert r["independent_clusters"] == 3
        assert r["inference_status"] == "very_few_clusters_no_strong_conclusion"


def test_single_dependence_cluster_prevents_inference():
    _, matches, features = statistics_fixture()
    matches = copy.deepcopy(matches)
    matches["ACH_FN"]["pairs"]["control_payer"] = 999
    result = run_primary_statistics(features, matches, n_perm=5000)
    t = result["tests"][0]
    assert t["independent_clusters"] == 1
    assert t["permutation_p"] is None and t["ci95"] is None


def test_weights_cases_not_number_of_controls():
    _, matches, features = statistics_fixture()
    p = matches["ACH_FN"]["pairs"]
    extra = p.iloc[[0]].copy()
    extra["control_edge_id"] = 9999
    extra["control_payer"] = int(p.iloc[0].case_payer)
    extra["control_weight"] = .5
    p.loc[p.case_edge_id == extra.iloc[0].case_edge_id, "control_weight"] = .5
    matches["ACH_FN"]["pairs"] = pd.concat([p, extra], ignore_index=True)
    features.loc[len(features)] = [9999] + [1] * 3
    test = run_primary_statistics(features, matches, n_perm=5000)["tests"][0]
    assert test["case_rate"] == 1
    assert test["matched_control_rate"] == pytest.approx(1/6)
    assert test["effect_rate_difference"] == pytest.approx(5/6)


def test_ach_tp_reference_is_descriptive_and_support_stratified():
    data = pd.DataFrame([row(1, 10, 1), row(2, 20, 1, outcome_lr_only="TP"),
                         row(3, 30, 1, outcome_lr_only="TP", v2_dst_count_7d=2),
                         row(4, 40, 1, outcome_lr_only="TP", r_pair_fwd_count_72h=1)])
    r = describe_ach_reference(data)
    assert r["totals"] == {"TP": 2, "FN": 1}
    assert len(r["strata"]) == 3


def test_missing_binary_feature_cannot_be_silently_excluded():
    _, matches, features = statistics_fixture()
    features.loc[0, PRIMARY_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="complete binary"):
        run_primary_statistics(features, matches, n_perm=5000)
