import pandas as pd
import pytest

from gnn_chain_audit_endpoint_sensitivity import endpoint_clusters, weighted_distribution, sensitivity, DESCRIPTIVE_FEATURES
from gnn_chain_audit_statistics import PRIMARY_FEATURES


def test_receiver_sharing_and_cross_role_account_overlap_merge_case_sets():
    pairs = pd.DataFrame(dict(case_edge_id=[1, 2, 3], control_edge_id=[11, 12, 13],
                             case_payer=[10, 20, 30], control_payer=[11, 21, 31]))
    identities = pd.DataFrame(dict(edge_id=[1, 2, 3, 11, 12, 13], src=[10, 20, 30, 11, 21, 31],
                                    dst=[100, 100, 21, 101, 102, 103]))
    output, report = endpoint_clusters(pairs, identities)
    assert report["clusters"] == 1
    assert report["largest_cluster_cases"] == 3
    assert output.endpoint_dependency_cluster.tolist() == [1, 1, 1]
    pd.testing.assert_frame_equal(output.drop(columns="endpoint_dependency_cluster"), pairs)


def test_disjoint_endpoints_preserve_components():
    pairs = pd.DataFrame(dict(case_edge_id=[1, 2], control_edge_id=[11, 12],
                             case_payer=[10, 20], control_payer=[11, 21]))
    identities = pd.DataFrame(dict(edge_id=[1, 2, 11, 12], src=[10, 20, 11, 21], dst=[100, 200, 101, 201]))
    _, report = endpoint_clusters(pairs, identities)
    assert report["clusters"] == 2
    assert report["largest_cluster_cases"] == 1


def test_payer_identity_mismatch_is_rejected():
    pairs = pd.DataFrame(dict(case_edge_id=[1], control_edge_id=[11], case_payer=[10], control_payer=[11]))
    identities = pd.DataFrame(dict(edge_id=[1, 11], src=[999, 11], dst=[100, 101]))
    with pytest.raises(ValueError, match="payer identity"):
        endpoint_clusters(pairs, identities)


def test_weighted_descriptions_preserve_missing_and_case_weights():
    r = weighted_distribution([1, 2, 100, float("nan")], [.5, .5, 1, 1])
    assert r["mean"] == 50.75
    assert r["missing_weight"] == 1
    assert r["quantiles"]["0.5"] == 2
    assert r["quantiles"]["0.9"] == 100


def test_complete_sensitivity_keeps_effect_and_reports_continuous_and_g_payers():
    f = pd.DataFrame(dict(edge_id=[1, 2, 11, 12], src=[10, 10, 20, 20], dst=[101, 102, 201, 202]))
    for col in PRIMARY_FEATURES:
        f[col] = [1, 0, 1, 0]
    for col in DESCRIPTIVE_FEATURES:
        f[col] = [1., .25, 1., .25]
    pairs = {}
    originals = []
    for cohort, case_id, control_id, payer in [("ACH_FN", 1, 2, 10), ("G_FN", 11, 12, 20)]:
        pairs[cohort] = pd.DataFrame(dict(case_edge_id=[case_id], control_edge_id=[control_id],
            case_payer=[payer], control_payer=[payer], control_weight=[1.]))
        for col in PRIMARY_FEATURES:
            originals.append(dict(cohort=cohort, feature=col, matched_cases=1,
                effect_rate_difference=1., case_rate=1., matched_control_rate=0.,
                ci95=None, permutation_p=None, holm6_p=None, independent_clusters=1))
    result = sensitivity(f, pairs, {"tests": originals}, n_perm=5000)
    assert len(result["tests"]) == 6
    assert len(result["g_per_payer"]) == 3
    assert len(result["exploratory_continuous_descriptions"]) == 8
    assert all(x["matched_mean_difference"] == .75 for x in result["exploratory_continuous_descriptions"])
    assert all(x["effect_rate_difference"] == 1 for x in result["tests"])
    assert "receiver" in result["tests"][0]["uncertainty_unit"]
