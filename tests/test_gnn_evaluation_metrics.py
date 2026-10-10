"""Hand-computed ranking metrics, indivisible ties and rare-positive limits."""
import json

import numpy as np
import pytest

from gnn_features.evaluation_metrics import binary_metrics


def test_tied_scores_exact_ap_auc_and_thresholds():
    result = binary_metrics([1, 0, 1, 0], [.8, .8, .3, .1])
    assert result["samples"] == 4
    assert result["positives"] == result["negatives"] == 2
    assert result["average_precision"] == pytest.approx(7 / 12)
    assert result["roc_auc"] == pytest.approx(5 / 8)
    assert result["threshold_0_5"] == {
        "threshold": .5, "precision": .5, "recall": .5, "f1": .5,
        "fpr": .5, "tp": 1, "fp": 1, "tn": 1, "fn": 1,
    }
    assert result["best_f1"]["threshold"] == .3
    assert result["best_f1"]["f1"] == pytest.approx(.8)
    # The highest-score positive cannot be taken without its tied negative.
    assert result["recall_at_fpr"]["0.01"]["recall"] == 0
    assert result["recall_at_fpr"]["0.01"]["threshold"] > .8
    json.dumps(result, allow_nan=False)


def test_all_scores_equal_preserves_group_and_permutation_invariance():
    result = binary_metrics([0, 0, 1, 0], [1, 1, 1, 1])
    permuted = binary_metrics([1, 0, 0, 0], [1, 1, 1, 1])
    assert result == permuted
    assert result["average_precision"] == .25
    assert result["roc_auc"] == .5
    assert result["best_f1"]["f1"] == .4
    assert result["recall_at_fpr"]["0.001"]["threshold"] > 1
    assert result["recall_at_fpr"]["0.001"]["tp"] == 0
    json.dumps(result, allow_nan=False)


def test_rare_positive_exact_fpr_boundary_and_tie_rejection():
    # 2 positives, 1,000 negatives. The .8 tie consumes exactly 0.001 FPR.
    labels = np.r_[1, 1, np.zeros(1000, dtype=int)]
    scores = np.r_[.9, .8, .8, np.full(999, .1)]
    result = binary_metrics(labels, scores)
    point = result["recall_at_fpr"]["0.001"]
    assert point["threshold"] == .8
    assert point["recall"] == 1
    assert point["precision"] == pytest.approx(2 / 3)
    assert point["fpr"] == .001
    # Two tied negatives exceed the 0.001 limit; never interpolate to 100% recall.
    scores[3] = .8
    result = binary_metrics(labels, scores)
    strict = result["recall_at_fpr"]["0.001"]
    assert strict["threshold"] == .9
    assert strict["recall"] == .5
    assert strict["fpr"] == 0
    assert result["recall_at_fpr"]["0.01"]["recall"] == 1
    # Equal recall should not add the final huge negative-only group.
    assert result["recall_at_fpr"]["0.01"]["threshold"] == .8


def test_f1_tie_prefers_highest_threshold():
    # At .9: TP=1, FP=0, FN=1 => F1=2/3. At .1: 2, 2, 0 => 2/3.
    result = binary_metrics([1, 0, 0, 1], [.9, .8, .7, .1])
    assert result["best_f1"]["threshold"] == .9


def test_perfect_and_reversed_rankings():
    perfect = binary_metrics([1, 1, 0, 0], [1, .5, .49, 0])
    assert perfect["average_precision"] == perfect["roc_auc"] == 1
    assert perfect["threshold_0_5"]["f1"] == 1
    reversed_result = binary_metrics([0, 0, 1, 1], [1, .5, .49, 0])
    assert reversed_result["roc_auc"] == 0
    assert reversed_result["average_precision"] == pytest.approx(5 / 12)


def test_no_positives_has_no_ranking_recall_or_f1():
    result = binary_metrics([0, 0], [.8, .2])
    assert result["average_precision"] is None
    assert result["roc_auc"] is None
    assert result["best_f1"] is None
    assert result["recall_at_fpr"] == {"0.001": None, "0.01": None}
    assert result["threshold_0_5"]["precision"] == 0
    assert result["threshold_0_5"]["recall"] is None
    assert result["threshold_0_5"]["f1"] is None
    assert result["threshold_0_5"]["fp"] == 1
    json.dumps(result, allow_nan=False)


def test_no_negatives_and_no_predictions():
    result = binary_metrics([1, 1], [.2, .1])
    assert result["average_precision"] == 1
    assert result["roc_auc"] is None
    assert result["threshold_0_5"]["precision"] is None
    assert result["threshold_0_5"]["recall"] == 0
    assert result["threshold_0_5"]["f1"] == 0
    assert result["threshold_0_5"]["fpr"] is None
    assert result["best_f1"]["f1"] == 1
    assert result["recall_at_fpr"] == {"0.001": None, "0.01": None}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([0, 1], [.1]), ([[0, 1]], [[.1, .8]]),
    ([0, 2], [.1, .8]), ([0, -.1], [.1, .8]),
    ([0, np.nan], [.1, .8]), ([0, np.inf], [.1, .8]),
    ([0, 1], [.1, np.nan]), ([0, 1], [.1, np.inf]),
    ([0, 1], [-.1, .8]), ([0, 1], [.1, 1.01]),
    ([0, 1], [1j, .8]), (["0", "1"], [.1, .8]),
    ([0, 1], ["0.1", "0.8"]),
])
def test_reject_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)
