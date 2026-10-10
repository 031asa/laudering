"""Exact binary ranking metrics with indivisible score ties.

This module consumes labels and probabilities already produced by evaluation; it
does not load a dataset or select a data split. ``average_precision`` is the
step-integral AP definition of PR-AUC, not trapezoidal PR-curve area. Thresholds
always mean ``score >= threshold``. A finite threshold just above the maximum
score denotes the option of predicting no positives.
"""
from __future__ import annotations

import numpy as np


def _operating_point(threshold, tp, fp, positives, negatives):
    tp, fp = int(tp), int(fp)
    fn, tn = int(positives - tp), int(negatives - fp)
    predicted = tp + fp
    return {
        "threshold": float(threshold),
        "precision": float(tp / predicted) if predicted else None,
        "recall": float(tp / positives) if positives else None,
        "f1": float(2 * tp / (positives + predicted)) if positives else None,
        "fpr": float(fp / negatives) if negatives else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def binary_metrics(labels, scores):
    """Return JSON-safe exact metrics for nonempty 1-D binary labels/scores.

    Probabilities must be finite and in [0, 1]. An absent positive class makes
    AP, recall, F1 and best-F1 unavailable (None). ROC AUC and recall-at-FPR
    require both classes. Precision is None when no positive is predicted;
    FPR is None when no negative label exists. Confusion counts remain valid.

    ``best_f1`` searches whole score groups on the supplied evaluation split;
    tied F1 values prefer the higher threshold. ``recall_at_fpr`` maximizes
    recall without interpolation or splitting tied scores at FPR limits 0.001
    and 0.01. For equal recall it prefers fewer false positives, then a higher
    threshold. A chosen validation threshold must be fixed before test use.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same one-dimensional shape")
    if not labels.size:
        raise ValueError("labels and scores must be nonempty")
    if labels.dtype.kind not in "buif" or scores.dtype.kind not in "buif":
        raise ValueError("labels and scores must be real numeric arrays")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0 and 1")
    scores = scores.astype(np.float64, copy=False)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    labels = labels.astype(np.int8, copy=False)
    positives = int(np.count_nonzero(labels))
    samples = int(labels.size)
    negatives = samples - positives

    predicted = scores >= 0.5
    tp_half = int(np.count_nonzero(labels[predicted]))
    at_half = _operating_point(0.5, tp_half,
                              int(np.count_nonzero(predicted)) - tp_half,
                              positives, negatives)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative_tp = np.cumsum(labels[order], dtype=np.int64)
    # Keep only the end of each equal-score group; a threshold cannot split it.
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), samples - 1]
    thresholds = sorted_scores[ends]
    tp = cumulative_tp[ends]
    fp = ends + 1 - tp
    previous_tp = np.r_[0, tp[:-1]]
    previous_fp = np.r_[0, fp[:-1]]
    precision = tp / (tp + fp)

    average_precision = None
    best_f1 = None
    if positives:
        average_precision = float(np.sum((tp - previous_tp) * precision) / positives)
        f1 = 2.0 * tp / (positives + tp + fp)
        # Thresholds descend, so the first maximizer is the highest threshold.
        best_index = int(np.argmax(f1))
        best_f1 = _operating_point(thresholds[best_index], tp[best_index],
                                   fp[best_index], positives, negatives)

    roc_auc = None
    at_fpr = {"0.001": None, "0.01": None}
    if positives and negatives:
        # Trapezoids in ROC space give half credit to positive/negative ties.
        roc_auc = float(np.sum((fp - previous_fp).astype(np.float64)
                               * (tp + previous_tp)) / (2.0 * positives * negatives))
        # Integer comparisons avoid floating-point tolerance at FPR boundaries.
        for key, denominator in (("0.001", 1000), ("0.01", 100)):
            eligible = np.flatnonzero(fp <= negatives // denominator)
            chosen = None
            if eligible.size:
                index = int(eligible[np.argmax(tp[eligible])])
                if tp[index] > 0:
                    chosen = _operating_point(thresholds[index], tp[index],
                                              fp[index], positives, negatives)
            if chosen is None:
                chosen = _operating_point(np.nextafter(sorted_scores[0], np.inf),
                                          0, 0, positives, negatives)
            at_fpr[key] = chosen

    return {
        "samples": samples,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / samples),
        "average_precision": average_precision,
        "pr_auc_definition": "average_precision (step integral)",
        "roc_auc": roc_auc,
        "threshold_0_5": at_half,
        "best_f1": best_f1,
        "recall_at_fpr": at_fpr,
    }
