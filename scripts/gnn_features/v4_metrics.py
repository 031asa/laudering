"""Source-bound v4 ranking metrics; each threshold includes complete score ties."""
from __future__ import annotations

import numpy as np

from .v2_evaluate import confusion, precision80_point


def score_metrics(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    if y.ndim != 1 or y.shape != s.shape or not len(y):
        raise ValueError('Nonempty matching one-dimensional inputs required')
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Invalid labels/probabilities')
    y, s = y.astype(np.int8, copy=False), s.astype(np.float64, copy=False)
    total, positive = len(y), int(y.sum())
    negative = total - positive
    order = np.argsort(-s, kind='stable')
    ordered = s[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), total - 1]
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    alerts = ends + 1
    fp = alerts - tp

    def point(i):
        t, f = int(tp[i]), int(fp[i])
        return dict(threshold=float(ordered[ends[i]]), samples=total, positives=positive,
                    negatives=negative, alerts=t+f, tp=t, fp=f, fn=positive-t, tn=negative-f,
                    precision=t/(t+f), recall=t/positive if positive else None,
                    f1=2*t/(positive+t+f) if positive else None,
                    fpr=f/negative if negative else None)

    best = p80 = ap = auc = None
    if positive:
        ap = float(np.sum(np.diff(np.r_[0, tp]) * (tp/alerts)) / positive)
        f1 = 2.0*tp/(positive+alerts)
        i = int(np.argmax(f1))
        # Exact integer cross multiplication: equal F1 prefers precision, then threshold.
        tied = np.flatnonzero(tp*(positive+alerts[i]) == tp[i]*(positive+alerts))
        i = int(tied[np.argmax(tp[tied]/alerts[tied])])
        best = point(i)
        eligible = np.flatnonzero((tp > 0) & (5*tp >= 4*alerts))
        if len(eligible):
            p80 = point(int(eligible[np.argmax(tp[eligible])]))
    if positive and negative:
        auc = float(np.sum(np.diff(np.r_[0, fp]).astype(np.float64) *
                           (tp+np.r_[0, tp[:-1]])) / (2.0*positive*negative))
    return dict(samples=total, positives=positive, negatives=negative,
                positive_rate=positive/total, average_precision=ap,
                pr_auc_definition='average_precision (step integral)', roc_auc=auc,
                best_f1=best, precision80=p80,
                threshold_0_5=confusion(y, s >= .5, .5),
                threshold_rule='score >= threshold; complete score ties; F1, precision, threshold')


def best_f1_point(labels, scores):
    return score_metrics(labels, scores)['best_f1']


binary_metrics = score_metrics
