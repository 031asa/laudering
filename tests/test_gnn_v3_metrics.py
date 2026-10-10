import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gnn_features.v3_metrics import score_metrics
from gnn_features.evaluation_metrics import binary_metrics


class MetricsTests(unittest.TestCase):
    def test_f1_tie_prefers_precision(self):
        r = score_metrics([1, 0, 0, 1], [.9, .8, .7, .6])
        self.assertEqual(r['best_f1']['threshold'], .9)
        self.assertEqual(r['best_f1']['precision'], 1)

    def test_whole_ties_and_exact_p80(self):
        r = score_metrics([1, 1, 1, 1, 0, 0], [.9, .8, .8, .8, .8, .7])
        self.assertEqual(r['precision80']['tp'], 4)
        self.assertEqual(r['precision80']['fp'], 1)
        self.assertEqual(r['precision80']['threshold'], .8)
        self.assertIsNone(score_metrics([1, 0], [.5, .5])['precision80'])

    def test_rank_metrics_match_frozen_definition(self):
        rng = np.random.default_rng(43)
        y, s = rng.integers(0, 2, 500), rng.integers(0, 25, 500)/25
        a, b = score_metrics(y, s), binary_metrics(y, s)
        self.assertEqual(a['average_precision'], b['average_precision'])
        self.assertEqual(a['roc_auc'], b['roc_auc'])
        self.assertEqual(a['best_f1']['f1'], b['best_f1']['f1'])

    def test_invalid_and_one_class(self):
        self.assertIsNone(score_metrics([0, 0], [.1, .2])['best_f1'])
        self.assertIsNone(score_metrics([1, 1], [.1, .2])['roc_auc'])
        for y, s in [([], []), ([1], [np.nan]), ([2], [.1]), ([1], [2])]:
            with self.assertRaises(ValueError): score_metrics(y, s)


if __name__ == '__main__': unittest.main()
