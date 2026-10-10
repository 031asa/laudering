"""Recipe-fidelity regression tests and real cached CUDA steps, no production training."""
import copy
import gc
from pathlib import Path
import unittest
import numpy as np
import torch
from gnn_features import repeat_common as common
from gnn_features.repeat_compare import summarize
from gnn_features.repeat_train import train_step, _adapt_batch, make_optimizer
from gnn_features.training_state import capture_rng, restore_rng
from gnn_features.v2_train import _assert_nested_equal
from gnn_features.v4_metrics import score_metrics

class ContractTests(unittest.TestCase):
    def test_seed_inventory(self):
        self.assertEqual(len(set(common.SEEDS)),3)
        self.assertNotIn(common.OLD_SEED,common.SEEDS)
        for seed in (None,20260920,1,True):
            with self.assertRaises(ValueError): common.check_seed(seed)
    def test_model_inventory(self):
        for trial in ('structure_three','amount_distribution','sqrt_weight','xgboost'):
            with self.assertRaises(ValueError): common.check_trial(trial)
    def test_output_isolation(self):
        paths=[common.trial_root(common.ROOT,t,s) for s in common.SEEDS for t in common.TRIALS]
        self.assertEqual(len(set(paths)),6)
        self.assertTrue(all(p.is_relative_to(common.ROOT/common.OUTPUT) for p in paths))
    def test_pairing_and_sd(self):
        rows=[]
        for seed,delta in zip(common.SEEDS,(.1,-.1,.03)):
            for t in common.TRIALS:
                value=.6+(delta if t=='features_only' else 0)
                rows.append(dict(seed=seed,trial=t,metrics=dict(best_f1=dict(f1=value),average_precision=value,precision80=dict(recall=value))))
        s=summarize(list(reversed(rows)),common.SEEDS)
        self.assertAlmostEqual(s['a_minus_lr']['f1']['mean'],.01)
        self.assertAlmostEqual(s['a_minus_lr']['f1']['sample_sd'],np.std([.1,-.1,.03],ddof=1))
        self.assertEqual(s['a_minus_lr']['f1']['a_wins'],2)
        self.assertEqual(s['preferred_recipe'],'features_only')
        self.assertFalse(s['checkpoint_selected'])
        with self.assertRaises(ValueError): summarize(rows[:-1],common.SEEDS)
        with self.assertRaises(ValueError): summarize(rows[:-1]+[rows[0]],common.SEEDS)
    def test_whole_ties(self):
        m=score_metrics(np.array([1,0,1,0]),np.array([.8,.8,.6,.2]))
        self.assertEqual(m['best_f1']['alerts'],3)
        self.assertEqual(m['best_f1']['tp'],2)
        self.assertIsNone(m['precision80'])
    def test_json_ascii(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'; common.atomic_json(p,dict(message='中文'))
            self.assertTrue(p.read_bytes().isascii())
            self.assertEqual(common.read(p)['message'],'中文')

class CudaRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gnn_features.train_model import require_cuda
        from gnn_features.v2_train import _open_loader
        from gnn_features.model_cache import ParquetBlockCache
        from gnn_features.v2_features import SidecarStore
        cls.device=require_cuda(); cls.project=common.ROOT
        cls.loader=_open_loader(cls.project,'2022-09-04')
        cls.block='transaction_date=2022-09-04/hour=12/block=0000'
        cache=cls.project/'data/gnn_train_cache_parquet'
        cls.cached=ParquetBlockCache(cache/cls.block,cls.loader,common.sha(cache/'config.json'))
        cls.side=SidecarStore(cls.project/'data/gnn_v2_target_features/train',cache,expected_split='train')
        cls.arrays,cls.labels,_=cls.cached.batch(0)
        vocab=common.read(cls.project/'data/gnn_v1_train/metadata/vocabulary.json')['values']
        cls.ach=vocab['payment_format'].index('ACH')+2
    @classmethod
    def tearDownClass(cls):
        del cls.loader,cls.cached,cls.side,cls.arrays,cls.labels
        gc.collect(); torch.cuda.empty_cache()
    def fidelity(self,trial):
        from gnn_features import train_model,tune_model,v2_adapter,v2_train
        cfg=common.read(common.reference_root(self.project,trial)/'config.json')
        if trial=='lr_only':
            train_model.seed_all(); original,opt,loss=tune_model.make_model(cfg,self.device)
        else:
            original=v2_adapter.make_initial_model(cfg,self.device); opt=v2_train.make_optimizer(original)
        original_rng=capture_rng()
        repeated=common.make_initial_model(cfg,self.device); repeat_rng=capture_rng(); ropt=make_optimizer(repeated)
        self.assertFalse(opt.state); self.assertFalse(ropt.state)
        for k,v in original.state_dict().items(): torch.testing.assert_close(v,repeated.state_dict()[k],rtol=0,atol=0)
        _assert_nested_equal(original_rng,repeat_rng)
        a,y,_=_adapt_batch(self.cached,0,self.side,self.block,trial)
        self.assertEqual(a['target_attr'].shape[1],110 if trial=='lr_only' else 169)
        if trial=='lr_only': self.assertIs(a['target_attr'].dtype,np.dtype('float32'))
        restore_rng(original_rng)
        if trial=='lr_only': expected=train_model.train_step(original,opt,loss,a,y,self.device)
        else: expected=v2_train.train_step(original,opt,a,y,self.device,self.ach,trial)
        after=capture_rng(); restore_rng(repeat_rng)
        actual=train_step(repeated,ropt,a,y,self.device,self.ach,trial)
        np.testing.assert_allclose(expected,actual,rtol=1e-4,atol=1e-5)
        _assert_nested_equal(original.state_dict(),repeated.state_dict())
        _assert_nested_equal(opt.state_dict(),ropt.state_dict()); _assert_nested_equal(after,capture_rng())
    def test_lr_original_seed_fidelity(self): self.fidelity('lr_only')
    def test_a_original_seed_fidelity(self): self.fidelity('features_only')
    def test_new_seed_is_repeatable_and_changes_initialization(self):
        for t in common.TRIALS:
            c=common.read(common.reference_root(self.project,t)/'config.json'); c['seed']=common.SEEDS[0]
            a=common.make_initial_model(c,self.device); rng=capture_rng()
            b=common.make_initial_model(c,self.device)
            for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
            _assert_nested_equal(rng,capture_rng())
            c['seed']=common.SEEDS[1]; b=common.make_initial_model(c,self.device)
            self.assertTrue(any(not torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),source_sha256=common.source_hashes(common.ROOT),
        cuda=torch.cuda.is_available(),original_seed_fidelity='real CUDA step: model, optimizer and RNG',
        full_run_acceptance='Each declared run additionally requires exact-config real Train ACH/non-ACH resume acceptance')
    common.atomic_json(common.ROOT/common.OUTPUT/'engineering_tests.json',report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
