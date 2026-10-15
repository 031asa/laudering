"""Full, label-isolated Validation/Test inference with resumable Parquet predictions."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gnn_features.build import load_archive
from gnn_features.cache_train import array_sha, sha
from gnn_features.contract import HOUR
from gnn_features.engine import FeatureEngine
from gnn_features.evaluation_metrics import binary_metrics
from gnn_features.loader import (ENUM, ECAT, CompactEdgeStore, TargetDay, TemporalBatchBuilder)
from gnn_features.model import BidirectionalGINE, to_torch
from gnn_features.model_cache import ParquetBlockCache
from gnn_features.training_state import ProcessLock, atomic_json
DATES = [f"2022-09-{d:02d}" for d in range(11,29)]
TOTAL = 4_689_006 + 6_883_233 + 6_987


class TestEdgeStore:
    """Reuse Train memmaps; decode only Validation/Test E to bounded (~463 MB) RAM."""
    def __init__(self, project, archive):
        project, archive = Path(project), Path(archive)
        train_cache = project / 'data/gnn_tensor_cache_v1'
        if not (train_cache / 'manifest.json').exists():
            raise ValueError('Existing verified Train E cache required')
        self.train = CompactEdgeStore(project / 'data/gnn_v1_train', train_cache)
        self.prefix = len(self.train.numeric)
        self.numeric = np.empty((TOTAL, len(ENUM)), dtype=np.float32)
        self.category = np.empty((TOTAL, len(ECAT)), dtype=np.int16)
        count = 0
        for date in DATES:
            for path in sorted((archive / f'graph/edges/transaction_date={date}').glob('*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=['edge_id'] + ENUM + ECAT):
                    tab = pa.Table.from_batches([batch])
                    n = len(tab)
                    if not np.array_equal(tab['edge_id'].to_numpy(), np.arange(self.prefix + count, self.prefix + count + n)):
                        raise ValueError('Validation/Test E edge IDs no longer match archive')
                    self.numeric[count:count+n] = np.column_stack([tab[k].to_numpy() for k in ENUM])
                    cats = np.column_stack([tab[k].to_numpy() for k in ECAT])
                    if np.any(cats < 0) or np.any(cats > 32767):
                        raise ValueError('Invalid encoded categories')
                    self.category[count:count+n] = cats
                    count += n
        if count != TOTAL or not np.isfinite(self.numeric).all():
            raise ValueError('Invalid/incomplete Validation/Test E')
        self.cache, self.reads = {}, 0

    def get(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.prefix + TOTAL):
            raise ValueError('Edge outside allowed Train/Validation/Test history')
        numeric = np.empty((len(ids), len(ENUM)), dtype=np.float32)
        category = np.empty((len(ids), len(ECAT)), dtype=np.int64)
        old = ids < self.prefix
        numeric[old], category[old] = self.train.get(ids[old])
        pos = ids[~old] - self.prefix
        numeric[~old], category[~old] = self.numeric[pos], self.category[pos]
        return numeric, category

