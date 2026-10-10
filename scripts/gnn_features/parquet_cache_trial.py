"""Lossless Parquet export and measured replay of the accepted one-hour cache."""
import json
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .contract import N_FIELDS
from .loader import TemporalLoader
from .prepare_hour import HourCache, signature


NAMES = ([f['name'] + '__encoded' for f in N_FIELDS]
         + [f['name'] + '__valid' for f in N_FIELDS if f['nullable']]
         + ['full_1h', 'full_24h', 'full_72h'])


class ParquetHourCache(HourCache):
    def __init__(self, out, loader):
        self.out = Path(out)
        m = json.loads((self.out / 'manifest.json').read_text(encoding='utf-8'))
        if m['status'] != 'passed':
            raise ValueError('Source cache was not accepted')
        if m['signature'] != signature(m['archive'], m['result_root'], m['start'], m['end'], m['batch_size']):
            raise ValueError('Stale cache')
        self.manifest = m
        self.loader = loader
        table = pq.read_table(self.out / 'nodes.parquet')
        self.keys = np.column_stack([table[n].to_numpy() for n in ['node_id', 'cutoff_us']])
        self.values = np.column_stack([table[n].to_numpy() for n in NAMES])
        self.lo = int(np.datetime64(m['start'], 'us').astype(np.int64))
        self.hi = int(np.datetime64(m['end'], 'us').astype(np.int64))
        self.span = self.hi - self.lo
        if int(self.keys[:, 0].max()) > (np.iinfo(np.int64).max - self.span) // self.span:
            raise ValueError('Key overflow')
        self.codes = self.keys[:, 0] * self.span + self.keys[:, 1] - self.lo
        if np.any(np.diff(self.codes) <= 0) or not np.isfinite(self.values).all():
            raise ValueError('Invalid node cache')
        table = pq.read_table(self.out / 'plans.parquet')
        if table['batch_id'].to_pylist() != list(range(m['batches'])):
            raise ValueError('Invalid batch sequence')
        self.plans = [{k: np.asarray(row[k], dtype=np.int64) for k in ['targets', 'nodes', 'edges']}
                      for row in table.to_pylist()]

    def batch(self, i):
        plan = self.plans[i]
        ids = plan['targets']
        p = self.loader.day.positions(ids)
        return self.loader.builder.build(ids, self.loader.day.r[p], self.loader.day.labels[p],
                                         prepared=(plan['nodes'], plan['edges']), node_lookup=self.lookup)


def run():
    source = Path('result/gnn_hour_cache_trial')
    out = Path('result/gnn_hour_cache_parquet_trial')
    out.mkdir(exist_ok=False)
    m = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if m['status'] != 'passed':
        raise ValueError('Unverified source')
    keys = np.load(source / 'node_keys.npy', mmap_mode='r')
    values = np.load(source / 'nodes.npy', mmap_mode='r')
    assert len(NAMES) == values.shape[1] == 63
    start = time.perf_counter()
    table = pa.table(dict(node_id=keys[:, 0], cutoff_us=keys[:, 1],
                          **{n: values[:, i] for i, n in enumerate(NAMES)}))
    table = table.replace_schema_metadata({b'feature_semantics': b'Encoded float32 model N; not raw statistics. Valid masks appended; source contract defines transforms.',
                                          b'source_signature': json.dumps(m['signature']).encode()})
    pq.write_table(table, out / 'nodes.parquet', compression='zstd', compression_level=3, row_group_size=65536)
    rows = []
    for i in range(m['batches']):
        with np.load(source / 'plans' / ('%05d.npz' % i), allow_pickle=False) as f:
            rows.append(dict(batch_id=i, **{k: f[k].tolist() for k in ['targets', 'nodes', 'edges']}))
    schema = pa.schema([('batch_id', pa.int32())] + [(k, pa.list_(pa.int64())) for k in ['targets', 'nodes', 'edges']])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / 'plans.parquet', compression='zstd', compression_level=3)
    export_seconds = time.perf_counter() - start
    (out / 'manifest.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='exported',seconds=export_seconds)), flush=True)
    loader = TemporalLoader(m['archive'], m['result_root'], m['start'][:10], edge_cache='data/gnn_tensor_cache_v1')
    start = time.perf_counter()
    cache = ParquetHourCache(out, loader)
    load_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(cache.keys, keys)
    np.testing.assert_array_equal(cache.values, values)
    for i, plan in enumerate(cache.plans):
        with np.load(source / 'plans' / ('%05d.npz' % i), allow_pickle=False) as f:
            for k in plan:
                np.testing.assert_array_equal(plan[k], f[k])
    old = HourCache(source, loader)
    # Independently assemble all batches from both formats; exact tensor/label equality.
    old_seconds = new_seconds = 0.0
    ids_seen = []
    for i in range(m['batches']):
        loader.builder.clear_cache()
        start = time.perf_counter()
        a, y, _ = old.batch(i)
        old_seconds += time.perf_counter() - start
        loader.builder.clear_cache()
        start = time.perf_counter()
        b, z, _ = cache.batch(i)
        new_seconds += time.perf_counter() - start
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(y, z)
        ids_seen.append(b['target_edge_id'])
        if i % 100 == 0:
            print(json.dumps(dict(stage='exact_batch_comparison', done=i+1, total=m['batches'])), flush=True)
    np.testing.assert_array_equal(np.sort(np.concatenate(ids_seen)), loader.day.ids[(loader.day.times >= cache.lo) & (loader.day.times < cache.hi)])
    loader.builder.clear_cache()
    start = time.perf_counter()
    for i in range(m['batches']):
        cache.batch(i)
    sequential_seconds = time.perf_counter() - start
    # Fresh direct-query comparisons on the same twelve stratified batches.
    direct = []
    for i in np.unique(np.linspace(0, m['batches']-1, 12, dtype=int)):
        loader.builder.clear_cache()
        start = time.perf_counter()
        a, y, _ = loader.batch(cache.plans[i]['targets'])
        elapsed = time.perf_counter() - start
        loader.builder.clear_cache()
        b, z, _ = cache.batch(i)
        for k in a:
            if k == 'x_num':
                np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-6)
            else:
                np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(y, z)
        direct.append(elapsed)
    source_bytes = sum((source / k).stat().st_size for k in ['nodes.npy', 'node_keys.npy']) + sum(p.stat().st_size for p in (source / 'plans').glob('*.npz'))
    sizes = {p.name: p.stat().st_size for p in out.glob('*.parquet')}
    ratio = 20318992 / m['targets']
    result = dict(status='passed', targets=m['targets'], node_states=len(keys), batches=m['batches'],
                  source_core_bytes=source_bytes, parquet_bytes=sizes, compression_ratio=source_bytes/sum(sizes.values()),
                  export_seconds=export_seconds, parquet_load_to_arrays_seconds=load_seconds,
                  paired_numpy_batch_seconds=old_seconds, paired_parquet_batch_seconds=new_seconds,
                  parquet_sequential_replay_seconds=sequential_seconds,
                  direct_query_sample_batches=len(direct), direct_query_sample_mean_seconds=float(np.mean(direct)),
                  direct_full_hour_estimate_seconds=float(np.mean(direct))*m['batches'],
                  projected_train_parquet_GB=sum(sizes.values())*ratio/1e9,
                  source_prepare_seconds=m['plan_seconds']+m['node_prepare_seconds'],
                  exact_all_encoded_values_and_plans=True, exact_all_batch_tensors_and_labels=True,
                  limitation='One hour only; OS caches not cleared. Direct full-hour timing extrapolated from 12 batches. No model training, validation or test. Compression does not reduce N calculation work.')
    (out / 'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
