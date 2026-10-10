"""Read-only acceptance and bounded replay of the frozen Train Parquet cache."""
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .build import load_archive
from .cache_train import chunks, array_sha
from .contract import HOUR, N_FIELDS
from .hybrid_cache_trial import ENCODED_NAMES
from .loader import TargetDay

ROOT = Path(__file__).resolve().parents[2]
DATES = [f'2022-09-{i:02d}' for i in range(1, 11)]
EXPECTED_TARGETS = 20_318_992
EXPECTED_POSITIVES = 17_794
NODE_COLUMNS = ['node_id', 'cutoff_us'] + ENCODED_NAMES
PLAN_COLUMNS = ['batch_id', 'cutoff_us', 'targets', 'nodes', 'edges']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def _inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Path escapes its input root: {relative}')
    return path


def _node_schema(pf):
    schema = pf.schema_arrow
    expected = [pa.int64(), pa.int64()] + [pa.float32()] * len(ENCODED_NAMES)
    if schema.names != NODE_COLUMNS or [f.type for f in schema] != expected:
        raise ValueError('Invalid cached node schema')


def _plans(folder):
    pf = pq.ParquetFile(Path(folder) / 'plans.parquet')
    expected = [pa.int32(), pa.int64()] + [pa.list_(pa.int64())] * 3
    if pf.schema_arrow.names != PLAN_COLUMNS or [f.type for f in pf.schema_arrow] != expected:
        raise ValueError('Invalid cached plan schema')
    rows = pf.read().to_pylist()
    if not rows or len(rows) > 512:
        raise ValueError('Invalid cached batch count')
    for i, row in enumerate(rows):
        if row['batch_id'] != i or row['cutoff_us'] is None:
            raise ValueError('Invalid cached batch identity')
        row['cutoff'] = int(row['cutoff_us'])
        for key in ('targets', 'nodes', 'edges'):
            if row[key] is None or any(value is None for value in row[key]):
                raise ValueError('Null cached plan identity')
            row[key] = np.asarray(row[key], dtype=np.int64)
            if np.any(np.diff(row[key]) <= 0):
                raise ValueError('Cached plan IDs must be unique ascending')
        if not 1 <= len(row['targets']) <= 64 or not len(row['nodes']):
            raise ValueError('Invalid cached target/node count')
        if i:
            previous = rows[i - 1]
            if row['cutoff'] < previous['cutoff'] or (
                row['cutoff'] == previous['cutoff'] and row['targets'][0] <= previous['targets'][-1]
            ):
                raise ValueError('Cached plans are not chronologically ordered')
    if rows[0]['cutoff'] // HOUR != rows[-1]['cutoff'] // HOUR:
        raise ValueError('Cached block crosses an hour boundary')
    return rows


def _validate_graph(plan, arrays, node_count):
    targets, nodes, edges = (plan[k] for k in ('targets', 'nodes', 'edges'))
    t = plan['cutoff']
    n = len(arrays['timestamp'])
    if np.any(targets < 0) or np.any(targets >= n) or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError('Cached edge identity outside archive')
    if np.any(nodes < 0) or np.any(nodes >= node_count):
        raise ValueError('Cached node identity outside archive')
    if not np.all(arrays['timestamp'][targets] == t):
        raise ValueError('Cached target cutoff mismatch')
    times = arrays['timestamp'][edges]
    if not np.all((times >= t - 72 * HOUR) & (times < t)):
        raise ValueError('Cached graph contains current/future or expired history')
    if np.any(arrays['src'][edges] == arrays['dst'][edges]):
        raise ValueError('Cached graph contains self-transfers')
    endpoints = np.r_[arrays['src'][targets], arrays['dst'][targets], arrays['src'][edges], arrays['dst'][edges]]
    positions = np.searchsorted(nodes, endpoints)
    if np.any(positions >= len(nodes)) or not np.array_equal(nodes[positions], endpoints):
        raise ValueError('Cached graph has missing endpoints')


def _block_manifest(folder, config_hash):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    if manifest.get('status') != 'verified' or manifest.get('config_sha256') != config_hash:
        raise ValueError('Cached block is not verified for this configuration')
    if set(manifest.get('file_sha256', {})) != {'nodes.parquet', 'plans.parquet'}:
        raise ValueError('Cached block file inventory differs')
    for name, digest in manifest['file_sha256'].items():
        if file_sha(folder / name) != digest:
            raise ValueError(f'Cached Parquet checksum mismatch: {folder / name}')
    if sum((folder / name).stat().st_size for name in manifest['file_sha256']) != manifest['parquet_bytes']:
        raise ValueError('Cached Parquet size mismatch')
    return manifest


def _read_labels(archive, date, day, arrays, split_name, record, tick):
    labels = np.empty(len(day.ids), dtype=np.uint8)
    seen = np.zeros(len(day.ids), dtype=bool)
    paths = sorted((archive / 'supervision/targets' / f'transaction_date={date}').glob('*.parquet'))
    if not paths:
        raise ValueError(f'Missing supervision for {date}')
    for path in paths:
        record(path)
        edges = archive / 'graph/edges' / path.parent.name / path.name
        lp, ep = pq.ParquetFile(path), pq.ParquetFile(edges)
        if lp.metadata.num_rows != ep.metadata.num_rows:
            raise ValueError('Supervision/edge row count mismatch')
        labels_iter = lp.iter_batches(batch_size=32768, columns=['edge_id', 'label', 'transaction_id', 'split'])
        edges_iter = ep.iter_batches(batch_size=32768, columns=['edge_id', 'transaction_id', 'timestamp'])
        for lb, eb in itertools.zip_longest(labels_iter, edges_iter):
            if lb is None or eb is None or lb.num_rows != eb.num_rows:
                raise ValueError('Supervision/edge batch coverage differs')
            ids = lb.column('edge_id').to_numpy().astype(np.int64)
            if lb.column('edge_id').null_count or not np.array_equal(ids, eb.column('edge_id').to_numpy()):
                raise ValueError('Supervision edge identity mismatch')
            if lb.column('transaction_id').null_count or not lb.column('transaction_id').equals(eb.column('transaction_id')):
                raise ValueError('Supervision transaction identity mismatch')
            if lb.column('split').null_count or set(lb.column('split').to_pylist()) != {split_name}:
                raise ValueError('Supervision outside frozen split')
            positions = day.positions(ids)
            y = lb.column('label').to_numpy()
            if lb.column('label').null_count or not np.isin(y, [0, 1]).all():
                raise ValueError('Invalid supervision label')
            if seen[positions].any() or len(np.unique(positions)) != len(positions):
                raise ValueError('Duplicate supervision identity')
            if not np.array_equal(eb.column('timestamp').cast(pa.int64()).to_numpy(), arrays['timestamp'][ids]):
                raise ValueError('Supervision edge timestamp mismatch')
            labels[positions] = y
            seen[positions] = True
            tick('supervision', date=date)
    if not seen.all():
        raise ValueError(f'Incomplete supervision for {date}')
    return labels


def verify_cache(cache_root, archive, feature_root, progress=None):
    """Verify all cached targets without loading whole-day R or computing features.

    Returns ordered block summaries with label counts and immutable input hashes.
    Hash verification establishes N payload integrity; all plans/labels are also
    checked against original target identities. No source or cache is modified.
    """
    cache_root, archive, feature_root = map(lambda p: Path(p).resolve(), (cache_root, archive, feature_root))
    inputs = {}
    last = 0.0

    def tick(stage, force=False, **fields):
        nonlocal last
        now = time.monotonic()
        if progress is not None and (force or now - last >= 15):
            progress(stage=stage, **fields)
            last = now

    def record(path):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        inputs[key] = file_sha(path)
        tick('hashing', file=key)
        return inputs[key]

    state = read_json(cache_root / 'status.json')
    root = read_json(cache_root / 'manifest.json')
    config = read_json(cache_root / 'config.json')
    if state.get('status') != 'completed' or root.get('status') != 'completed':
        raise ValueError('Full cache is not completed')
    for obj, key in ((state, 'verified_targets'), (root, 'targets'), (config, 'expected_targets')):
        if obj.get(key) != EXPECTED_TARGETS:
            raise ValueError('Full Train target count differs')
    if config.get('dates') != DATES or state.get('dates') != DATES:
        raise ValueError('Frozen Train dates differ')
    for key, value in {'batch_size': 64, 'max_batches_per_block': 512, 'seed': 20260920, 'fanout': [10, 5], 'end_exclusive': '2022-09-11T00:00:00'}.items():
        if config.get(key) != value:
            raise ValueError(f'Frozen cache setting differs: {key}')
    if Path(root['feature_tables']).resolve() != feature_root or Path(root['edge_and_identity_tables']).resolve() != archive:
        raise ValueError('Cache input locations differ')
    if root.get('node_columns') != ENCODED_NAMES:
        raise ValueError('Cache node feature contract differs')
    config_hash = record(cache_root / 'config.json')
    manifest_hash = record(cache_root / 'manifest.json')
    if root.get('config_sha256') != config_hash:
        raise ValueError('Root cache configuration hash differs')
    active = cache_root / '_work/active'
    if active.exists() and any(active.rglob('*')):
        raise ValueError('Completed cache has remaining active work; preserve it for review')
    if sorted(p.name for p in cache_root.glob('transaction_date=*') if p.is_dir()) != ['transaction_date=' + date for date in DATES]:
        raise ValueError('Cache date folder inventory differs')
    required = {ROOT / 'config/dataset_split.json', archive / 'build.json'}
    required.update(Path(__file__).with_name(name) for name in ('cache_train.py', 'hybrid_cache_trial.py', 'build.py', 'engine.py', 'rolling.py', 'contract.py', 'loader.py'))
    required.update(feature_root / f'transaction_date={date}' / name for date in DATES for name in ('target_edge_ids.npy', 'merge_phase.json', 'verification.json', 'run_config.json'))
    frozen = {_inside(ROOT, rel): digest for rel, digest in config.get('hashes', {}).items()}
    if not required.issubset(frozen):
        raise ValueError('Frozen input hash inventory is incomplete')
    for path, digest in frozen.items():
        if record(path) != digest:
            raise ValueError(f'Frozen code/input hash changed: {path}')
    archive_hash = file_sha(archive / 'build.json')
    meta, arrays, bank, _ = load_archive(archive)
    if meta['rows'] != EXPECTED_TARGETS or meta['end_exclusive'] != config['end_exclusive']:
        raise ValueError('Archive outside frozen Train scope')
    split = read_json(ROOT / 'config/dataset_split.json')
    split_dates = {date: item['name'] for item in split['splits'] for date in item['dates']}
    root_blocks = root['blocks']
    if len(root_blocks) != state.get('completed_blocks') or len({b['block'] for b in root_blocks}) != len(root_blocks):
        raise ValueError('Root block inventory is duplicate or incomplete')
    actual_blocks = {str(p.parent.relative_to(cache_root)).replace('\\', '/') for p in cache_root.glob('transaction_date=*/hour=*/block=*/manifest.json')}
    if actual_blocks != {b['block'] for b in root_blocks}:
        raise ValueError('Published block inventory differs from root manifest')
    verified = []
    seen_targets = np.zeros(EXPECTED_TARGETS, dtype=bool)
    last_target_key = None
    positives = total = parquet_bytes = 0
    for date in DATES:
        folder = feature_root / f'transaction_date={date}'
        run = read_json(folder / 'daily_run.json')
        source_check = read_json(folder / 'verification.json')
        if run.get('status') != 'verified' or source_check.get('status') != 'passed':
            raise ValueError(f'Source day is not verified: {date}')
        record(folder / 'daily_run.json')
        if read_json(folder / 'run_config.json')['archive_manifest_sha256'] != archive_hash:
            raise ValueError('Source day archive identity mismatch')
        day = TargetDay.__new__(TargetDay)
        day.ids = np.load(folder / 'target_edge_ids.npy').astype(np.int64)
        if not len(day.ids) or np.any(day.ids < 0) or np.any(day.ids >= EXPECTED_TARGETS) or np.any(np.diff(day.ids) <= 0):
            raise ValueError('Invalid daily target identities')
        if seen_targets[day.ids].any():
            raise ValueError('Targets repeated across days')
        seen_targets[day.ids] = True
        day.times = np.asarray(arrays['timestamp'][day.ids])
        day.order = np.lexsort((day.ids, day.times))
        start = int(np.datetime64(date, 'us').astype(np.int64))
        if not np.all((day.times >= start) & (day.times < start + 24 * HOUR)) or len(day.ids) != source_check['full_day_rows']:
            raise ValueError('Daily target date/count mismatch')
        labels = _read_labels(archive, date, day, arrays, split_dates[date], record, tick)
        daily_path = cache_root / f'transaction_date={date}' / 'manifest.json'
        daily = read_json(daily_path)
        record(daily_path)
        expected_daily = [b for b in root_blocks if b['block'].startswith(f'transaction_date={date}/')]
        if daily.get('status') != 'verified' or daily.get('targets') != len(day.ids) or daily.get('blocks') != expected_daily:
            raise ValueError('Daily cache manifest differs from root/source')
        daily_count = 0
        for hour, index, batches in chunks(day):
            rel = f'transaction_date={date}/hour={hour % 24:02d}/block={index:04d}'
            block_index = len(verified)
            if block_index >= len(root_blocks) or root_blocks[block_index]['block'] != rel:
                raise ValueError('Root cache block order/coverage differs')
            block = _inside(cache_root, rel)
            m = _block_manifest(block, config_hash)
            block_manifest_sha = record(block / 'manifest.json')
            expected_ids = np.concatenate(batches)
            if m.get('block') != rel or m.get('target_ids_sha256') != array_sha(expected_ids):
                raise ValueError('Cache block target identity differs')
            for key in ('targets', 'node_states', 'parquet_bytes'):
                if m[key] != root_blocks[block_index][key]:
                    raise ValueError('Root/block summary differs')
            if m['targets'] != len(expected_ids) or m['batches'] != len(batches):
                raise ValueError('Block target/batch count differs')
            nodes_file = pq.ParquetFile(block / 'nodes.parquet')
            _node_schema(nodes_file)
            if nodes_file.metadata.num_rows != m['node_states']:
                raise ValueError('Node state count differs')
            plans = _plans(block)
            if len(plans) != len(batches):
                raise ValueError('Plan batch coverage differs')
            for plan, ids in zip(plans, batches):
                if not np.array_equal(plan['targets'], ids):
                    raise ValueError('Plan target order/coverage differs')
                _validate_graph(plan, arrays, len(bank))
                first = (plan['cutoff'], int(ids[0]))
                if last_target_key is not None and first <= last_target_key:
                    raise ValueError('Target chronology repeated or reversed')
                last_target_key = (plan['cutoff'], int(ids[-1]))
            if m['first_cutoff_us'] != plans[0]['cutoff'] or m['last_cutoff_us'] != plans[-1]['cutoff']:
                raise ValueError('Block cutoff summary differs')
            count_positive = int(labels[day.positions(expected_ids)].sum())
            verified.append(dict(root_blocks[block_index], positives=count_positive, negatives=len(expected_ids) - count_positive, manifest_sha256=block_manifest_sha))
            daily_count += len(expected_ids)
            total += len(expected_ids)
            positives += count_positive
            parquet_bytes += m['parquet_bytes']
            tick('verify_cache_blocks', blocks=len(verified), targets=total, date=date)
        if daily_count != len(day.ids):
            raise ValueError('Incomplete daily cache coverage')
    if len(verified) != len(root_blocks) or not seen_targets.all() or total != EXPECTED_TARGETS:
        raise ValueError('Incomplete full Train cache coverage')
    if positives != EXPECTED_POSITIVES:
        raise ValueError(f'Full Train positive count differs: {positives}')
    if parquet_bytes != state.get('parquet_bytes'):
        raise ValueError('Completed cache byte count differs')
    tick('cache_verified', force=True, blocks=len(verified), targets=total, positives=positives)
    return dict(blocks=verified, targets=total, positives=positives, negatives=total - positives,
                parquet_bytes=parquet_bytes, config_sha256=config_hash, manifest_sha256=manifest_hash,
                dates=DATES.copy(), input_hashes=inputs, archive_manifest_sha256=archive_hash,
                active_work_files=0, node_columns=ENCODED_NAMES.copy())


class ParquetBlockCache:
    """One bounded block of encoded N and immutable sampling plans, read only."""
    def __init__(self, folder, loader, config_hash):
        self.folder = Path(folder)
        self.loader = loader
        self.manifest = _block_manifest(self.folder, config_hash)
        self.plans = _plans(self.folder)
        self.targets = np.concatenate([p['targets'] for p in self.plans])
        if len(np.unique(self.targets)) != len(self.targets) or len(self.targets) != self.manifest['targets']:
            raise ValueError('Cached block duplicate/missing target identity')
        if array_sha(self.targets) != self.manifest['target_ids_sha256'] or len(self.plans) != self.manifest['batches']:
            raise ValueError('Cached block target hash/batch count mismatch')
        pf = pq.ParquetFile(self.folder / 'nodes.parquet')
        _node_schema(pf)
        table = pf.read()
        if table.num_rows != self.manifest['node_states'] or any(table[name].null_count for name in NODE_COLUMNS):
            raise ValueError('Cached node count/null mismatch')
        nodes = table['node_id'].to_numpy()
        cutoffs = table['cutoff_us'].to_numpy()
        self.values = np.column_stack([table[name].to_numpy() for name in ENCODED_NAMES])
        if not len(nodes) or not np.isfinite(self.values).all():
            raise ValueError('Invalid/nonfinite cached node values')
        if not np.isin(self.values[:, len(N_FIELDS):], [0, 1]).all():
            raise ValueError('Invalid cached validity/coverage flag')
        self.origin = int(cutoffs.min())
        self.end = int(cutoffs.max())
        self.span = self.end - self.origin + 1
        if self.span > HOUR or self.origin != self.plans[0]['cutoff'] or self.end != self.plans[-1]['cutoff']:
            raise ValueError('Cached node cutoff range differs from plans')
        if self.origin != self.manifest['first_cutoff_us'] or self.end != self.manifest['last_cutoff_us']:
            raise ValueError('Cached node cutoff summary mismatch')
        self.node_count = len(loader.builder.bank_cat)
        if np.any(nodes < 0) or np.any(nodes >= self.node_count):
            raise ValueError('Cached node ID outside archive')
        if int(nodes.max()) > (np.iinfo(np.int64).max - (self.span - 1)) // self.span:
            raise ValueError('Cached node lookup key overflow')
        self.codes = nodes * self.span + (cutoffs - self.origin)
        if np.any(np.diff(self.codes) <= 0):
            raise ValueError('Cached node keys must be unique ascending')
        used = np.zeros(len(nodes), dtype=bool)
        for plan in self.plans:
            _validate_graph(plan, loader.builder.a, self.node_count)
            loader.day.positions(plan['targets'])
            used[self._positions(plan['nodes'], plan['cutoff'])] = True
        if not used.all():
            raise ValueError('Cached nodes contain unrequested keys')
        # The existing builder's LRU must not carry a different block's values.
        loader.builder.clear_cache()

    def _positions(self, nodes, cutoff):
        nodes = np.asarray(nodes, dtype=np.int64)
        if cutoff < self.origin or cutoff > self.end or np.any(nodes < 0) or np.any(nodes >= self.node_count):
            raise ValueError('Node lookup outside cached block')
        if len(nodes) and int(nodes.max()) > (np.iinfo(np.int64).max - (self.span - 1)) // self.span:
            raise ValueError('Node lookup key overflow')
        codes = nodes * self.span + (cutoff - self.origin)
        positions = np.searchsorted(self.codes, codes)
        if np.any(positions >= len(self.codes)) or not np.array_equal(self.codes[positions], codes):
            raise ValueError('Missing cached node/cutoff key')
        return positions

    def node_lookup(self, node, cutoff):
        if not self.origin <= cutoff <= self.end or not 0 <= node < self.node_count:
            raise ValueError('Node lookup outside cached block')
        code = int(node) * self.span + (int(cutoff) - self.origin)
        if code > np.iinfo(np.int64).max:
            raise ValueError('Node lookup key overflow')
        position = int(np.searchsorted(self.codes, code))
        if position >= len(self.codes) or int(self.codes[position]) != code:
            raise ValueError('Missing cached node/cutoff key')
        return self.values[position]

    def __len__(self):
        return len(self.plans)

    def batch(self, i):
        if not isinstance(i, (int, np.integer)) or i < 0 or i >= len(self):
            raise IndexError('Cached batch index outside block')
        plan = self.plans[i]
        positions = self.loader.day.positions(plan['targets'])
        return self.loader.builder.build(plan['targets'], self.loader.day.r[positions], self.loader.day.labels[positions],
                                         prepared=(plan['nodes'], plan['edges']), node_lookup=self.node_lookup)
