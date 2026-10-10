"""Exact, causal three-motif target sidecars over immutable A archives.

No labels, path sampling, witness export, or new graph cache. Counts are saved
as int64 (-1 means inapplicable); six float32 model columns are derived on read.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gnn_causal_structures import CausalStructures, REQUIRED, amount_bounds, _between
from . import v2_features as v2
from .training_state import ProcessLock, atomic_json

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('cycle2_count_72h', 'cycle3_path_count_72h', 'chain3_amount_match_path_count_72h')
EXTRA_COLUMNS = ['v4_' + n + '__encoded' for n in NAMES] + ['v4_' + n + '__valid' for n in NAMES]
EXTRA_DIM = 6
SPECS = v2.SPECS
PROBE = 'result/gnn_causal_structure_probe_20261006'
read, file_sha, digest, array_sha = v2.read, v2.file_sha, v2.digest, v2.array_sha
inside, ensure_space = v2.inside, v2.ensure_space


def contract():
    return dict(version='gnn-v4-three-structures-1', names=list(NAMES), columns=EXTRA_COLUMNS,
        extra_dim=6, window_us=72 * 3600_000000,
        history='[t-72h,t); strict time increase; exclude every timestamp peer and historical self transfer',
        paths=['v->u then current u->v', 'v->w->u then current u->v', 'a->w->u then current u->v'],
        accounts='Distinct vertices within each path (except the closing cycle endpoint)',
        currency='Every historical edge paid/received currency equals current payment currency; cycles require same-currency target',
        amount='Chain only: previous received within inclusive [ceil(.9*next_paid),floor(1.1*next_paid)]; exact integers; zero matches zero',
        encoding='Three log1p counts then three valid masks, float32; -1 raw sentinel becomes zero value/mask',
        applicability='Self target: all missing; cross-currency target: cycles missing, chain eligible; eligible empty history: valid zero',
        coverage='Reuse frozen A full_72h based on archive data_start_us; retain partial windows',
        labels='Never accessed; no Test; all payment types use the same definition',
        graph='Only target R appended; N63/R175; existing graph unchanged')


class ThreeStructures(CausalStructures):
    """Prune nonselected motifs; exact time joins and Fenwick amount joins."""
    def counts(self, edge_id):
        a,i=self.a,int(edge_id)
        if not 0<=i<len(a['timestamp']):raise ValueError('Target outside archive')
        u,v,t=int(a['src'][i]),int(a['dst'][i]),int(a['timestamp'][i])
        if u==v:return np.full(3,-1,np.int64)
        self.work=dict(adjacency_rows_examined=0,pair_first_rows=0,pair_second_rows=0,
            middle_accounts=0,amount_range_queries=0,cache_hits=0)
        c=int(a['payment_currency_cat'][i]);same=c==int(a['receiving_currency_cat'][i])
        incoming=self._currency(self._incident(u,'in',t),c)
        reverse=int(np.count_nonzero(a['src'][incoming]==v)) if same else -1
        triangle=0 if same else -1
        if same and len(incoming):
            outgoing=self._currency(self._incident(v,'out',t),c)
            common=np.intersect1d(a['src'][incoming],a['dst'][outgoing])
            common=common[(common!=u)&(common!=v)]
            for middle in common:
                first=outgoing[a['dst'][outgoing]==middle]
                second=incoming[a['src'][incoming]==middle]
                triangle+=sum(map(int,np.searchsorted(a['timestamp'][first],a['timestamp'][second],side='left')))
        lo,hi=amount_bounds(a['paid_scaled'][i]);chain=0
        compatible=incoming[_between(a['received_scaled'][incoming],lo,hi)]
        for middle,second in self._groups(compatible):
            if middle in (u,v):continue
            first=self._currency(self._incident(middle,'in',t),c)
            first=first[(a['src'][first]!=u)&(a['src'][first]!=v)]
            if len(first):chain+=self._pairs(first,second)[1]
        counts=[reverse,triangle,chain]
        if any(n>np.iinfo(np.int64).max for n in counts):raise OverflowError('Count overflow')
        return np.asarray(counts,np.int64)

def encode_extra(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype != np.int64 or np.any(raw < -1):
        raise ValueError('Invalid exact structure counts')
    valid = raw >= 0
    out = np.column_stack((np.log1p(np.maximum(raw, 0)), valid)).astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite structure encoding')
    return out


def normalized_arrays(arrays):
    return {k: arrays[k] if k in arrays else arrays[k.replace('_order', '_edge_ids')] for k in REQUIRED}


def compute_selected(arrays, data_start_us, target_ids):
    engine = ThreeStructures(normalized_arrays(arrays))
    raw = np.asarray([engine.counts(i) for i in target_ids], dtype=np.int64).reshape(-1, 3)
    return encode_extra(raw)


def source_hashes(project):
    return {str(p.relative_to(project)): file_sha(p) for p in
            (Path(__file__), Path(project)/'scripts/gnn_causal_structures.py',
             Path(v2.__file__), Path(__file__).with_name('training_state.py'))}


def open_arrays(project, split):
    archive = Path(project)/SPECS[split]['archive']
    paths = {n: archive/'history_index'/f'{n}.npy' for n in REQUIRED[:7]}
    for d in ('in', 'out'):
        paths[d+'_ptr'] = archive/'graph/adj_index'/f'{d}_ptr.npy'
        paths[d+'_order'] = archive/'graph/adj_index'/f'{d}_edge_ids.npy'
    return {n: np.load(p, mmap_mode='r') for n, p in paths.items()}, paths


def _pause(project):
    p = Path(project)/'result/gnn_tuning_v4/pause_request.json'
    if p.exists() and read(p).get('enabled') is True:
        raise Paused('Explicit v4 pause requested')


class Paused(RuntimeError): pass


def _validate_sidecar(folder, config_hash, base_manifest_hash, ids, cutoffs):
    m = read(folder/'manifest.json')
    if (m.get('status') != 'verified' or m.get('config_sha256') != config_hash
            or m.get('base_manifest_sha256') != base_manifest_hash
            or m.get('contract_sha256') != digest(contract())
            or m.get('targets') != len(ids) or m.get('target_ids_sha256') != array_sha(ids)
            or file_sha(folder/'features.parquet') != m.get('file_sha256')):
        raise ValueError('Structure block manifest/hash differs')
    tab = pq.ParquetFile(folder/'features.parquet').read()
    if tab.column_names != ['edge_id', 'cutoff_us'] + list(NAMES) or tab.num_rows != len(ids):
        raise ValueError('Structure block schema/count differs')
    if not np.array_equal(tab['edge_id'].to_numpy(), ids) or not np.array_equal(tab['cutoff_us'].to_numpy(), cutoffs):
        raise ValueError('Structure target ID/cutoff mismatch')
    raw = np.column_stack([tab[n].to_numpy() for n in NAMES])
    matrix = encode_extra(raw)
    return m, matrix, {n: raw[:, j] for j, n in enumerate(NAMES)}


class SidecarStore(v2.SidecarStore):
    def __init__(self, root, base_cache_root, expected_split=None):
        self.root, self.base_cache_root = Path(root).resolve(), Path(base_cache_root).resolve()
        self.config = read(self.root/'config.json')
        self.config_sha256 = file_sha(self.root/'config.json')
        self.contract_sha256 = digest(contract())
        self.manifest = read(self.root/'manifest.json')
        self.manifest_sha256 = file_sha(self.root/'manifest.json')
        if (self.manifest.get('status') != 'completed' or self.manifest.get('config_sha256') != self.config_sha256
                or self.config.get('contract') != contract() or self.config.get('contract_sha256') != self.contract_sha256):
            raise ValueError('Structure store contract/completion identity differs')
        project = Path(self.config['project'])
        if self.config['source_sha256'] != source_hashes(project):
            raise ValueError('Structure feature source changed')
        for n in ('pilot', 'crosscheck'):
            saved = read(self.root/(n+'.json'))
            if (saved.get('status') != 'passed' or saved.get('config_sha256') != self.config_sha256
                    or file_sha(self.root/(n+'.json')) != self.manifest.get(n+'_sha256')):
                raise ValueError('Structure acceptance changed')
        if expected_split is not None and expected_split != self.config['split']:
            raise ValueError('Structure split differs')
        if (self.base_cache_root != Path(self.config['cache_root']).resolve()
                or file_sha(self.base_cache_root/'manifest.json') != self.config['base_manifest_sha256']):
            raise ValueError('Structure base cache identity changed')
        self.base = read(self.base_cache_root/'manifest.json')
        self.blocks = {b['block']: b for b in self.manifest['blocks']}
        if (self.base.get('status') != 'completed' or self.base['targets'] != self.manifest['targets']
                or len(self.blocks) != len(self.manifest['blocks'])
                or list(self.blocks) != [b['block'] for b in self.base['blocks']]
                or sum(b['targets'] for b in self.blocks.values()) != self.manifest['targets']):
            raise ValueError('Structure full block coverage mismatch')
        self._loaded = None

    def _lookup(self, block, target_ids, cutoff_us):
        ids, times = np.asarray(target_ids, dtype=np.int64), np.asarray(cutoff_us, dtype=np.int64)
        if times.ndim == 0:
            times = np.full(len(ids), int(times), dtype=np.int64)
        if ids.ndim != 1 or ids.shape != times.shape or len(np.unique(ids)) != len(ids) or block not in self.blocks:
            raise ValueError('Invalid structure target request')
        if self._loaded != block:
            folder = inside(self.root, block)
            if file_sha(folder/'manifest.json') != self.blocks[block]['manifest_sha256']:
                raise ValueError('Structure block manifest changed')
            _, all_ids, all_times = v2._base_plan(self.base_cache_root, block, self.base['config_sha256'])
            _, matrix, support = _validate_sidecar(folder, self.config_sha256,
                file_sha(inside(self.base_cache_root, block)/'manifest.json'), all_ids, all_times)
            order = np.argsort(all_ids)
            self._ids, self._times, self._matrix = all_ids[order], all_times[order], matrix[order]
            self._support = {k: v[order] for k, v in support.items()}
            self._loaded = block
        positions = np.searchsorted(self._ids, ids)
        if (np.any(positions >= len(self._ids)) or not np.array_equal(self._ids[positions], ids)
                or not np.array_equal(self._times[positions], times)):
            raise ValueError('Structure requested ID/cutoff mismatch')
        return positions


def pilot(arrays, spec, update, project):
    import psutil
    engine, reference = ThreeStructures(arrays), CausalStructures(arrays)
    groups = []
    ts = arrays['timestamp']
    for date in spec['dates']:
        begin = int(np.datetime64(date, 'us').astype(np.int64))
        lo = max(spec['offset'], int(np.searchsorted(ts, begin)))
        hi = min(spec['offset']+spec['targets'], int(np.searchsorted(ts, begin+v2.DAY)))
        groups.append(('day_'+date, np.unique(np.linspace(lo, hi-1, min(1024, hi-lo), dtype=np.int64))))
    for d in ('in', 'out'):
        degree = np.diff(arrays[d+'_ptr'])
        for node in np.argsort(degree)[-4:]:
            ids = arrays[d+'_order'][int(arrays[d+'_ptr'][node]):int(arrays[d+'_ptr'][node+1])]
            ids = ids[(ids >= spec['offset']) & (ids < spec['offset']+spec['targets'])]
            if len(ids):
                groups.append((f'high_degree_{d}_{int(node)}', ids[np.unique(np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=np.int64))]))
    process, peak, checks, records = psutil.Process(), 0, 0, []
    for name, ids in groups:
        _pause(project)
        began = time.perf_counter()
        for j, i in enumerate(ids):
            raw = engine.counts(i)
            if j in (0, len(ids)//2, len(ids)-1):
                expected = reference.extract(i)['features']
                np.testing.assert_array_equal(raw, [-1 if expected[n] is None else expected[n] for n in NAMES])
                checks += 1
            peak = max(peak, process.memory_info().rss)
        elapsed = time.perf_counter()-began
        records.append(dict(group=name, targets=len(ids), seconds=elapsed, per_second=len(ids)/max(elapsed,1e-9)))
        update(stage='pilot', pilot_groups=len(records), group=name, seconds=elapsed, peak_rss=peak)
    if not checks:
        raise ValueError('No pilot checks')
    return dict(status='passed', groups=records, reference_checks=checks, peak_rss_sampled_bytes=peak,
                raw_bytes_per_target=40, encoding='Six model columns derived from three raw int64 counts',
                timing_scope='Representative day targets and highest-degree account targets; not a full-run guarantee')


def probe_table(project):
    root = Path(project)/PROBE
    m = read(root/'structures_manifest.json')
    if m.get('status') != 'completed' or m['binding']['engine_sha256'] != file_sha(Path(project)/'scripts/gnn_causal_structures.py'):
        raise ValueError('Audited structure reference identity differs')
    if file_sha(root/'structures.parquet') != m['sha256']:
        raise ValueError('Audited structure values changed')
    tab = pq.ParquetFile(root/'structures.parquet').read(columns=['edge_id']+list(NAMES)).combine_chunks()
    ids = tab['edge_id'].to_numpy().astype(np.int64)
    raw = np.column_stack([np.asarray([-1 if x is None else x for x in tab[n].to_pylist()], np.int64) for n in NAMES])
    order = np.argsort(ids)
    if len(ids) != 79872 or len(np.unique(ids)) != len(ids):
        raise ValueError('Audited probe coverage differs')
    return ids[order], raw[order]


def run(project=ROOT, split='train', pilot_only=False):
    project = Path(project).resolve()
    if split not in SPECS:
        raise ValueError('Only Train/Validation allowed')
    spec = SPECS[split]
    archive, cache = project/spec['archive'], project/spec['cache']
    out = project/'data/gnn_v4_structure_features'/split
    ensure_space(project); out.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', pid=os.getpid(), split=split, started_at=datetime.now(timezone.utc).isoformat(), expected_targets=spec['targets'])
    began = time.perf_counter()
    def update(**kw):
        state.update(kw, updated_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-began)
        atomic_json(out/'status.json', state)
        print(json.dumps(state, ensure_ascii=True), flush=True)
    with ProcessLock(out):
        try:
            _pause(project); update(stage='verify_inputs')
            meta, base = read(archive/'build.json'), read(cache/'manifest.json')
            if (meta['rows'] != spec['offset']+spec['targets'] or meta['end_exclusive'] != spec['end']
                    or base.get('status') != 'completed' or base['targets'] != spec['targets']
                    or base['config_sha256'] != file_sha(cache/'config.json')):
                raise ValueError('Frozen full archive/cache scope differs')
            old = v2.SidecarStore(project/'data/gnn_v2_target_features'/split, cache, split)
            arrays, paths = open_arrays(project, split)
            inputs = {}
            for n, p in paths.items():
                relative, h = str(p.relative_to(project)), file_sha(p)
                if old.config['input_hashes'].get(relative) != h:
                    raise ValueError('Frozen history differs: '+relative)
                inputs[relative] = h
            for p in (archive/'build.json', cache/'config.json', cache/'manifest.json'):
                inputs[str(p.relative_to(project))] = file_sha(p)
            if split == 'train':
                for n in ('structures.parquet', 'structures_manifest.json'):
                    p = project/PROBE/n; inputs[str(p.relative_to(project))] = file_sha(p)
            config = dict(version=4, split=split, project=str(project), contract=contract(), contract_sha256=digest(contract()),
                cache_root=str(cache), base_manifest_sha256=file_sha(cache/'manifest.json'), input_hashes=inputs,
                source_sha256=source_hashes(project), v2_manifest_sha256=old.manifest_sha256,
                offset=spec['offset'], targets=spec['targets'])
            if (out/'config.json').exists():
                if read(out/'config.json') != config:
                    raise ValueError('Structure configuration/source/input changed; preserve for audit')
            else:
                atomic_json(out/'config.json', config)
            cfg_hash = file_sha(out/'config.json')
            if (out/'pilot.json').exists():
                p = read(out/'pilot.json')
                if p.get('status') != 'passed' or p.get('config_sha256') != cfg_hash:
                    raise ValueError('Pilot identity mismatch')
            else:
                p = pilot(arrays, spec, update, project)
                p['config_sha256'] = cfg_hash
                ensure_space(project); atomic_json(out/'pilot.json', p)
            if pilot_only:
                update(status='ready', stage='pilot_complete'); return p
            engine = ThreeStructures(arrays)
            reference = CausalStructures(arrays)
            probe_ids, probe_raw = probe_table(project) if split == 'train' else (np.empty(0,np.int64), np.empty((0,3),np.int64))
            total, overlap, checked, blocks = 0, 0, 0, []
            seen = np.zeros(spec['targets'], bool)
            for bi, entry in enumerate(base['blocks']):
                _pause(project); ensure_space(project)
                block = entry['block']; folder = inside(out, block)
                bm, ids, times = v2._base_plan(cache, block, base['config_sha256'])
                pos = ids-spec['offset']
                if (np.any(pos < 0) or np.any(pos >= len(seen)) or seen[pos].any()
                        or not np.array_equal(arrays['timestamp'][ids], times)):
                    raise ValueError('Full target/cutoff identity mismatch')
                old_values = old.block(block, ids, times)
                full = (times-int(meta['data_start_us']) >= 72*3600_000000).astype(np.float32)
                np.testing.assert_array_equal(old_values[:, v2.EXTRA_COLUMNS.index('full_72h')], full)
                bh = file_sha(inside(cache, block)/'manifest.json')
                if (folder/'manifest.json').exists():
                    record, _, supports = _validate_sidecar(folder,cfg_hash,bh,ids,times)
                    raw = np.column_stack([supports[n] for n in NAMES])
                else:
                    raw = np.empty((len(ids),3),np.int64)
                    tick = time.monotonic()
                    for j, i in enumerate(ids):
                        raw[j] = engine.counts(i)
                        if time.monotonic()-tick > 15:
                            _pause(project)
                            update(status='running', stage='extract', current_block=block, block_target=j+1,
                                   processed_targets=total+j+1, committed_targets=total)
                            tick = time.monotonic()
                    # Independent legacy reference on deterministic block endpoints.
                    for j in sorted(set((0,len(ids)//2,len(ids)-1))):
                        expected = reference.extract(ids[j])['features']
                        np.testing.assert_array_equal(raw[j], [-1 if expected[n] is None else expected[n] for n in NAMES])
                    folder.mkdir(parents=True, exist_ok=True)
                    table = pa.table({'edge_id':ids,'cutoff_us':times,**{n:raw[:,j] for j,n in enumerate(NAMES)}})
                    ensure_space(project, table.nbytes*2)
                    v2._atomic_parquet(folder/'features.parquet',table)
                    record = dict(status='verified',config_sha256=cfg_hash,base_manifest_sha256=bh,
                        contract_sha256=digest(contract()),targets=len(ids),target_ids_sha256=array_sha(ids),
                        file_sha256=file_sha(folder/'features.parquet'),reference_checks=3)
                    atomic_json(folder/'manifest.json',record)
                self_mask = arrays['src'][ids] == arrays['dst'][ids]
                same = arrays['payment_currency_cat'][ids] == arrays['receiving_currency_cat'][ids]
                np.testing.assert_array_equal(raw[:,2] < 0,self_mask)
                for j in (0,1): np.testing.assert_array_equal(raw[:,j] < 0,self_mask | ~same)
                _, locations, known = np.intersect1d(ids, probe_ids, assume_unique=True, return_indices=True)
                if len(locations):
                    np.testing.assert_array_equal(raw[locations],probe_raw[known]); overlap += len(locations)
                checked += record['reference_checks']
                seen[pos] = True; total += len(ids)
                blocks.append(dict(block=block,targets=len(ids),manifest_sha256=file_sha(folder/'manifest.json')))
                update(status='running',stage='block_committed',completed_blocks=bi+1,committed_targets=total,
                       processed_targets=total,probe_overlap_checked=overlap)
            if not seen.all() or total != spec['targets'] or (split=='train' and overlap!=79872):
                raise ValueError('Full structure coverage/probe replay incomplete')
            check = dict(status='passed',config_sha256=cfg_hash,targets=total,coverage_mask_checked=total,
                         applicability_checked=total,probe_overlap_checked=overlap,reference_checks=checked)
            atomic_json(out/'crosscheck.json',check)
            manifest = dict(status='completed',config_sha256=cfg_hash,targets=total,blocks=blocks,
                pilot_sha256=file_sha(out/'pilot.json'),crosscheck_sha256=file_sha(out/'crosscheck.json'))
            if (out/'manifest.json').exists() and read(out/'manifest.json') != manifest:
                raise ValueError('Completed structure manifest replay differs')
            atomic_json(out/'manifest.json',manifest)
            SidecarStore(out,cache,split)
            update(status='completed',stage='complete',committed_targets=total)
            return manifest
        except Paused as exc:
            update(status='paused',reason=str(exc)); return dict(status='paused')
        except BaseException as exc:
            update(status='failed',error=repr(exc),traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--split',choices=tuple(SPECS),required=True);p.add_argument('--pilot-only',action='store_true')
    a=p.parse_args(); result=run(a.project,a.split,a.pilot_only)
    if result.get('status')=='paused': raise SystemExit(75)
