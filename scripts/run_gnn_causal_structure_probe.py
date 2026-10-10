"""Train-only causal structure discovery; no training, Test access, or frozen writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/gnn_causal_structure_probe_20261006'
START = int(np.datetime64('2022-09-01', 'us').astype('int64'))
DAY = 86400_000000
COUNT = 20_318_992
SEED = 20260920
FIELDS = ('src', 'dst', 'timestamp', 'paid_scaled', 'received_scaled',
          'payment_currency_cat', 'receiving_currency_cat', 'payment_format_cat',
          'in_ptr', 'out_ptr', 'in_order', 'out_order', 'bank_key')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def space():
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Disk reserve below 20 GiB')
    if OUT.exists() and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) > 2*1024**3:
        raise RuntimeError('Diagnostic output exceeds its 2 GiB budget')


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write(name, value):
    space()
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=True, indent=2,
                              allow_nan=False), 'utf-8')
    os.replace(tmp, p)


def progress(**kw):
    d = dict(pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), **kw)
    write('status.json', d)
    print(json.dumps(clean(d), ensure_ascii=True), flush=True)


def arrays_verified():
    config = ROOT / 'result/gnn_train_gpu_v1/config.json'
    result = read(ROOT / 'result/gnn_train_gpu_v1/training_result.json')
    assert result['config_sha256'] == sha(config)
    frozen = {k.replace('\\', '/'): v for k, v in read(config)['input_hashes'].items()}
    a, hashes = {}, {}
    for key in FIELDS:
        p = ROOT / 'data/gnn_v1_train/history_index' / (key + '.npy')
        rel = p.relative_to(ROOT).as_posix()
        digest = sha(p)
        assert digest == frozen[rel], rel
        hashes[rel] = digest
        a[key] = np.load(p, mmap_mode='r', allow_pickle=False)
    assert len(a['timestamp']) == COUNT
    assert int(a['timestamp'].min()) >= START
    assert int(a['timestamp'].max()) < START + 10*DAY
    return a, frozen, hashes


def prepare(bank_matched=False):
    prefix = 'final_' if bank_matched else ''
    manifest_name = prefix+'targets_manifest.json'
    target_name, pair_name = prefix+'targets.parquet', prefix+'pairs.parquet'
    unmatched_name = prefix+'unmatched_positives.csv'
    design_name = 'design_bank_ach.json' if bank_matched else 'design.json'
    if (OUT/manifest_name).exists():
        m = read(OUT/manifest_name)
        for n, h in m['outputs'].items(): assert sha(OUT/n) == h
        progress(stage='prepare', status='verified_reuse', **m['counts'])
        return
    design = dict(
        version=1, scope='Train Sep01..10 only; fixed matched case-control diagnostic; no fitting',
        history='Unlabeled external directed graph [t-72h,t); exclude all timestamp ties',
        motifs=['incoming turnover', 'two historical edges preceding current edge',
                'current-edge closure of 2/3-edge cycles', 'historical parallel two-hop paths'],
        amount_matching='Each comparable consecutive amount within 10%; integer-safe boundaries; zero only matches zero',
        controls='Up to five deterministic random negatives per positive: exact day, payer, payment, both currencies and self status; both amounts within factor two. Reuse between cases allowed. No score/structure matching.',
        discovery='Sep01..07', intermediate='Sep08', later_check='Sep09..10',
        cohorts=['ACH', 'bank70_nonACH', 'other_nonACH'], seed=SEED,
        selection='Direction fixed on discovery; effect >=0.03 and same later direction; discovery/later Holm p<0.05 and endpoint-cluster CI excluding 0.5 with >=5 clusters. Otherwise exploratory only.',
        limitation='Sampled controls do not estimate population precision or recall. Matched subset may omit positive-only payer strata. No temporal holdout is an untouched Test.',
        new_storage_budget_gib=2, frozen_assets_modified=False,
        history_labels_as_features=False, validation_test_read=False)
    if bank_matched:
        design['controls'] += ' ACH main analysis replaces exact payer by exact payer bank; strict-payer analysis retained separately for sensitivity. Non-ACH stays exact-payer matched.'
        design['reason'] = 'Only 384/15067 ACH positives have eligible exact-payer controls; adjustment based solely on coverage, before structure extraction.'
        design['strict_manifest_sha256'] = sha(OUT/'targets_manifest.json')
    if (OUT/design_name).exists(): assert read(OUT/design_name) == design
    else: write(design_name, design)
    a, frozen, hashes = arrays_verified()
    y = np.full(COUNT, -1, dtype=np.int8)
    paths = sorted((ROOT/'data/gnn_v1_train/supervision/targets').rglob('*.parquet'))
    expected = sorted(k for k in frozen if k.startswith('data/gnn_v1_train/supervision/targets/') and k.endswith('.parquet'))
    assert [p.relative_to(ROOT).as_posix() for p in paths] == expected
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        assert sha(p) == frozen[rel]
        hashes[rel] = frozen[rel]
        for b in pq.ParquetFile(p).iter_batches(columns=['edge_id','label'], use_threads=False):
            ids, labels = b['edge_id'].to_numpy(), b['label'].to_numpy()
            assert np.isin(labels,[0,1]).all() and len(np.unique(ids)) == len(ids)
            assert np.all(y[ids] == -1)
            y[ids] = labels
    assert np.isin(y,[0,1]).all() and int(y.sum()) == 17794
    vp = ROOT/'data/gnn_v1_train/metadata/vocabulary.json'
    assert sha(vp) == frozen[vp.relative_to(ROOT).as_posix()]
    hashes[vp.relative_to(ROOT).as_posix()] = sha(vp)
    vocab = read(vp)
    ach = vocab['values']['payment_format'].index('ACH') + 2
    identity = ROOT/'data/gnn_v1_train/metadata/bank_identity.json'
    # Use the frozen raw-bank identity mapping, never compare encoded ID to 70.
    bank = read(identity)
    assert isinstance(bank,list) and sha(identity) == frozen[identity.relative_to(ROOT).as_posix()]
    hashes[identity.relative_to(ROOT).as_posix()] = sha(identity)
    bank70 = bank.index('70')
    all_positive = np.flatnonzero(y == 1)
    assert int(((a['bank_key'][a['src'][all_positive]] == bank70) &
                (a['payment_format_cat'][all_positive] != ach)).sum()) == 2551
    rng = np.random.default_rng(SEED)
    pairs, coverage = [], []
    selected = set(map(int, all_positive))
    unmatched = []
    for day in range(1,11):
        di = np.flatnonzero((a['timestamp'] >= START+(day-1)*DAY) & (a['timestamp'] < START+day*DAY))
        src = a['src'][di].astype(np.int64)
        pay = a['payment_format_cat'][di].astype(np.int64)
        pc = a['payment_currency_cat'][di].astype(np.int64)
        rc = a['receiving_currency_cat'][di].astype(np.int64)
        assert np.all(pay < 16) and np.all(pc < 32) and np.all(rc < 32)
        self_flag = a['src'][di] == a['dst'][di]
        match_source = np.where(pay == ach,a['bank_key'][src],src) if bank_matched else src
        key = (((match_source*16+pay)*32+pc)*32+rc)*2+self_flag
        pos = np.flatnonzero(y[di] == 1)
        neg = np.flatnonzero((y[di] == 0) & np.isin(key, np.unique(key[pos])))
        order = np.lexsort((a['paid_scaled'][di[neg]],key[neg]))
        ni, nk = di[neg[order]], key[neg[order]]
        controls_day = 0
        for pi in pos:
            eid = int(di[pi]); k = key[pi]
            lo, hi = np.searchsorted(nk,[k,k],side='left')[0], np.searchsorted(nk,k,side='right')
            group = ni[lo:hi]
            pval, rval = int(a['paid_scaled'][eid]), int(a['received_scaled'][eid])
            pvalues = a['paid_scaled'][group]
            left = int(np.searchsorted(pvalues,(pval+1)//2,'left'))
            upper = min(pval*2,np.iinfo(np.int64).max)
            right = int(np.searchsorted(pvalues,upper,'right'))
            eligible = group[left:right]
            rv = a['received_scaled'][eligible]
            eligible = eligible[(rv >= (rval+1)//2) & (rv <= min(rval*2,np.iinfo(np.int64).max))]
            cohort = 'ACH' if pay[pi] == ach else ('bank70_nonACH' if int(a['bank_key'][src[pi]]) == bank70 else 'other_nonACH')
            if not len(eligible):
                unmatched.append(dict(edge_id=eid,day=day,src=int(src[pi]),cohort=cohort))
                continue
            chosen = np.sort(rng.choice(eligible,size=min(5,len(eligible)),replace=False))
            for c in chosen:
                pairs.append(dict(case_edge_id=eid,control_edge_id=int(c),day=day,src=int(src[pi]),cohort=cohort,
                                  matching_level='payer_bank' if bank_matched and cohort=='ACH' else 'payer'))
                selected.add(int(c))
            controls_day += len(chosen)
        coverage.append(dict(day=day,rows=len(di),positives=len(pos),pairs=controls_day,eligible_negative_pool=len(ni)))
        progress(stage='matching',status='running',day=day,selected_targets=len(selected),pairs=len(pairs))
    ids = np.asarray(sorted(selected),np.int64)
    frame = pd.DataFrame({k:np.asarray(a[k][ids]) for k in ['src','dst','timestamp','paid_scaled','received_scaled','payment_format_cat','payment_currency_cat','receiving_currency_cat']})
    frame.insert(0,'edge_id',ids)
    frame['label'] = y[ids]
    frame['day'] = ((frame.timestamp-START)//DAY+1).astype('int8')
    frame['cohort'] = np.where(frame.payment_format_cat == ach,'ACH',np.where(a['bank_key'][frame.src] == bank70,'bank70_nonACH','other_nonACH'))
    pair_frame = pd.DataFrame(pairs)
    frame.to_parquet(OUT/target_name,index=False)
    pair_frame.to_parquet(OUT/pair_name,index=False)
    pd.DataFrame(unmatched).to_csv(OUT/unmatched_name,index=False)
    counts = dict(targets=len(frame),positives=int(frame.label.sum()),matched_positives=int(pair_frame.case_edge_id.nunique()),
                  unique_controls=int(pair_frame.control_edge_id.nunique()),pairs=len(pair_frame),
                  cohort_positive_counts=frame.loc[frame.label==1,'cohort'].value_counts().to_dict(),
                  cohort_matched_positive_counts=pair_frame.drop_duplicates('case_edge_id').cohort.value_counts().to_dict())
    write(manifest_name,dict(status='verified',counts=counts,daily=coverage,
          outputs={n:sha(OUT/n) for n in [target_name,pair_name,unmatched_name]},
          source_hashes=hashes,bank70_identity=bank70,bank_identity_sha256=sha(identity),
          design_sha256=sha(OUT/design_name),script_sha256=sha(__file__)))
    progress(stage='prepared',status='completed',**counts)


def extract(pilot=False):
    from gnn_causal_structures import CausalStructures
    m = read(OUT/'final_targets_manifest.json')
    for n,h in m['outputs'].items(): assert sha(OUT/n) == h
    a,_,hashes = arrays_verified()
    t = pd.read_parquet(OUT/'final_targets.parquet')
    # Include original exact-payer controls for the separate ACH sensitivity.
    strict = pd.read_parquet(OUT/'targets.parquet')
    t = pd.concat([t,strict]).drop_duplicates('edge_id').sort_values('edge_id')
    engine = CausalStructures({k:v for k,v in a.items() if k not in ('bank_key','payment_format_cat')})
    source = ROOT/'scripts/gnn_causal_structures.py'
    binding = dict(engine_sha256=sha(source),targets_sha256=sha(OUT/'final_targets.parquet'),strict_targets_sha256=sha(OUT/'targets.parquet'),history_hashes=hashes)
    if pilot:
        # Worst total lifetime endpoint degrees plus cohort/date coverage.
        deg = a['in_ptr'][t.src+1]-a['in_ptr'][t.src]+a['in_ptr'][t.dst+1]-a['in_ptr'][t.dst]
        picks = set(t.iloc[np.argsort(deg)[-10:]].edge_id.astype(int))
        for _,g in t.groupby(['cohort','label','day']): picks.update(g.edge_id.iloc[:1].astype(int))
        selected = t[t.edge_id.isin(picks)]
        rows=[]; began=time.monotonic()
        for eid in selected.edge_id:
            tick=time.monotonic(); result=engine.extract(int(eid))
            rows.append(dict(edge_id=int(eid),seconds=time.monotonic()-tick,result=result))
            progress(stage='pilot',status='running',done=len(rows),total=len(selected),seconds=time.monotonic()-began)
        write('pilot.json',dict(status='completed',binding=binding,rows=rows,seconds=time.monotonic()-began))
        return
    p = read(OUT/'pilot.json'); assert p['status']=='completed' and p['binding']==binding
    began=time.monotonic(); done=0
    for day,g in t.groupby('day'):
        name=f'day_{int(day):02d}'
        manifest=OUT/'parts'/(name+'.json')
        if manifest.exists():
            d=read(manifest); assert d['binding']==binding
            for n,h in d['outputs'].items(): assert sha(OUT/'parts'/n)==h
            done+=len(g); continue
        features=[]; witnesses=[]; tick=time.monotonic()
        for eid in g.edge_id:
            result=engine.extract(int(eid))
            features.append(dict(edge_id=int(eid),**result['features']))
            witnesses.append(dict(edge_id=int(eid),witnesses=result.get('witnesses',{}),work=result.get('work',{})))
            if len(features)%250==0:
                progress(stage='extract',status='running',day=int(day),day_done=len(features),day_total=len(g),completed=done+len(features),total=len(t),seconds=time.monotonic()-began)
        space(); (OUT/'parts').mkdir(exist_ok=True)
        pd.DataFrame(features).to_parquet(OUT/'parts'/(name+'.parquet'),index=False)
        write('parts/'+name+'_witnesses.json',witnesses)
        outputs={n:sha(OUT/'parts'/n) for n in [name+'.parquet',name+'_witnesses.json']}
        write('parts/'+name+'.json',dict(status='verified',rows=len(g),binding=binding,outputs=outputs,seconds=time.monotonic()-tick))
        done+=len(g)
        progress(stage='extract',status='running',completed=done,total=len(t),seconds=time.monotonic()-began)
    frames=[pd.read_parquet(OUT/'parts'/f'day_{d:02d}.parquet') for d in range(1,11)]
    f=pd.concat(frames,ignore_index=True)
    assert f.edge_id.is_unique and set(f.edge_id)==set(t.edge_id)
    f.to_parquet(OUT/'structures.parquet',index=False)
    write('structures_manifest.json',dict(status='completed',binding=binding,rows=len(f),columns=list(f),
         sha256=sha(OUT/'structures.parquet'),parts={f'day_{d:02d}.json':sha(OUT/'parts'/f'day_{d:02d}.json') for d in range(1,11)},seconds=time.monotonic()-began))
    progress(stage='extracted',status='completed',targets=len(f),seconds=time.monotonic()-began)


def main():
    from gnn_features.training_state import ProcessLock
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','prepare_bank','pilot','extract'],required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);space()
    with ProcessLock(OUT):
        if args.stage in ('prepare','prepare_bank'): prepare(args.stage=='prepare_bank')
        else: extract(args.stage=='pilot')


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        progress(stage='failed',status='failed',error=repr(exc))
        raise
