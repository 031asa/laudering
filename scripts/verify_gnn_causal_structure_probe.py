"""Independent read-only input/matching checks for the Train structure probe.

Writes only a new verification report in the probe output directory. It never
imports the extraction runner, fits a model, reads Validation/Test, or mutates
any source data, model, or existing manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/gnn_causal_structure_probe_20261006'
COUNT, POSITIVES = 20_318_992, 17_794
START = int(np.datetime64('2022-09-01', 'us').astype(np.int64))
DAY = 86_400_000000
FIELDS = ('src', 'dst', 'timestamp', 'paid_scaled', 'received_scaled',
          'payment_currency_cat', 'receiving_currency_cat',
          'payment_format_cat', 'in_ptr', 'out_ptr', 'in_order',
          'out_order', 'bank_key')
ROW_FIELDS = ('src', 'dst', 'timestamp', 'paid_scaled', 'received_scaled',
              'payment_format_cat', 'payment_currency_cat',
              'receiving_currency_cat')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_targets(final=False):
    prefix = 'final_' if final else ''
    manifest_path = OUT / (prefix + 'targets_manifest.json')
    manifest_sha = sha(manifest_path)
    manifest = read(manifest_path)
    require(manifest['status'] == 'verified', 'Targets manifest is not verified')
    require(set(manifest['outputs']) == {
        prefix + 'targets.parquet', prefix + 'pairs.parquet', prefix + 'unmatched_positives.csv'},
        'Unexpected target output set')
    for name, digest in manifest['outputs'].items():
        require(sha(OUT / name) == digest, 'Target output SHA mismatch: ' + name)
    design_path = OUT / ('design_bank_ach.json' if final else 'design.json')
    require(sha(design_path) == manifest['design_sha256'], 'Design SHA mismatch')
    source_path = OUT / 'source_archive' / ('runner_bank_prepare.py' if final else 'runner_strict_prepare.py')
    require(sha(source_path) == manifest['script_sha256'],
            'Preparation source SHA mismatch')
    design = read(design_path)
    if final:
        require(design['strict_manifest_sha256'] == sha(OUT / 'targets_manifest.json'),
                'Final design is not bound to the original strict preparation')
    require(design['validation_test_read'] is False and
            design['history_labels_as_features'] is False and
            design['frozen_assets_modified'] is False, 'Unexpected diagnostic scope')

    config_path = ROOT / 'result/gnn_train_gpu_v1/config.json'
    training_path = ROOT / 'result/gnn_train_gpu_v1/training_result.json'
    config_sha = sha(config_path)
    require(read(training_path)['config_sha256'] == config_sha, 'Frozen training config binding mismatch')
    frozen = {name.replace('\\', '/'): digest for name, digest in read(config_path)['input_hashes'].items()}
    sources = {name.replace('\\', '/'): digest for name, digest in manifest['source_hashes'].items()}
    verified = {}
    arrays = {}
    for name in FIELDS:
        rel = 'data/gnn_v1_train/history_index/' + name + '.npy'
        require(rel in sources and sources[rel] == frozen[rel], 'History source binding mismatch: ' + rel)
        require(sha(ROOT / rel) == sources[rel], 'History actual SHA mismatch: ' + rel)
        arrays[name] = np.load(ROOT / rel, mmap_mode='r', allow_pickle=False)
        verified[rel] = sources[rel]
    for name in ROW_FIELDS:
        require(arrays[name].shape == (COUNT,), 'Unexpected historical array shape: ' + name)
    require(arrays['timestamp'].dtype == np.dtype('int64'), 'Timestamp precision changed')
    for name in ('paid_scaled', 'received_scaled'):
        require(arrays[name].dtype == np.dtype('int64'), 'Amount precision changed: ' + name)
    require(int(arrays['timestamp'].min()) >= START and
            int(arrays['timestamp'].max()) < START + 10 * DAY,
            'Historical timestamps outside Train')

    bank_path = ROOT / 'data/gnn_v1_train/metadata/bank_identity.json'
    bank_rel = bank_path.relative_to(ROOT).as_posix()
    require(sha(bank_path) == frozen[bank_rel] == manifest['bank_identity_sha256'],
            'Bank identity SHA mismatch')
    bank = read(bank_path)
    require(isinstance(bank, list) and bank.count('70') == 1, 'Raw bank 70 identity is not unique')
    bank70 = bank.index('70')
    require(bank70 == 122017 == manifest['bank70_identity'], 'Encoded bank 70 mismatch')
    vocab_path = ROOT / 'data/gnn_v1_train/metadata/vocabulary.json'
    vocab_rel = vocab_path.relative_to(ROOT).as_posix()
    require(sha(vocab_path) == frozen[vocab_rel], 'Frozen vocabulary mismatch')
    vocab = read(vocab_path)
    ach = vocab['values']['payment_format'].index('ACH') + 2
    verified.update({bank_rel: frozen[bank_rel], vocab_rel: frozen[vocab_rel]})

    # Re-read frozen Train supervision independently: no use of target labels
    # or any previous derived positive list to establish complete coverage.
    label_paths = sorted((ROOT / 'data/gnn_v1_train/supervision/targets').rglob('*.parquet'))
    expected = sorted(name for name in frozen
                      if name.startswith('data/gnn_v1_train/supervision/targets/')
                      and name.endswith('.parquet'))
    require([path.relative_to(ROOT).as_posix() for path in label_paths] == expected,
            'Train supervision file inventory mismatch')
    labels = np.full(COUNT, -1, dtype=np.int8)
    for path in label_paths:
        rel = path.relative_to(ROOT).as_posix()
        require(sources[rel] == frozen[rel] == sha(path), 'Supervision SHA mismatch: ' + rel)
        verified[rel] = frozen[rel]
        for batch in pq.ParquetFile(path).iter_batches(columns=['edge_id', 'label', 'split'], use_threads=False):
            ids = batch['edge_id'].to_numpy().astype(np.int64)
            values = batch['label'].to_numpy()
            require(np.all((ids >= 0) & (ids < COUNT)), 'Supervision edge ID outside Train')
            require(len(np.unique(ids)) == len(ids) and np.all(labels[ids] == -1),
                    'Duplicate supervision edge IDs')
            require(np.isin(values, [0, 1]).all(), 'Invalid supervision labels')
            expected_split = np.where(arrays['timestamp'][ids] < START + 7*DAY,
                                      'discovery', 'internal_validation')
            require(np.array_equal(np.asarray(batch['split'].to_pylist()), expected_split),
                    'Supervision leaf split does not match its Train date')
            labels[ids] = values
    require(np.isin(labels, [0, 1]).all() and int(labels.sum()) == POSITIVES,
            'Full Train label coverage/count mismatch')
    require(set(sources) == (set(verified) if final else (set(verified) - {bank_rel, vocab_rel})),
            'Unexpected preparation source outside Train history/supervision')

    targets = pd.read_parquet(OUT / (prefix + 'targets.parquet'))
    pairs = pd.read_parquet(OUT / (prefix + 'pairs.parquet'))
    ids = targets['edge_id'].to_numpy(np.int64)
    require(len(ids) > 0 and np.all(np.diff(ids) > 0), 'Targets are not strictly sorted unique IDs')
    require(np.all((ids >= 0) & (ids < COUNT)), 'Target edge ID outside Train')
    for name in ROW_FIELDS:
        require(np.array_equal(targets[name].to_numpy(), arrays[name][ids]),
                'Target identity mismatch: ' + name)
    require(np.array_equal(targets['label'].to_numpy(), labels[ids]), 'Target labels mismatch')
    positive_ids = np.flatnonzero(labels == 1)
    require(np.array_equal(ids[targets['label'].to_numpy() == 1], positive_ids),
            'Not all 17794 Train positives are covered exactly once')
    days = ((arrays['timestamp'][ids] - START) // DAY + 1).astype(np.int8)
    require(np.array_equal(targets['day'].to_numpy(), days), 'Target dates mismatch')
    require(np.all((days >= 1) & (days <= 10)), 'Target dates outside Train')
    cohorts = np.where(arrays['payment_format_cat'][ids] == ach, 'ACH',
                       np.where(arrays['bank_key'][arrays['src'][ids]] == bank70,
                                'bank70_nonACH', 'other_nonACH'))
    require(np.array_equal(targets['cohort'].to_numpy(), cohorts), 'Target cohorts mismatch')
    require(not pairs.duplicated(['case_edge_id', 'control_edge_id']).any(), 'Repeated control within case')
    pair_counts = pairs.groupby('case_edge_id').size()
    require(pair_counts.between(1, 5).all(), 'Controls per case outside 1..5')
    cases = pairs.case_edge_id.to_numpy(np.int64)
    controls = pairs.control_edge_id.to_numpy(np.int64)
    require(np.isin(cases, ids).all() and np.isin(controls, ids).all(), 'Pair member not in targets')
    require(np.all(cases != controls) and np.all(labels[cases] == 1) and np.all(labels[controls] == 0),
            'Pair label/identity mismatch')
    for name in ('payment_format_cat', 'payment_currency_cat', 'receiving_currency_cat'):
        require(np.array_equal(arrays[name][cases], arrays[name][controls]),
                'Exact pair matching failed: ' + name)
    exact_payer = np.ones(len(cases), dtype=bool)
    if final:
        exact_payer = arrays['payment_format_cat'][cases] != ach
        require(np.array_equal(pairs['matching_level'].to_numpy(),
                               np.where(exact_payer, 'payer', 'payer_bank')), 'Final matching level metadata mismatch')
        require(np.array_equal(arrays['bank_key'][arrays['src'][cases[~exact_payer]]],
                               arrays['bank_key'][arrays['src'][controls[~exact_payer]]]),
                'Exact ACH payer-bank match failed')
    require(np.array_equal(arrays['src'][cases[exact_payer]], arrays['src'][controls[exact_payer]]),
            'Exact payer match failed')
    case_days = ((arrays['timestamp'][cases] - START) // DAY + 1).astype(np.int8)
    control_days = ((arrays['timestamp'][controls] - START) // DAY + 1).astype(np.int8)
    require(np.array_equal(case_days, control_days) and np.array_equal(case_days, pairs.day.to_numpy()),
            'Pair date mismatch')
    require(np.array_equal(arrays['src'][cases], pairs.src.to_numpy()), 'Pair payer metadata mismatch')
    require(np.array_equal(arrays['src'][cases] == arrays['dst'][cases],
                           arrays['src'][controls] == arrays['dst'][controls]), 'Pair self-status mismatch')
    indexed = targets.set_index('edge_id')
    require(np.array_equal(indexed.loc[cases, 'cohort'].to_numpy(), pairs.cohort.to_numpy()),
            'Pair cohort metadata mismatch')
    for name in ('paid_scaled', 'received_scaled'):
        # Python integers ensure exact half/double checks even near int64 max.
        require(all(0 <= int(c) and 0 <= int(n) and int(c) <= 2*int(n) and int(n) <= 2*int(c)
                    for c, n in zip(arrays[name][cases], arrays[name][controls])),
                'Exact integer amount caliper failed: ' + name)
    require(np.array_equal(np.sort(np.unique(controls)), ids[targets['label'].to_numpy() == 0]),
            'Unpaired or missing target controls')
    unmatched = pd.read_csv(OUT / (prefix + 'unmatched_positives.csv'))
    unmatched_ids = unmatched.edge_id.to_numpy(np.int64)
    require(len(np.unique(unmatched_ids)) == len(unmatched_ids), 'Duplicate unmatched positives')
    require(np.array_equal(np.sort(unmatched_ids), np.setdiff1d(positive_ids, np.unique(cases))),
            'Unmatched positive partition mismatch')
    for name in ('day', 'src', 'cohort'):
        require(np.array_equal(indexed.loc[unmatched_ids, name].to_numpy(), unmatched[name].to_numpy()),
                'Unmatched positive metadata mismatch: ' + name)

    actual = dict(targets=len(targets), positives=int(targets.label.sum()),
                  matched_positives=int(len(pair_counts)), unique_controls=int(len(np.unique(controls))),
                  pairs=len(pairs), cohort_positive_counts=targets.loc[targets.label == 1, 'cohort'].value_counts().to_dict(),
                  cohort_matched_positive_counts=pairs.drop_duplicates('case_edge_id').cohort.value_counts().to_dict())
    require(actual == manifest['counts'], 'Manifest counts mismatch')
    require(actual['cohort_positive_counts'] == {'ACH': 15067, 'bank70_nonACH': 2551, 'other_nonACH': 176},
            'Full Train positive cohort totals changed')
    for daily in manifest['daily']:
        day = int(daily['day'])
        all_day = (arrays['timestamp'] >= START+(day-1)*DAY) & (arrays['timestamp'] < START+day*DAY)
        require(int(all_day.sum()) == daily['rows'] and int(labels[all_day].sum()) == daily['positives'],
                'Manifest daily source counts mismatch')
        require(int((pairs.day == day).sum()) == daily['pairs'], 'Manifest daily pair count mismatch')
    require(sha(manifest_path) == manifest_sha, 'Targets manifest changed during verification')
    for name, digest in manifest['outputs'].items():
        require(sha(OUT/name) == digest, 'Target output changed during verification: ' + name)
    return dict(status='passed', stage='targets_final' if final else 'targets_strict', completed_at=datetime.now(timezone.utc).isoformat(),
                verifier_sha256=sha(__file__), targets_manifest_sha256=manifest_sha,
                frozen_training_config_sha256=config_sha, source_hashes=verified,
                counts=actual, bank70_identity={'raw': '70', 'encoded': bank70},
                checks=['all target and pair output hashes', 'full frozen Train source identity',
                        'all 17794 Train positives exactly once', 'full Train supervision reconstruction',
                        'exact target array identity', 'case/control opposite labels and distinct IDs',
                        ('day/payer-bank(ACH)/payer(non-ACH)/payment/both currencies/self exact matches'
                         if final else 'day/payer/payment/both currencies/self exact matches'),
                        'integer-safe factor-two paid/received amounts', 'one to five unique controls per case',
                        'all unmatched positives accounted for', 'raw bank 70 frozen identity mapping'],
                limitations=[('ACH uses payer-bank matching; same-payer sensitivity remains separate and account confounding remains possible.'
                              if final else 'ACH matching covers a small subset; no claim of ACH-wide conditional utility.'),
                             'Matched controls do not estimate population precision, recall, or F1.',
                             'No Validation/Test read; no model trained or frozen source modified.'],
                validation_test_read=False, frozen_assets_modified=False)


def verify_witnesses():
    """Check all stored paths, then independently enumerate low-degree cases."""
    manifest_path = OUT / 'structures_manifest.json'
    manifest_sha = sha(manifest_path)
    manifest = read(manifest_path)
    require(manifest['status'] == 'completed', 'Structures are incomplete')
    require(sha(OUT/'structures.parquet') == manifest['sha256'], 'Structure table SHA mismatch')
    binding = manifest['binding']
    require(sha(OUT/'final_targets.parquet') == binding['targets_sha256'] and
            sha(OUT/'targets.parquet') == binding['strict_targets_sha256'], 'Structure target binding mismatch')
    require(sha(ROOT/'scripts/gnn_causal_structures.py') == binding['engine_sha256'], 'Engine SHA mismatch')
    for final, report_name in [(False, 'verification_targets_strict.json'), (True, 'verification_targets_final.json')]:
        report = read(OUT/report_name)
        require(report['status'] == 'passed', 'Target verification not passed')
        require(report['frozen_training_config_sha256'] == sha(ROOT/'result/gnn_train_gpu_v1/config.json'),
                'Frozen config changed after target verification')
        require(report['targets_manifest_sha256'] == sha(OUT/(('final_' if final else '')+'targets_manifest.json')),
                'Target verification manifest binding mismatch')
    cfg = read(ROOT/'result/gnn_train_gpu_v1/config.json')
    frozen = {name.replace('\\', '/'): value for name, value in cfg['input_hashes'].items()}
    a = {}
    for name in FIELDS:
        rel = 'data/gnn_v1_train/history_index/' + name + '.npy'
        require(binding['history_hashes'][rel] == frozen[rel] == sha(ROOT/rel), 'Witness source identity mismatch: '+rel)
        a[name] = np.load(ROOT/rel, mmap_mode='r', allow_pickle=False)
    t = pd.concat([pd.read_parquet(OUT/'final_targets.parquet'), pd.read_parquet(OUT/'targets.parquet')])
    # Duplicate identities must agree before deduplication.
    require(not t.drop_duplicates().duplicated('edge_id').any(), 'Strict/final target identities disagree')
    t = t.drop_duplicates('edge_id').sort_values('edge_id').reset_index(drop=True)
    f = pd.read_parquet(OUT/'structures.parquet').set_index('edge_id').sort_index()
    require(f.index.is_unique and np.array_equal(f.index.to_numpy(), t.edge_id.to_numpy()), 'Structure target coverage mismatch')
    require(len(f) == manifest['rows'] and ['edge_id']+list(f.columns) == manifest['columns'], 'Structure schema/count mismatch')
    mapping = {
        'prior_in': 'prior_in_same_currency_count_72h',
        'prior_in_amount_match': 'prior_in_amount_match_count_72h',
        'chain3': 'chain3_path_count_72h', 'chain3_amount_match': 'chain3_amount_match_path_count_72h',
        'cycle2': 'cycle2_count_72h', 'cycle2_amount_match': 'cycle2_amount_match_count_72h',
        'cycle3': 'cycle3_path_count_72h', 'cycle3_amount_match': 'cycle3_amount_match_path_count_72h',
        'parallel2': 'parallel2_path_count_72h', 'parallel2_amount_match': 'parallel2_amount_match_path_count_72h'}
    count_names = ['prior_in_count_72h'] + list(mapping.values()) + [
        'parallel2_middle_accounts_72h', 'parallel2_amount_match_middle_accounts_72h']
    window = 72 * 3_600_000000
    history_start = int(a['timestamp'].min())
    seen = set()
    counts = {name: 0 for name in mapping}
    parts = []

    def amount_match(receipt, paid):
        # A different expression from the production bounds implementation.
        return 9 * int(paid) <= 10 * int(receipt) <= 11 * int(paid)

    for day in range(1,11):
        name = f'day_{day:02d}.json'
        require(sha(OUT/'parts'/name) == manifest['parts'][name], 'Part manifest SHA mismatch')
        part = read(OUT/'parts'/name)
        require(part['status'] == 'verified' and part['binding'] == binding, 'Part binding mismatch')
        for output, digest in part['outputs'].items():
            require(sha(OUT/'parts'/output) == digest, 'Part payload SHA mismatch: '+output)
        part_frame = pd.read_parquet(OUT/'parts'/f'day_{day:02d}.parquet').set_index('edge_id')
        require(np.array_equal(part_frame.index.to_numpy(), t.loc[t.day == day, 'edge_id'].to_numpy()), 'Part day/target coverage mismatch')
        pd.testing.assert_frame_equal(f.loc[part_frame.index], part_frame, check_dtype=False)
        records = read(OUT/'parts'/f'day_{day:02d}_witnesses.json')
        require(len(records) == len(part_frame) == part['rows'], 'Witness part row count mismatch')
        require([int(row['edge_id']) for row in records] == part_frame.index.tolist(), 'Witness row identity mismatch')
        for record in records:
            eid = int(record['edge_id'])
            require(eid not in seen, 'Duplicate witness target')
            seen.add(eid)
            u, v, now = int(a['src'][eid]), int(a['dst'][eid]), int(a['timestamp'][eid])
            currency = int(a['payment_currency_cat'][eid])
            same = currency == int(a['receiving_currency_cat'][eid])
            row = f.loc[eid]
            require(bool(row.eligible) == (u != v) and bool(row.not_applicable) == (u == v), 'Eligibility mask mismatch')
            require(bool(row.same_currency_target) == same, 'Target currency mask mismatch')
            require(bool(row.full_window_72h) == (now-window >= history_start), 'Full-window mask mismatch')
            require(set(record['witnesses']) == set(mapping), 'Unexpected witness family set')
            for key in count_names:
                value = row[key]
                missing = u == v or (not same and key.startswith(('cycle', 'parallel')))
                require(bool(pd.isna(value)) == missing, 'Count applicability mismatch: '+key)
                if not missing:
                    require(np.isfinite(value) and value >= 0 and float(value).is_integer(), 'Invalid count: '+key)
            for smaller,larger in [
                ('prior_in_same_currency_count_72h','prior_in_count_72h'),
                ('prior_in_amount_match_count_72h','prior_in_same_currency_count_72h'),
                ('chain3_amount_match_path_count_72h','chain3_path_count_72h'),
                ('cycle2_amount_match_count_72h','cycle2_count_72h'),
                ('cycle3_amount_match_path_count_72h','cycle3_path_count_72h'),
                ('parallel2_amount_match_path_count_72h','parallel2_path_count_72h'),
                ('parallel2_middle_accounts_72h','parallel2_path_count_72h'),
                ('parallel2_amount_match_middle_accounts_72h','parallel2_amount_match_path_count_72h'),
                ('parallel2_amount_match_middle_accounts_72h','parallel2_middle_accounts_72h')]:
                if not pd.isna(row[smaller]):
                    require(row[smaller] <= row[larger], 'Count nesting relation failed: '+smaller)
            for family, count_name in mapping.items():
                witness = record['witnesses'][family]
                expected = not pd.isna(row[count_name]) and row[count_name] > 0
                require(bool(witness) == bool(expected), 'Count/witness existence mismatch: '+family)
                if not witness:
                    continue
                counts[family] += 1
                path = [int(x) for x in witness]
                require(len(set(path)) == len(path) and all(0 <= x < COUNT for x in path), 'Invalid witness IDs')
                parallel = family.startswith('parallel2')
                expected_length = 3 if family.startswith(('chain3', 'cycle3')) else 2
                require(len(path) == expected_length, 'Wrong witness path length')
                historical = path if parallel else path[:-1]
                require(eid not in historical and (parallel or path[-1] == eid), 'Target position in witness is wrong')
                times = [int(a['timestamp'][x]) for x in path]
                require(all(now-window <= int(a['timestamp'][x]) < now for x in historical), 'History timestamp is outside [t-72h,t)')
                require(all(left < right for left, right in zip(times, times[1:])), 'Witness time is not strictly increasing')
                require(all(int(a['src'][x]) != int(a['dst'][x]) for x in path), 'Self edge in witness')
                require(all(int(a['dst'][left]) == int(a['src'][right]) for left,right in zip(path,path[1:])), 'Broken path direction')
                nodes = [int(a['src'][path[0]])] + [int(a['dst'][x]) for x in path]
                if family.startswith('chain3'):
                    require(len(set(nodes)) == 4 and nodes[-2:] == [u,v], 'Chain is not a simple target-ending path')
                elif family.startswith('cycle'):
                    require(nodes[0] == nodes[-1] == v and len(set(nodes[:-1])) == len(nodes)-1,
                            'Cycle is not simple/closed at current receiver')
                elif parallel:
                    require(nodes[0] == u and nodes[-1] == v and len(set(nodes)) == 3,
                            'Parallel historical path is not simple from current payer to receiver')
                else:
                    require(nodes[-2:] == [u,v], 'Incoming receipt does not precede current target')
                if family.startswith('prior_in'):
                    require(int(a['receiving_currency_cat'][historical[0]]) == currency, 'Incoming receipt currency mismatch')
                else:
                    require(all(int(a['payment_currency_cat'][x]) == currency == int(a['receiving_currency_cat'][x])
                                for x in historical), 'Historical path currency mismatch')
                if family.startswith(('cycle', 'parallel')):
                    require(same, 'Cycle/parallel witness for a cross-currency target')
                if 'amount_match' in family:
                    require(all(amount_match(a['received_scaled'][left], a['paid_scaled'][right])
                                for left,right in zip(path,path[1:])), 'Consecutive exact amount match failed')
                    if family.startswith('cycle'):
                        require(amount_match(a['received_scaled'][eid], a['paid_scaled'][path[0]]), 'Closing cycle amount join failed')
                    if family == 'prior_in_amount_match':
                        expected_gap = (now-int(a['timestamp'][path[0]]))/60_000000
                        require(float(row.prior_in_match_min_gap_minutes_72h) == expected_gap, 'Incoming gap witness mismatch')
            gap = row.prior_in_match_min_gap_minutes_72h
            has_match = not pd.isna(row.prior_in_amount_match_count_72h) and row.prior_in_amount_match_count_72h > 0
            require(bool(pd.isna(gap)) == (not has_match), 'Incoming gap applicability mismatch')
        parts.append({'day': day, 'targets': len(records)})
    require(seen == set(t.edge_id), 'Incomplete witness target coverage')

    def brute(eid):
        """Literal bounded nested loops, no production engine/helper import."""
        u,v,now = int(a['src'][eid]),int(a['dst'][eid]),int(a['timestamp'][eid])
        cur=int(a['payment_currency_cat'][eid]); same=cur == int(a['receiving_currency_cat'][eid])
        vals={name:0 for name in count_names}; vals['prior_in_match_min_gap_minutes_72h']=None
        if u == v:
            return {name:None for name in vals},0
        cache={}; work=0
        def incoming(node):
            if node not in cache:
                begin,end=int(a['in_ptr'][node]),int(a['in_ptr'][node+1])
                if end-begin > 20000:
                    raise OverflowError('Lifetime adjacency exceeds independent-brute budget')
                edges=a['in_order'][begin:end].astype(np.int64)
                keep=(a['timestamp'][edges]>=now-window)&(a['timestamp'][edges]<now)&(a['src'][edges]!=a['dst'][edges])
                cache[node]=[int(x) for x in edges[keep]]
            return cache[node]
        def currency_ok(edge):
            return int(a['payment_currency_cat'][edge]) == cur == int(a['receiving_currency_cat'][edge])
        matches=[]
        for second in incoming(u):
            vals['prior_in_count_72h']+=1
            if int(a['receiving_currency_cat'][second])==cur:
                vals['prior_in_same_currency_count_72h']+=1
                if amount_match(a['received_scaled'][second],a['paid_scaled'][eid]):
                    vals['prior_in_amount_match_count_72h']+=1;matches.append(second)
            if not currency_ok(second):continue
            middle=int(a['src'][second])
            if same and middle==v:
                vals['cycle2_count_72h']+=1
                if amount_match(a['received_scaled'][second],a['paid_scaled'][eid]) and amount_match(a['received_scaled'][eid],a['paid_scaled'][second]):
                    vals['cycle2_amount_match_count_72h']+=1
            if middle in (u,v):continue
            for first in incoming(middle):
                work+=1
                if work>200000:raise OverflowError('Pair budget exceeded')
                if not currency_ok(first) or int(a['timestamp'][first])>=int(a['timestamp'][second]):continue
                first_src=int(a['src'][first])
                joined=amount_match(a['received_scaled'][first],a['paid_scaled'][second]) and amount_match(a['received_scaled'][second],a['paid_scaled'][eid])
                if first_src not in (u,v):
                    vals['chain3_path_count_72h']+=1
                    vals['chain3_amount_match_path_count_72h']+=int(joined)
                elif same and first_src==v:
                    vals['cycle3_path_count_72h']+=1
                    vals['cycle3_amount_match_path_count_72h']+=int(joined and amount_match(a['received_scaled'][eid],a['paid_scaled'][first]))
        if matches:vals['prior_in_match_min_gap_minutes_72h']=min((now-int(a['timestamp'][x]))/60_000000 for x in matches)
        if same:
            middles=set();matched_middles=set()
            for second in incoming(v):
                middle=int(a['src'][second])
                if middle in (u,v) or not currency_ok(second):continue
                for first in incoming(middle):
                    work+=1
                    if work>200000:raise OverflowError('Pair budget exceeded')
                    if int(a['src'][first])!=u or not currency_ok(first) or int(a['timestamp'][first])>=int(a['timestamp'][second]):continue
                    vals['parallel2_path_count_72h']+=1;middles.add(middle)
                    if amount_match(a['received_scaled'][first],a['paid_scaled'][second]):
                        vals['parallel2_amount_match_path_count_72h']+=1;matched_middles.add(middle)
            vals['parallel2_middle_accounts_72h']=len(middles)
            vals['parallel2_amount_match_middle_accounts_72h']=len(matched_middles)
        else:
            for name in vals:
                if name.startswith(('cycle','parallel')):vals[name]=None
        return vals,work

    degree=(a['in_ptr'][t.src.to_numpy()+1]-a['in_ptr'][t.src.to_numpy()]+
            a['in_ptr'][t.dst.to_numpy()+1]-a['in_ptr'][t.dst.to_numpy()])
    candidates=t[(degree<=1000)&(t.src!=t.dst)].copy()
    rng=np.random.default_rng(20260920);checked=[];skipped=[]
    # Round-robin cohort/label sampling avoids letting a large ACH cohort occupy
    # every independent check, while preserving a fixed seed and bounded work.
    grouped=[rng.permutation(g.edge_id.to_numpy())[:40].tolist() for _,g in candidates.groupby(['cohort','label'])]
    ordered=[g[i] for i in range(max(map(len,grouped))) for g in grouped if i<len(g)]
    for eid in ordered:
        try:expected,work=brute(int(eid))
        except OverflowError as error:
            skipped.append({'edge_id':int(eid),'reason':str(error)});continue
        for name,value in expected.items():
            actual=f.loc[eid,name]
            require(pd.isna(actual) if value is None else (not pd.isna(actual) and float(actual)==float(value)),
                    f'Independent brute count mismatch: {eid} {name}, expected={value}, actual={actual}')
        info=t.loc[t.edge_id==eid].iloc[0]
        checked.append({'edge_id':int(eid),'cohort':str(info.cohort),'label':int(info.label),'day':int(info.day),'enumerated_pairs':work})
        if len(checked)>=24:break
    require(len(checked)>=20, 'Insufficient independent bounded brute checks')
    require(sha(manifest_path)==manifest_sha,'Structure manifest changed during verification')
    return dict(status='passed',stage='witnesses',completed_at=datetime.now(timezone.utc).isoformat(),
                verifier_sha256=sha(__file__),structures_manifest_sha256=manifest_sha,
                targets=len(t),all_stored_witness_paths_checked=sum(counts.values()),witness_counts=counts,
                parts=parts,all_target_masks_checked=True,all_count_witness_existence_checked=True,
                exact_brute_sample=checked,brute_skipped_budget=skipped,
                checks=['all part and source SHA bindings','all saved witnesses: edge identity/direction/simple motif/time/currency',
                        'exact integer amount joins including closed-cycle join','all full-window/applicability masks',
                        'all positive motif counts have witnesses and zero/missing counts have none',
                        '24 fixed-seed low-degree targets counted independently with bounded literal loops'],
                coverage_semantics='full_window_72h is conservative: earliest observed Train timestamp, not midnight boundary',
                limitations=['All stored witness paths verified; complete counts independently enumerated only for listed low-degree sample.',
                             'Existence of a path or close amounts is not proof that funds followed that path.'],
                validation_test_read=False,frozen_assets_modified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['targets', 'targets_strict', 'targets_final', 'witnesses'], required=True)
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    result = verify_witnesses() if args.stage == 'witnesses' else verify_targets(final=args.stage == 'targets_final')
    summary = ({'counts': result['counts']} if 'counts' in result else {
        'targets': result['targets'],
        'all_stored_witness_paths_checked': result['all_stored_witness_paths_checked'],
        'independent_brute_targets': len(result['exact_brute_sample']),
    })
    require(shutil.disk_usage(ROOT).free >= 20 * 1024**3, 'Disk reserve below 20 GiB')
    path = OUT / ('verification_targets_strict.json' if args.stage in ('targets', 'targets_strict') else 'verification_' + args.stage + '.json')
    if path.exists():
        require(read(path)['status'] == 'passed', 'Existing verification is not passed; preserve it')
        print(json.dumps({'status': 'passed_reverified', 'stage': args.stage,
                          **summary}, ensure_ascii=True))
        return
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)
    print(json.dumps({'status': 'passed', 'stage': args.stage,
                      **summary}, ensure_ascii=True))


if __name__ == '__main__':
    main()
