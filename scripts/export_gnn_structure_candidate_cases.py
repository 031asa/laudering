"""Export completed Train structure-screen candidates and their fixed controls.

This is a traceable diagnostic case pack, not a detection rule, model gain, or
causal finding. No historical labels, Test data, model fitting, or edits to
existing preparation/extraction/statistics outputs are permitted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import msvcrt
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
BASELINE = {'prior_in_count_72h', 'prior_in_same_currency_count_72h'}
WITNESS = {
    'prior_in_amount_match_count_72h': 'prior_in_amount_match',
    'prior_in_match_min_gap_minutes_72h': 'prior_in_amount_match',
    'chain3_path_count_72h': 'chain3',
    'chain3_amount_match_path_count_72h': 'chain3_amount_match',
    'cycle2_count_72h': 'cycle2',
    'cycle2_amount_match_count_72h': 'cycle2_amount_match',
    'cycle3_path_count_72h': 'cycle3',
    'cycle3_amount_match_path_count_72h': 'cycle3_amount_match',
    'parallel2_path_count_72h': 'parallel2',
    'parallel2_middle_accounts_72h': 'parallel2',
    'parallel2_amount_match_path_count_72h': 'parallel2_amount_match',
    'parallel2_amount_match_middle_accounts_72h': 'parallel2_amount_match',
}
EDGE_FIELDS = ('src', 'dst', 'timestamp', 'paid_scaled', 'received_scaled',
               'payment_format_cat', 'payment_currency_cat', 'receiving_currency_cat')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def payload(value):
    return json.dumps(clean(value), ensure_ascii=True, indent=2, allow_nan=False).encode('utf-8')


def exact_decimal(value):
    value = int(value)
    require(value >= 0, 'Negative amount outside archive contract')
    whole, fraction = divmod(value, 1_000_000)
    return f'{whole}.{fraction:06d}'


def safe_source(relative):
    require(isinstance(relative, str), 'Input path must be a string')
    path = (ROOT/relative.replace('\\','/')).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Source outside project')
    return path


@contextmanager
def export_lock():
    stream = (OUT/'candidate_export.lock').open('a+b')
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'\0'); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


def _periods(report, cohort, feature):
    keep = ('period', 'ncase', 'total_positive_inventory', 'unmatched_cases',
            'conditional_auc', 'positive_nonzero_rate', 'matched_negative_nonzero_rate',
            'nonzero_rate_difference', 'positive_nonzero_count', 'nonzero_case_payers',
            'payer', 'endpoint', 'full_window_sensitivity')
    return [{key: row.get(key) for key in keep} for row in report['results']
            if row['cohort'] == cohort and row['feature'] == feature]


def selected_groups(statistics, strict_statistics):
    """Deduplicate only by cohort/base; retain all preselected predicates."""
    indexed = {(row['cohort'], row['feature']): row for row in statistics['screening']}
    groups = {}
    for tier, name in [('formal_screen_passed', 'promising'), ('exploratory_sparse', 'sparse_candidates')]:
        for row in statistics[name]:
            require(indexed[(row['cohort'], row['feature'])] == row, 'Screened candidate differs from complete screening')
            require(row['promising'] if tier == 'formal_screen_passed' else row['sparse_descriptive_candidate'],
                    'Candidate does not satisfy its declared screen')
            feature = row['feature']
            definition = statistics['predicate_definitions'].get(feature)
            base = definition['source'] if definition else feature
            require(feature.split('__',1)[0] == base, 'Unexpected predicate base')
            if base in BASELINE: continue
            require(base in WITNESS, 'No witness contract for selected feature: '+base)
            if definition:
                require((definition['operator'],definition['threshold']) in [('>',0),('>=',2)],
                        'Unrecognized fixed predicate')
            key = (row['cohort'], base)
            group = groups.setdefault(key, dict(cohort=row['cohort'], base_feature=base,
                candidate_key=row['cohort']+'::'+base, witness_family=WITNESS[base],
                evidence_tier=tier, selected_statistics=[]))
            if tier == 'formal_screen_passed':group['evidence_tier']=tier
            group['selected_statistics'].append(dict(
                feature=feature, evidence_tier=tier, screen=row,
                predicate=definition or dict(source=base,operator='>',threshold=0,
                    interpretation='Export nonzero observed structure only; not a learned risk threshold'),
                main_periods=_periods(statistics,row['cohort'],feature),
                strict_payer_sensitivity_periods=(_periods(strict_statistics,'ACH',feature) if row['cohort']=='ACH' else [])))
    return [groups[key] for key in sorted(groups)]


def run():
    # These guards execute before any export output or lock is created.
    require(read(OUT/'status.json')['status'] == 'completed', 'Structure extraction is incomplete')
    require(read(OUT/'status_analysis.json')['status'] == 'completed', 'Statistical analysis is incomplete')
    am = read(OUT/'analysis_manifest.json')
    require(am['status'] == 'completed', 'Analysis manifest is incomplete')
    inputs = {}
    def bind(path, expected=None):
        path=Path(path); digest=sha(path)
        if expected is not None:require(digest == expected, 'SHA mismatch: '+str(path))
        inputs[path.relative_to(ROOT).as_posix()] = digest
        return digest
    analysis_manifest_sha = bind(OUT/'analysis_manifest.json')
    statistics_verification = read(OUT/'verification_statistics.json')
    require(statistics_verification['status'] == 'passed' and
            statistics_verification['analysis_manifest_sha256'] == analysis_manifest_sha,
            'Independent statistics verification is missing or bound to another analysis')
    bind(OUT/'verification_statistics.json')
    for name,digest in am['outputs'].items():bind(OUT/name,digest)
    for name,digest in am['binding']['inputs_and_sources'].items():bind(safe_source(name),digest)
    statistics=read(OUT/'statistics.json');strict_statistics=read(OUT/'statistics_strict_ach.json')
    require(statistics['status'] == strict_statistics['status'] == 'completed', 'Statistics are incomplete')
    for report in (statistics, strict_statistics):
        require(report['analysis_binding_sha256'] == am['binding_sha256'], 'Statistics binding mismatch')
        require(report['validation_test_read'] is False and report['model_trained'] is False and
                report['history_labels_as_features'] is False, 'Unexpected analysis scope')
    sm=read(OUT/'structures_manifest.json');sm_sha=bind(OUT/'structures_manifest.json')
    require(sm['status']=='completed', 'Structure manifest not complete')
    for name in ('verification_targets_strict.json','verification_targets_final.json','verification_witnesses.json','engineering_tests.json'):
        report=read(OUT/name);require(report['status']=='passed','Verification not passed: '+name);bind(OUT/name)
        if name=='verification_witnesses.json':
            require(report['structures_manifest_sha256']==sm_sha,'Witness verification binding mismatch')
        elif name.startswith('verification_targets_'):
            prefix='final_' if name=='verification_targets_final.json' else ''
            require(report['targets_manifest_sha256']==bind(OUT/(prefix+'targets_manifest.json')),'Target verification binding mismatch')
        elif name=='engineering_tests.json':
            for source,digest in report['source_hashes'].items():bind(safe_source(source),digest)
    groups=selected_groups(statistics,strict_statistics)
    features=pd.read_parquet(OUT/'structures.parquet').set_index('edge_id');bind(OUT/'structures.parquet',sm['sha256'])
    targets=pd.read_parquet(OUT/'final_targets.parquet').set_index('edge_id')
    pairs=pd.read_parquet(OUT/'final_pairs.parquet')
    require(targets.index.is_unique and int(targets.label.sum())==17794,'Positive target inventory incomplete')
    grouped_controls={int(eid):g.control_edge_id.astype(int).tolist() for eid,g in pairs.groupby('case_edge_id')}
    pair_level={int(eid):str(g.matching_level.iloc[0]) for eid,g in pairs.groupby('case_edge_id')}
    saved={}
    for day in range(1,11):
        name=f'day_{day:02d}.json';part=read(OUT/'parts'/name)
        bind(OUT/'parts'/name,sm['parts'][name]);require(part['binding']==sm['binding'],'Witness partition binding mismatch')
        wp=f'day_{day:02d}_witnesses.json';bind(OUT/'parts'/wp,part['outputs'][wp])
        for record in read(OUT/'parts'/wp):
            eid=int(record['edge_id']);require(eid not in saved,'Repeated witness target');saved[eid]=record['witnesses']
    cfg=read(ROOT/'result/gnn_train_gpu_v1/config.json')
    frozen={name.replace('\\','/'):digest for name,digest in cfg['input_hashes'].items()}
    a={}
    for name in EDGE_FIELDS+('bank_key',):
        rel='data/gnn_v1_train/history_index/'+name+'.npy';bind(ROOT/rel,frozen[rel]);a[name]=np.load(ROOT/rel,mmap_mode='r',allow_pickle=False)
    bank_rel='data/gnn_v1_train/metadata/bank_identity.json';vocab_rel='data/gnn_v1_train/metadata/vocabulary.json'
    bind(ROOT/bank_rel,frozen[bank_rel]);bind(ROOT/vocab_rel,frozen[vocab_rel])
    bank=read(ROOT/bank_rel);vocab=read(ROOT/vocab_rel)['values']
    require(bank.index('70')==122017,'Raw bank 70 identity mismatch')

    def evaluated(value, selected):
        if pd.isna(value):return None
        rule=selected['predicate']
        return bool(value>rule['threshold']) if rule['operator']=='>' else bool(value>=rule['threshold'])

    # Resolve selection before loading the small raw account dictionary.
    inventory=[];needed_edges=set()
    for group in groups:
        base,family=group['base_feature'],group['witness_family']
        cases=targets[(targets.label==1)&(targets.cohort==group['cohort'])]
        for eid,row in cases.iterrows():
            eid=int(eid);value=features.at[eid,base]
            hits=[s for s in group['selected_statistics'] if evaluated(value,s) is True]
            if not hits:continue
            witness=saved[eid][family]
            # Lower-direction results are represented only by existing paths,
            # never by invented witnesses for absence or zero-valued risk.
            if not witness:continue
            control_ids=grouped_controls.get(eid,[])
            needed_edges.add(eid);needed_edges.update(int(x) for x in witness)
            for cid in control_ids:
                needed_edges.add(cid);needed_edges.update(int(x) for x in (saved[cid][family] or []))
            inventory.append((group,eid,hits,control_ids))
    nodes={int(a[role][eid]) for eid in needed_edges for role in ('src','dst')}
    accounts={}
    account_path=ROOT/'data/gnn_v1_train/graph/accounts/accounts.parquet'
    account_rel=account_path.relative_to(ROOT).as_posix()
    account_expected=frozen.get(account_rel)
    account_actual=bind(account_path,account_expected)
    account_identity=dict(path=account_rel,sha256=account_actual,
        frozen_anchor_available=account_expected is not None,
        identity_check=('Frozen SHA verified and raw bank identities cross-checked against frozen bank_key mapping.'
                        if account_expected is not None else
                        'Actual SHA recorded; no frozen SHA anchor exists for this raw account dictionary. Exported bank identities are cross-checked against frozen bank_key mapping.'))
    wanted=np.asarray(sorted(nodes),dtype=np.uint32)
    for batch in pq.ParquetFile(account_path).iter_batches(columns=['node_id','bank_id','account_number'],batch_size=65536,use_threads=False):
        ids=batch['node_id'].to_numpy();keep=np.isin(ids,wanted)
        if not keep.any():continue
        for record in pa.Table.from_batches([batch]).filter(pa.array(keep)).to_pylist():
            node=int(record['node_id']);require(node not in accounts,'Duplicate account identity')
            require(record['bank_id']==bank[int(a['bank_key'][node])],'Raw account bank conflicts with frozen bank mapping')
            accounts[node]=dict(bank=record['bank_id'],account=record['account_number'])
    require(set(accounts)==nodes,'Missing raw account identity')
    def category(field,value):
        index=int(value)-2;require(0<=index<len(vocab[field]),'Category outside frozen vocabulary');return vocab[field][index]
    def edge_info(eid):
        eid=int(eid);u,v=int(a['src'][eid]),int(a['dst'][eid]);stamp=int(a['timestamp'][eid])
        require(0<=eid<20318992,'Non-Train edge in case export')
        return dict(edge_id=eid,timestamp_us=stamp,timestamp=str(np.datetime64(stamp,'us')),
            src=u,dst=v,from_bank=accounts[u]['bank'],from_account=accounts[u]['account'],
            to_bank=accounts[v]['bank'],to_account=accounts[v]['account'],
            amount_paid=exact_decimal(a['paid_scaled'][eid]),amount_received=exact_decimal(a['received_scaled'][eid]),
            payment_format=category('payment_format',a['payment_format_cat'][eid]),
            payment_currency=category('currency',a['payment_currency_cat'][eid]),
            receiving_currency=category('currency',a['receiving_currency_cat'][eid]))
    def witness_info(eid,group):
        ids=saved[eid][group['witness_family']] or []
        parallel=group['witness_family'].startswith('parallel2')
        return dict(family=group['witness_family'],edge_ids=ids,edges=[edge_info(x) for x in ids],
            history_labels_included=False,only_one_path_saved=bool(parallel and ids),
            interpretation=('One saved two-edge historical path only; it does not exhibit every counted path or two distinct middle accounts.'
                            if parallel and ids else 'Complete saved witness for this motif; all historical fields are label-free.'))

    formal=[];exploratory=[];csv_rows=[]
    for group,eid,hits,control_ids in inventory:
        base=group['base_feature'];value=features.at[eid,base];target=targets.loc[eid]
        applicable=lambda x: bool(features.at[x,'eligible']) and not bool(features.at[x,'not_applicable']) and not pd.isna(features.at[x,base])
        denominator=bool(control_ids and applicable(eid) and all(applicable(cid) for cid in control_ids))
        controls=[]
        for cid in control_ids:
            controls.append(dict(**edge_info(cid),label=0,base_feature_value=clean(features.at[cid,base]),
                selected_predicates={s['feature']:evaluated(features.at[cid,base],s) for s in group['selected_statistics']},
                full_window_72h=bool(features.at[cid,'full_window_72h']),witness=witness_info(cid,group)))
        tier='formal_screen_passed' if any(s['evidence_tier']=='formal_screen_passed' for s in hits) else 'exploratory_sparse'
        low=any(s['screen']['discovery_direction']=='lower' for s in hits)
        record=dict(candidate_key=group['candidate_key'],cohort=group['cohort'],base_feature=base,
            evidence_tier=tier,exploratory=(tier!='formal_screen_passed'),
            matching_design=('same payer bank/day/payment/both currencies/self; both amounts within factor two'
                             if group['cohort']=='ACH' else 'same payer/day/payment/both currencies/self; both amounts within factor two'),
            matching_level=pair_level.get(eid,'unmatched'),matched=bool(control_ids),
            in_matched_effect_denominator=denominator,
            unmatched_note=(None if control_ids else 'Unmatched positive retained as an example; excluded from matched effect denominators.'),
            target=dict(**edge_info(eid),label=1,day=int(target.day)),base_feature_value=clean(value),
            full_window_72h=bool(features.at[eid,'full_window_72h']),
            selected_predicates_fired=[s['feature'] for s in hits],
            selected_predicate_values={s['feature']:evaluated(value,s) for s in group['selected_statistics']},
            lower_direction_existing_witness_only=low,is_detection_rule=False,model_improvement_demonstrated=False,
            witness=witness_info(eid,group),controls=controls,
            limitations=['Formal means passed the prespecified association screen only; it is not proven model benefit.',
                         'Low-value direction, if present, is not converted into a risk rule; this pack exports existing nonzero paths only.',
                         'ACH bank-matched effects can reflect account differences; compare the separately attached strict-payer sensitivity.'])
        (formal if tier=='formal_screen_passed' else exploratory).append(record)
        for control in controls or [None]:
            csvrow=dict(candidate_key=record['candidate_key'],cohort=record['cohort'],base_feature=base,
                evidence_tier=tier,exploratory=record['exploratory'],matching_design=record['matching_design'],
                matching_level=record['matching_level'],in_matched_effect_denominator=denominator,
                base_feature_value=record['base_feature_value'],selected_predicates_fired=json.dumps(record['selected_predicates_fired']),
                full_window_72h=record['full_window_72h'],lower_direction_existing_witness_only=low,
                is_detection_rule=False,model_improvement_demonstrated=False,
                case_witness_edge_ids=json.dumps(record['witness']['edge_ids']),
                only_one_path_saved=record['witness']['only_one_path_saved'],
                control_weight_within_case=(1/len(controls) if controls else None))
            csvrow.update({'case_'+key:value for key,value in record['target'].items()})
            if control is not None:
                csvrow.update({'control_'+key:value for key,value in control.items() if key not in ('witness','selected_predicates')})
                csvrow['control_selected_predicates']=json.dumps(control['selected_predicates'])
                csvrow['control_witness_edge_ids']=json.dumps(control['witness']['edge_ids'])
            csv_rows.append(csvrow)
    pack=dict(status='completed',created_at=datetime.now(timezone.utc).isoformat(),scope='Train Sep01..10 diagnostic case export',
        timezone='Source timestamps have no declared timezone; no timezone conversion applied.',
        selection='Only official promising plus prespecified sparse_candidates, excluding baseline duplicate counts; deduplicated by cohort/base feature.',
        candidates=groups,formal_screen_passed_cases=formal,exploratory_sparse_cases=exploratory,
        raw_account_dictionary_identity=account_identity,
        row_semantics='JSON has one positive target per candidate base; CSV has one fixed case/control reference per row, with one empty-control row for unmatched cases.',
        historical_labels_included=False,validation_test_read=False,models_trained=False,
        improvement_claim=False,
        limitations=['All 17794 Train positives were eligible for export, but only selected predicates with stored witnesses are included.',
                     'This is not population precision/recall/AP/F1 and not statistical confirmation of model improvement.',
                     'Controls are the original fixed matched controls; no rematching or new threshold optimization.',
                     'One saved parallel path does not demonstrate all counted paths or an actual money split.',
                     'Unmatched hits are retained only as examples and never added to the matched effect denominator.'])
    bind(Path(__file__))
    binding=dict(inputs=inputs,selection_groups=[dict(cohort=g['cohort'],base_feature=g['base_feature'],
        selected_features=[s['feature'] for s in g['selected_statistics']]) for g in groups])
    json_bytes=payload(pack)
    csv_buffer=io.StringIO(newline='')
    csv_frame = (pd.DataFrame(csv_rows) if csv_rows else pd.DataFrame(columns=[
        'candidate_key','cohort','base_feature','evidence_tier','exploratory',
        'case_edge_id','control_edge_id','in_matched_effect_denominator']))
    csv_frame.to_csv(csv_buffer,index=False)
    csv_bytes=csv_buffer.getvalue().encode('utf-8-sig')
    with export_lock():
        mp=OUT/'candidate_cases_manifest.json'
        if mp.exists():
            existing=read(mp);require(existing['status']=='completed' and existing['binding']==binding,'Existing export binding differs')
            for name,digest in existing['outputs'].items():require(sha(OUT/name)==digest,'Existing case output SHA mismatch')
            print(json.dumps(dict(status='verified_reuse',**existing['counts']),ensure_ascii=True));return
        require(not (OUT/'candidate_cases.json').exists() and not (OUT/'candidate_cases.csv').exists(),'Unmanifested case outputs exist; preserve them')
        for rel,digest in inputs.items():require(sha(safe_source(rel))==digest,'Input changed during case export: '+rel)
        extra=len(json_bytes)+len(csv_bytes)
        require(shutil.disk_usage(ROOT).free-extra>=20*1024**3,'Disk reserve below 20 GiB')
        used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        require(used+extra+1_000_000<=2*1024**3,'Case export exceeds 2 GiB diagnostic budget')
        outputs={}
        for name,data in [('candidate_cases.json',json_bytes),('candidate_cases.csv',csv_bytes)]:
            temp=OUT/(name+'.tmp');temp.write_bytes(data);os.replace(temp,OUT/name);outputs[name]=sha(OUT/name)
        counts=dict(candidate_groups=len(groups),formal_case_records=len(formal),exploratory_case_records=len(exploratory),
            unique_positive_targets=len({r['target']['edge_id'] for r in formal+exploratory}),
            unmatched_case_records=sum(not r['matched'] for r in formal+exploratory),csv_pair_rows=len(csv_rows))
        manifest=dict(status='completed',created_at=datetime.now(timezone.utc).isoformat(),binding=binding,outputs=outputs,
            counts=counts,validation_test_read=False,models_trained=False,historical_labels_exported=False,frozen_assets_modified=False)
        temp=mp.with_suffix('.json.tmp');temp.write_bytes(payload(manifest));os.replace(temp,mp)
        print(json.dumps(dict(status='completed',**counts),ensure_ascii=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    run()


if __name__=='__main__':
    main()
