"""Export audited Train-only structure findings and inspectable path examples."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_gnn_causal_structure_probe import OUT, read, sha, write, arrays_verified, space
from gnn_features.training_state import ProcessLock

BASELINE = {'prior_in_count_72h', 'prior_in_same_currency_count_72h'}
NAMES = {
    'prior_in_count_72h': '付款方此前入账笔数（已有）',
    'prior_in_same_currency_count_72h': '付款方同币种入账笔数（已有）',
    'prior_in_amount_match_count_72h': '与当前金额接近的此前入账',
    'prior_in_match_min_gap_minutes_72h': '距最近金额匹配入账的分钟数',
    'chain3_path_count_72h': '两笔历史转付接当前交易',
    'chain3_amount_match_path_count_72h': '金额连续的三边转付链',
    'cycle2_count_72h': '当前交易闭合两边回流',
    'cycle2_amount_match_count_72h': '金额匹配的两边回流',
    'cycle3_path_count_72h': '当前交易闭合三角回流',
    'cycle3_amount_match_path_count_72h': '金额连续的三角回流',
    'parallel2_path_count_72h': '双方此前存在有序两跳路径',
    'parallel2_middle_accounts_72h': '历史两跳路径的不同中间账户数',
    'parallel2_amount_match_path_count_72h': '金额连续的历史两跳路径',
    'parallel2_amount_match_middle_accounts_72h': '金额连续路径的不同中间账户数',
}
WITNESS_FEATURE = {
    'prior_in': 'prior_in_count_72h',
    'prior_in_amount_match': 'prior_in_amount_match_count_72h',
    'chain3': 'chain3_path_count_72h',
    'chain3_amount_match': 'chain3_amount_match_path_count_72h',
    'cycle2': 'cycle2_count_72h',
    'cycle2_amount_match': 'cycle2_amount_match_count_72h',
    'cycle3': 'cycle3_path_count_72h',
    'cycle3_amount_match': 'cycle3_amount_match_path_count_72h',
    'parallel2': 'parallel2_path_count_72h',
    'parallel2_amount_match': 'parallel2_amount_match_path_count_72h',
}
COHORT_NAMES = {'ACH': 'ACH', 'bank70_nonACH': '付款银行70且非ACH', 'other_nonACH': '其他非ACH'}


def base_feature(name):
    return name.split('__')[0]


def metric_label(name):
    base = base_feature(name)
    suffix = '是否出现' if name.endswith('__present') else '至少两个中间账户' if name.endswith('__at_least_2') else '计数/数值'
    return NAMES[base] + '：' + suffix


def pct(x):
    return '不适用' if x is None else f'{x*100:.2f}%'


def val(x):
    return '不适用' if x is None else f'{x:.4f}'


def main():
    with ProcessLock(OUT/'export_lock'):
        space()
        sm = read(OUT/'structures_manifest.json')
        assert sm['status'] == 'completed' and sha(OUT/'structures.parquet') == sm['sha256']
        for stage in ['strict', 'final']:
            verification = read(OUT/f'verification_targets_{stage}.json')
            assert verification['status'] in ('passed', 'completed', 'verified'), verification
            manifest_name = ('final_' if stage == 'final' else '')+'targets_manifest.json'
            assert verification['targets_manifest_sha256'] == sha(OUT/manifest_name)
            for name, digest in read(OUT/manifest_name)['outputs'].items():
                assert sha(OUT/name) == digest
        witness_review = read(OUT/'verification_witnesses.json')
        assert witness_review['status'] in ('passed', 'completed', 'verified'), witness_review
        assert witness_review['structures_manifest_sha256'] == sha(OUT/'structures_manifest.json')
        for name, digest in sm['parts'].items():
            assert sha(OUT/'parts'/name) == digest
            for payload, payload_sha in read(OUT/'parts'/name)['outputs'].items():
                assert sha(OUT/'parts'/payload) == payload_sha
        am = read(OUT/'analysis_manifest.json')
        assert am['status'] == 'completed'
        statistic_review = read(OUT/'verification_statistics.json')
        assert statistic_review['status'] == 'passed'
        assert statistic_review['analysis_manifest_sha256'] == sha(OUT/'analysis_manifest.json')
        for name, digest in am['outputs'].items():
            assert sha(OUT/name) == digest
        for name, digest in am['binding']['inputs_and_sources'].items():
            p = (ROOT/name).resolve()
            assert p.is_relative_to(ROOT.resolve()) and sha(p) == digest
        main_stats = read(OUT/'statistics.json')
        strict_stats = read(OUT/'statistics_strict_ach.json')
        assert main_stats['status'] == strict_stats['status'] == 'completed'
        assert main_stats['analysis_binding_sha256'] == strict_stats['analysis_binding_sha256'] == am['binding_sha256']
        t = pd.read_parquet(OUT/'final_targets.parquet')
        pairs = pd.read_parquet(OUT/'final_pairs.parquet')
        all_features = pd.read_parquet(OUT/'structures.parquet')
        f = all_features[all_features.edge_id.isin(t.edge_id)]
        assert f.edge_id.is_unique and set(f.edge_id) == set(t.edge_id)
        assert int(t.label.sum()) == 17794 and int(pairs.case_edge_id.nunique()) == 11974
        indexed = {(r['cohort'], r['feature'], r['period']): r for r in main_stats['results']}
        strict_index = {(r['cohort'], r['feature'], r['period']): r for r in strict_stats['results']}
        approved, exploratory = [], []
        for item in main_stats['screening']:
            base = base_feature(item['feature'])
            if base in BASELINE:
                continue
            d = indexed[item['cohort'], item['feature'], 'discovery_sep01_07']
            full = d['full_window_sensitivity']
            direction = 1 if item['discovery_direction'] == 'higher' else -1
            full_reproduced = (full and full['conditional_auc'] is not None
                               and direction*(full['conditional_auc']-.5) >= .03)
            record = dict(item, name=metric_label(item['feature']),
                          discovery_full_window_effect_reproduced=bool(full_reproduced))
            if item['cohort'] == 'ACH':
                sr = strict_index.get(('ACH', item['feature'], 'later_sep09_10'))
                record['strict_payer_later_auc'] = sr['conditional_auc'] if sr else None
                record['strict_payer_later_ncase'] = sr['ncase'] if sr else 0
            if item['promising'] and full_reproduced:
                approved.append(record)
            elif item['promising'] or item['sparse_descriptive_candidate']:
                exploratory.append(record)
        write('selected_features.json', dict(status='completed', selected=approved,
              selection='Predeclared main gate plus discovery complete-window effect; pre-existing counts excluded',
              scope='Matched Train association only, no new training or demonstrated validation improvement',
              exploratory_candidates=exploratory, model_trained=False, validation_test_read=False))

        rows = []
        for r in main_stats['results']:
            full = r['full_window_sensitivity']
            rows.append(dict(cohort=r['cohort'], period=r['period'], feature=r['feature'],
                description=metric_label(r['feature']), ncase=r['ncase'],
                inventory=r['total_positive_inventory'], unmatched=r['unmatched_cases'],
                positive_present=r['positive_nonzero_count'], positive_rate=r['positive_nonzero_rate'],
                matched_control_rate=r['matched_negative_nonzero_rate'],
                conditional_auc=r['conditional_auc'], endpoint_clusters=r['endpoint']['clusters'],
                endpoint_holm_p=r['endpoint']['holm_p'], endpoint_ci95=json.dumps(r['endpoint']['ci95']),
                full_window_ncase=full['ncase'] if full else None,
                full_window_auc=full['conditional_auc'] if full else None))
        pd.DataFrame(rows).to_csv(OUT/'structure_statistics.csv', index=False, encoding='utf-8-sig')
        joined = t.merge(f, on='edge_id', validate='one_to_one')
        joined['has_matched_controls'] = joined.edge_id.isin(pairs.case_edge_id)
        joined.to_csv(OUT/'target_structure_values.csv', index=False, encoding='utf-8-sig')
        selected_bases = {(x['cohort'], base_feature(x['feature'])) for x in approved}
        candidate_rows = []
        for cohort, name in sorted(selected_bases):
            for row in joined[joined.cohort.eq(cohort)].itertuples(index=False):
                value = getattr(row, name)
                if pd.notna(value):
                    candidate_rows.append(dict(edge_id=row.edge_id, label=row.label, day=row.day,
                        src=row.src, dst=row.dst, cohort=cohort, feature=name, value=value,
                        full_window_72h=row.full_window_72h,
                        has_matched_controls=row.has_matched_controls))
        pd.DataFrame(candidate_rows, columns=['edge_id','label','day','src','dst','cohort','feature',
            'value','full_window_72h','has_matched_controls']).to_csv(
            OUT/'candidate_structure_values.csv', index=False, encoding='utf-8-sig')

        arrays, frozen, _ = arrays_verified()
        for rel in ['data/gnn_v1_train/metadata/vocabulary.json',
                    'data/gnn_v1_train/metadata/bank_identity.json',
                    'data/gnn_v1_train/metadata/feature_manifest.json']:
            assert sha(ROOT/rel) == frozen[rel]
        vocab = read(ROOT/'data/gnn_v1_train/metadata/vocabulary.json')['values']
        bank_identity = read(ROOT/'data/gnn_v1_train/metadata/bank_identity.json')
        ti = t.set_index('edge_id')
        fi = f.set_index('edge_id')
        groups = pairs.groupby('case_edge_id').control_edge_id.apply(list).to_dict()
        examples, used, edge_rows = [], {}, []

        def edge_record(eid):
            eid = int(eid)
            src, dst = int(arrays['src'][eid]), int(arrays['dst'][eid])
            def currency(key):
                code = int(arrays[key][eid])
                return vocab['currency'][code-2] if code >= 2 else str(code)
            def amount(key):
                whole, fraction = divmod(int(arrays[key][eid]), 1_000_000)
                return f'{whole}.{fraction:06d}'
            return dict(edge_id=eid, timestamp=str(np.datetime64(int(arrays['timestamp'][eid]), 'us')),
                src=src, dst=dst, payer_bank=bank_identity[int(arrays['bank_key'][src])],
                receiver_bank=bank_identity[int(arrays['bank_key'][dst])],
                paid_scaled=str(int(arrays['paid_scaled'][eid])),
                received_scaled=str(int(arrays['received_scaled'][eid])),
                paid_amount=amount('paid_scaled'), received_amount=amount('received_scaled'),
                payment_currency=currency('payment_currency_cat'),
                receiving_currency=currency('receiving_currency_cat'))

        # Fixed small balanced gallery; cases are illustrative, not selected for
        # diagnostic classification strength. All raw feature rows are exported.
        for day in range(10, 0, -1):
            ws = read(OUT/'parts'/f'day_{day:02d}_witnesses.json')
            for w in ws:
                eid = int(w['edge_id'])
                if eid not in ti.index or not bool(fi.at[eid, 'full_window_72h']):
                    continue
                row = ti.loc[eid]
                for kind, path in w['witnesses'].items():
                    if kind not in WITNESS_FEATURE or kind == 'prior_in' or not path:
                        continue
                    key = (row.cohort, int(row.label), kind)
                    if used.get(key, 0) >= 2:
                        continue
                    used[key] = used.get(key, 0)+1
                    control_ids = groups.get(eid, [])
                    feature = WITNESS_FEATURE[kind]
                    family_bases = {'prior_in_amount_match_count_72h', 'prior_in_match_min_gap_minutes_72h'} if kind == 'prior_in_amount_match' else (
                        {'parallel2_path_count_72h', 'parallel2_middle_accounts_72h'} if kind == 'parallel2' else (
                        {'parallel2_amount_match_path_count_72h', 'parallel2_amount_match_middle_accounts_72h'} if kind == 'parallel2_amount_match' else {feature}))
                    family_candidates = [x['feature'] for x in approved if x['cohort'] == row.cohort and base_feature(x['feature']) in family_bases]
                    example = dict(cohort=row.cohort, target_label=int(row.label), kind=kind,
                        feature=feature, feature_value=int(fi.at[eid, feature]),
                        selected_candidate_family=family_candidates,
                        witness_proves='One concrete path only; does not by itself prove two different middle accounts or a min-gap optimum',
                        target=edge_record(eid), path=[edge_record(x) for x in path],
                        matched_control_feature_values=[dict(edge_id=int(c), value=None if pd.isna(fi.at[c, feature]) else float(fi.at[c, feature])) for c in control_ids],
                        caution='Illustrative historical path; historical labels were not read; it does not prove actual fund provenance')
                    examples.append(example)
                    for step, edge in enumerate(example['path'], 1):
                        edge_rows.append(dict(target_edge_id=eid, target_label=int(row.label),
                            cohort=row.cohort, motif_type=kind, path_id=len(examples), step=step,
                            is_target=edge['edge_id']==eid, **edge))
        pd.DataFrame(edge_rows).to_csv(OUT/'structure_example_edges.csv', index=False, encoding='utf-8-sig')
        write('structure_examples.json', dict(status='completed', examples=examples,
              exact_scaled_integers_are_strings=True, scaled_amount_unit='1e-6 currency units (frozen feature_manifest decimal(38,6)); decimal strings without floating conversion',
              path_semantics='chain/cycle paths include target last; parallel2 has two historical edges and separate target',
              selection='At most two examples per cohort/label/motif, complete windows, latest day then ascending edge_id'))

        bank_chain = indexed['bank70_nonACH', 'chain3_path_count_72h', 'later_sep09_10']
        ach_chain = indexed['ACH', 'chain3_amount_match_path_count_72h', 'later_sep09_10']
        lines = ['交易结构检查结果（Train开发诊断）', '',
            f'主匹配设计下完整筛选通过{len({(x["cohort"],base_feature(x["feature"])) for x in approved})}种候选；探索性线索{len({(x["cohort"],base_feature(x["feature"])) for x in exploratory})}种。尚未训练模型验证增益。',
            '主分析通过的是ACH的72h同币种反向回流，属于已有24h反向计数的细化；其严格同付款账户敏感性没有达到显著标准。',
            f'银行70非ACH后两天：有序三边转付链正例出现率{pct(bank_chain["positive_nonzero_rate"])}，匹配正常{pct(bank_chain["matched_negative_nonzero_rate"])}。',
            f'ACH后两天：金额连续三边链正例{ach_chain["positive_nonzero_count"]}/{ach_chain["ncase"]}（{pct(ach_chain["positive_nonzero_rate"])}），同银行匹配正常{pct(ach_chain["matched_negative_nonzero_rate"])}；同账户敏感性见下文。', '',
            '范围：2022-09-01至10；检查全部17,794笔正例和固定匹配正常对照。未读取Validation/Test，未训练新模型。',
            '历史：仅当前交易前72h无标签交易，排除同刻和自转账；路径内时间严格递增。另设金额连续版本，使用逐段±10%整数金额匹配；普通结构版本不要求金额相近。',
            f'提取目标合计{len(all_features):,}笔（含两套匹配的并集），主分析{len(t):,}笔。',
            '主匹配覆盖：银行70非ACH 2,551/2,551，ACH 9,410/15,067，其他非ACH 13/176。',
            '非ACH按同付款账户匹配；ACH因同付款账户仅384笔可配对，主分析改为同银行，对照支付方式/日期/双币种相同，双边金额均在半倍至两倍内。',
            'ACH原同付款账户384笔另作敏感性。缺乏对照的正例不用于宣称组间区别。', '',
            f'通过预定主分析筛选且发现期完整窗口效应一致的候选表达：{len(approved)}项（包含同一结构的计数/是否出现表达）。',
            f'仅描述性线索：{len(exploratory)}项，不等于显著或可提高召回。',
            '比较指标：每个正例与自己的正常对照比大小，正例等权，平分计半；0.5表示无排序差异。',
            '检验按账户关联组件聚类，29项×3群体逐时段Holm校正，并要求前7天、8日、9—10日方向一致。', '',
            '后两天主要结构：群体 | 结构 | 正例出现率 | 匹配正常出现率 | 条件AUC | 可用正例 | 关联组件数']
        main_features = ['prior_in_amount_match_count_72h', 'chain3_amount_match_path_count_72h',
                         'cycle2_count_72h', 'cycle2_amount_match_count_72h',
                         'cycle3_path_count_72h', 'cycle3_amount_match_path_count_72h',
                         'parallel2_amount_match_middle_accounts_72h__at_least_2']
        for cohort in ('bank70_nonACH', 'ACH', 'other_nonACH'):
            for name in main_features:
                r = indexed[cohort, name, 'later_sep09_10']
                lines.append(f"{COHORT_NAMES[cohort]} | {metric_label(name)} | {pct(r['positive_nonzero_rate'])} | {pct(r['matched_negative_nonzero_rate'])} | {val(r['conditional_auc'])} | {r['ncase']} | {r['endpoint']['clusters']}")
        lines += ['', '新候选：']
        lines += [f"{COHORT_NAMES[x['cohort']]}：{x['name']}（{x['discovery_direction']}）" for x in approved] or ['无。']
        lines += ['', 'ACH更严格的同付款账户敏感性（覆盖很少，不是主样本的随机子样本）：',
                  '时期 | 结构 | 正例出现数/可用正例 | 匹配正常出现率 | 条件AUC']
        for period in ('discovery_sep01_07', 'sep08', 'later_sep09_10'):
            for name in ('prior_in_amount_match_count_72h', 'chain3_amount_match_path_count_72h',
                         'cycle2_count_72h', 'cycle2_amount_match_count_72h', 'cycle3_path_count_72h'):
                r = strict_index['ACH', name, period]
                lines.append(f"{period} | {NAMES[name]} | {r['positive_nonzero_count']}/{r['ncase']} | {pct(r['matched_negative_nonzero_rate'])} | {val(r['conditional_auc'])}")
        lines += ['', '解释边界：',
            '1. 这轮最长是两笔历史边接当前目标的三边链；未穷举任意深度链条。',
            '2. 并行两跳路径表示多个中间账户，不证明同一笔资金被分拆后汇集；金额连续也不是资金来源证明。',
            '3. 付款方入账笔数/同币种入账笔数已有，本轮只把它们当支持数对照；两边回流属于原反向计数的72h同币种细化。',
            '4. 匹配样本不能换算全量精确率、AP或召回。账户关联可能形成单大组件，不能把交易数量当独立样本量。',
            '5. ACH同银行对照仍可能受账户差异影响；需同时读statistics_strict_ach.json。',
            '6. 即便出现结构关联，也只产生候选；是否有独立于已有特征的增益仍需后续模型对照。', '',
            '7. 描述性稀疏候选只要求前7天和后2天出现率差异重复达到1个百分点并有基本账户支持；不等于通过完整显著性/8日一致性筛选。', '',
            '文件：structure_statistics.csv（各时期各群体统计）；target_structure_values.csv（主目标逐笔值）；',
            'selected_features.json（筛选及描述性线索）；candidate_structure_values.csv（通过筛选结构逐笔值，若无则空表）；',
            'structure_examples.json / structure_example_edges.csv（正负路径见证）；',
            'statistics_strict_ach.json（ACH同账户敏感性）；verification_*.json（独立核验）。']
        (OUT/'结构检验结果.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
        inputs = ['final_targets_manifest.json', 'targets_manifest.json', 'structures_manifest.json',
                  'analysis_manifest.json', 'statistics.json', 'statistics_strict_ach.json',
                  'verification_targets_strict.json', 'verification_targets_final.json', 'verification_witnesses.json',
                  'verification_statistics.json', 'engineering_tests.json']
        outputs = ['selected_features.json', 'structure_statistics.csv', 'target_structure_values.csv',
                   'candidate_structure_values.csv', 'structure_examples.json', 'structure_example_edges.csv', '结构检验结果.txt']
        write('delivery_manifest.json', dict(status='completed', at=datetime.now(timezone.utc).isoformat(),
              input_sha256={n:sha(OUT/n) for n in inputs}, output_sha256={n:sha(OUT/n) for n in outputs},
              exporter_sha256=sha(__file__), examples=len(examples), selected_features=len(approved),
              exploratory_features=len(exploratory), frozen_assets_modified=False, model_trained=False,
              validation_test_read=False))
        print(json.dumps(dict(status='completed', selected=len(approved), exploratory=len(exploratory), examples=len(examples)), ensure_ascii=True))


if __name__ == '__main__':
    main()
