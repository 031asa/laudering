"""Write a concise Chinese report only after all chain-audit gates pass."""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'result/gnn_chain_audit_20261003'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def percent(value):
    return f'{100 * value:.2f}%'


def interval(value):
    return '不可估计' if value is None else f'[{100*value[0]:.2f},{100*value[1]:.2f}]'


def pvalue(value):
    return '不可估计' if value is None else f'{value:.4g}'


def main():
    destination = OUT / '链条审计结果.txt'
    if destination.exists():
        raise FileExistsError('Preserve the existing report')
    # All three completion gates are mandatory. Never render partial results.
    run, sensitivity, verified = (read(n) for n in [
        'run_manifest.json', 'endpoint_cluster_sensitivity.json', 'verification.json'])
    assert run['status'] == 'completed' and sensitivity['status'] == 'completed'
    assert verified['status'] == 'passed'
    run_sha = sha(OUT / 'run_manifest.json')
    assert verified['run_manifest_sha256'] == run_sha
    assert verified['global_predeclared_positive_cohort_counts'] == {
        'ACH_FN': 772, 'ACH_TP': 1508, 'G_FN': 662}
    for name, expected in run['output_sha256'].items():
        assert sha(OUT / name) == expected, name
        assert verified['verified_output_sha256'][name] == expected, name
    for name, expected in sensitivity['audit']['inputs_sha256'].items():
        assert sha(OUT / name) == expected, name
    assert sensitivity['audit']['inputs_sha256']['run_manifest.json'] == run_sha

    summary = read('cohort_summary.json')
    primary = read('matched_statistics.json')
    deep = read('ach_deep_walk_witnesses.json')
    matching = {c: read(c + '_matching.json')['summary'] for c in ['ACH_FN', 'G_FN']}
    assert len(primary['tests']) == len(sensitivity['tests']) == 6
    candidates = pd.read_parquet(OUT / 'candidates.parquet', columns=[
        'edge_id', 'label', 'payment_format', 'from_bank', 'outcome_lr_only'])
    pos = candidates[candidates.label == 1]
    ach = pos.payment_format == 'ACH'
    group_g = (~ach) & (pos.from_bank.astype(str) == '70')
    other = (~ach) & (~group_g)
    fn = pos.outcome_lr_only == 'FN'
    assert [len(pos), int(fn.sum()), int(ach.sum()), int((ach & fn).sum()),
            int(group_g.sum()), int((group_g & fn).sum()), int(other.sum()),
            int((other & fn).sum())] == [6420, 2887, 5705, 2172, 662, 662, 53, 53]
    old = json.loads((PROJECT / 'result/gnn_non_ach_score_diagnosis_20261003/summary.json').read_text(encoding='utf-8'))
    assert [old['groups'][k]['samples'] for k in [
        'ACH', 'non_ACH_sender_bank70', 'non_ACH_other_sender_banks']] == [584437, 477760, 3626809]
    assert deep['counts']['cases'] == summary['ACH_FN']['counts']['src_temporal_walk_ge2_7d'] == 36
    for c, count in [('ACH_FN', 772), ('ACH_TP', 1508), ('G_FN', 662)]:
        assert summary[c]['transactions'] == count

    labels = {'ACH_FN': 'ACH漏检子组', 'ACH_TP': '同条件ACH检出组', 'G_FN': 'G漏检组'}
    features = [('prior_positive_any_7d', '历史正例接触'),
                ('src_temporal_walk_ge2_7d', '付款方有序≥2边'),
                ('mixed_2hop_positive_any_7d', '拓扑二跳正例（去除直接边）')]
    lines = [
        '链条审计结果（2026-10-03）',
        '',
        '本次复盘冻结lr_only的全局P80阈值，未重训或修改阈值。完整Validation共4,689,006笔、6,420正例，漏检2,887笔；互不重叠的三组为：ACH漏检2,172／正例5,705／交易584,437；G（付款银行70且非ACH）662／662／477,760；其他非ACH 53／53／3,626,809。',
        '',
        '重点审计的是ACH漏检中的772笔：同向交易对72小时无历史、当前金额超过收款方7天历史P90；另有满足同条件的1,508笔ACH已检出作为描述性参照。772不是全部ACH漏检。G审计覆盖全部662笔。所有历史只取[t−7天,t)，排除全部同时间戳及自转账；采样对照另统一到72小时。',
        '',
        '全组历史证据（依次为：正例接触／付款方有序≥2边／拓扑二跳正例（去除直接边））：',
    ]
    for c in ['ACH_FN', 'ACH_TP', 'G_FN']:
        s = summary[c]
        values = '／'.join(str(s['counts'][f]) for f, _ in features)
        lines.append(f"{labels[c]}{s['transactions']}笔：{values}。")
    lines.extend([
        '因此，772笔中只有36笔具有付款方有序多边历史，不能把全部漏检归因于深链。已检出组也常有历史接触，不能把“有链”与“模型没看到”画等号。',
        '',
        f"匹配对照：ACH成功{matching['ACH_FN']['matched_cases']}/772笔，未匹配{matching['ACH_FN']['unmatched_cases']}笔；G成功{matching['G_FN']['matched_cases']}/662笔，未匹配{matching['G_FN']['unmatched_cases']}笔。固定日期、支付方式、两币种及双方样本数档位，并限制金额在两倍范围内；每例最多3个正常对照、每例等权。ACH匹配均跨付款账户，G要求同付款账户。匹配不使用模型分数、链条或历史标签。",
        '',
        '六项主比较（病例率/加权正常率；差值95%区间单位为百分点；p为六项Holm校正）：',
    ])
    for c in ['ACH_FN', 'G_FN']:
        for feature, label in features:
            test = next(t for t in primary['tests'] if t['cohort'] == c and t['feature'] == feature)
            lines.append(f"{'ACH' if c == 'ACH_FN' else 'G'}·{label}：{percent(test['case_rate'])}/{percent(test['matched_control_rate'])}；CI{interval(test['ci95'])}；p={pvalue(test['holm6_p'])}。")
    ac = sensitivity['components']['ACH_FN']['clusters']
    gc = sensitivity['components']['G_FN']['clusters']
    sen = [next(t for t in sensitivity['tests'] if t['cohort'] == 'ACH_FN' and t['feature'] == f)
           for f, _ in features]
    lines.extend([
        f"考虑付款和收款账户共用后的敏感性分析，ACH/G分别为{ac}/{gc}个依赖组。ACH三项区间依次为{'、'.join(interval(t['ci95']) for t in sen)}，校正p依次为{'、'.join(pvalue(t['sensitivity_holm6_p']) for t in sen)}；方向一致，但仍是匹配后的观察性关联，不证明因果或入模收益。G只有15个付款账户，不能把662笔当662个独立证据。",
        '',
        '实际保存的72小时批次图（符合历史条件／保留／采样丢失）：',
    ])
    for c in ['ACH_FN', 'ACH_TP', 'G_FN']:
        vals = []
        for key, label in [('direct_positive_72h', '直接'), ('mixed_2hop_72h', '二跳'),
                           ('temporal_walk2_72h', '有序2边')]:
            v = summary[c]['sampling'][key]
            assert v['eligible'] == v['retained'] + v['lost']
            vals.append(f"{label}{v['eligible']}/{v['retained']}/{v['lost']}")
        lines.append(labels[c] + '：' + '；'.join(vals) + '。')
    density = next(x for x in sensitivity['exploratory_continuous_descriptions']
                   if x['cohort'] == 'G_FN' and x['feature'] == 'endpoint_positive_density_7d')
    receiver_followup = read('bank70_receiver_followup.json')
    assert receiver_followup['status'] == 'descriptive_only' and receiver_followup['matched_cases'] == 647
    for name, expected in receiver_followup['source_hashes'].items():
        assert sha(OUT / name) == expected
    receiver = receiver_followup['measures']['dst_history_positive_incident_unique_edges_7d']
    lines.extend([
        'ACH的36笔七天有序多边历史中，19笔不满足原72小时的两条边窗口；窗口内17笔又因采样丢3笔。窗口排除与采样丢失是两件事。已采进图也不等于历史标签被入模，不能声称修复采样就能找回这些交易。',
        f"G在病例与匹配正常交易中，接触和二跳存在性均饱和为100%，有序≥2边均为0；这些二值量无法区分两者。正例接触密度均值也接近（{100*density['cases']['mean']:.4f}%/{100*density['matched_controls']['mean']:.4f}%）；单独看收款方接触率为{percent(receiver['case_nonzero_rate'])}/{percent(receiver['control_nonzero_rate'])}。密度、数量及收款方拆分只作探索性描述，未新增显著性检验；这不是“所有方向、金额或图结构均无信号”的证明，也不能把提高正例覆盖当作提高召回的保证。",
        '',
        '建议与现有A/B/C逐笔得失对照：ACH优先研究时序链的无泄漏表示；银行70本轮汇总信号不足以支持简单风险传播，下一步聚焦组内可区分的方向、金额和时间结构。统计细节、逐笔证据和哈希见同目录JSON/Parquet。',
        '',
        '独立验收已通过：全量正例标签身份、匹配规则与权重、全部5,138目标及双方10,276组端点记录（含空历史）均复核，并用另一套动态规划确认最长有序walk；六项率差及Holm校正也已重算。独立验收未重新遍历全部无标签二跳图，也未重放bootstrap；相应结果由冻结输入、原始计算输出哈希及小图边界测试约束。匹配未成功的案例不进入主统计，结果不可无条件外推至全部漏检。',
        '',
        '边界：有序walk允许重复账户，不要求币种连续或金额守恒，因此不是已还原的连续资金链；拓扑二跳先经任意无标签历史边到达邻居，再接正例边，并排除直接关联目标端点的正例边；首条连接边允许是正常交易，且路径内部不要求时间有序。本次历史最终标签只作事后oracle叠加，从未作为输入传入图消息，不能当交易时已知风险直接入模。本报告来自已用于诊断的Validation开发分析，Test未读取，尚无独立泛化结论。',
    ])
    text = '\n'.join(lines) + '\n'
    assert shutil.disk_usage(OUT).free >= 20 * 1024**3
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(json.dumps(dict(status='completed', path=str(destination), sha256=sha(destination),
                          characters=len(text), chinese_characters=sum('\u4e00' <= c <= '\u9fff' for c in text)),
                     ensure_ascii=True))


if __name__ == '__main__':
    main()
