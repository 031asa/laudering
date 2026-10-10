"""Publish the verified v2 handoff addendum without changing frozen reports."""
from pathlib import Path
from datetime import datetime, timezone
import json

from analyze_gnn_non_ach_scores import read, sha
from correct_gnn_v2_bank70_report import assert_only_bank_groups_changed
from gnn_features.training_state import ProcessLock, atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = ROOT/'result/gnn_tuning_v2'
    audit = result/'recovery/20261003_final_delivery'
    with ProcessLock(result):
        integrity = read(result/'final_integrity_verification.json')
        correction = read(result/'recovery/20261003_bank70_correction/verification.json')
        assert integrity['status'] == 'passed_with_separate_bank70_correction_required'
        assert correction['status'] == 'passed'
        assert correction['original_files_preserved'] is True
        assert correction['corrected_only_two_bank_groups'] is True
        assert integrity['comparison_sha256'] == sha(result/'comparison.json')
        assert integrity['best_model_sha256'] == sha(result/'best_model.json')
        assert correction['original_files_sha256']['comparison.json'] == integrity['comparison_sha256']
        assert correction['full_prediction_blocks_verified'] == 788
        assert correction['samples_per_model'] == 4689006
        assert correction['positives_per_model'] == 6420
        original = read(result/'comparison.json')
        corrected = read(result/'comparison.corrected.json')
        assert_only_bank_groups_changed(original, corrected)
        for name, digest in correction['original_files_sha256'].items():
            assert sha(result/name) == digest
        for name, digest in correction['output_sha256'].items():
            assert sha(result/name) == digest
        diagnosis_path = ROOT/'result/gnn_bank70_v2_diagnosis_20261003/summary.json'
        diagnosis = read(diagnosis_path)
        assert diagnosis['status'] == 'completed'
        assert diagnosis['comparison_sha256'] == sha(result/'comparison.json')
        assert diagnosis['script_sha256'] == sha(ROOT/'scripts/analyze_gnn_v2_bank70_scores.py')
        selected = read(result/'best_model.json')
        assert selected['name'] == 'lr_only' and selected['comparison_sha256'] == sha(result/'comparison.json')
        assert selected['sha256'] == sha(ROOT/selected['path'])
        handoff = ROOT/'GPTPro交接_GNN交易反洗钱项目_2026-10-02.md'
        previous = handoff.read_bytes()
        marker = '<!-- GNN_V2_FINAL_REVIEW_BEGIN -->'
        assert marker.encode() not in previous
        lines = [marker, '', '## 2026-10-03 第二轮最终核验与银行70诊断补充', '',
            '三组完整Train各1 epoch及完整Validation均已完成；模型SHA与严格载入、四组全量预测、全局整同分组P80阈值、分层覆盖和配对账户统计已经复核。',
            '**没有新候选同时满足三个平衡目标，继续保留旧lr_only。旧模型也未达到非ACH召回目标，保留代表按事先规则维持基线，并不表示该问题已解决。**', '',
            '| 模型 | AP | P80精确率 | P80召回 | TP | FP | 非ACH TP/715 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for item in original['results']:
            p = item['precision80']
            lines.append(f"| {item['name']} | {item['metrics']['average_precision']:.6f} | {p['precision']:.4%} | {p['recall']:.4%} | {p['tp']} | {p['fp']} | {item['non_ach_precision80']['tp']} |")
        lines += ['', '相对旧模型，A净增加2个TP和2个FP；B减少63个TP及16个FP；C减少95个TP及21个FP。',
            '顺序对照A→B新增102个TP、损失167个TP，新增261个FP、消除279个FP；B→C新增94个TP、损失126个TP，新增243个FP、消除248个FP。各自使用自己的单一全局P80阈值。',
            '保留参数：学习率0.0003，基础正例权重1140.9013150500168，width64、dropout0.1、AdamW权重衰减0.0001、clip5、种子20260920，完整Train 1 epoch。',
            '模型路径：`'+selected['path'].replace('\\','/')+'`。模型SHA256：`'+selected['sha256']+'`。', '',
            '### 银行分层阅读口径', '',
            '正式compare的原始版本将编码70误当作原始银行编号70；已用冻结身份词典解析正确编码122017，在主子进程退出后持锁完成独立更正。',
            '**付款/收款银行70内外分层请以更正版为准。** 原comparison.json、best_model.json、调参结果.md字节保留，以维持原流水线复核链；模型、整体/非ACH指标、阈值、选优及其他分层未变。',
            '本Validation中真实收款银行70为0笔，更正件的收款分层因此只有other组；这不代表漏掉了银行70收款记录。',
            '阅读：[更正版三组报告](result/gnn_tuning_v2/调参结果_银行分层更正.md)；[完整更正版比较](result/gnn_tuning_v2/comparison.corrected.json)；[银行更正审计](result/gnn_tuning_v2/bank70_correction.json)；[最终独立验收](result/gnn_tuning_v2/final_integrity_verification.json)。', '',
            '### 付款银行70且非ACH交叉群体G', '',
            'G在Validation有477,760笔、662正例、477,098负例。每个模型均复用全部197个已核验预测块；以下组内阈值仅用于诊断，不替代全局阈值。', '',
            '| 模型 | G组内AP | G组内ROC-AUC | 检出133个正例所需FP | 原全局P80下G检出 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name, model in diagnosis['models'].items():
            group = model['groups']['G']
            lines.append(f"| {name} | {group['average_precision']:.8f} | {group['roc_auc']:.6f} | {group['recall_operating_points']['0.2']['fp']} | {model['G_global_operating_point']['tp']} |")
        lines += ['', '新增特征及两项调整带来轻微组内排序变化，但AUC仍约0.512–0.513，检出133例仍需8.5万至8.9万误报；未解决主要漏检。此结果不能单独证明具体特征的因果作用，也不能据此认定现有输入不存在有用信息。',
            '完整分布、固定误报预算、分付款账户/支付方式/日期结果见[四模型G专项复核](result/gnn_bank70_v2_diagnosis_20261003/summary.json)。', '',
            '### 已完成的定位与仍待执行的后续诊断', '',
            '已核验：Train G为1,801,617笔、2,551正例，正例率0.141595%；Validation正例率0.138563%。正例付款账户15个完全重叠。Train G自转账为0；Validation的662个G正例也全部非自转账。',
            '已有均匀正常对照样本中G有10,177笔，也无自转账；这只是负例样本，不是全量负例。付款方24小时同币种外部笔数中位数，正例28,903、该正常样本28,870。',
            '这些结果排除了“没有训练正例”“付款方均为新账户”“这662例全部落入自转账特征不适用分支”等解释；不能排除组内模式漂移、收款方变化或模型未利用有效行为差异。',
            '下一步尚未全部实施：真实loader输入逐笔审计、同账户/支付方式/币种/日期的严格匹配对照、冻结最终模型在Train G上的诊断推理。当前没有最终模型Train G分数；在线训练BCE不能代替这些指标。尚未自动新增其他候选或epoch。', '',
            '**以上均为已用于诊断、特征设计和选型的Validation开发结果；Test及尾部仍未处理、未独立评估。**', '',
            '<!-- GNN_V2_FINAL_REVIEW_END -->', '', '---', '']
        audit.mkdir(parents=True, exist_ok=True)
        backup = audit/'gptpro_handoff_before_final_review.md'
        with backup.open('xb') as stream:
            stream.write(previous)
        updated = ('\n'.join(lines)+'\n').encode('utf-8')+previous
        temporary = audit/'gptpro_handoff_updated.tmp'
        with temporary.open('xb') as stream:
            stream.write(updated)
        assert handoff.read_bytes() == previous
        temporary.replace(handoff)
        assert handoff.read_bytes() == updated
        artifacts = [result/'comparison.json', result/'best_model.json', result/'调参结果.md',
            result/'comparison.corrected.json', result/'bank70_correction.json', result/'调参结果_银行分层更正.md',
            result/'final_integrity_verification.json', result/'recovery/20261003_bank70_correction/verification.json',
            diagnosis_path, handoff]
        final = audit/'delivery_verification.json'
        assert not final.exists()
        atomic_json(final, dict(status='passed', completed_at=datetime.now(timezone.utc).isoformat(),
            selected_model=selected, handoff_updated=True, bank_groups_corrected=True,
            artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in artifacts},
            prior_handoff_sha256=sha(backup), test_evaluated=False,
            scope='v2 A/B/C complete; further input/matched/Train-inference diagnosis remains future work.'))
        print(json.dumps({'status':'passed','handoff':str(handoff),'delivery_verification':str(final)},ensure_ascii=True))


if __name__ == '__main__':
    main()
