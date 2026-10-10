"""Full-Validation, additive research comparison; no deployment-pointer mutation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import traceback

import numpy as np
import pyarrow.parquet as pq

from .cache_train import sha, array_sha
from .training_state import ProcessLock, atomic_json
from .v2_evaluate import checked_path, confusion, group_report, paired_changes, support_bucket, _disk
from .v2_compare import load_predictions as old_predictions, qualification
from .v3_metrics import score_metrics
from .v3_pipeline import TRIALS, verify_freeze, source_hashes, read, now

TRAIN_TOTAL, TOTAL, POSITIVES = 20318992, 4689006, 6420
NAMES = ('lr_only', 'features_only')+TRIALS


def choose_best(results):
    if len(results) != len(NAMES) or {x['name'] for x in results} != set(NAMES):
        raise ValueError('Exactly the two frozen references and two new GNNs required')
    def key(r):
        m = r['metrics']; p = m['precision80']
        return (m['best_f1']['f1'], m['average_precision'], -1 if p is None else p['recall'])
    # Stable order retains an earlier reference if every ranking metric ties.
    return max(results, key=key)


def load_new_predictions(project, folder, expected_times, labels, model_sha):
    report, config = read(folder/'metrics.json'), read(folder/'config.json')
    config_hash = sha(folder/'config.json')
    if (report.get('status') != 'completed' or report['config_sha256'] != config_hash
            or report['model_sha256'] != model_sha):
        raise ValueError('New prediction configuration/model mismatch')
    if config.get('model_sha256') != model_sha:
        raise ValueError('Evaluation config model identity differs')
    for rel, digest in config.get('code_sha256', {}).items():
        path = project/'scripts/gnn_features'/rel
        if sha(path) != digest: raise ValueError('Prediction code changed: '+rel)
    base = read(project/'data/gnn_validation_cache_parquet/manifest.json')['blocks']
    blocks = report['blocks']
    if (len(blocks) != 197 or [b['block'] for b in blocks] != [b['block'] for b in base]
            or [b['targets'] for b in blocks] != [b['targets'] for b in base]):
        raise ValueError('Validation block coverage/order differs')
    scores, seen = np.empty(TOTAL, np.float32), np.zeros(TOTAL, bool)
    for entry in blocks:
        block = checked_path(folder/'predictions', entry['block'])
        record = read(block/'manifest.json')
        if (sha(block/'manifest.json') != entry['manifest_sha256'] or record['config_sha256'] != config_hash
                or sha(block/'predictions.parquet') != record['file_sha256']):
            raise ValueError('Prediction block hash differs')
        tab = pq.ParquetFile(block/'predictions.parquet').read()
        ids = tab['edge_id'].to_numpy(); pos = ids-TRAIN_TOTAL
        if (len(ids) != entry['targets'] or len(ids) != record['targets']
                or array_sha(ids) != record['target_ids_sha256'] or np.any(pos < 0) or np.any(pos >= TOTAL)
                or len(np.unique(pos)) != len(pos) or seen[pos].any()):
            raise ValueError('Prediction ID coverage differs')
        if (not np.array_equal(tab['cutoff_us'].to_numpy(), expected_times[pos])
                or not np.array_equal(tab['label'].to_numpy(), labels[pos])):
            raise ValueError('Prediction timestamps/labels differ')
        values = tab['probability'].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Nonfinite/out-of-range prediction')
        scores[pos], seen[pos] = values, True
    if not seen.all(): raise ValueError('Missing Validation predictions')
    metrics = score_metrics(labels, scores)
    if metrics != report['metrics']: raise ValueError('Stored metrics differ from prediction replay')
    return scores, dict(config_sha256=config_hash, metrics_sha256=sha(folder/'metrics.json'),
                        blocks=blocks, config=config, model_sha256=model_sha)


def _markdown(report):
    lines = ['# 第三轮：金额与时间分布统计GNN对照', '',
             '完整Train 20,318,992笔／17,794正例；完整Validation 4,689,006笔／6,420正例。',
             '主指标为全局最大F1，阈值同分整组纳入；同时列P80。Validation已参与诊断与选择，Test未评估。',
             '| 模型 | AP | ROC-AUC | 最大F1 | F1点精确率 | F1点召回 | TP/FP/FN/告警 | P80召回 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in report['results']:
        m = r['metrics']; f, p = m['best_f1'], m['precision80']
        pr = '无可行阈值' if p is None else f"{p['recall']:.4%}"
        lines.append(f"| {r['name']} | {m['average_precision']:.6f} | {m['roc_auc']:.6f} | {f['f1']:.4%} | "
                     f"{f['precision']:.4%} | {f['recall']:.4%} | {f['tp']}/{f['fp']}/{f['fn']}/{f['alerts']} | {pr} |")
    lines += ['', '研究排名第一：**'+report['best']['name']+'**。冻结旧模型lr_only及其部署选择文件未替换。',
              '模型：`'+report['best']['model_path']+'`；SHA256：`'+report['best']['model_sha256']+'`。', '',
              '## 顺序对照', '', '| 对照 | 口径 | F1变化（百分点） | 新检出 | 损失检出 | 新增误报 | 消除误报 | 新检出独立账户 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pair, points in report['paired_changes'].items():
        for name, c in points.items():
            if not c['available']:
                lines.append(f'| {pair} | {name} | 无可比阈值 | — | — | — | — | — |'); continue
            lines.append(f"| {pair} | {name} | {100*c['f1_delta']:+.4f} | {c['gained_tp']['transactions']} | {c['lost_tp']['transactions']} | "
                         f"{c['new_fp']['transactions']} | {c['removed_fp']['transactions']} | {c['gained_tp']['either_accounts']} |")
    lines += ['', '## 非ACH与银行70交叉组', '', '| 模型 | 口径 | 分组 | 正例 | TP | FP | FN | 精确率 | 召回 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in report['results']:
        for point, grouped in r['groups'].items():
            for name, value in [('ach_group', 'non_ACH'), ('payer_bank70_non_ach', 'bank70_non_ACH')]:
                g = grouped[name][value]; c = g['operating_point']
                if c is None: continue
                p = '无告警' if c['precision'] is None else f"{c['precision']:.4%}"
                rec = '无正例' if c['recall'] is None else f"{c['recall']:.4%}"
                lines.append(f"| {r['name']} | {point} | {value} | {g['positives']} | {c['tp']} | {c['fp']} | {c['fn']} | {p} | {rec} |")
    lines += ['', '各支付方式、日期、银行70内外、四方向历史支持数0／1–4／≥5及独立账户覆盖，详见comparison.json。',
              '全部分层复用全局阈值；银行70通过冻结bank_identity解析。',
              '原业务达标条件（P≥80%、TP≥3469、非ACH TP≥143）单列qualification，不作为本轮F1研究排名门槛。',
              '不宣称单种子涨点统计显著，不与论文Test成绩作直接胜负判断。']
    return '\n'.join(lines)+'\n'


def run(project):
    project = Path(project).resolve(); root = project/'result/gnn_tuning_v3'
    state = dict(status='running', pid=os.getpid(), stage='verify_inputs', started_at=now())
    def update(**kw):
        state.update(kw, updated_at=now()); atomic_json(root/'comparison_status.json', state)
    with ProcessLock(root/'comparison_lock'):
        try:
            update()
            config = read(root/'config.json')
            if verify_freeze(project) != config['frozen_anchors'] or source_hashes(project) != config['source_sha256']:
                raise ValueError('Root source/frozen input mismatch')
            archive = project/'data/gnn_v1_validation'
            cache = project/'data/gnn_validation_cache_parquet'
            hashes = read(cache/'config.json')['hashes']
            keys = ('timestamp', 'src', 'dst', 'payment_format_cat', 'bank_key')
            for relative in [f'data/gnn_v1_validation/history_index/{x}.npy' for x in keys]+[
                    'data/gnn_v1_validation/metadata/bank_identity.json', 'data/gnn_v1_validation/metadata/vocabulary.json']:
                expected = hashes.get(str(Path(relative)))
                if expected is None or sha(project/relative) != expected: raise ValueError('Group source changed: '+relative)
            arrays = {k: np.load(archive/'history_index'/f'{k}.npy', mmap_mode='r') for k in keys}
            times = arrays['timestamp'][TRAIN_TOTAL:]
            src, dst, fmt = (arrays[k][TRAIN_TOTAL:] for k in ('src', 'dst', 'payment_format_cat'))
            if times.shape != (TOTAL,): raise ValueError('Validation archive size differs')
            vocab = read(archive/'metadata/vocabulary.json')['values']
            ach = vocab['payment_format'].index('ACH')+2
            non_ach = fmt != ach
            identity = read(archive/'metadata/bank_identity.json')
            if identity.count('70') != 1: raise ValueError('Ambiguous raw bank70 identity')
            bank70 = identity.index('70'); payer70 = arrays['bank_key'][src] == bank70
            groups = dict(ach_group=np.where(non_ach, 'non_ACH', 'ACH'),
                payment_format=np.asarray(['missing', 'unknown']+vocab['payment_format'])[fmt],
                payer_bank70=np.where(payer70, 'bank70', 'other'),
                receiver_bank70=np.where(arrays['bank_key'][dst] == bank70, 'bank70', 'other'),
                payer_bank70_non_ach=np.where(payer70 & non_ach, 'bank70_non_ACH', 'other'),
                date=times.astype('datetime64[us]').astype('datetime64[D]').astype(str))
            from .v3_features import SidecarStore
            side = SidecarStore(project/'data/gnn_v3_distribution_features/validation', cache, expected_split='validation')
            supports = {role: np.full(TOTAL, -2, np.int64) for role in ('src_in', 'src_out', 'dst_in', 'dst_out')}
            for i, entry in enumerate(read(cache/'manifest.json')['blocks']):
                tab = pq.ParquetFile(checked_path(cache, entry['block'])/'plans.parquet').read(columns=['targets'])
                ids = np.concatenate([np.asarray(x, np.int64) for x in tab['targets'].to_pylist()]); pos = ids-TRAIN_TOTAL
                if (np.any(pos < 0) or np.any(pos >= TOTAL) or len(np.unique(pos)) != len(pos)
                        or np.any(supports['src_in'][pos] != -2)):
                    raise ValueError('Support block target overlap/scope differs')
                counts = side.support(entry['block'], ids, times[pos])
                side.block(entry['block'], ids, times[pos])
                for role in supports: supports[role][pos] = counts[role+'_count_24h']
                update(stage='verify_support', blocks=i+1)
            for role, count in supports.items():
                if np.any(count < -1) or not np.array_equal(count == -1, src == dst):
                    raise ValueError('Support coverage/applicability differs')
                groups[role+'_support_24h'] = support_bucket(count)
            y = None; scores = {}; results = []
            from .v2_evaluate import verified_training as old_training
            from .v3_evaluate import verified_training
            for name in NAMES:
                update(stage='verify_predictions', model=name)
                if name in ('lr_only', 'features_only'):
                    trained = old_training(project, name)
                    y, s, provenance = old_predictions(project, name, times, y)
                    path, digest = str(trained['model_path'].relative_to(project)), trained['model_sha256']
                else:
                    trained = verified_training(project, name)
                    path, digest = str(trained['model_path'].relative_to(project)), trained['model_sha256']
                    s, provenance = load_new_predictions(project, root/'evaluations'/name, times, y, digest)
                    for key in ('manifest_sha256', 'config_sha256', 'contract_sha256'):
                        if provenance['config']['distribution_'+key] != getattr(side, key):
                            raise ValueError('GNN prediction uses different distribution sidecar')
                scores[name] = s
                metrics = score_metrics(y, s)
                if metrics['samples'] != TOTAL or metrics['positives'] != POSITIVES or int(y[non_ach].sum()) != 715:
                    raise ValueError('Full Validation/class coverage differs')
                p = metrics['precision80']
                non = None if p is None else confusion(y[non_ach], s[non_ach] >= p['threshold'], p['threshold'])
                item = dict(name=name, model_path=path, model_sha256=digest, metrics=metrics,
                            non_ach_precision80=non, qualification=qualification(p, non), provenance=provenance)
                item['groups'] = {point: group_report(y, s, metrics[point], groups, src, dst)
                                  for point in ('best_f1', 'precision80')}
                results.append(item)
            lookup = {r['name']: r for r in results}
            pairs = [('features_only', TRIALS[0]), (TRIALS[0], TRIALS[1])]+[('lr_only', n) for n in NAMES[2:]]
            changes = {a+' -> '+b: {p: paired_changes(y, scores[a], lookup[a]['metrics'][p], scores[b],
                       lookup[b]['metrics'][p], src, dst) for p in ('best_f1', 'precision80')} for a, b in pairs}
            for a, b in pairs:
                for point, change in changes[a+' -> '+b].items():
                    if change['available']:
                        for metric in ('f1', 'precision', 'recall'):
                            change[metric+'_delta'] = lookup[b]['metrics'][point][metric]-lookup[a]['metrics'][point][metric]
            winner = choose_best(results)
            report = dict(status='completed', version=3, results=results, paired_changes=changes,
                best={k: winner[k] for k in ('name', 'model_path', 'model_sha256')},
                bank70_identity=dict(raw='70', encoded=bank70, sha256=sha(archive/'metadata/bank_identity.json')),
                research_ranking=['best_f1', 'average_precision', 'precision80_recall'],
                original_business_gates=dict(precision=.8, min_tp=3469, min_non_ach_tp=143),
                deployment_unchanged=True, test_evaluated=False, completed_at=now(),
                root_config_sha256=sha(root/'config.json'),
                scope='Reused Validation development, one seed; no independent Test or significance claim')
            _disk(project)
            if (root/'comparison.json').exists():
                saved = read(root/'comparison.json'); report['completed_at'] = saved.get('completed_at')
                if not report['completed_at'] or report != saved: raise ValueError('Comparison replay differs')
            else: atomic_json(root/'comparison.json', report)
            selected = dict(name=winner['name'], path=winner['model_path'], sha256=winner['model_sha256'],
                validation_threshold=winner['metrics']['best_f1']['threshold'],
                comparison_sha256=sha(root/'comparison.json'), research_only=True, deployment_replaced=False, test_evaluated=False)
            if (root/'best_model.json').exists() and read(root/'best_model.json') != selected:
                raise ValueError('Research selection replay differs')
            atomic_json(root/'best_model.json', selected)
            content = _markdown(report); path = root/'调参结果.md'
            if path.exists() and path.read_text(encoding='utf-8') != content: raise ValueError('Report replay differs')
            if not path.exists():
                temp = path.with_suffix('.md.tmp'); temp.write_text(content, encoding='utf-8'); temp.replace(path)
            update(status='completed', stage='complete', winner=winner['name'])
            return report
        except BaseException as exc:
            update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, default=Path.cwd())
    run(p.parse_args().project)
