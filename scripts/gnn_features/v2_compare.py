"""Verified three-way v2 comparison at one global P80 point per model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .build import load_archive
from .cache_train import array_sha, sha
from .evaluation_metrics import binary_metrics
from .training_state import ProcessLock, atomic_json
from .validation_cache import DATES, END, TOTAL, POSITIVES, read
from .v2_evaluate import (TRIALS, TRAIN_TOTAL, checked_path, confusion, group_report,
                          paired_changes, precision80_point,
                          verified_training, _disk)

MIN_TP = 3469  # ceil(6420 * 0.540312), rather than rounded percentage checks.
MIN_NON_ACH_TP = 143
NAMES = ('lr_only',) + TRIALS
SUPPORT_LABELS = {-1: 'not_applicable', 0: '0', 1: '1-4', 2: '>=5'}


def support_codes(counts):
    """Compact categorical codes, never reconstructed count estimates."""
    values = np.asarray(counts)
    if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values < -1)
            or np.any(values != np.floor(values))):
        raise ValueError('Invalid raw historical support count')
    return np.where(values < 0, -1, np.where(values == 0, 0, np.where(values < 5, 1, 2))).astype(np.int8)


def encoded_support_codes(encoded, valid):
    """Bucket verified log1p(count) values; do not invert float32 into integers.

    The upper small-sample boundary lies halfway between encoded counts 4 and
    5. It is far from float32 round-off on either side. Zero is exactly encoded.
    """
    values, masks = np.asarray(encoded), np.asarray(valid)
    if (values.ndim != 1 or masks.shape != values.shape or not np.isfinite(values).all()
            or np.any(values < 0) or not np.isin(masks, (0, 1)).all()
            or np.any(values[masks == 0] != 0)):
        raise ValueError('Invalid encoded support count or validity mask')
    boundary = (np.log1p(4.) + np.log1p(5.))/2
    return np.where(masks == 0, -1, np.where(values == 0, 0, np.where(values < boundary, 1, 2))).astype(np.int8)


def _bound_input(project, path, expected_hashes, feature_root=None):
    path, project = Path(path).resolve(), Path(project).resolve()
    if not path.is_relative_to(project) or (feature_root is not None and not path.is_relative_to(Path(feature_root).resolve())):
        raise ValueError('Historical support input escapes its declared root')
    digest = expected_hashes.get(str(path.relative_to(project)))
    if digest is None or sha(path) != digest:
        raise ValueError('Historical support source is missing from frozen identity or changed')
    return path


def _support24(project, times, self_transfer, sidecar_config, cache_config, progress=None):
    """Join two frozen raw 24h counts by complete edge/timestamp identities."""
    project = Path(project).resolve()
    times, self_transfer = np.asarray(times), np.asarray(self_transfer, bool)
    if times.shape != (TOTAL,) or self_transfer.shape != (TOTAL,):
        raise ValueError('24h support target scope differs')
    features = project/'result/gnn_validation_v1'
    source_hashes = {str(Path(k)): v for k, v in sidecar_config['input_hashes'].items()}
    manifest_hashes = {str(Path(k)): v for k, v in cache_config['hashes'].items()}
    groups = {f'{role}_count_24h': np.full(TOTAL, -2, np.int8) for role in ('src', 'dst')}
    seen = np.zeros(TOTAL, bool)
    names = dict(src='r_src_ext_out_count_24h', dst='r_dst_ext_in_count_24h')
    for date in DATES:
        folder = features/f'transaction_date={date}'
        manifest_path = _bound_input(project, folder/'merge_phase.json', manifest_hashes, features)
        day_start = int(np.datetime64(date, 'us').astype(np.int64))
        day_end = day_start+86_400_000_000
        for part in read(manifest_path)['parts']:
            path = _bound_input(project, part['path'], source_hashes, folder)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=['edge_id', 'timestamp']+list(names.values())):
                ids = batch.column('edge_id').to_numpy().astype(np.int64)
                pos = ids-TRAIN_TOTAL
                if (np.any(pos < 0) or np.any(pos >= TOTAL) or len(np.unique(pos)) != len(pos)
                        or np.any(seen[pos])):
                    raise ValueError('24h support duplicated or out-of-scope target')
                cutoff = batch.column('timestamp').cast(pa.int64()).to_numpy()
                if (not np.array_equal(cutoff, times[pos]) or np.any(cutoff < day_start) or np.any(cutoff >= day_end)):
                    raise ValueError('24h support timestamp/date identity differs')
                for role, name in names.items():
                    column = batch.column(name)
                    valid = column.is_valid().to_numpy(zero_copy_only=False)
                    if not np.array_equal(valid, ~self_transfer[pos]):
                        raise ValueError('24h support applicability differs from target identity')
                    count = np.full(len(ids), -1., np.float64)
                    count[valid] = column.to_numpy(zero_copy_only=False)[valid]
                    groups[f'{role}_count_24h'][pos] = support_codes(count)
                seen[pos] = True
            if progress:
                progress(stage='verify_support_24h', date=date, verified_targets=int(seen.sum()))
    if not seen.all():
        raise ValueError('Incomplete 24h support target coverage')
    return groups


def qualification(point, non_ach_point):
    if point is None or non_ach_point is None:
        return dict(passed=False, precision=False, overall_recall=False, non_ach_recall=False,
                    reason='No nonempty global P80 operating point')
    if point.get('positives') != POSITIVES or non_ach_point.get('positives') != 715:
        raise ValueError('Selection requires full Validation class counts')
    checks = dict(precision=(point['alerts'] > 0 and 5*point['tp'] >= 4*point['alerts']),
                  overall_recall=point['tp'] >= MIN_TP,
                  non_ach_recall=non_ach_point['tp'] >= MIN_NON_ACH_TP)
    return dict(passed=all(checks.values()), **checks)


def choose_best(results):
    by_name = {item['name']: item for item in results}
    if len(by_name) != len(results) or set(by_name) != set(NAMES):
        raise ValueError('Exactly the frozen reference and three declared candidates are required')
    candidates = [by_name[name] for name in TRIALS
                  if qualification(by_name[name]['precision80'], by_name[name]['non_ach_precision80'])['passed']]
    if not candidates:
        return by_name['lr_only'], 'No new candidate meets all three balance gates; retain frozen lr_only'
    selected = sorted(candidates, key=lambda item: (-item['precision80']['tp'],
                      -item['non_ach_precision80']['tp'], -item['metrics']['average_precision'], item['name']))[0]
    return selected, 'Qualified candidates ranked by overall recall, non-ACH recall, AP, then stable trial name'


def load_predictions(project, name, expected_times, reference_labels=None):
    """Read only committed, hash-bound, complete prediction blocks by edge ID."""
    project = Path(project).resolve()
    folder = project/('result/gnn_tuning_v1' if name == 'lr_only' else 'result/gnn_tuning_v2')/'evaluations'/name
    report, config = read(folder/'metrics.json'), read(folder/'config.json')
    config_hash = sha(folder/'config.json')
    if (report.get('status') != 'completed' or report.get('config_sha256') != config_hash
            or config['dates'] != DATES or config['expected_targets'] != TOTAL or config['expected_positives'] != POSITIVES
            or report['model_sha256'] != config['model_sha256']):
        raise ValueError('Incomplete/changed evaluation configuration: '+name)
    for source, digest in config['code_sha256'].items():
        if sha(Path(__file__).with_name(source)) != digest:
            raise ValueError('Frozen evaluation source changed: '+source)
    cache = project/'data/gnn_validation_cache_parquet'
    if (sha(cache/'config.json') != config['cache_config_sha256']
            or sha(cache/'manifest.json') != config['cache_manifest_sha256']):
        raise ValueError('Evaluation frozen cache identity differs')
    blocks = report['blocks']
    if len(blocks) != 197 or len({b['block'] for b in blocks}) != 197 or sum(b['targets'] for b in blocks) != TOTAL:
        raise ValueError('Evaluation block inventory differs')
    y, scores, seen = np.empty(TOTAL, np.int8), np.empty(TOTAL, np.float32), np.zeros(TOTAL, bool)
    for entry in blocks:
        block = entry['block']
        if block.split('/')[0].split('=')[1] not in DATES:
            raise ValueError('Non-Validation prediction block')
        block_root = checked_path(folder/'predictions', block)
        record, path = block_root/'manifest.json', block_root/'predictions.parquet'
        manifest = read(record)
        if (sha(record) != entry['manifest_sha256'] or manifest['config_sha256'] != config_hash
                or sha(path) != manifest['file_sha256']):
            raise ValueError('Prediction block hash/config changed')
        table = pq.ParquetFile(path).read()
        ids = table['edge_id'].to_numpy()
        local = ids-TRAIN_TOTAL
        if (len(ids) != entry['targets'] or len(ids) != manifest['targets']
                or array_sha(ids) != manifest['target_ids_sha256'] or np.any(local < 0) or np.any(local >= TOTAL)
                or len(np.unique(local)) != len(local) or np.any(seen[local])):
            raise ValueError('Prediction target identity/coverage differs')
        if not np.array_equal(table['cutoff_us'].to_numpy(), expected_times[local]):
            raise ValueError('Prediction cutoff identity differs')
        values, labels = table['probability'].to_numpy(), table['label'].to_numpy()
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid prediction labels/probabilities')
        if reference_labels is not None and not np.array_equal(labels, reference_labels[local]):
            raise ValueError('Prediction labels differ across models')
        y[local], scores[local], seen[local] = labels, values, True
    if not seen.all() or int(y.sum()) != POSITIVES:
        raise ValueError('Incomplete full Validation prediction coverage')
    metrics = binary_metrics(y, scores)
    for key in ('samples', 'positives', 'average_precision', 'roc_auc'):
        if metrics[key] != report['metrics'][key]:
            raise ValueError('Stored overall metric differs from verified probabilities: '+key)
    return y, scores, dict(metrics_sha256=sha(folder/'metrics.json'), config_sha256=config_hash,
                          model_sha256=config['model_sha256'], blocks=blocks, config=config, metrics=metrics)


def _support(project, blocks, times, self_transfer, progress=None):
    from .v2_features import EXTRA_NAMES, SidecarStore
    cache = project/'data/gnn_validation_cache_parquet'
    store = SidecarStore(project/'data/gnn_v2_target_features/validation', cache, expected_split='validation')
    cache_config = read(cache/'config.json')
    if sha(cache/'config.json') != store.config['base_config_sha256']:
        raise ValueError('Historical support base-cache config differs')
    groups = {f'{role}_count_{window}': np.full(TOTAL, -2, np.int8)
              for role in ('src', 'dst') for window in ('72h', '7d')}
    # Use the cached plans' exact target IDs, independently of prediction order.
    for index, entry in enumerate(blocks):
        base = checked_path(cache, entry['block'])
        tab = pq.ParquetFile(base/'plans.parquet').read(columns=['targets'])
        ids = np.concatenate([np.asarray(v, np.int64) for v in tab['targets'].to_pylist()])
        pos = ids-TRAIN_TOTAL
        if len(ids) != entry['targets'] or np.any(pos < 0) or np.any(pos >= TOTAL):
            raise ValueError('Support-cache target identity differs')
        found = store.support(entry['block'], ids, times[pos])
        extra = store.block(entry['block'], ids, times[pos])
        for role in ('src', 'dst'):
            seven, three = f'{role}_count_7d', f'{role}_count_72h'
            if np.any(groups[seven][pos] != -2):
                raise ValueError('Repeated support target')
            groups[seven][pos] = support_codes(found[seven])
            column = EXTRA_NAMES.index(f'v2_{role}_count_72h')
            groups[three][pos] = encoded_support_codes(extra[:, column], extra[:, len(EXTRA_NAMES)+column])
            if not np.array_equal(groups[three][pos] == -1, self_transfer[pos]):
                raise ValueError('72h support applicability differs from target identity')
            if not np.array_equal(groups[seven][pos] == -1, self_transfer[pos]):
                raise ValueError('7d support applicability differs from target identity')
        if progress and (index % 25 == 0 or index+1 == len(blocks)):
            progress(stage='verify_support', completed_blocks=index+1)
    if any(np.any(value < -1) for value in groups.values()):
        raise ValueError('Incomplete support feature coverage')
    groups.update(_support24(project, times, self_transfer, store.config, cache_config, progress))
    for role in ('src', 'dst'):
        if (np.any(groups[f'{role}_count_24h'] > groups[f'{role}_count_72h'])
                or np.any(groups[f'{role}_count_72h'] > groups[f'{role}_count_7d'])):
            raise ValueError('Historical support buckets decrease across longer same-currency windows')
    return groups, dict(manifest_sha256=store.manifest_sha256, config_sha256=store.config_sha256,
                        contract_sha256=store.contract_sha256)


def _markdown(report):
    lines = ['# GNN 第二轮三组对照结果', '',
             '完整 Train：20,318,992 笔／17,794 正例；每组从头训练 1 epoch。',
             '完整 Validation：2022-09-11 至 13，4,689,006 笔／6,420 正例；其中非 ACH 正例 715 笔。',
             '每个模型只选择一个全局阈值；同分交易整组纳入。在精确率至少 80% 的阈值中取整体召回最高者。', '',
             '| 模型 | AP | ROC-AUC | 阈值 | 精确率 | 整体召回 | 非 ACH 召回 | TP | FP | FN | 告警量 | 平衡达标 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for item in report['results']:
        p, n, m = item['precision80'], item['non_ach_precision80'], item['metrics']
        if p is None:
            detail = '| 不存在非空 P80 阈值 | — | — | — | — | — | — | — | 否 |'
            lines.append(f"| {item['name']} | {m['average_precision']:.6f} | {m['roc_auc']:.6f} "+detail)
        else:
            lines.append(f"| {item['name']} | {m['average_precision']:.6f} | {m['roc_auc']:.6f} | {p['threshold']:.9g} | "
                         f"{p['precision']:.2%} | {p['recall']:.2%} | {n['recall']:.2%} | {p['tp']} | {p['fp']} | "
                         f"{p['fn']} | {p['alerts']} | {'是' if item['qualification']['passed'] else '否'} |")
    lines += ['', '达标要求：整体精确率 ≥80%，整体 TP ≥3,469（召回 ≥54.0312%），非 ACH TP ≥143（召回 ≥20%）。',
              '参考模型旧 lr_only 的同口径 TP=3,533、FP=880、召回=55.03115264797508%。', '',
              '选择结果：**'+report['best']['name']+'**。'+('没有新候选全部达标，保留冻结旧模型。' if report['best']['name'] == 'lr_only'
                else '在达标候选中依次按整体召回、非 ACH 召回、AP 排序。'),
              '模型：`'+report['best']['model_path']+'`',
              '模型 SHA256：`'+report['best']['model_sha256']+'`', '',
              '## 顺序对照变化', '',
              '| 对照 | 新检出 TP | 损失 TP | 新增 FP | 消除 FP | 新检出付款账户 | 新检出收款账户 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, change in report['paired_changes'].items():
        if not change['available']:
            lines.append(f'| {name} | 无可比 P80 阈值 | — | — | — | — | — |')
        else:
            g = change['gained_tp']
            lines.append(f"| {name} | {g['transactions']} | {change['lost_tp']['transactions']} | {change['new_fp']['transactions']} | "
                         f"{change['removed_fp']['transactions']} | {g['payer_accounts']} | {g['receiver_accounts']} |")
    lines += ['', 'features_only → features_pruned 检验删减 6 项 ACH 专属统计的效果；',
              'features_pruned → features_balanced 检验温和正例权重调整的效果。每个模型采用各自选定的同口径全局阈值。', '',
              '## 分层结果', '',
              '完整分层及独立账户覆盖见 comparison.json 的 groups：支付方式、付款银行 70 内外、收款银行 70 内外、',
              '日期，以及付款/收款两个角色各自的 24 小时、72 小时、7 天历史支持数（0、1–4、≥5；自转账为不适用）。',
              '24 小时分组读取已冻结原 R 的同币种外部笔数，核对源文件 SHA、全部交易 ID、时间和适用性；',
              '72 小时直接将已核验的 log1p 笔数归入稳定边界桶，不反算或宣称恢复精确整数；7 天使用已核验原始整数笔数。',
              '所有分层复用整体阈值，不为分组单独调阈值；不以少数账户的重复交易代替独立账户覆盖。', '',
              '## 解释边界', '',
              'Validation 已用于发现问题、特征设计、阈值选择和模型选择，因此结果是开发集改进证据。',
              '**Test 2022-09-14 至 16 及尾部未处理、未评估，尚无独立 Test 性能结论。**',
              '没有增加候选、epoch、环特征或 GNN 层数；没有加入标签型交易对手风险。',
              '不同权重下的在线训练 BCE 不用于模型横向排名。']
    return '\n'.join(lines)+'\n'


def _run(project, output=None, progress=None):
    project = Path(project).resolve()
    root = project/'result/gnn_tuning_v2'
    out = Path(output).resolve() if output else root
    if out != root:
        raise ValueError('Comparison output must be the canonical v2 result directory')
    _disk(project)
    # A separate child lock allows the already-locked pipeline to invoke this CLI.
    with ProcessLock(root/'comparison_lock'):
        archive = project/'data/gnn_v1_validation'
        meta, arrays, bank, vocab = load_archive(archive)
        if meta['rows'] != TRAIN_TOTAL+TOTAL or meta['end_exclusive'] != END:
            raise ValueError('Unexpected Validation archive scope')
        times = arrays['timestamp'][TRAIN_TOTAL:]
        src, dst = arrays['src'][TRAIN_TOTAL:], arrays['dst'][TRAIN_TOTAL:]
        fmt = arrays['payment_format_cat'][TRAIN_TOTAL:]
        ach = vocab['payment_format'].index('ACH')+2
        non_ach = fmt != ach
        score_sets, results, y, prediction_blocks = {}, [], None, None
        evaluation_configs = {}
        for name in NAMES:
            if progress:
                progress(stage='verify_predictions', current_model=name)
            trained = verified_training(project, name)
            del trained['model']
            current_y, scores, verified = load_predictions(project, name, times, y)
            if verified['model_sha256'] != trained['model_sha256']:
                raise ValueError('Evaluation used a different final model')
            if y is None:
                y = current_y
                prediction_blocks = verified['blocks']
            if int(y[non_ach].sum()) != 715:
                raise ValueError('Non-ACH positive population differs')
            point = precision80_point(y, scores)
            npnt = None if point is None else confusion(y[non_ach], scores[non_ach] >= point['threshold'], point['threshold'])
            if name == 'lr_only' and (point is None or point['tp'] != 3533 or point['fp'] != 880):
                raise ValueError('Frozen reference P80 result differs')
            results.append(dict(name=name, model_path=str(trained['model_path'].relative_to(project)),
                                model_sha256=trained['model_sha256'], training_config_sha256=trained['config_sha256'],
                                training_result_sha256=trained['training_result_sha256'],
                                evaluation_config_sha256=verified['config_sha256'], metrics_sha256=verified['metrics_sha256'],
                                metrics=verified['metrics'], precision80=point, non_ach_precision80=npnt,
                                qualification=qualification(point, npnt), groups={}))
            score_sets[name] = scores
            evaluation_configs[name] = verified['config']
        support_groups, sidecar_identity = _support(project, prediction_blocks, times, src == dst, progress)
        for name in TRIALS:
            cfg = evaluation_configs[name]
            for key in ('manifest_sha256', 'config_sha256', 'contract_sha256'):
                if cfg['sidecar_'+key] != sidecar_identity[key]:
                    raise ValueError('Evaluation used a different sidecar: '+name)
        dates = (times.astype('datetime64[us]').astype('datetime64[D]')).astype(str)
        format_names = np.array(['missing', 'unknown']+list(vocab['payment_format']))
        if np.any(fmt < 0) or np.any(fmt >= len(format_names)):
            raise ValueError('Invalid payment format encoding')
        groups = dict(payment_format=format_names[fmt], ach_group=np.where(non_ach, 'non_ACH', 'ACH'),
                      payer_bank70=np.where(bank[src] == 70, 'bank70', 'other'),
                      receiver_bank70=np.where(bank[dst] == 70, 'bank70', 'other'), date=dates)
        support_labels = {}
        for role, label in (('src', 'payer'), ('dst', 'receiver')):
            for window in ('24h', '72h', '7d'):
                key = f'{label}_support_{window}'
                groups[key] = support_groups[f'{role}_count_{window}']
                support_labels[key] = SUPPORT_LABELS
        for item in results:
            if progress:
                progress(stage='group_metrics', current_model=item['name'])
            item['groups'] = group_report(y, score_sets[item['name']], item['precision80'], groups, src, dst, support_labels)
        lookup = {x['name']: x for x in results}
        pairs = [('features_only', 'features_pruned'), ('features_pruned', 'features_balanced')]+[('lr_only', n) for n in TRIALS]
        changes = {a+' -> '+b: paired_changes(y, score_sets[a], lookup[a]['precision80'], score_sets[b],
                   lookup[b]['precision80'], src, dst) for a, b in pairs}
        best, reason = choose_best(results)
        report = dict(status='completed', version=2, results=results, best={k: best[k] for k in
                      ('name', 'model_path', 'model_sha256', 'precision80', 'non_ach_precision80')},
                      selection_reason=reason, requirements=dict(precision_min=.8, overall_tp_min=MIN_TP,
                      non_ach_tp_min=MIN_NON_ACH_TP, validation_targets=TOTAL, validation_positives=POSITIVES,
                      non_ach_positives=715), paired_changes=changes, sidecar_identity=sidecar_identity,
                      support_grouping=dict(windows=['24h', '72h', '7d'], roles=['payer', 'receiver'],
                          buckets=['0', '1-4', '>=5', 'not_applicable'],
                          count_24h='Frozen original R; SHA, full edge ID, timestamp and applicability verified',
                          count_72h='Verified sidecar log1p count bucket; no inverse integer reconstruction',
                          count_7d='Verified sidecar raw integer count'),
                      code_sha256={n: sha(Path(__file__).with_name(n)) for n in ('v2_evaluate.py', 'v2_compare.py')},
                      completed_at=datetime.now(timezone.utc).isoformat(),
                      scope='Validation development/selection evidence only; Test 2022-09-14..16 and tail untouched')
        out.mkdir(parents=True, exist_ok=True)
        _disk(project)
        if (out/'comparison.json').exists():
            committed = read(out/'comparison.json')
            report['completed_at'] = committed.get('completed_at')
            if not report['completed_at'] or committed != report:
                raise ValueError('Committed comparison changed on verification replay')
        else:
            atomic_json(out/'comparison.json', report)
        selected = dict(name=best['name'], path=best['model_path'], sha256=best['model_sha256'],
                    validation_threshold=best['precision80']['threshold'], selection_reason=reason,
                    comparison_sha256=sha(out/'comparison.json'), test_evaluated=False)
        if (out/'best_model.json').exists():
            if read(out/'best_model.json') != selected:
                raise ValueError('Committed model selection differs')
        else:
            atomic_json(out/'best_model.json', selected)
        markdown = _markdown(report)
        if (out/'调参结果.md').exists():
            if (out/'调参结果.md').read_text(encoding='utf-8') != markdown:
                raise ValueError('Committed report text differs')
        else:
            temp = out/'调参结果.md.tmp'
            temp.write_text(markdown, encoding='utf-8')
            temp.replace(out/'调参结果.md')
        return report


def run(project, output=None):
    project = Path(project).resolve()
    root = project/'result/gnn_tuning_v2'
    if output is not None and Path(output).resolve() != root:
        raise ValueError('Comparison output must be the canonical v2 result directory')
    state = dict(status='running', stage='verify_inputs', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())

    def update(**fields):
        state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(root/'comparison_status.json', state)

    update()
    try:
        report = _run(project, output, update)
        update(status='completed', stage='complete', best_trial=report['best']['name'])
        return report
    except BaseException as exc:
        update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.project, args.output)
