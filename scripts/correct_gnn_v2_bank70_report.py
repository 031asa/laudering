"""Additive, audited correction of v2 bank-70 reporting; frozen files stay intact."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import psutil

from gnn_features.cache_train import sha
from gnn_features.training_state import ProcessLock
from gnn_features.v2_compare import NAMES, load_predictions, choose_best, _markdown
from gnn_features.v2_evaluate import group_report, precision80_point, verified_training, _disk
from gnn_features.v2_pipeline import source_hashes, verify_freeze

ROOT = Path(__file__).resolve().parents[1]
TOTAL, POSITIVES, PREFIX = 4689006, 6420, 20318992
GROUPS = ('payer_bank70', 'receiver_bank70')
AUDIT = 'recovery/20261003_bank70_correction'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_bank70(bank_identity, bank_keys, src, dst):
    if (not isinstance(bank_identity, list) or not all(isinstance(x, str) for x in bank_identity)
            or len(set(bank_identity)) != len(bank_identity) or bank_identity.count('70') != 1):
        raise ValueError('Unique raw bank 70 identity is required')
    keys, src, dst = np.asarray(bank_keys), np.asarray(src), np.asarray(dst)
    if (keys.ndim != 1 or keys.dtype.kind not in 'iu' or np.any(keys < 0)
            or np.any(keys >= len(bank_identity)) or src.ndim != 1 or dst.shape != src.shape
            or src.dtype.kind not in 'iu' or dst.dtype.kind not in 'iu'
            or np.any(src < 0) or np.any(dst < 0) or np.any(src >= len(keys)) or np.any(dst >= len(keys))):
        raise ValueError('Invalid bank/account identity encoding')
    code = bank_identity.index('70')
    return code, dict(payer_bank70=np.where(keys[src] == code, 'bank70', 'other'),
                      receiver_bank70=np.where(keys[dst] == code, 'bank70', 'other'))


def assert_only_bank_groups_changed(original, corrected):
    restored = copy.deepcopy(corrected)
    restored.pop('bank70_correction', None)
    if [x['name'] for x in restored['results']] != [x['name'] for x in original['results']]:
        raise ValueError('Model inventory changed')
    for before, after in zip(original['results'], restored['results']):
        for group in GROUPS:
            if group not in after['groups']:
                raise ValueError('Missing corrected bank group')
            after['groups'][group] = copy.deepcopy(before['groups'][group])
    if restored != original:
        raise ValueError('Correction changed content outside the two bank groups')


def write_new(path, data):
    """Exclusive publication; never replace an already committed artifact."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def ensure_stopped(root):
    state = read(root/'status.json')
    if state.get('status') != 'completed' or state.get('child_pid') is not None:
        raise ValueError('Complete, stopped pipeline required before report correction')
    live = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info.get('cmdline') or []
        if '-m' in command and any(x.startswith('gnn_features.') for x in command):
            live.append(process.info['pid'])
    if live:
        raise RuntimeError('Active gnn module processes: '+repr(live))
    return state


def correction_markdown(report, original_sha, correction_sha):
    text = _markdown(report).replace('# GNN 第二轮三组对照结果', '# GNN 第二轮三组对照结果（银行70分层更正版）', 1)
    text = text.replace('comparison.json', 'comparison.corrected.json')
    lines = [text, '', '## 银行70分层更正与审计', '',
             '原报告将银行身份编码70误当作原始银行编号“70”。本更正版使用已冻结词典解析原始编号，正确编码为122017。',
             '仅更正付款银行70内外、收款银行70内外两个分层；模型、预测、整体及非ACH指标、全局阈值、选优和顺序对照均未改变。',
             '原comparison.json、best_model.json和调参结果.md保持字节不变；best_model.json继续绑定原comparison的SHA。',
             '完整更正统计见[comparison.corrected.json](comparison.corrected.json)，更正记录见[bank70_correction.json](bank70_correction.json)。',
             '原comparison SHA256：`'+original_sha+'`。更正记录 SHA256：`'+correction_sha+'`。', '',
             '| 模型 | 方向 | 分组 | 交易数 | 正例 | TP | FP | FN | 精确率 | 召回率 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for result in report['results']:
        for name in GROUPS:
            for category, item in result['groups'][name].items():
                p = item['operating_point']
                fields = ['—']*5 if p is None else [str(p['tp']), str(p['fp']), str(p['fn']),
                          '—' if p['precision'] is None else f"{p['precision']:.4%}",
                          '—' if p['recall'] is None else f"{p['recall']:.4%}"]
                lines.append('| '+' | '.join([result['name'], name, category, str(item['samples']), str(item['positives'])]+fields)+' |')
    lines += ['', '所有分组沿用各模型原有的单一全局阈值，没有按银行另选阈值。Validation为开发结果，Test尚未评估。']
    return '\n'.join(lines)+'\n'


def run(project=ROOT):
    project = Path(project).resolve()
    root = project/'result/gnn_tuning_v2'
    audit = root/AUDIT
    destinations = [root/'bank70_correction.json', root/'comparison.corrected.json', root/'调参结果_银行分层更正.md', audit/'verification.json']
    _disk(project)
    with ProcessLock(root), ProcessLock(root/'comparison_lock'):
        state = ensure_stopped(root)
        if any(p.exists() or p.is_symlink() for p in destinations):
            raise FileExistsError('Correction outputs already exist; preserve and inspect before replay')
        if not all(p.resolve().is_relative_to(root.resolve()) for p in destinations):
            raise ValueError('Correction destination escaped result root')
        frozen_config = read(root/'config.json')
        if source_hashes(project) != frozen_config['source_sha256'] or verify_freeze(project) != frozen_config['frozen_v1_anchors']:
            raise ValueError('Frozen pipeline/source/input identity differs')
        tests = read(audit/'tests.json')
        if (tests.get('status') != 'passed' or tests.get('script_sha256') != sha(Path(__file__))
                or tests.get('test_sha256') != sha(project/'tests/test_gnn_v2_bank70_correction.py')):
            raise ValueError('Passing tests bound to this exact correction required')
        originals = {name:sha(root/name) for name in ['comparison.json', 'best_model.json', '调参结果.md', 'config.json', 'engineering_tests.json']}
        original, selected = read(root/'comparison.json'), read(root/'best_model.json')
        if original.get('status') != 'completed' or selected['comparison_sha256'] != originals['comparison.json']:
            raise ValueError('Original comparison/selected-pointer chain differs')
        if original['code_sha256'] != {name:sha(project/'scripts/gnn_features'/name) for name in original['code_sha256']}:
            raise ValueError('Original report source identity differs')
        winner, reason = choose_best(original['results'])
        expected_selected = dict(name=winner['name'], path=winner['model_path'], sha256=winner['model_sha256'],
            validation_threshold=winner['precision80']['threshold'], selection_reason=reason,
            comparison_sha256=originals['comparison.json'], test_evaluated=False)
        if selected != expected_selected or original['best'] != {k:winner[k] for k in original['best']}:
            raise ValueError('Original selected model or criteria differ')
        if (root/'调参结果.md').read_text(encoding='utf-8') != _markdown(original):
            raise ValueError('Original Markdown does not match frozen report renderer')
        cache = project/'data/gnn_validation_cache_parquet'
        cache_config = read(cache/'config.json')
        identities = {}
        archive = project/'data/gnn_v1_validation'
        paths = [archive/'build.json', archive/'metadata/bank_identity.json']
        paths += [archive/'history_index'/f'{k}.npy' for k in ['timestamp', 'src', 'dst', 'bank_key']]
        for path in paths:
            relative = str(path.relative_to(project))
            digest = sha(path)
            if cache_config['hashes'].get(relative) != digest:
                raise ValueError('Frozen correction input changed: '+relative)
            identities[relative] = digest
        build = read(archive/'build.json')
        if build['rows'] != PREFIX+TOTAL or build['end_exclusive'] != '2022-09-14T00:00:00':
            raise ValueError('Validation archive scope differs')
        arrays = {k:np.load(archive/'history_index'/f'{k}.npy', mmap_mode='r') for k in ['timestamp', 'src', 'dst', 'bank_key']}
        times, src, dst = (arrays[k][PREFIX:] for k in ['timestamp', 'src', 'dst'])
        code, groups = resolve_bank70(read(archive/'metadata/bank_identity.json'), arrays['bank_key'], src, dst)
        if code != 122017:
            raise ValueError('Audited raw bank identity changed unexpectedly')
        mistaken = dict(payer_bank70=np.where(arrays['bank_key'][src] == 70, 'bank70', 'other'),
                        receiver_bank70=np.where(arrays['bank_key'][dst] == 70, 'bank70', 'other'))
        corrected = copy.deepcopy(original)
        inventory, replacements, labels = {}, {}, None
        if [x['name'] for x in original['results']] != list(NAMES):
            raise ValueError('Original four-model order differs')
        for item in corrected['results']:
            name = item['name']
            print(json.dumps(dict(stage='verify_model_predictions_and_bank_groups', model=name)), flush=True)
            trained = verified_training(project, name)
            del trained['model']
            y, scores, verified = load_predictions(project, name, times, labels)
            labels = y
            if len(y) != TOTAL or int(y.sum()) != POSITIVES:
                raise ValueError('Full Validation population differs')
            point = precision80_point(y, scores)
            if point != item['precision80'] or verified['metrics'] != item['metrics']:
                raise ValueError('Unchanged overall metrics/global threshold differ')
            expected_fields = dict(model_sha256=trained['model_sha256'], training_config_sha256=trained['config_sha256'],
                training_result_sha256=trained['training_result_sha256'], evaluation_config_sha256=verified['config_sha256'],
                metrics_sha256=verified['metrics_sha256'])
            if any(item[k] != value for k,value in expected_fields.items()):
                raise ValueError('Original report model/evaluation identity differs')
            old_groups = group_report(y, scores, point, mistaken, src, dst)
            if old_groups != {k:item['groups'][k] for k in GROUPS}:
                raise ValueError('Original bank groups do not reproduce known coding defect')
            new_groups = group_report(y, scores, point, groups, src, dst)
            for group in GROUPS:
                if (sum(x['samples'] for x in new_groups[group].values()) != TOTAL
                        or sum(x['positives'] for x in new_groups[group].values()) != POSITIVES):
                    raise ValueError('Corrected bank partition coverage differs')
            if (new_groups['payer_bank70']['bank70']['samples'] != 477760
                    or new_groups['payer_bank70']['bank70']['positives'] != 662):
                raise ValueError('Raw bank70 does not match independently checked history diagnosis')
            item['groups'].update(new_groups)
            replacements[name] = dict(previous=old_groups, corrected=new_groups)
            inventory[name] = {**expected_fields, 'prediction_blocks_verified':len(verified['blocks']),
                               'prediction_blocks':verified['blocks'], 'model_strictly_loadable':True}
        assert_only_bank_groups_changed(original, corrected)
        audit.mkdir(parents=True, exist_ok=True)
        snapshot = audit/'originals'
        snapshot.mkdir(exist_ok=False)
        for name in originals:
            write_new(snapshot/name, (root/name).read_bytes())
        write_new(audit/'correction_script.py', Path(__file__).read_bytes())
        write_new(audit/'regression_tests.py', (project/'tests/test_gnn_v2_bank70_correction.py').read_bytes())
        correction = dict(status='completed', created_at=datetime.now(timezone.utc).isoformat(),
            correction_type='additive_bank_identity_reporting_correction', raw_bank='70', correct_encoded_bank_key=code,
            previous_erroneous_literal_key=70, original_files_sha256=originals,
            input_sha256=identities, models=inventory, bank_group_replacements=replacements,
            correction_script_sha256=sha(Path(__file__)), tests_sha256=sha(audit/'tests.json'),
            frozen_source_sha256=frozen_config['source_sha256'], original_pipeline_completed_at=state['updated_at'],
            preserved=['All frozen code/config/model/prediction files', 'Original comparison, best_model and Markdown bytes',
                       'All report semantics except payer_bank70/receiver_bank70 groups'],
            scope='Full Validation reporting correction only; no inference, retraining, threshold changes or Test access')
        write_new(root/'bank70_correction.json', json_bytes(correction))
        corrected['bank70_correction'] = dict(path='bank70_correction.json', sha256=sha(root/'bank70_correction.json'),
            original_comparison_sha256=originals['comparison.json'], raw_bank='70', encoded_bank_key=code,
            changed_groups=list(GROUPS), canonical_originals_preserved=True)
        assert_only_bank_groups_changed(original, corrected)
        write_new(root/'comparison.corrected.json', json_bytes(corrected))
        markdown = correction_markdown(corrected, originals['comparison.json'], sha(root/'bank70_correction.json'))
        write_new(root/'调参结果_银行分层更正.md', markdown.encode('utf-8'))
        actual = {name:sha(root/name) for name in originals}
        if actual != originals or source_hashes(project) != frozen_config['source_sha256']:
            raise ValueError('Frozen original report/config/source changed during correction')
        assert_only_bank_groups_changed(original, read(root/'comparison.corrected.json'))
        verification = dict(status='passed', original_files_preserved=True, original_files_sha256=actual,
            corrected_only_two_bank_groups=True, four_models_strictly_loaded=True,
            full_prediction_blocks_verified=197*4, samples_per_model=TOTAL, positives_per_model=POSITIVES,
            raw_bank70_code=code, root_and_comparison_locks_held=True, no_training_or_inference=True,
            output_sha256={p.name:sha(p) for p in destinations[:-1]}, tests=tests,
            completed_at=datetime.now(timezone.utc).isoformat())
        write_new(audit/'verification.json', json_bytes(verification))
        print(json.dumps(verification, ensure_ascii=True), flush=True)
        return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=ROOT)
    run(parser.parse_args().project)
