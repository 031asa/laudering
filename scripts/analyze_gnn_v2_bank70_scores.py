"""Read-only four-model subgroup diagnosis over verified Validation predictions."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pyarrow as pa
import torch

from analyze_gnn_non_ach_scores import describe, curve, point, read, sha
from gnn_features.v2_compare import load_predictions, NAMES
from gnn_features.v2_evaluate import confusion, precision80_point
from gnn_features.training_state import ProcessLock, atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/gnn_bank70_v2_diagnosis_20261003'
PREFIX, TOTAL = 20318992, 4689006


def at_fp_budgets(y, scores):
    thresholds, tp, fp, alerts = curve(y, scores)
    result = {}
    for budget in (100, 1000, 5000):
        eligible = np.flatnonzero(fp <= budget)
        if not len(eligible) or int(tp[eligible].max()) == 0:
            result[str(budget)] = confusion(y, np.zeros(len(y), bool))
        else:
            # First maximum TP minimizes FP and then prefers higher threshold.
            index = int(eligible[np.argmax(tp[eligible])])
            result[str(budget)] = point(thresholds[index], tp[index], fp[index], len(y), int(y.sum()))
    return result


def main():
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    result_root = ROOT/'result/gnn_tuning_v2'
    assert read(result_root/'status.json')['status'] == 'completed'
    comparison = read(result_root/'comparison.json')
    comparison_sha = sha(result_root/'comparison.json')
    rows = {x['name']: x for x in comparison['results']}
    archive = ROOT/'data/gnn_v1_validation'
    cache = ROOT/'data/gnn_validation_cache_parquet'
    frozen = read(cache/'config.json')['hashes']
    relative = ['build.json', 'metadata/bank_identity.json', 'metadata/vocabulary.json']
    relative += ['history_index/'+name+'.npy' for name in ('timestamp', 'src', 'bank_key', 'payment_format_cat')]
    input_hashes = {}
    for name in relative:
        p = archive/name
        digest = sha(p)
        assert digest == frozen[str(p.relative_to(ROOT))]
        input_hashes[str(p.relative_to(ROOT))] = digest
    build = read(archive/'build.json')
    assert build['rows'] == PREFIX+TOTAL and build['end_exclusive'] == '2022-09-14T00:00:00'
    arrays = {n: np.load(archive/'history_index'/f'{n}.npy', mmap_mode='r')
              for n in ('timestamp', 'src', 'bank_key', 'payment_format_cat')}
    times, src = arrays['timestamp'][PREFIX:], arrays['src'][PREFIX:]
    fmt = arrays['payment_format_cat'][PREFIX:]
    methods = read(archive/'metadata/vocabulary.json')['values']['payment_format']
    bank70 = read(archive/'metadata/bank_identity.json').index('70')
    non = fmt != methods.index('ACH')+2
    g = non & (arrays['bank_key'][src] == bank70)
    assert int(g.sum()) == 477760
    with ProcessLock(OUT):
        output = OUT/'summary.json'
        if output.exists():
            raise FileExistsError('Preserve existing diagnosis')
        reference_y, models = None, {}
        for name in NAMES:
            y, scores, verified = load_predictions(ROOT, name, times, reference_y)
            if reference_y is None:
                reference_y = y
            assert int(y[g].sum()) == 662 and int(y[non].sum()) == 715
            assert verified['model_sha256'] == rows[name]['model_sha256'] == sha(ROOT/rows[name]['model_path'])
            assert verified['metrics_sha256'] == rows[name]['metrics_sha256']
            global_point = precision80_point(y, scores)
            assert global_point == rows[name]['precision80']
            local_y, local_scores = y[g], scores[g]
            groups = {'G': describe(local_y, local_scores), 'non_ACH': describe(y[non], scores[non])}
            local_payment, local_src, local_times = fmt[g], src[g], times[g]
            dates = local_times.astype('datetime64[us]').astype('datetime64[D]').astype(str)
            strata = {}
            for kind, values in [('payment', local_payment), ('payer_node_id', local_src), ('date', dates)]:
                strata[kind] = {}
                for value in np.unique(values):
                    mask = values == value
                    label = methods[int(value)-2] if kind == 'payment' else str(value)
                    details = describe(local_y[mask], local_scores[mask])
                    strata[kind][label] = {k: details[k] for k in ('samples', 'positives', 'negatives', 'average_precision', 'roc_auc')}
                    strata[kind][label]['global_operating_point'] = confusion(local_y[mask], local_scores[mask] >= global_point['threshold'], global_point['threshold'])
            models[name] = dict(model_sha256=verified['model_sha256'], metrics_sha256=verified['metrics_sha256'],
                                config_sha256=verified['config_sha256'], verified_prediction_blocks=verified['blocks'],
                                global_threshold=global_point['threshold'], groups=groups, G_strata=strata,
                                G_fixed_fp_budgets=at_fp_budgets(local_y, local_scores),
                                G_global_operating_point=confusion(local_y, local_scores >= global_point['threshold'], global_point['threshold']))
            if name == 'lr_only':
                old = read(ROOT/'result/gnn_non_ach_score_diagnosis_20261003/summary.json')['groups']['non_ACH_sender_bank70']
                for field in ('samples', 'positives', 'average_precision', 'roc_auc'):
                    assert groups['G'][field] == old[field]
            print(json.dumps({'verified': name, 'G_AP': groups['G']['average_precision'], 'G_AUC': groups['G']['roc_auc'],
                              'G_recall20': groups['G']['recall_operating_points']['0.2']}, ensure_ascii=True), flush=True)
        assert sha(result_root/'comparison.json') == comparison_sha
        report = dict(status='completed', created_at=datetime.now(timezone.utc).isoformat(),
                      scope='Full Validation retrospective diagnosis; no training or deployed subgroup thresholds; Test untouched.',
                      definition='G = raw payer bank 70 and payment format != ACH', raw_bank70_identity_key=bank70,
                      comparison_sha256=comparison_sha, script_sha256=sha(Path(__file__)), input_hashes=input_hashes,
                      models=models, limitations=['Existing Validation is a development set.',
                        'Group recall20 requires >=133 of 662; non-ACH recall20 requires >=143 of 715.',
                        'Fixed FP budgets and subgroup thresholds are diagnostic only; original global selection remains unchanged.',
                        'Score differences alone do not prove a feature-level cause.'])
        atomic_json(output, report)


if __name__ == '__main__':
    main()
