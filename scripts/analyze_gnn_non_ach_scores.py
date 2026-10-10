"""Read-only diagnosis of frozen lr_only Validation scores; no model changes."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/gnn_non_ach_score_diagnosis_20261003'
PREFIX, TOTAL, POS = 20318992, 4689006, 6420


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def quantiles(values):
    levels = [0, .01, .1, .25, .5, .75, .9, .99, .999, 1]
    return {str(q): float(v) for q, v in zip(levels, np.quantile(values, levels))} if len(values) else {}


def curve(y, scores):
    order = np.argsort(-scores, kind='stable')
    s, yy = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    tp = np.cumsum(yy, dtype=np.int64)[ends]
    alerts = ends + 1
    fp = alerts - tp
    return s[ends], tp, fp, alerts


def point(threshold, tp, fp, total, positives):
    tp, fp = int(tp), int(fp)
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=positives-tp,
                alerts=tp+fp, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/positives if positives else None,
                fpr=fp/(total-positives) if total>positives else None)


def describe(y, scores):
    n, p = len(y), int(y.sum())
    thresholds, tp, fp, alerts = curve(y, scores)
    precision = tp / alerts
    prev_tp, prev_fp = np.r_[0, tp[:-1]], np.r_[0, fp[:-1]]
    ap = float(np.sum((tp-prev_tp) * precision) / p) if p else None
    auc = float(np.sum((fp-prev_fp) * (tp+prev_tp) / 2) / (p*(n-p))) if 0<p<n else None
    valid = np.flatnonzero((5*tp >= 4*alerts) & (tp>0))
    best = None
    if len(valid):
        max_tp = tp[valid].max()
        tied = valid[tp[valid] == max_tp]
        i = int(tied[np.argmax(precision[tied])])
        best = point(thresholds[i], tp[i], fp[i], n, p)
    targets = {}
    if p:
        for recall in [.2, .5, .8, 1.]:
            required = int(np.ceil(recall*p))
            i = int(np.searchsorted(tp, required))
            targets[str(recall)] = point(thresholds[i], tp[i], fp[i], n, p)
    budgets = {}
    for budget in [100, 715, 1000, 5000, 10000, 50000]:
        allowed = np.flatnonzero(alerts <= budget)
        if len(allowed):
            i = int(allowed[-1])
            budgets[str(budget)] = point(thresholds[i], tp[i], fp[i], n, p)
        else:
            budgets[str(budget)] = None
    neg = np.sort(scores[y==0])
    pos = scores[y==1]
    fraction_below = np.searchsorted(neg, pos, side='left') / len(neg) if len(neg) else np.array([])
    return dict(samples=n, positives=p, negatives=n-p, prevalence=p/n,
                average_precision=ap, ap_over_prevalence=ap/(p/n) if p else None,
                roc_auc=auc, positive_scores=quantiles(pos), negative_scores=quantiles(neg),
                positive_fraction_of_negatives_strictly_below=quantiles(fraction_below),
                best_precision80=best, recall_operating_points=targets,
                alert_budget_points=budgets)


def main():
    began = time.monotonic()
    if (OUT/'summary.json').exists():
        raise FileExistsError('Existing diagnosis must be preserved')
    folder = ROOT/'result/gnn_tuning_v1/evaluations/lr_only'
    config, metrics = read(folder/'config.json'), read(folder/'metrics.json')
    config_sha = sha(folder/'config.json')
    assert metrics['status']=='completed' and metrics['config_sha256']==config_sha
    assert sha(ROOT/config['model_path'])==config['model_sha256']==metrics['model_sha256']
    cache = ROOT/'data/gnn_validation_cache_parquet'
    assert sha(cache/'config.json')==config['cache_config_sha256']
    assert sha(cache/'manifest.json')==config['cache_manifest_sha256']
    frozen = read(cache/'config.json')['hashes']
    archive = ROOT/'data/gnn_v1_validation'
    files = [archive/'build.json', archive/'metadata/vocabulary.json',
             archive/'metadata/bank_identity.json']
    files += [archive/'history_index'/f'{key}.npy' for key in ['timestamp','payment_format_cat','src','bank_key']]
    identities = {}
    for path in files:
        digest = sha(path)
        assert digest==frozen[str(path.relative_to(ROOT))]
        identities[str(path.relative_to(ROOT))] = digest
    build = read(archive/'build.json')
    assert build['rows']==PREFIX+TOTAL and build['end_exclusive']=='2022-09-14T00:00:00'
    vocab = read(archive/'metadata/vocabulary.json')['values']['payment_format']
    arr = {k:np.load(archive/'history_index'/f'{k}.npy',mmap_mode='r')
           for k in ['timestamp','payment_format_cat','src','bank_key']}
    y = np.full(TOTAL,-1,dtype=np.int8)
    scores = np.full(TOTAL,np.nan,dtype=np.float32)
    seen = np.zeros(TOTAL,bool)
    for entry in metrics['blocks']:
        path = folder/'predictions'/entry['block']
        manifest = read(path/'manifest.json')
        assert sha(path/'manifest.json')==entry['manifest_sha256']
        assert manifest['config_sha256']==config_sha
        assert sha(path/'predictions.parquet')==manifest['file_sha256']
        tab = pq.ParquetFile(path/'predictions.parquet').read(use_threads=False)
        ids = tab['edge_id'].to_numpy(); positions=ids-PREFIX
        assert len(ids)==entry['targets']==manifest['targets']
        assert (positions>=0).all() and (positions<TOTAL).all()
        assert len(np.unique(positions))==len(positions) and not seen[positions].any()
        assert hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()==manifest['target_ids_sha256']
        assert np.array_equal(tab['cutoff_us'].to_numpy(),arr['timestamp'][ids])
        y[positions]=tab['label'].to_numpy();scores[positions]=tab['probability'].to_numpy();seen[positions]=True
    assert seen.all() and np.isfinite(scores).all() and np.isin(y,[0,1]).all() and y.sum()==POS
    assert (scores>=0).all() and (scores<=1).all() and len(metrics['blocks'])==197
    fmt=np.asarray(arr['payment_format_cat'][PREFIX:])
    source=np.asarray(arr['src'][PREFIX:])
    bank=np.asarray(arr['bank_key'][source])
    bank_identity=read(archive/'metadata/bank_identity.json')
    bank70_key=bank_identity.index('70')
    ach_code=vocab.index('ACH')+2
    non=fmt!=ach_code
    assert int(y[non].sum())==715
    assert int(y[non&(bank==bank70_key)].sum())==662
    groups={'all':np.ones(TOTAL,bool),'ACH':~non,'non_ACH':non,
            'non_ACH_sender_bank70':non&(bank==bank70_key),
            'non_ACH_other_sender_banks':non&(bank!=bank70_key)}
    for code in np.unique(fmt):
        name=vocab[int(code)-2]
        if name!='ACH':groups['payment_'+name]=fmt==code
    results={name:describe(y[mask],scores[mask]) for name,mask in groups.items() if mask.any()}
    assert abs(results['all']['average_precision']-metrics['metrics']['average_precision'])<1e-12
    assert abs(results['all']['roc_auc']-metrics['metrics']['roc_auc'])<1e-12
    non20=results['non_ACH']['recall_operating_points']['0.2']['threshold']
    fixed={}
    for name,th in [('old_best_f1',metrics['metrics']['best_f1']['threshold']),
                    ('old_global_P80',.945133626461029),('threshold_0_5',.5),
                    ('non_ACH_20pct_recall_diagnostic_only',non20)]:
        fixed[name]={}
        for group,mask in [('all',np.ones(TOTAL,bool)),('ACH',~non),('non_ACH',non)]:
            hit=mask&(scores>=th)
            fixed[name][group]=point(th,np.sum(y[hit]),int(hit.sum()-y[hit].sum()),int(mask.sum()),int(y[mask].sum()))
    neg=np.sort(scores[non&(y==0)])
    positions=np.flatnonzero(non&(y==1))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'non_ach_positive_score_ranks.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['edge_id','payment_format','sender_bank','sender_bank_key','score','non_ach_negative_scores_ge','fraction_non_ach_negatives_strictly_below'])
        writer.writeheader()
        for pos in positions:
            below=int(np.searchsorted(neg,scores[pos],side='left'))
            writer.writerow(dict(edge_id=int(pos+PREFIX),payment_format=vocab[int(fmt[pos])-2],sender_bank=bank_identity[int(bank[pos])],sender_bank_key=int(bank[pos]),score=float(scores[pos]),non_ach_negative_scores_ge=len(neg)-below,fraction_non_ach_negatives_strictly_below=below/len(neg)))
    report=dict(status='completed',created_at=datetime.now(timezone.utc).isoformat(),
                scope='Frozen lr_only full Validation retrospective diagnosis only; no retraining, threshold deployment, new candidate, or Test access.',
                source=dict(model_sha256=config['model_sha256'],config_sha256=config_sha,metrics_sha256=sha(folder/'metrics.json'),prediction_blocks_verified=197,input_hashes=identities,script_sha256=sha(Path(__file__))),
                groups=results,fixed_thresholds=fixed,
                bank70_identity_key=bank70_key,
                non_ach_positive_distinct_senders=int(np.unique(source[positions]).size),
                limitations=['Group-specific operating points are diagnostic only, not deployed thresholds.','Validation has been used for development; this is not independent Test evidence.','This score analysis cannot establish which feature or training choice caused the failure.'],
                seconds=time.monotonic()-began)
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'groups':results,'fixed_thresholds':fixed,'seconds':report['seconds']},ensure_ascii=True))


if __name__=='__main__':
    main()
