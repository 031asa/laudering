"""Read-only, matched positive/negative diagnostics on the verified Train feature snapshot.

No fitting, threshold deployment, Validation/Test access, or frozen input writes.
Exact matching: date + payer + payment type + both transaction currencies.
Numerical missing values are excluded from the value test and masks tested
separately. Every negative in a positive-containing stratum is retained.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'result/gnn_non_ach_matched_differences_20261004/input_snapshot'
OUT = ROOT / 'result/gnn_non_ach_matched_differences_20261004'
SEED = 20260920


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Disk reserve below 20 GiB')
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')


def bh(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(q, 1.)
    return result


def conditional_rank(positive, negative):
    """For each positive, P(x_positive > x_negative) + 0.5 P(tie)."""
    negative = np.sort(np.asarray(negative))
    if not len(negative):
        raise ValueError('No controls')
    return (np.searchsorted(negative, positive, side='left') +
            np.searchsorted(negative, positive, side='right')) / (2 * len(negative))


def interval(num, den, rng_counts):
    total = float(den.sum())
    if not total:
        return None
    draws_den = rng_counts @ den
    keep = draws_den > 0
    draws = (rng_counts[keep] @ num) / draws_den[keep]
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def analyze(values, valid, groups, signs, draws, positive_rows, want_details=False):
    # Rows are date (1..10), columns are payer (15). AUC counts ties as half.
    num = np.zeros((10, 15)); den = np.zeros((10, 15))
    neg_total = np.zeros((10, 15), dtype=np.int64)
    pos_mean = np.zeros((10, 15)); neg_mean = np.zeros((10, 15))
    details = []
    for d, account, pos, neg in groups:
        p = pos[valid[pos]]; n = neg[valid[neg]]
        if not len(p) or not len(n):
            continue
        px = values[p]; nx = values[n]
        ranks = conditional_rank(px, nx)
        num[d-1, account] += ranks.sum()
        den[d-1, account] += len(p)
        neg_total[d-1, account] += len(n)
        pos_mean[d-1, account] += float(px.sum(dtype=np.float64))
        neg_mean[d-1, account] += len(p) * float(nx.mean(dtype=np.float64))
        if want_details:
            details.append(dict(day=d, payer_index=account, positives=len(p), negatives=len(n),
                                rank_auc=float(ranks.mean()), positive_mean=float(px.mean()),
                                matched_negative_mean=float(nx.mean())))
    periods = {}
    for name, sl in [('discovery_sep01_07', slice(0,7)), ('sep08', slice(7,8)),
                     ('later_sep09_10', slice(8,10)), ('all_train', slice(0,10))]:
        a = num[sl].sum(axis=0); b = den[sl].sum(axis=0)
        z = a - .5*b
        # Exact cluster sign flip: assumes independent payer clusters with
        # symmetric cluster effects under the null. Not a causal test.
        p = float(np.mean(np.abs(signs @ z) >= abs(z.sum()) - 1e-12)) if b.sum() else 1.
        periods[name] = dict(matched_positives=int(b.sum()),
            negatives=int(neg_total[sl].sum()), payer_clusters=int((b>0).sum()),
            conditional_auc=float(a.sum()/b.sum()) if b.sum() else None,
            cluster_bootstrap_ci95=interval(a,b,draws), cluster_signflip_p=p,
            positive_mean_encoded=float(pos_mean[sl].sum()/b.sum()) if b.sum() else None,
            matched_negative_mean_encoded=float(neg_mean[sl].sum()/b.sum()) if b.sum() else None,
            by_payer=[dict(payer_index=i, positives=int(b[i]), auc=float(a[i]/b[i]))
                      for i in range(15) if b[i]])
    daily = [dict(day=i+1, positives=int(den[i].sum()),
                  auc=float(num[i].sum()/den[i].sum()) if den[i].sum() else None)
             for i in range(10)]
    out = dict(periods=periods, by_day=daily,
               valid_positive_rows=int(valid[positive_rows].sum()))
    if want_details:
        out['strata'] = details
    return out


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise RuntimeError('Completed diagnostic exists; preserve it')
    manifest = json.loads((SRC/'dataset_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'verified'
    for rel, expected in manifest['outputs'].items():
        assert sha(SRC/rel) == expected, rel
    print('Verified five derived input hashes', flush=True)
    x = np.load(SRC/'features.npy', mmap_mode='r')
    y = np.load(SRC/'labels.npy')
    day = np.load(SRC/'days.npy')
    payer = np.load(SRC/'payer_ids_for_audit_only.npy')
    ids = np.load(SRC/'edge_ids.npy')
    names = manifest['identity']['feature_names']
    assert x.shape == (1801617,309) and int(y.sum()) == 2551
    assert np.all(np.diff(ids) > 0) and np.isin(day,np.arange(1,11)).all()
    payers, pi = np.unique(payer, return_inverse=True)
    assert len(payers) == 15
    keys = np.column_stack([day, pi, x[:,8:11].astype(np.int64)])
    strata, gi = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(gi, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(gi[order]))+1, len(y)]
    groups=[]; missing_positive=0; matched_negative=0
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        idx=order[lo:hi]; key=strata[gi[idx[0]]]
        pos=idx[y[idx]==1]; neg=idx[y[idx]==0]
        if len(pos) and len(neg):
            groups.append((int(key[0]),int(key[1]),pos,neg))
            matched_negative += len(neg)
        elif len(pos):
            missing_positive += len(pos)
    pr=np.flatnonzero(y==1)
    signs=(2*((np.arange(2**15)[:,None] >> np.arange(15)) & 1)-1).astype(float)
    draws=np.random.default_rng(SEED).multinomial(15,np.full(15,1/15),size=2000)
    coverage = dict(rows=len(y), positives=int(y.sum()), negatives=int((y==0).sum()),
        payer_accounts=len(payers), total_strata=len(strata), positive_strata=len(groups),
        positive_rows_without_controls=missing_positive, matched_negative_rows=matched_negative,
        matching=['date','payer','payment_format','payment_currency','receiving_currency'],
        daily=[dict(day=int(d),rows=int((day==d).sum()),positives=int(y[day==d].sum()))
               for d in range(1,11)])
    write('design.json',dict(coverage=coverage,source_manifest_sha256=sha(SRC/'dataset_manifest.json'),
        source_outputs=manifest['outputs'],script_sha256=sha(__file__),
        scope='Train bank70 payer non-ACH only',
        periods='Sep1-7 discover direction; Sep8 intermediate; Sep9-10 later-date check; all are development data',
        controls='All negatives in the exact positive-containing stratum; no negative sampling',
        value_missing='Only compare valid numerical values; test all validity masks separately',
        statistic='Positive-weighted within-stratum Mann-Whitney conditional AUC; ties half',
        uncertainty='2000 bootstrap resamples of 15 payer clusters; sign-flip assumes cluster independence and symmetry; no causal interpretation',
        multiplicity='BH over all tested numerical/mask/coverage columns and specified binary predicates, separately per period',
        selection='At least 1000 eligible discovery positives, discovery effect >=0.03, later effect same direction >=0.03, later BH q<0.05',
        validation_test_read=False,history_labels_as_features=False))
    results=[]
    indices=[i for i in range(len(names)) if i not in range(8,14)]
    lookup={name:i for i,name in enumerate(names)}
    # Read manageable column blocks, rather than materializing the 2.2 GB matrix.
    for begin in range(0,len(indices),12):
        cols=indices[begin:begin+12]
        block=np.asarray(x[:,cols],order='F')
        for jj,j in enumerate(cols):
            name=names[j]
            mask_name=(name.replace('__encoded','')+'__valid') if name.endswith('__encoded') else name+'__valid'
            valid=np.asarray(x[:,lookup[mask_name]] == 1) if mask_name in lookup else np.ones(len(y),bool)
            result=analyze(block[:,jj],valid,groups,signs,draws,pr)
            result.update(name=name,index=j,kind='column',valid_mask=mask_name if mask_name in lookup else None)
            results.append(result)
        print(json.dumps(dict(columns=min(begin+12,len(indices)),total=len(indices),seconds=round(time.monotonic()-started)),ensure_ascii=True),flush=True)
        write('status.json',dict(status='running',pid=os.getpid(),columns=min(begin+12,len(indices)),seconds=time.monotonic()-started))

    predicates=[]
    def predicate(name, column, op, value):
        j=lookup[column]; v=np.asarray(x[:,j])
        mask_name=column.replace('__encoded','')+'__valid' if column.endswith('__encoded') else column+'__valid'
        valid=np.asarray(x[:,lookup[mask_name]]==1) if mask_name in lookup else np.ones(len(y),bool)
        condition=(v==value) if op=='eq' else (v>value) if op=='gt' else (v<=value)
        result=analyze(condition.astype(float),valid,groups,signs,draws,pr,True)
        result.update(name=name,kind='predicate',source_column=column,operator=op,encoded_value=value)
        results.append(result); predicates.append(name)
    predicate('no_forward_pair_72h','r_pair_fwd_count_72h__encoded','eq',0.)
    predicate('has_reverse_pair_24h','r_pair_rev_count_24h__encoded','gt',0.)
    predicate('receiver_no_external_out_24h','dst_n_out_ext_count_24h__encoded','eq',0.)
    predicate('receiver_no_external_out_72h','dst_n_out_ext_count_72h__encoded','eq',0.)
    predicate('receiver_no_previous_in_24h','r_dst_ext_in_count_24h__encoded','eq',0.)
    predicate('receiver_at_least5_in_24h','r_dst_ext_in_count_24h__encoded','gt',float(np.log1p(4.5)))
    predicate('receiver_no_same_payment_24h','r_dst_current_payment_share_24h__encoded','eq',0.)
    predicate('payer_no_same_currency_in_72h','r_src_ext_in_count_72h__encoded','eq',0.)
    for role in ('src','dst'):
        for window in ('24h','72h','7d'):
            predicate(f'{role}_amount_above_p90_{window}',f'extra_v2_{role}_current_to_q90_{window}','gt',float(np.log(2.)))
    for period in results[0]['periods']:
        q=bh([r['periods'][period]['cluster_signflip_p'] for r in results])
        for r,value in zip(results,q):
            r['periods'][period]['bh_q']=float(value)
    eligible=[]
    for r in results:
        fit=r['periods']['discovery_sep01_07']; late=r['periods']['later_sep09_10']
        a=fit['conditional_auc']; b=late['conditional_auc']
        direction=1 if a is not None and a>=.5 else -1
        r['discovery_direction']='higher' if direction==1 else 'lower'
        r['later_auc_in_discovery_direction']=(.5+direction*(b-.5)) if b is not None else None
        r['repeated_direction']=(a is not None and b is not None and (a-.5)*(b-.5)>0)
        r['passes_screen']=(fit['matched_positives']>=1000 and a is not None and b is not None
            and abs(a-.5)>=.03 and direction*(b-.5)>=.03 and late['bh_q']<.05)
        if r['passes_screen']: eligible.append(r['name'])
    ranked=sorted([r for r in results if r['periods']['discovery_sep01_07']['matched_positives']>=1000],
                  key=lambda r:abs(r['periods']['discovery_sep01_07']['conditional_auc']-.5),reverse=True)
    write('summary.json',dict(coverage=coverage,screen_pass=eligible,discovery_ranking=[r['name'] for r in ranked],
        results=results,seconds=time.monotonic()-started,validation_test_read=False,
        note='Conditional AUC is matched discrimination, not population precision/AP; bank code not treated as ordinal feature'))
    write('status.json',dict(status='completed',pid=os.getpid(),seconds=time.monotonic()-started,features=len(results)))
    print(json.dumps(dict(completed=True,screen_pass=eligible,seconds=time.monotonic()-started),ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
