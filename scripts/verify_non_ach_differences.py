"""Independent pooled-rank cross-check and rare/tied-feature sensitivity.

Cluster sign-flip symmetry can fail for nearly constant flags. Require also
the conditional rank null-variance test, taking the larger p before BH. This
is a sensitivity screen, not evidence of independent transactions/causality.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
from analyze_non_ach_differences import ROOT, SRC, OUT, bh, sha, write


def pooled_rank_moments(p, n):
    """Rank sum and finite-population null variance, with exact score ties."""
    values=np.r_[p,n]
    _,inverse,count=np.unique(values,return_inverse=True,return_counts=True)
    mean_rank=np.cumsum(count)-(count-1)/2
    P=len(p); N=len(n); total=P+N
    u=float(mean_rank[inverse[:P]].sum()-P*(P+1)/2)
    ss=float(np.sum(count*(mean_rank-(total+1)/2)**2))
    variance=P*N/(total*(total-1))*ss/(N*N)
    return u/N,variance


def main():
    if (OUT/'verification.json').exists():
        raise RuntimeError('Preserve completed verification')
    d=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    meta=json.loads((SRC/'dataset_manifest.json').read_text(encoding='utf-8'))
    names=meta['identity']['feature_names']; ix={n:i for i,n in enumerate(names)}
    for file,expected in meta['outputs'].items():
        assert sha(SRC/file)==expected
    x=np.load(SRC/'features.npy',mmap_mode='r'); y=np.load(SRC/'labels.npy'); day=np.load(SRC/'days.npy')
    payer=np.load(SRC/'payer_ids_for_audit_only.npy'); _,pi=np.unique(payer,return_inverse=True)
    keys=np.column_stack([day,pi,x[:,8:11].astype(np.int64)])
    strata,gi=np.unique(keys,axis=0,return_inverse=True); order=np.argsort(gi,kind='stable')
    ends=np.r_[0,np.flatnonzero(np.diff(gi[order]))+1,len(y)]
    groups=[]
    for lo,hi in zip(ends[:-1],ends[1:]):
        idx=order[lo:hi]; pos=idx[y[idx]==1]; neg=idx[y[idx]==0]
        if len(pos) and len(neg):groups.append((int(day[idx[0]]),pos,neg))
    output=[]; maximum_error=0.
    for k,r in enumerate(d['results']):
        name=r['name'] if r['kind']=='column' else r['source_column']
        value=np.asarray(x[:,ix[name]])
        mask_name=name.replace('__encoded','')+'__valid' if name.endswith('__encoded') else name+'__valid'
        valid=np.asarray(x[:,ix[mask_name]]==1) if mask_name in ix else np.ones(len(y),bool)
        if r['kind']=='predicate':
            cut=r['encoded_value']; op=r['operator']
            value=((value==cut) if op=='eq' else (value>cut) if op=='gt' else (value<=cut)).astype(float)
        numer=np.zeros(10); denom=np.zeros(10); variance=np.zeros(10)
        for date,p,n in groups:
            p=p[valid[p]]; n=n[valid[n]]
            if not len(p) or not len(n):continue
            a,v=pooled_rank_moments(value[p],value[n])
            numer[date-1]+=a; variance[date-1]+=v; denom[date-1]+=len(p)
        periods={}
        for period,sl in [('discovery_sep01_07',slice(0,7)),('sep08',slice(7,8)),
                          ('later_sep09_10',slice(8,10)),('all_train',slice(0,10))]:
            n=denom[sl].sum(); z=numer[sl].sum()-.5*n; v=variance[sl].sum()
            orig=r['periods'][period]
            assert n==orig['matched_positives']
            error=abs(numer[sl].sum()/n-orig['conditional_auc']) if n else 0.
            assert error<1e-12, (r['name'],period,error)
            maximum_error=max(maximum_error,error)
            p=math.erfc(abs(z)/math.sqrt(2*v)) if v>0 else 1.
            periods[period]=dict(independent_rank_auc=float(numer[sl].sum()/n) if n else None,
                conditional_rank_null_variance=float(v),conditional_rank_normal_p=p,
                conservative_p=max(p,orig['cluster_signflip_p']))
        output.append(dict(name=r['name'],periods=periods))
        if k%40==0:print(json.dumps(dict(verified=k+1,total=len(d['results']))),flush=True)
    for period in output[0]['periods']:
        q=bh([r['periods'][period]['conservative_p'] for r in output])
        for r,v in zip(output,q):r['periods'][period]['conservative_bh_q']=float(v)
    significant={period:[r['name'] for r in output if r['periods'][period]['conservative_bh_q']<.05]
                 for period in output[0]['periods']}
    write('significance_sensitivity.json',dict(results=output,significant_q05=significant,
        method='Max(cluster sign-flip p, conditional tie-corrected rank normal p), then BH over 317 tests per period',
        caveat='Neither test is a causal test; only 15 payer clusters and repeated receivers. Normal approximation is a safeguard for sparse flags, not an exact small-sample test.',
        reason='One-sided tiny effects in almost-constant flags can violate sign-flip symmetry; do not interpret original flag p alone.',
        summary_sha256=sha(OUT/'summary.json')))
    write('verification.json',dict(status='passed',columns_and_predicates=317,
        input_hashes_verified=True,all_period_counts_and_auc_independently_reproduced=True,
        maximum_auc_discrepancy=maximum_error,script_sha256=sha(__file__),
        rare_flag_significance_sensitivity_complete=True,significant_q05=significant,
        validation_test_read=False))
    print(json.dumps(dict(completed=True,significant_q05=significant,maximum_error=maximum_error)),flush=True)


if __name__=='__main__':main()
