"""Exact raw-int payment signatures; retrospective label audit, never features."""
from pathlib import Path
import json
import numpy as np
from analyze_non_ach_differences import ROOT, SRC, OUT, sha, write


def prior_counts(group, timestamp, label):
    """Number of earlier observations and positives of same key; all ties excluded."""
    order=np.lexsort((timestamp,group)); g=group[order]; t=timestamp[order]
    z=label[order].astype(np.int64)
    i=np.arange(len(g)); start=np.r_[True,(g[1:]!=g[:-1])]
    base=np.maximum.accumulate(np.where(start,i,0))
    event=np.maximum.accumulate(np.where(np.r_[True,(g[1:]!=g[:-1])|(t[1:]!=t[:-1])],i,0))
    cum=np.r_[0,np.cumsum(z)]
    n=np.empty(len(g),dtype=np.int64); p=np.empty(len(g),dtype=np.int64)
    n[order]=event-base; p[order]=cum[event]-cum[base]
    return n,p


def main():
    if (OUT/'repeated_payment_audit.json').exists():
        raise RuntimeError('Preserve existing completed audit')
    m=json.loads((SRC/'dataset_manifest.json').read_text(encoding='utf-8'))
    # Parent diagnostic has verified the matrix; bind this audit to the same
    # identities and labels, and verify raw inputs against frozen baseline.
    for key in ['edge_ids.npy','labels.npy','payer_ids_for_audit_only.npy','days.npy']:
        assert sha(SRC/key)==m['outputs'][key]
    frozen=json.loads((ROOT/'result/gnn_train_gpu_v1/config.json').read_text(encoding='utf-8'))['input_hashes']
    ids=np.load(SRC/'edge_ids.npy'); y=np.load(SRC/'labels.npy'); day=np.load(SRC/'days.npy')
    a={}; hashes={}
    for name in ['src','dst','paid_scaled','received_scaled','timestamp','payment_format_cat','payment_currency_cat','receiving_currency_cat']:
        p=ROOT/'data/gnn_v1_train/history_index'/f'{name}.npy'; rel=str(p.relative_to(ROOT))
        assert sha(p)==frozen[rel], rel
        hashes[rel]=frozen[rel]; a[name]=np.asarray(np.load(p,mmap_mode='r')[ids])
    assert np.array_equal(a['src'],np.load(SRC/'payer_ids_for_audit_only.npy'))
    results=[]
    for columns in [('src','dst','payment_format_cat','payment_currency_cat','receiving_currency_cat'),
                    ('src','dst','payment_format_cat','payment_currency_cat','receiving_currency_cat','paid_scaled','received_scaled')]:
        rec=np.rec.fromarrays([a[n] for n in columns],names=list(columns))
        _,gi=np.unique(rec,return_inverse=True)
        size=np.bincount(gi); positives=np.bincount(gi,weights=y).astype(np.int64)
        mixed=(positives>0)&(positives<size)
        prior_n,prior_p=prior_counts(gi,a['timestamp'],y)
        stats={}
        for period,mask in [('all_train',np.ones(len(y),bool)),('early_sep01_07',day<=7),('later_sep09_10',day>=9)]:
            stats[period]={}
            for label in [0,1]:
                t=mask&(y==label)
                stats[period][str(label)]=dict(rows=int(t.sum()),
                    mixed_label_signature_any_train_date=int(mixed[gi[t]].sum()),
                    has_strictly_prior_same_signature=int((prior_n[t]>0).sum()),
                    has_strictly_prior_negative_same_signature=int((prior_n[t]-prior_p[t]>0).sum()),
                    has_strictly_prior_positive_same_signature=int((prior_p[t]>0).sum()))
        results.append(dict(signature_columns=list(columns),distinct_signatures=len(size),
            mixed_label_signatures=int(mixed.sum()),periods=stats))
        print(json.dumps(results[-1],ensure_ascii=True),flush=True)
    # Population denominators: category can look distinctive only because of
    # prevalence, so record full counts, not just positive shares.
    v=json.loads((ROOT/'data/gnn_v1_train/metadata/vocabulary.json').read_text(encoding='utf-8'))
    payments=[]
    for c in np.unique(a['payment_format_cat']):
        mask=a['payment_format_cat']==c
        payments.append(dict(payment=v['values']['payment_format'][int(c)-2],rows=int(mask.sum()),
            positives=int(y[mask].sum()),prevalence=float(y[mask].mean())))
    write('repeated_payment_audit.json',dict(results=results,payment_counts=payments,
        raw_hashes=hashes,script_sha256=sha(__file__),source_manifest_sha256=sha(SRC/'dataset_manifest.json'),
        interpretation='Identical listed payment fields can have different labels; this does not establish identical full prior histories or erroneous labels.',
        label_usage='Retrospective audit only. Prior event labels were not necessarily known then. No label-derived feature is proposed or trained.',
        time_scope='Train Sep1-10 only; any-date mixed signatures are retrospective. Prior counts strictly exclude same timestamps.',
        validation_test_read=False))


if __name__=='__main__':
    main()
