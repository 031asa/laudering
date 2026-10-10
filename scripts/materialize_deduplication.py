"""Apply exact Spark decisions; copy untouched parts and verify every rewritten value."""
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    source=ROOT/'data/bronze/transactions'
    output=ROOT/'data/silver/transactions_deduplicated'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite deduplicated output')
    decisions=json.loads((ROOT/'result/deduplication/decisions.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'_manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='verified' or manifest['source_sha256']!=decisions['source_sha256']:
        raise ValueError('Source version/verification mismatch')
    removed={d['removed_transaction_id']:d for d in decisions['decisions']}
    kept={d['kept_transaction_id']:d for d in decisions['decisions']}
    if len(removed)!=decisions['removed_rows'] or set(removed)&set(kept):
        raise ValueError('Invalid duplicate decisions')
    output.mkdir(parents=True,exist_ok=True)
    rewritten=copied=total=positives=0
    observed_removed=set(); observed_kept=set(); parts=[]
    for part in manifest['parts']:
        origin=source/part['name']; target=output/part['name']
        hits=[d for d in removed.values() if part['first_record']<=d['removed_source_record_number']<=part['last_record']]
        keep_hits=[d for d in kept.values() if part['first_record']<=d['kept_source_record_number']<=part['last_record']]
        if hits or keep_hits:
            table=pq.read_table(origin)
            for d,field in [(d,'removed') for d in hits]+[(d,'kept') for d in keep_hits]:
                pos=d[field+'_source_record_number']-part['first_record']
                row=table.slice(pos,1).to_pylist()[0]
                if row['transaction_id']!=d[field+'_transaction_id'] or any(row[c]!=v for c,v in d['original_fields'].items()):
                    raise ValueError('Decision does not match original 11 field values')
                (observed_removed if field=='removed' else observed_kept).add(row['transaction_id'])
        if hits:
            mask=pc.invert(pc.is_in(table['transaction_id'],value_set=pa.array([d['removed_transaction_id'] for d in hits])))
            expected=table.filter(mask)
            pq.write_table(expected,target,compression='zstd',row_group_size=100000)
            actual=pq.read_table(target)
            if not actual.equals(expected,check_metadata=True):
                raise ValueError('Written Parquet differs from retained source values')
            if actual.num_rows!=part['rows']-len(hits):
                raise ValueError('Incorrect number of removals')
            rewritten+=1
        else:
            shutil.copy2(origin,target)
            if digest(origin)!=digest(target):
                raise ValueError('Copied part SHA-256 mismatch')
            copied+=1
        count=pq.ParquetFile(target).metadata.num_rows
        total+=count
        for batch in pq.ParquetFile(target).iter_batches(columns=['is_laundering']):
            positives+=pc.sum(pc.cast(pc.equal(batch.column(0),'1'),pa.int64())).as_py()
        parts.append({'name':part['name'],'rows':count,'bytes':target.stat().st_size})
        if len(parts)%16==0:
            print(f'Verified {len(parts)}/{len(manifest["parts"])} parts',flush=True)
    if observed_removed!=set(removed) or observed_kept!=set(kept):
        raise ValueError('Some kept/removed records were not located')
    if total!=manifest['rows']-len(removed):
        raise ValueError('Total row reconciliation failed')
    original_positive=json.loads((ROOT/'result/cleaning_audit/audit.json').read_text(encoding='utf-8'))['stages']['transaction_quality']['positive_rows']
    removed_positive=sum(d['original_fields']['is_laundering']=='1' for d in removed.values())
    if positives!=original_positive-removed_positive:
        raise ValueError('Label count reconciliation failed')
    result={'status':'verified','stage':'exact_deduplication_only; remaining silver rules not applied',
            'input_rows':manifest['rows'],'removed_rows':len(removed),'output_rows':total,
            'output_positive_rows':positives,'removed_positive_rows':removed_positive,
            'compared_fields':decisions['compared_fields'],'source_sha256':manifest['source_sha256'],
            'copied_parts':copied,'rewritten_parts':rewritten,'parts':parts,
            'verification':'Untouched parts SHA-256 identical; rewritten parts all retained columns and metadata equal source after exact exclusions; every removal and keeper independently compared on 11 original fields.',
            'decision_log':'result/deduplication/decisions.json',
            'execution':'Spark 4.1.2 exact grouping and provenance; PyArrow '+pa.__version__+' partial rewrite and value verification'}
    (output/'_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (ROOT/'result/deduplication/result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='parts'},indent=2))


if __name__=='__main__':
    main()
