"""Identify exact 11-field duplicates with Spark; record every kept/removed ID."""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.audit import TX


def main():
    output = ROOT/'result/deduplication'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='aml_dedup_') as temporary:
        scratch=Path(temporary)
        spark=(SparkSession.builder.master('local[2]').appName('AML-exact-deduplication')
               .config('spark.ui.enabled','false').config('spark.ui.showConsoleProgress','false')
               .config('spark.sql.shuffle.partitions','128')
               .config('spark.sql.warehouse.dir',(scratch/'warehouse').as_uri())
               .config('spark.local.dir',str(scratch/'shuffle'))
               .config('spark.hadoop.fs.file.impl','com.globalmentor.apache.hadoop.fs.BareLocalFileSystem')
               .getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        try:
            source=spark.read.parquet((ROOT/'data/bronze/transactions').as_posix())
            groups=(source.groupBy(*TX).count().filter(F.col('count')>1).persist(StorageLevel.DISK_ONLY))
            group_rows=groups.collect()
            print(f'Exact duplicate groups: {len(group_rows)}',flush=True)
            repeated=source.join(F.broadcast(groups.select(*TX)),TX,'inner').collect()
            by_key={}
            for row in repeated:
                value=row.asDict()
                by_key.setdefault(tuple(value[c] for c in TX),[]).append(value)
            decisions=[]
            for key, rows in by_key.items():
                rows.sort(key=lambda r:(r['source_file'],r['source_record_number'],r['transaction_id']))
                kept=rows[0]
                for removed in rows[1:]:
                    decisions.append({'kept_transaction_id':kept['transaction_id'],
                                      'kept_source_record_number':kept['source_record_number'],
                                      'removed_transaction_id':removed['transaction_id'],
                                      'removed_source_record_number':removed['source_record_number'],
                                      'source_file':removed['source_file'],
                                      'original_fields':dict(zip(TX,key))})
            if len(decisions)!=sum(row['count']-1 for row in group_rows):
                raise RuntimeError('Duplicate decision count mismatch')
            decisions.sort(key=lambda x:x['removed_source_record_number'])
            manifest=json.loads((ROOT/'data/bronze/transactions/_manifest.json').read_text())
            result={'status':'identified','spark_version':spark.version,'compared_fields':TX,
                    'source_sha256':manifest['source_sha256'],'input_rows':manifest['rows'],
                    'duplicate_groups':len(group_rows),'removed_rows':len(decisions),
                    'keep_rule':'smallest (source_file, source_record_number, transaction_id)',
                    'created_at_utc':datetime.now(timezone.utc).isoformat(),'decisions':decisions}
            (output/'decisions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(f'Identified {len(decisions)} removals; original files unchanged.',flush=True)
        finally:
            spark.stop()


if __name__=='__main__':
    main()
