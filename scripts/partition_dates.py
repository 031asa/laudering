"""One calendar date per Hive-style directory; no random/time-window partitions."""
import json
import tempfile
import time
import uuid
from datetime import timedelta
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/'data/silver/transactions'
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite published transactions')
    source_dir = ROOT/'data/silver/transactions_deduplicated'
    source_manifest = json.loads((source_dir/'_manifest.json').read_text(encoding='utf-8'))
    if source_manifest['status'] != 'verified':
        raise ValueError('Input must be verified')
    staging = destination.with_name('transactions_partitioning_'+uuid.uuid4().hex)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='aml_dates_') as temporary:
        scratch = Path(temporary)
        spark = (SparkSession.builder.master('local[2]').appName('AML-continuous-daily-partitions')
                 .config('spark.ui.enabled','false').config('spark.ui.showConsoleProgress','false')
                 .config('spark.sql.shuffle.partitions','32')
                 .config('spark.sql.session.timeZone','UTC')
                 .config('spark.sql.warehouse.dir',(scratch/'warehouse').as_uri())
                 .config('spark.local.dir',str(scratch/'shuffle'))
                 .config('spark.hadoop.fs.file.impl','com.globalmentor.apache.hadoop.fs.BareLocalFileSystem')
                 .getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        try:
            source = spark.read.parquet(source_dir.as_posix())
            columns = source.columns
            time_expr = F.expr("try_cast(replace(timestamp, '/', '-') AS TIMESTAMP_NTZ)")
            dated = source.withColumn('transaction_date',F.to_date(time_expr))
            invalid = dated.agg(
                F.count('*').alias('input_rows'),
                F.sum(F.when(F.col('transaction_date').isNull(),1).otherwise(0)).alias('invalid_dates'),
                F.sum(F.when(F.col('is_laundering').isin('0','1'),0).otherwise(1)).alias('invalid_labels')
            ).first().asDict()
            if invalid['input_rows'] != source_manifest['output_rows'] or invalid['invalid_dates']:
                raise ValueError('Invalid dates or input row count mismatch; no records silently dropped')
            dated = dated.filter(F.col('is_laundering').isin('0','1'))
            def fingerprint(frame):
                return [row.asDict() for row in frame.groupBy('transaction_date').agg(
                    F.count('*').alias('rows'),
                    F.sum(F.when(F.col('is_laundering')=='1',1).otherwise(0)).alias('positive_rows'),
                    F.sum(F.xxhash64(*columns).cast('decimal(38,0)')).alias('row_checksum_sum'),
                    F.min('source_record_number').alias('min_source_record_number'),
                    F.max('source_record_number').alias('max_source_record_number')
                ).orderBy('transaction_date').collect()]
            before = fingerprint(dated)
            dates = [row['transaction_date'] for row in before]
            if not dates:
                raise ValueError('No valid rows')
            expected_dates = [dates[0]+timedelta(days=i) for i in range((dates[-1]-dates[0]).days+1)]
            if dates != expected_dates:
                raise ValueError('Source dates contain gaps; refusing to invent transactions for missing dates')
            print(f'Validated {len(dates)} consecutive dates; writing daily directories.',flush=True)
            (dated.write.mode('errorifexists').option('compression','zstd')
             .option('maxRecordsPerFile',500000).partitionBy('transaction_date').parquet(staging.as_posix()))
            print('Daily Parquet written; checking every row against its directory date.',flush=True)
            restored = spark.read.parquet(staging.as_posix())
            wrong = restored.filter(F.col('transaction_date').isNull() |
                     (F.col('transaction_date') != F.to_date(time_expr))).count()
            if wrong:
                raise ValueError('Rows found in incorrect date partitions')
            after = fingerprint(restored)
            if before != after:
                raise ValueError('Daily counts, label counts, row checksums or source ranges differ')
            directories = sorted(path.name for path in staging.iterdir() if path.is_dir())
            if directories != ['transaction_date='+str(day) for day in dates]:
                raise ValueError('Missing or unexpected partition directories')
            files = []
            for name in directories:
                parts = list((staging/name).glob('*.parquet'))
                if not parts:
                    raise ValueError('Empty date partition')
                files.append({'date':name.split('=',1)[1],'parquet_files':len(parts),
                              'bytes':sum(p.stat().st_size for p in parts)})
            report = {'status':'verified','spark_version':spark.version,
                      'stage':'deduplicated and daily-partitioned; other silver standardization pending',
                      **invalid,'output_rows':sum(row['rows'] for row in after),
                      'output_positive_rows':sum(row['positive_rows'] for row in after),
                      'partition_column':'transaction_date','date_start':str(dates[0]),'date_end':str(dates[-1]),
                      'partition_count':len(dates),'dates_continuous':True,'wrong_partition_rows':wrong,
                      'daily_fingerprints':after,'files':files,
                      'verification':'Full row/date check plus per-day counts, labels, source ranges and xxHash64 sums (checksum comparison, not exact set comparison).',
                      'source':'data/silver/transactions_deduplicated',
                      'seconds':round(time.perf_counter()-started,3)}
            (staging/'_manifest.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
        finally:
            spark.stop()
    if destination.exists():
        destination.rmdir()  # Empty directory only; never recursively remove existing data.
    staging.rename(destination)
    result = ROOT/'result/date_partitioning.json'
    result.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['files','daily_fingerprints']},indent=2))


if __name__ == '__main__':
    main()
