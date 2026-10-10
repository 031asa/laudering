"""Validate and publish typed silver tables; retain the preceding daily table."""
import json
import tempfile
import time
import uuid
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]


def bank(column):
    return F.regexp_replace(F.col(column), r'^0+(?!$)', '')


def failures(frame, conditions):
    return frame.agg(*[F.sum(F.when(F.coalesce(condition, F.lit(False)), 0)
                                  .otherwise(1)).alias(name)
                       for name, condition in conditions.items()]).first().asDict()


def fingerprint(frame, group_columns):
    return [r.asDict() for r in frame.groupBy(*group_columns).agg(
        F.count('*').alias('rows'),
        F.sum(F.xxhash64(*frame.columns).cast('decimal(38,0)')).alias('checksum')
    ).orderBy(*group_columns).collect()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    started = time.perf_counter()
    silver = ROOT/'data/silver'
    source = silver/'transactions'
    backup = silver/'transactions_before_standardization'
    accounts_dest = silver/'accounts'
    manifest = json.loads((source/'_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'verified' or manifest.get('stage') == 'standardized':
        raise ValueError('Expected verified, unstandardized daily input')
    if backup.exists() or (accounts_dest.exists() and any(accounts_dest.iterdir())):
        raise FileExistsError('Refusing to overwrite previous outputs')
    staging = silver/('standardizing_'+uuid.uuid4().hex)
    staging.mkdir()
    report = {}
    with tempfile.TemporaryDirectory(prefix='aml_standardize_') as temporary:
        scratch = Path(temporary)
        spark = (SparkSession.builder.master('local[2]').appName('AML-silver-standardization')
                 .config('spark.ui.enabled','false').config('spark.ui.showConsoleProgress','false')
                 .config('spark.sql.shuffle.partitions','32')
                 .config('spark.sql.session.timeZone','UTC')
                 .config('spark.sql.warehouse.dir',(scratch/'warehouse').as_uri())
                 .config('spark.local.dir',str(scratch/'shuffle'))
                 .config('spark.hadoop.fs.file.impl','com.globalmentor.apache.hadoop.fs.BareLocalFileSystem')
                 .getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        try:
            # Boundary check: never strip account IDs or turn an all-zero bank into empty text.
            examples = spark.createDataFrame([('020',),('000',),('0',),('20',)],['bank'])
            assert [r[0] for r in examples.select(bank('bank')).collect()] == ['20','0','0','20']
            tx = spark.read.parquet(source.as_posix())
            ac = spark.read.parquet((ROOT/'data/bronze/accounts').as_posix())
            parsed_time = F.expr("try_cast(replace(timestamp, '/', '-') AS TIMESTAMP_NTZ)")
            conditions = {
                'invalid_from_bank': F.col('from_bank').rlike(r'^[0-9]+$'),
                'invalid_to_bank': F.col('to_bank').rlike(r'^[0-9]+$'),
                'invalid_from_account': F.length('from_account') > 0,
                'invalid_to_account': F.length('to_account') > 0,
                'invalid_time': F.col('timestamp').rlike(r'^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$') & parsed_time.isNotNull(),
                'wrong_date': F.to_date(parsed_time) == F.col('transaction_date'),
                'missing_payment_currency': F.length('payment_currency') > 0,
                'missing_receiving_currency': F.length('receiving_currency') > 0,
            }
            for column in ['amount_paid','amount_received']:
                decimal_value = F.expr(f'try_cast({column} AS DECIMAL(38,6))')
                conditions['invalid_'+column] = (F.col(column).rlike(r'^[0-9]+(?:\.[0-9]{1,6})?$')
                                                 & decimal_value.isNotNull() & (decimal_value >= 0))
            report['input_failures'] = failures(tx, conditions)
            report['account_failures'] = failures(ac, {
                'invalid_bank': F.col('bank_id').rlike(r'^[0-9]+$'),
                'invalid_account': F.length('account_number') > 0,
            })
            if any(report['input_failures'].values()) or any(report['account_failures'].values()):
                raise ValueError('Unexpected malformed values: refusing partial publication; see failure report')
            input_rows = tx.count()
            valid = tx.filter(F.col('is_laundering').isin('0','1'))
            report['excluded_invalid_labels'] = input_rows-valid.count()
            if input_rows != manifest['output_rows']:
                raise ValueError('Input count disagrees with manifest')
            typed = (valid.withColumn('from_bank',bank('from_bank'))
                     .withColumn('to_bank',bank('to_bank'))
                     .withColumn('timestamp',parsed_time)
                     .withColumn('amount_paid',F.col('amount_paid').cast('decimal(38,6)'))
                     .withColumn('amount_received',F.col('amount_received').cast('decimal(38,6)'))
                     .withColumn('is_laundering',F.col('is_laundering').cast('int')))
            typed_ac = ac.withColumn('bank_id',bank('bank_id'))
            print('Source values validated; calculating expected typed content and writing both tables.',flush=True)
            tx_before = fingerprint(typed,['transaction_date','is_laundering'])
            ac_before = fingerprint(typed_ac,['source_file'])
            (typed.write.mode('errorifexists').option('compression','zstd')
             .option('maxRecordsPerFile',500000).partitionBy('transaction_date')
             .parquet((staging/'transactions').as_posix()))
            typed_ac.write.mode('errorifexists').option('compression','zstd').parquet((staging/'accounts').as_posix())
            print('Typed tables written; checking schemas, daily content and account identity joins.',flush=True)
            restored = spark.read.parquet((staging/'transactions').as_posix()).select(*typed.columns)
            restored_ac = spark.read.parquet((staging/'accounts').as_posix()).select(*typed_ac.columns)
            if restored.dtypes != typed.dtypes or restored_ac.dtypes != typed_ac.dtypes:
                raise ValueError('Output types changed on readback')
            if fingerprint(restored,['transaction_date','is_laundering']) != tx_before:
                raise ValueError('Transaction daily counts or typed content checksums differ')
            if fingerprint(restored_ac,['source_file']) != ac_before:
                raise ValueError('Account counts or content checksums differ')
            report['wrong_partition_rows'] = restored.filter(
                F.col('timestamp').isNull() | F.col('transaction_date').isNull() |
                (F.to_date('timestamp') != F.col('transaction_date'))).count()
            keys = restored_ac.select(F.col('bank_id').alias('bank'),F.col('account_number').alias('account'))
            report['duplicate_account_keys'] = keys.groupBy('bank','account').count().filter('count > 1').count()
            tx_keys = (restored.select(F.col('from_bank').alias('bank'),F.col('from_account').alias('account'))
                       .union(restored.select(F.col('to_bank').alias('bank'),F.col('to_account').alias('account')))
                       .distinct())
            report['unmatched_account_keys'] = tx_keys.join(keys,['bank','account'],'left_anti').count()
            if report['wrong_partition_rows'] or report['duplicate_account_keys'] or report['unmatched_account_keys']:
                raise ValueError('Output date/identity checks failed')
            directories = sorted(p.name for p in (staging/'transactions').iterdir() if p.is_dir())
            if directories != sorted(p.name for p in source.iterdir() if p.is_dir()):
                raise ValueError('Date directory inventory changed')
            report.update(status='verified',stage='standardized',spark_version=spark.version,
                          input_rows=input_rows,output_rows=sum(r['rows'] for r in tx_before),
                          output_positive_rows=sum(r['rows'] for r in tx_before if r['is_laundering']==1),
                          account_rows=sum(r['rows'] for r in ac_before),partition_count=len(directories),
                          transaction_schema=restored.dtypes,account_schema=restored_ac.dtypes,
                          date_start=manifest['date_start'],date_end=manifest['date_end'],
                          daily_fingerprints=tx_before,account_fingerprints=ac_before,
                          verification='Validated source syntax/range before casting; readback schema, daily counts and typed xxHash64 sums; full date check and account-key joins. Checksums are not exact set comparison.',
                          seconds=round(time.perf_counter()-started,3))
            dump(staging/'transactions/_manifest.json',report)
            dump(staging/'accounts/_manifest.json',dict(status='verified',stage='standardized',output_rows=report['account_rows'],schema=restored_ac.dtypes))
        except Exception:
            dump(staging/'failure.json',report)
            raise
        finally:
            spark.stop()
    # All publication paths are fixed workspace children; retain the previous table for recovery.
    for path in [source,backup,accounts_dest,staging]:
        if not path.resolve().is_relative_to(silver.resolve()):
            raise ValueError('Publication path escaped the silver directory')
    if accounts_dest.exists():
        accounts_dest.rmdir()  # Empty directory only.
    source.rename(backup)
    try:
        (staging/'transactions').rename(source)
    except Exception:
        backup.rename(source)
        raise
    (staging/'accounts').rename(accounts_dest)
    staging.rmdir()
    dump(ROOT/'result/standardization.json',report)
    print(json.dumps({k:v for k,v in report.items() if 'fingerprints' not in k},indent=2))


if __name__ == '__main__':
    main()
