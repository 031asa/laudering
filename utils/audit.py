"""Basic quality evidence for cleaning decisions; all changes are projections only."""
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

TX = ['timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
      'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
      'payment_format', 'is_laundering']
AC = ['bank_name', 'bank_id', 'account_number', 'entity_id', 'entity_name']


def n(condition, name):
    return F.sum(F.when(condition, 1).otherwise(0)).alias(name)


def bank_key(column):
    value = F.trim(F.col(column))
    return F.when(value.rlike('^[0-9]+$'), F.regexp_replace(value, '^0+(?!$)', '')).otherwise(value)


def time_value():
    return F.expr("try_cast(replace(trim(timestamp), '/', '-') AS TIMESTAMP_NTZ)")


def quality(frame, columns):
    expressions = [F.count('*').alias('rows')]
    for column in columns:
        value = F.col(column)
        expressions.extend([
            n(value.isNull(), column + '__null'),
            n(F.trim(value) == '', column + '__blank'),
            n(value != F.trim(value), column + '__outer_spaces'),
            n(F.upper(F.trim(value)).isin('NULL', 'NAN', 'N/A', 'NONE'), column + '__placeholder'),
        ])
    return expressions


def run(root: Path):
    output = root / 'result' / 'cleaning_audit'
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'running', 'scope': 'full bronze transactions and accounts; no mutation',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'stages': {}}
    def save(stage, value):
        report['stages'][stage] = value
        report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        temporary = output / 'audit.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
        temporary.replace(output / 'audit.json')
        print('COMPLETED '+stage+': '+json.dumps(value, ensure_ascii=True, default=str)[:1300], flush=True)

    with tempfile.TemporaryDirectory(prefix='aml_audit_') as temporary:
        scratch = Path(temporary)
        spark = (SparkSession.builder.master('local[2]').appName('AML-basic-cleaning-audit')
                 .config('spark.ui.enabled', 'false').config('spark.ui.showConsoleProgress', 'false')
                 .config('spark.sql.shuffle.partitions', '128')
                 .config('spark.sql.adaptive.enabled', 'true')
                 .config('spark.sql.session.timeZone', 'UTC')
                 .config('spark.sql.warehouse.dir', (scratch/'warehouse').as_uri())
                 .config('spark.local.dir', str(scratch/'shuffle'))
                 .config('spark.hadoop.fs.file.impl', 'com.globalmentor.apache.hadoop.fs.BareLocalFileSystem')
                 .getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        report['engine'] = {'spark': spark.version, 'master': 'local[2]', 'driver_memory': '2g',
                            'timestamp_type': 'TIMESTAMP_NTZ; timezone unknown, no UTC conversion'}
        try:
            checks = spark.sql("""SELECT
                try_cast('2022-02-30 00:00' AS TIMESTAMP_NTZ) IS NULL AS invalid_date_rejected,
                try_cast('0001.2300' AS DECIMAL(38,10)) = 1.23 AS decimal_ok,
                regexp_replace('000', '^0+(?!$)', '') = '0' AS zero_bank_preserved,
                regexp_replace('020', '^0+(?!$)', '') = '20' AS bank_padding_removed""").first().asDict()
            if not all(checks.values()):
                raise RuntimeError('Audit expression checks failed')
            save('expression_checks', checks)
            # Hadoop Path handles Unicode directly; percent-encoded file URIs
            # are interpreted literally by the installed Windows filesystem adapter.
            t = spark.read.parquet((root/'data/bronze/transactions').as_posix()).select(*TX, 'transaction_id', 'source_record_number')
            a = spark.read.parquet((root/'data/bronze/accounts').as_posix()).select(*AC)
            t = (t.withColumn('_time', time_value())
                 .withColumn('_paid', F.expr('try_cast(trim(amount_paid) AS DECIMAL(38,10))'))
                 .withColumn('_received', F.expr('try_cast(trim(amount_received) AS DECIMAL(38,10))')))
            base = quality(t, TX)
            base.extend([
                n(~F.col('is_laundering').isin('0','1'), 'invalid_label'),
                n(F.col('is_laundering') == '1', 'positive_rows'),
                n(F.col('_time').isNull(), 'timestamp_unparseable'),
                n(~F.col('timestamp').rlike(r'^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$'), 'timestamp_nonstandard_format'),
                F.min('_time').alias('time_min'), F.max('_time').alias('time_max'),
                n((F.col('from_bank') == F.col('to_bank')) & (F.col('from_account') == F.col('to_account')), 'self_transfers'),
                n(F.col('from_bank') != F.col('to_bank'), 'cross_bank_rows'),
                n(F.col('payment_currency') != F.col('receiving_currency'), 'cross_currency_rows'),
                n((F.col('payment_currency') == F.col('receiving_currency')) & (F.col('_paid') != F.col('_received')), 'same_currency_amount_mismatch'),
            ])
            for column, parsed in [('amount_paid','_paid'), ('amount_received','_received')]:
                base.extend([n(F.col(parsed).isNull(), column+'__unparseable'),
                             n(F.col(parsed) < 0, column+'__negative'), n(F.col(parsed) == 0, column+'__zero'),
                             n(~F.trim(F.col(column)).rlike(r'^[+-]?\d+(\.\d+)?$'),column+'__nonplain_number'),
                             F.max(F.length(F.regexp_extract(column, r'\.(\d+)$',1))).alias(column+'__max_fraction_digits')])
            for column in ['from_bank','to_bank']:
                base.extend([n(~F.col(column).rlike('^[0-9]+$'), column+'__nondigit'),
                             n(F.col(column).rlike('^0[0-9]+$'), column+'__leading_zero')])
            for column in ['from_account','to_account']:
                base.append(n(~F.col(column).rlike('^[0-9A-F]+$'),column+'__non_upper_hex'))
            summary = t.agg(*base).first().asDict()
            save('transaction_quality', summary)

            def distributions(frame, columns, extra=()):
                return [row.asDict() for row in frame.groupBy(*columns).agg(
                    F.count('*').alias('rows'), n(F.col('is_laundering')=='1','positive_rows'), *extra
                ).orderBy(*columns).collect()]
            save('daily', distributions(t.withColumn('date',F.to_date('_time')), ['date']))
            save('payment_formats', distributions(t, ['payment_format']))
            save('currency_pairs', distributions(t, ['payment_currency','receiving_currency']))
            for currency, amount in [('payment_currency','_paid'),('receiving_currency','_received')]:
                save(currency+'_amounts', distributions(t,[currency],[
                    F.min(amount).alias('min'),F.max(amount).alias('max'),
                    F.percentile_approx(F.col(amount).cast('double'),[0.5,0.99,0.999],10000).alias('approx_p50_p99_p999')]))
            save('account_quality',a.agg(*quality(a,AC),
                n(~F.col('bank_id').rlike('^[0-9]+$'),'bank_id__nondigit'),
                n(F.col('bank_id').rlike('^0[0-9]+$'),'bank_id__leading_zero'),
                n(~F.col('account_number').rlike('^[0-9A-F]+$'),'account_number__non_upper_hex')
            ).first().asDict())

            print('STARTING account identity and coverage checks',flush=True)
            ag = a.withColumn('bank',bank_key('bank_id')).groupBy('bank','account_number').agg(
                F.count('*').alias('n'),F.countDistinct('entity_id').alias('entity_ids'))
            save('account_key_uniqueness', ag.agg(F.count('*').alias('normalized_unique_keys'),
                n(F.col('n')>1,'repeated_key_groups'),n(F.col('entity_ids')>1,'conflicting_entity_key_groups'),
                F.sum(F.col('n')-1).alias('extra_records_per_key')).first().asDict())
            keys = t.select(F.col('from_bank').alias('bank'),F.col('from_account').alias('account')).union(
                t.select(F.col('to_bank').alias('bank'),F.col('to_account').alias('account'))).distinct()
            keys = keys.persist(StorageLevel.DISK_ONLY)
            raw_accounts = a.select(F.col('bank_id').alias('bank'),F.col('account_number').alias('account')).distinct()
            normalized_accounts = a.select(bank_key('bank_id').alias('bank'),F.col('account_number').alias('account')).distinct()
            normalized = keys.select(bank_key('bank').alias('bank'),'account').distinct().persist(StorageLevel.DISK_ONLY)
            unmatched = normalized.join(normalized_accounts,['bank','account'],'left_anti').persist(StorageLevel.DISK_ONLY)
            coverage = {'raw_transaction_keys': keys.count(),
                        'raw_unmatched_transaction_keys':keys.join(raw_accounts,['bank','account'],'left_anti').count(),
                        'normalized_transaction_keys': normalized.count(),
                        'normalized_unmatched_transaction_keys': unmatched.count(),
                        'unmatched_key_samples':[r.asDict() for r in unmatched.orderBy('bank','account').limit(10).collect()]}
            save('account_coverage',coverage)
            unmatched.unpersist(); normalized.unpersist(); keys.unpersist()

            print('STARTING full transaction duplicate and label conflict check',flush=True)
            repeated = (t.groupBy(*[c for c in TX if c!='is_laundering'])
                        .agg(F.count('*').alias('n'),F.collect_set('is_laundering').alias('labels'))
                        .filter(F.col('n')>1).persist(StorageLevel.DISK_ONLY))
            duplicates = repeated.agg(
                F.count('*').alias('repeated_attribute_groups'),
                F.coalesce(F.sum('n'),F.lit(0)).alias('rows_in_repeated_attribute_groups'),
                F.coalesce(F.sum(F.col('n')-F.size('labels')),F.lit(0)).alias('exact_duplicate_excess_rows'),
                n(F.size('labels')>1,'label_conflict_groups')
            ).first().asDict()
            duplicates['definition']='Group by all 10 raw transaction attributes (excluding label and provenance); exact duplicate excess = sum(n - distinct labels). No deletion.'
            duplicates['samples']=[r.asDict() for r in repeated.orderBy(F.desc('n'),'timestamp','from_bank','from_account').limit(10).collect()]
            save('transaction_duplicates',duplicates)
            repeated.unpersist()
            report['status']='completed'
            save('completion',{'no_data_modified':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()})
        except Exception as error:
            report['status']='failed'
            save('failure',{'error':str(error)[:2000]})
            raise
        finally:
            spark.stop()
