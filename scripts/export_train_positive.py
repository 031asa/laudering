"""Export all label-1 transactions from the frozen training union, without sampling."""
import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'config/dataset_split.json').read_text(encoding='utf-8'))
    source = ROOT / config['source']
    assert hashlib.sha256((source / '_manifest.json').read_bytes()).hexdigest() == config['source_manifest_sha256']
    members = [s for s in config['splits'] if s['name'] in config['train_members']]
    assert {s['name'] for s in members} == set(config['train_members'])
    dates = sorted(d for s in members for d in s['dates'])
    assert len(dates) == len(set(dates)) == 10
    assert dates[0] == '2022-09-01' and dates[-1] == '2022-09-10'
    pieces, scanned = [], 0
    for date in dates:
        paths = sorted((source / f'transaction_date={date}').glob('*.parquet'))
        assert paths
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=150000):
                scanned += batch.num_rows
                selected = pa.Table.from_batches([batch]).filter(pc.equal(batch.column('is_laundering'), 1))
                if selected.num_rows:
                    selected = selected.append_column('transaction_date', pc.cast(selected['timestamp'], pa.date32()))
                    assert all(str(d) == date for d in selected['transaction_date'].to_pylist())
                    pieces.append(selected)
        print(f'Export scan completed: {date}', flush=True)
    table = pa.concat_tables(pieces).sort_by([('timestamp', 'ascending'), ('source_record_number', 'ascending')])
    assert scanned == sum(s['rows'] for s in members) == 20318992
    assert table.num_rows == sum(s['positive_rows'] for s in members) == 17794
    assert pc.count_distinct(table['transaction_id']).as_py() == table.num_rows
    out = ROOT / 'result/train_laundering'
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / '训练集_全部洗钱标签交易.parquet'
    csv_path = out / '训练集_全部洗钱标签交易.csv'
    pq.write_table(table, parquet, compression='snappy')
    rows = table.to_pylist()
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=table.column_names)
        writer.writeheader()
        for row in rows:
            row['timestamp'] = row['timestamp'].isoformat(sep=' ', timespec='seconds')
            writer.writerow(row)
    assert pq.ParquetFile(parquet).read().equals(table)
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert all(all(a[k] == str(b[k]) for k in table.column_names) for a, b in zip(csv_rows, rows))
    report = dict(train_members=config['train_members'], dates=dates, scanned_rows=scanned,
                  exported_rows=table.num_rows, filter='is_laundering = 1', sampled=False,
                  csv_exact_roundtrip=True, parquet_exact_roundtrip=True,
                  source_manifest_sha256=config['source_manifest_sha256'],
                  files=[csv_path.name, parquet.name])
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
