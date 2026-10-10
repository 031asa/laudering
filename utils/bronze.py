"""Lossless CSV-field ingestion. No business values are parsed or cleaned."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TABLES = {
    "transactions": (
        "HI-Medium_Trans.csv",
        ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
         "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"],
        ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_received",
         "receiving_currency", "amount_paid", "payment_currency", "payment_format", "is_laundering"],
        "transaction_id",
    ),
    "accounts": (
        "HI-Medium_accounts.csv",
        ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"],
        ["bank_name", "bank_id", "account_number", "entity_id", "entity_name"],
        "account_record_id",
    ),
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(reader, count: int, width: int, offset: int):
    rows = list(itertools.islice(reader, count))
    for index, row in enumerate(rows, offset + 1):
        if len(row) != width:
            raise ValueError(f"Record {index}: expected {width} fields, got {len(row)}; no row skipped")
    return rows


def convert(source: Path, destination: Path, kind: str, expected_sha256: str,
            batch_rows: int = 250_000) -> dict:
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    filename, header, names, id_name = TABLES[kind]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty destination: {destination}")
    started = time.perf_counter()
    source_hash = sha256_file(source)
    if source_hash != expected_sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source}")
    source_stat = source.stat()
    destination.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [pa.field(name, pa.string(), nullable=False) for name in names]
        + [pa.field(id_name, pa.string(), nullable=False),
           pa.field("source_file", pa.string(), nullable=False),
           pa.field("source_record_number", pa.int64(), nullable=False)],
        metadata={"source_sha256": source_hash, "original_header": json.dumps(header),
                  "record_number": "1-based CSV data record, excludes header; not physical line",
                  "lossless_scope": "decoded UTF-8 CSV field values; original bytes remain in raw"},
    )
    parts, offset = [], 0
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != header:
            raise ValueError(f"Unexpected source header: {source}")
        while rows := read_rows(reader, batch_rows, len(names), offset):
            count = len(rows)
            arrays = [pa.array(column, type=pa.string()) for column in zip(*rows)]
            arrays.extend([
                pa.array([f"{source_hash}:{i}" for i in range(offset + 1, offset + count + 1)]),
                pa.array([filename] * count),
                pa.array(range(offset + 1, offset + count + 1), type=pa.int64()),
            ])
            table = pa.Table.from_arrays(arrays, schema=schema)
            name = f"part-{len(parts):05d}.parquet"
            partial = destination / (name + ".partial")
            pq.write_table(table, partial, compression="zstd", row_group_size=100_000)
            partial.rename(destination / name)
            parts.append({"name": name, "rows": count, "first_record": offset + 1,
                          "last_record": offset + count, "bytes": (destination / name).stat().st_size})
            offset += count
            if len(parts) % 4 == 0:
                print(f"{kind}: converted {offset:,} records", flush=True)
    if (source.stat().st_size, source.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise RuntimeError("Source changed during conversion")
    report = {
        "status": "converted_pending_verification", "kind": kind, "source_file": filename,
        "source_sha256": source_hash, "source_bytes": source_stat.st_size,
        "rows": offset, "columns": schema.names, "business_field_type": "string (no null coercion)",
        "field_mapping": [{"position": i + 1, "original": old, "bronze": new}
                          for i, (old, new) in enumerate(zip(header, names))],
        "id_rule": "source SHA-256 + ':' + 1-based CSV data record number",
        "engine": f"PyArrow {pa.__version__}; Python csv reader", "compression": "zstd",
        "parquet_bytes": sum(part["bytes"] for part in parts), "parts": parts,
        "conversion_seconds": round(time.perf_counter() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(destination / "_manifest.json", report)
    return report


def verify(source: Path, destination: Path, kind: str) -> dict:
    """Independently reread the original CSV and compare every field of every row."""
    started = time.perf_counter()
    filename, header, names, id_name = TABLES[kind]
    manifest = json.loads((destination / "_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(source) != manifest["source_sha256"]:
        raise ValueError("Source SHA-256 changed")
    expected_parts = [part["name"] for part in manifest["parts"]]
    if sorted(path.name for path in destination.glob("*.parquet")) != expected_parts:
        raise ValueError("Parquet part inventory mismatch")
    offset = 0
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != header:
            raise ValueError("Source header mismatch")
        for part in manifest["parts"]:
            parquet = pq.ParquetFile(destination / part["name"])
            if parquet.schema_arrow.names != names + [id_name, "source_file", "source_record_number"]:
                raise ValueError("Parquet schema mismatch")
            part_count = 0
            for batch in parquet.iter_batches(batch_size=100_000):
                rows = read_rows(reader, batch.num_rows, len(names), offset)
                if len(rows) != batch.num_rows:
                    raise ValueError("Parquet has extra records")
                for index, column in enumerate(zip(*rows)):
                    if not batch.column(index).equals(pa.array(column, type=pa.string())):
                        raise ValueError(f"Original field mismatch: {names[index]} near record {offset + 1}")
                sequence = range(offset + 1, offset + batch.num_rows + 1)
                expected = [
                    pa.array([f"{manifest['source_sha256']}:{i}" for i in sequence]),
                    pa.array([filename] * batch.num_rows),
                    pa.array(sequence, type=pa.int64()),
                ]
                for index, values in enumerate(expected, len(names)):
                    if not batch.column(index).equals(values):
                        raise ValueError(f"Provenance/ID mismatch near record {offset + 1}")
                offset += batch.num_rows
                part_count += batch.num_rows
            if part_count != part["rows"]:
                raise ValueError("Part row count mismatch")
            if offset % 1_000_000 == 0:
                print(f"{kind}: verified {offset:,} records, all fields", flush=True)
        if next(reader, None) is not None or offset != manifest["rows"]:
            raise ValueError("CSV/Parquet record count mismatch")
    result = {"status": "passed", "rows": offset, "original_fields_compared": offset * len(names),
              "checks": ["source_sha256", "all_original_field_values", "record_order",
                         "row_count", "part_inventory", "stable_unique_record_ids", "source_provenance"],
              "seconds": round(time.perf_counter() - started, 3),
              "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(destination / "_verification.json", result)
    manifest["status"] = "verified"
    write_json(destination / "_manifest.json", manifest)
    return result
