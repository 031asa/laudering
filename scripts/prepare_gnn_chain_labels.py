"""Export frozen positive edges for retrospective Train/Validation chain audit.

Final labels are an oracle for after-the-fact analysis, not information known at
the transaction time. This script never reads Test or changes a model/cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/gnn_chain_audit_20261003"
TRAIN_ROWS, TRAIN_POS = 20_318_992, 17_794
VAL_ROWS, VAL_POS = 4_689_006, 6_420
TOTAL = TRAIN_ROWS + VAL_ROWS
START = int(np.datetime64("2022-09-01", "us").astype("int64"))
CUT = int(np.datetime64("2022-09-11", "us").astype("int64"))
END = int(np.datetime64("2022-09-14", "us").astype("int64"))
FIELDS = ("src", "dst", "timestamp", "payment_currency_cat",
          "receiving_currency_cat", "paid_scaled", "received_scaled")
EXPECTED_DTYPES = {"src": "uint32", "dst": "uint32", "timestamp": "int64",
                   "payment_currency_cat": "int16", "receiving_currency_cat": "int16",
                   "paid_scaled": "int64", "received_scaled": "int64"}
BATCH = 131_072


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def normalized_hashes(mapping):
    return {str(k).replace("\\", "/"): value for k, value in mapping.items()}


def disk_check():
    require(shutil.disk_usage(ROOT).free >= 20 * 1024 ** 3,
            "Less than 20 GiB free: stop creating audit data")


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    began = time.monotonic()
    disk_check()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "positive_edges.parquet"
    manifest_path = OUT / "labels_manifest.json"
    status_path = OUT / "labels_status.json"
    require(not destination.exists() and not manifest_path.exists(),
            "Existing label outputs must be preserved")
    # Exclusive status creation is also the claim for this one-shot export.
    status = {"status": "running", "pid": os.getpid(), "stage": "verify_sources"}
    with status_path.open("x", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=True, indent=2)

    def progress(**changes):
        status.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = OUT / (".labels_status_" + uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=True, indent=2)
        os.replace(temporary, status_path)

    verified = {}
    source_stats = {}

    def verify(path, expected):
        path = Path(path)
        key = rel(path)
        fingerprint = (path.stat().st_size, path.stat().st_mtime_ns)
        digest = sha(path)
        require(digest == expected, "SHA mismatch: " + key)
        require(fingerprint == (path.stat().st_size, path.stat().st_mtime_ns),
                "Source changed during SHA verification: " + key)
        verified[key] = digest
        source_stats[key] = fingerprint
        return digest

    try:
        cache = ROOT / "data/gnn_validation_cache_parquet/config.json"
        evaluation = ROOT / "result/gnn_tuning_v1/evaluations/lr_only"
        eval_config, eval_metrics = read(evaluation / "config.json"), read(evaluation / "metrics.json")
        require(eval_metrics["status"] == "completed", "Frozen lr_only evaluation incomplete")
        verify(evaluation / "config.json", eval_metrics["config_sha256"])
        verify(cache, eval_config["cache_config_sha256"])
        frozen = normalized_hashes(read(cache)["hashes"])
        archive, train = ROOT / "data/gnn_v1_validation", ROOT / "data/gnn_v1_train"
        for path in (archive / "build.json", archive / "metadata/vocabulary.json",
                     archive / "metadata/history_identity_vocabulary.json"):
            verify(path, frozen[rel(path)])
        build = read(archive / "build.json")
        require(build["rows"] == TOTAL and build["train_rows"] == TRAIN_ROWS
                and build["validation_rows"] == VAL_ROWS and build["validation_positives"] == VAL_POS
                and build["end_exclusive"] == "2022-09-14T00:00:00", "Validation build scope mismatch")
        verify(archive / "config.json", build["config_sha256"])
        verify(archive / "stages.json", build["stages_sha256"])
        original_hashes = normalized_hashes(read(archive / "config.json")["input_hashes"])
        verify(train / "build.json", build["train_archive_sha256"])
        require(read(train / "build.json")["rows"] == TRAIN_ROWS, "Train build count mismatch")

        arrays = {}
        progress(stage="verify_history_arrays")
        for field in FIELDS:
            path = archive / "history_index" / (field + ".npy")
            original_path = train / "history_index" / (field + ".npy")
            verify(path, frozen[rel(path)])
            verify(original_path, original_hashes[rel(original_path)])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            original = np.load(original_path, mmap_mode="r", allow_pickle=False)
            require(array.shape == (TOTAL,) and original.shape == (TRAIN_ROWS,),
                    "History length mismatch: " + field)
            require(str(array.dtype) == EXPECTED_DTYPES[field] and array.dtype == original.dtype,
                    "History dtype mismatch: " + field)
            for lo in range(0, TRAIN_ROWS, BATCH):
                hi = min(lo + BATCH, TRAIN_ROWS)
                require(np.array_equal(array[lo:hi], original[lo:hi]),
                        "Train prefix mapping mismatch: " + field)
            arrays[field] = array
            progress(stage="verify_history_arrays", verified_field=field)

        expected_files = sorted(k for k in frozen if k.startswith(
            "data/gnn_v1_validation/supervision/targets/") and k.endswith(".parquet"))
        actual_files = sorted(rel(p) for p in (archive / "supervision/targets").rglob("*.parquet"))
        require(expected_files == actual_files and len(expected_files) == 102,
                "Frozen supervision file inventory mismatch")
        seen = np.zeros(TOTAL, dtype=np.bool_)
        positive_ids = []
        counts = {"train": {"rows": 0, "positives": 0, "files": 0},
                  "validation": {"rows": 0, "positives": 0, "files": 0}}
        per_date = {}
        file_audit = []
        for file_index, key in enumerate(expected_files):
            path = ROOT / key
            date = path.parent.name.removeprefix("transaction_date=")
            require("2022-09-01" <= date <= "2022-09-13", "Forbidden supervision date")
            split = "train" if date < "2022-09-11" else "validation"
            expected_day = int(np.datetime64(date, "us").astype("int64"))
            digest = verify(path, frozen[key])
            if split == "train":
                original_path = train / path.relative_to(archive)
                require(original_hashes[rel(original_path)] == digest,
                        "Original Train supervision identity mismatch")
                verify(original_path, digest)
            file_rows, file_positive = 0, 0
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=BATCH, columns=["edge_id", "label"], use_threads=False):
                require(batch["edge_id"].null_count == 0 and batch["label"].null_count == 0,
                        "Null supervision identity or label")
                ids, labels = batch["edge_id"].to_numpy(), batch["label"].to_numpy()
                require(np.issubdtype(ids.dtype, np.integer) and np.isin(labels, [0, 1]).all(),
                        "Invalid edge id or nonbinary label")
                lo, hi = (0, TRAIN_ROWS) if split == "train" else (TRAIN_ROWS, TOTAL)
                require(((ids >= lo) & (ids < hi)).all(), "Label outside its fixed split")
                require(len(np.unique(ids)) == len(ids) and not seen[ids].any(),
                        "Duplicate supervision edge")
                seen[ids] = True
                timestamps = arrays["timestamp"][ids]
                require(((timestamps >= expected_day)
                         & (timestamps < expected_day + 86_400_000_000)).all(),
                        "Supervision edge timestamp/date mismatch")
                selected = np.asarray(ids[labels == 1], dtype=np.int64)
                positive_ids.append(selected)
                file_rows += len(ids)
                file_positive += len(selected)
            counts[split]["rows"] += file_rows
            counts[split]["positives"] += file_positive
            counts[split]["files"] += 1
            day_counts = per_date.setdefault(date, {"rows": 0, "positives": 0})
            day_counts["rows"] += file_rows
            day_counts["positives"] += file_positive
            file_audit.append({"path": key, "sha256": digest, "rows": file_rows,
                               "positives": file_positive, "split": split,
                               "original_train_copy_sha_equal": split == "train"})
            progress(stage="read_supervision", files_completed=file_index + 1,
                     rows_completed=sum(v["rows"] for v in counts.values()))
        require(seen.all(), "Incomplete Train/Validation edge coverage")
        require(counts == {"train": {"rows": TRAIN_ROWS, "positives": TRAIN_POS, "files": 78},
                           "validation": {"rows": VAL_ROWS, "positives": VAL_POS, "files": 24}},
                "Supervision counts mismatch")
        ids = np.sort(np.concatenate(positive_ids))
        require(len(ids) == TRAIN_POS + VAL_POS and len(np.unique(ids)) == len(ids),
                "Positive identity count mismatch")
        data = {"edge_id": pa.array(ids, type=pa.int64())}
        for field in FIELDS:
            data["timestamp_us" if field == "timestamp" else field] = pa.array(arrays[field][ids])
        data["split"] = pa.array(np.where(ids < TRAIN_ROWS, "train", "validation"))
        table = pa.table(data).replace_schema_metadata({
            b"label_scope": b"Final synthetic labels: retrospective oracle, not point-in-time-known risk",
            b"history_scope": b"Train Sep01-10 and Validation Sep11-13 only; Test never read",
            b"amount_encoding": b"Frozen original int64 scaled amounts; no float conversion",
        })
        timestamps = table["timestamp_us"].to_numpy()
        require(((timestamps >= START) & (timestamps < END)).all(), "Positive timestamp outside scope")
        require(((timestamps[ids < TRAIN_ROWS] < CUT)).all()
                and ((timestamps[ids >= TRAIN_ROWS] >= CUT)).all(), "Positive split/time mismatch")
        require(table["paid_scaled"].type == pa.int64()
                and table["received_scaled"].type == pa.int64(), "Loss of integer amount precision")
        for key, fingerprint in source_stats.items():
            stat = (ROOT / key).stat()
            require((stat.st_size, stat.st_mtime_ns) == fingerprint, "Source changed during export: " + key)
        disk_check()
        temporary = OUT / (".positive_edges_" + uuid.uuid4().hex + ".parquet")
        with temporary.open("xb") as stream:
            pq.write_table(table, stream, compression="zstd")
        reread = pq.ParquetFile(temporary).read(use_threads=False)
        require(reread.equals(table, check_metadata=True), "Parquet roundtrip changed positive edges")
        require(not destination.exists() and not manifest_path.exists(), "Output appeared during export")
        temporary.rename(destination)
        manifest = {
            "status": "verified", "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Only frozen Train 2022-09-01..10 and Validation 2022-09-11..13; Test never read",
            "label_semantics": "Oracle final synthetic labels for retrospective chain audit only; label confirmation times unavailable, not point-in-time-known or deployable risk features",
            "counts": counts, "per_date": per_date,
            "train_prefix_exact_fields": list(FIELDS), "full_unique_edge_coverage": True,
            "original_train_supervision_copy_sha_equal": True,
            "positive_edges": {"path": rel(destination), "rows": len(ids), "sha256": sha(destination),
                               "edge_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                               "columns": {f.name: str(f.type) for f in table.schema},
                               "amount_precision": "original int64 scaled amounts, verified lossless Parquet roundtrip"},
            "source_sha256": verified, "supervision_files": file_audit,
            "script_sha256": sha(Path(__file__)), "runtime_seconds": time.monotonic() - began,
        }
        with manifest_path.open("x", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
            f.write("\n")
        progress(status="completed", stage="published", positive_edges=len(ids),
                 manifest_sha256=sha(manifest_path), elapsed_seconds=time.monotonic() - began)
        print(json.dumps({"status": "completed", "counts": counts,
                          "positive_edges": manifest["positive_edges"],
                          "manifest": rel(manifest_path), "seconds": time.monotonic() - began},
                         ensure_ascii=True))
    except BaseException as error:
        progress(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - began)
        raise


if __name__ == "__main__":
    main()
