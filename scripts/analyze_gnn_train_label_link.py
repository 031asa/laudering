"""Reproduce the fixed-Train, retrospective label-link error analysis.

This is an analysis entry point, not a training feature pipeline. Historical
counts use only 2022-09-01..10 labels from the frozen Train archive. Validation
labels are used only to select FN/TP/negative-control/FP groups. There are no
label-confirmation timestamps, so these counts must not be described as risk
information known when a transaction occurred. Test is never read.

Run from any working directory with the project's existing Python environment:
    .venv-gnn-gpu/Scripts/python.exe scripts/analyze_gnn_train_label_link.py

Requires the previously generated inspection_sample.parquet. Recomputes and
overwrites only result/gnn_error_analysis_v1/train_label_link.json and .parquet.
The implementation below preserves the diagnostics and validations used for
the original analysis; saving this script did not rerun that analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TRAIN_ROWS = 20_318_992
TRAIN_POSITIVES = 17_794
THRESHOLD = 0.9529672861099243


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=True), flush=True)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def proportion(values):
    return {
        "yes": int(values.sum()),
        "denominator": len(values),
        "rate": float(values.mean()) if len(values) else None,
    }


def describe_count(values):
    return {
        "sum": int(values.sum()),
        "zero": int((values == 0).sum()),
        "median": float(values.median()) if len(values) else None,
        "p90": float(values.quantile(0.9)) if len(values) else None,
        "max": int(values.max()) if len(values) else None,
    }


def summarize(frame):
    result = {
        "n": len(frame),
        "unique_payers": frame.src.nunique(),
        "unique_payees": frame.dst.nunique(),
    }
    for endpoint in ["payer", "payee"]:
        for direction in ["in", "out"]:
            prefix = f"{endpoint}_train_{direction}"
            result[f"{prefix}_has_positive"] = proportion(
                frame[f"{prefix}_positive"] > 0
            )
            result[f"{prefix}_positive_count"] = describe_count(
                frame[f"{prefix}_positive"]
            )
            result[f"{prefix}_all_count"] = describe_count(frame[f"{prefix}_all"])
        key = f"{endpoint}_train_any_positive"
        result[key] = proportion(frame[key])
    for key in [
        "either_endpoint_train_any_positive",
        "both_endpoints_train_any_positive",
    ]:
        result[key] = proportion(frame[key])
    result["receiver_train_negative_in"] = describe_count(
        frame.receiver_train_negative_in
    )
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = root / "data/gnn_v1_train"
    validation = root / "data/gnn_v1_validation"
    output = root / "result/gnn_error_analysis_v1"
    build = json.loads((train / "build.json").read_text(encoding="utf-8"))
    assert build["rows"] == TRAIN_ROWS
    assert build["end_exclusive"] == "2022-09-11T00:00:00"
    validation_build = json.loads(
        (validation / "build.json").read_text(encoding="utf-8")
    )
    assert validation_build["train_rows"] == TRAIN_ROWS
    assert validation_build["end_exclusive"] == "2022-09-14T00:00:00"
    node_count = build["node_count"]
    history = validation / "history_index"
    original_history = train / "history_index"
    src = np.load(history / "src.npy", mmap_mode="r")
    dst = np.load(history / "dst.npy", mmap_mode="r")
    timestamps = np.load(history / "timestamp.npy", mmap_mode="r")

    # Compare the whole original Train prefix, with bounded temporary arrays.
    for name, array in [("src", src), ("dst", dst), ("timestamp", timestamps)]:
        original = np.load(original_history / f"{name}.npy", mmap_mode="r")
        assert len(original) == TRAIN_ROWS
        for start in range(0, TRAIN_ROWS, 500_000):
            stop = min(TRAIN_ROWS, start + 500_000)
            assert np.array_equal(original[start:stop], array[start:stop]), (
                name,
                start,
            )
    emit({"phase": "identity_prefix_verified", "rows": TRAIN_ROWS})

    counts = {
        key: np.zeros(node_count, dtype=np.int64)
        for key in ["out_all", "in_all", "out_positive", "in_positive"]
    }
    seen = np.zeros(TRAIN_ROWS, dtype=bool)
    labels = np.zeros(TRAIN_ROWS, dtype=np.int8)
    file_audits = []
    samples = []
    for day in range(1, 11):
        date = f"2022-09-{day:02d}"
        partition = f"transaction_date={date}"
        for path in sorted((train / "supervision/targets" / partition).glob("*.parquet")):
            relative = path.relative_to(train)
            archive_copy = validation / relative
            digest = sha256(path)
            assert archive_copy.is_file() and sha256(archive_copy) == digest, str(relative)
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=100_000,
                columns=["edge_id", "label", "split", "transaction_id"],
            ):
                data = batch.to_pydict()
                ids = np.array(data["edge_id"], dtype=np.int64)
                y = np.array(data["label"], dtype=np.int8)
                assert np.all((ids >= 0) & (ids < TRAIN_ROWS))
                assert len(np.unique(ids)) == len(ids) and not seen[ids].any()
                assert np.isin(y, [0, 1]).all()
                # Original split names predate this GNN's complete Train range.
                expected_split = {"discovery"} if day <= 7 else {"internal_validation"}
                assert set(data["split"]) == expected_split
                seen[ids] = True
                labels[ids] = y
                payer = src[ids]
                payee = dst[ids]
                np.add.at(counts["out_all"], payer, 1)
                np.add.at(counts["in_all"], payee, 1)
                np.add.at(counts["out_positive"], payer[y == 1], 1)
                np.add.at(counts["in_positive"], payee[y == 1], 1)
                minimum = np.datetime64(date, "us").astype(np.int64)
                maximum = minimum + 86_400_000_000
                assert np.all((timestamps[ids] >= minimum) & (timestamps[ids] < maximum))
                if not any(sample["date"] == date for sample in samples) and y.any():
                    index = int(np.flatnonzero(y == 1)[0])
                    samples.append(
                        {
                            "date": date,
                            "edge_id": int(ids[index]),
                            "label": 1,
                            "transaction_id": data["transaction_id"][index],
                            "src": int(payer[index]),
                            "dst": int(payee[index]),
                            "timestamp_us": int(timestamps[ids[index]]),
                            "source_file": str(relative),
                        }
                    )
            file_audits.append(
                {
                    "path": str(relative),
                    "rows": parquet.metadata.num_rows,
                    "sha256_original_and_copy": digest,
                }
            )
        emit(
            {
                "phase": "day_done",
                "day": day,
                "covered": int(seen.sum()),
                "positives": int(labels.sum()),
            }
        )
    assert seen.all() and int(labels.sum()) == TRAIN_POSITIVES

    # One positive per Train day: cross-check supervision against graph identity.
    for sample in samples:
        path = (
            train
            / "graph/edges"
            / f"transaction_date={sample['date']}"
            / Path(sample["source_file"]).name
        )
        found = False
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=100_000,
            columns=["edge_id", "transaction_id", "src", "dst", "timestamp"],
        ):
            frame = batch.to_pandas()
            selected = frame.loc[frame.edge_id == sample["edge_id"]]
            if len(selected):
                assert len(selected) == 1
                row = selected.iloc[0]
                assert str(row.transaction_id) == sample["transaction_id"]
                assert int(row.src) == sample["src"] and int(row.dst) == sample["dst"]
                assert int(row.timestamp.value // 1000) == sample["timestamp_us"]
                found = True
                break
        assert found, sample
    emit({"phase": "sample_graph_identity_verified", "samples": len(samples)})

    columns = [
        "edge_id", "transaction_id", "group", "fn", "tp", "fp",
        "uniform_negative_control", "src", "dst", "from_bank", "from_account",
        "to_bank", "to_account", "payment_format", "score_lr_only", "timestamp",
    ]
    sample_frame = pq.read_table(
        output / "inspection_sample.parquet", columns=columns
    ).to_pandas()
    assert len(sample_frame.loc[sample_frame.fn]) == 2905
    assert len(sample_frame.loc[sample_frame.tp]) == 3515
    assert sample_frame.uniform_negative_control.sum() == 100_000
    assert len(sample_frame.loc[sample_frame.fp]) == 810
    target_ids = sample_frame.edge_id.to_numpy()
    assert np.array_equal(src[target_ids], sample_frame.src.to_numpy())
    assert np.array_equal(dst[target_ids], sample_frame.dst.to_numpy())
    assert (
        sample_frame.timestamp.to_numpy().astype("datetime64[us]").astype(np.int64)
        > timestamps[:TRAIN_ROWS].max()
    ).all()
    for endpoint, column in [("payer", "src"), ("payee", "dst")]:
        for kind, array in counts.items():
            sample_frame[f"{endpoint}_train_{kind}"] = array[sample_frame[column].to_numpy()]
        sample_frame[f"{endpoint}_train_any_positive"] = (
            sample_frame[f"{endpoint}_train_out_positive"]
            + sample_frame[f"{endpoint}_train_in_positive"]
        ) > 0
    sample_frame["either_endpoint_train_any_positive"] = (
        sample_frame.payer_train_any_positive | sample_frame.payee_train_any_positive
    )
    sample_frame["both_endpoints_train_any_positive"] = (
        sample_frame.payer_train_any_positive & sample_frame.payee_train_any_positive
    )
    sample_frame["receiver_train_negative_in"] = (
        sample_frame.payee_train_in_all - sample_frame.payee_train_in_positive
    )
    sample_frame["receiver_train_negative_out"] = (
        sample_frame.payee_train_out_all - sample_frame.payee_train_out_positive
    )
    pq.write_table(
        pa.Table.from_pandas(sample_frame, preserve_index=False),
        output / "train_label_link.parquet",
        compression="zstd",
    )

    groups = {
        "FN": sample_frame.fn,
        "TP": sample_frame.tp,
        "uniform_negative_control": sample_frame.uniform_negative_control,
        "FP": sample_frame.fp,
    }
    slices = {
        "all": np.ones(len(sample_frame), dtype=bool),
        "ACH": sample_frame.payment_format == "ACH",
        "non_ACH": sample_frame.payment_format != "ACH",
        "payer_bank70": sample_frame.from_bank == "70",
        "payee_bank70": sample_frame.to_bank == "70",
        "either_bank70": (sample_frame.from_bank == "70") | (sample_frame.to_bank == "70"),
        "payer_bank70_ACH": (sample_frame.from_bank == "70") & (sample_frame.payment_format == "ACH"),
        "payer_bank70_non_ACH": (sample_frame.from_bank == "70") & (sample_frame.payment_format != "ACH"),
    }
    results = {
        name: {
            group: summarize(sample_frame.loc[mask & group_mask])
            for group, group_mask in groups.items()
        }
        for name, mask in slices.items()
    }
    receivers = []
    for node, frame in sample_frame.loc[sample_frame.fn].groupby("dst", sort=False):
        row = frame.iloc[0]
        node = int(node)
        receivers.append(
            {
                "node_id": node,
                "to_bank": str(row.to_bank),
                "to_account": str(row.to_account),
                "fn_transactions": len(frame),
                "fn_ach": int((frame.payment_format == "ACH").sum()),
                "fn_from_bank70": int((frame.from_bank == "70").sum()),
                "fn_min_score": float(frame.score_lr_only.min()),
                "fn_median_score": float(frame.score_lr_only.median()),
                "train_in_all": int(counts["in_all"][node]),
                "train_in_positive": int(counts["in_positive"][node]),
                "train_in_negative": int(counts["in_all"][node] - counts["in_positive"][node]),
                "train_out_all": int(counts["out_all"][node]),
                "train_out_positive": int(counts["out_positive"][node]),
                "train_out_negative": int(counts["out_all"][node] - counts["out_positive"][node]),
            }
        )
    receivers.sort(key=lambda value: (-value["fn_transactions"], value["node_id"]))
    result = {
        "scope": (
            "Retrospective association with fixed Train transaction labels only. "
            "Train dates 2022-09-01 through 10. Validation labels used only for error "
            "grouping, never for historical label counts. No label-confirmation "
            "timestamps available; this is not point-in-time-known risk information "
            "and does not establish deployable feature gains. Test not read."
        ),
        "identity_audit": {
            "train_rows": TRAIN_ROWS,
            "train_positive": TRAIN_POSITIVES,
            "node_count": node_count,
            "supervision_files_identical_to_validation_prefix": len(file_audits),
            "complete_unique_train_edge_coverage": bool(seen.all()),
            "original_train_src_dst_timestamp_prefix_exact_match": True,
            "validation_target_src_dst_match": True,
            "all_validation_targets_strictly_after_train_history": True,
            "graph_identity_spotchecks": samples,
            "supervision_file_sha256": file_audits,
        },
        "definitions": {
            "in_positive": "Account was payee in a Train transaction with synthetic label 1",
            "out_positive": "Account was payer in a Train transaction with synthetic label 1",
            "any_positive": "in_positive + out_positive > 0; self-transfer counts in both roles, not two distinct events",
            "counts": "Raw unsmoothed counts; no fitted rate, no target encoding, no score model trained",
            "controls": "All 6420 validation positives; all 810 FP at saved threshold; 100000 uniformly sampled negatives. Negative control and FP overlap in 15 rows.",
            "FN_threshold": THRESHOLD,
            "bank70": "Actual bank identity string 70, never a claim that every account of a bank has the same risk",
            "legacy_split_names": "Original supervision split names discovery for Sep1-7 and internal_validation for Sep8-10; all dates are within this GNN complete Train definition",
        },
        "groups_and_slices": results,
        "fn_receiver_profiles": receivers,
        "saved_parquet": "result/gnn_error_analysis_v1/train_label_link.parquet",
    }

    # These account-level summaries avoid repeatedly adding one account's Train
    # counts for each Validation transaction in which it appears.
    bank70 = sample_frame.loc[
        sample_frame.from_bank == "70",
        [
            "src", "from_bank", "from_account", "payer_train_out_all",
            "payer_train_out_positive", "payer_train_in_all", "payer_train_in_positive",
        ],
    ].drop_duplicates().sort_values("payer_train_out_all", ascending=False)
    result["bank70_payer_train_profiles"] = bank70.to_dict("records")
    result["bank70_payer_train_aggregate"] = {
        "accounts": len(bank70),
        "out_transactions": int(bank70.payer_train_out_all.sum()),
        "out_positive": int(bank70.payer_train_out_positive.sum()),
        "out_negative": int((bank70.payer_train_out_all - bank70.payer_train_out_positive).sum()),
        "in_transactions": int(bank70.payer_train_in_all.sum()),
        "in_positive": int(bank70.payer_train_in_positive.sum()),
        "note": "Same account counts deduplicated, not transaction-weighted replication. Presence of prior positive does not distinguish FN from sampled normal transfers from these accounts.",
    }
    result["unique_account_linkage"] = {
        group: {
            endpoint: {
                "unique_accounts": int(sample_frame.loc[mask, column].nunique()),
                "unique_with_train_positive": int(
                    sample_frame.loc[
                        mask & sample_frame[f"{endpoint}_train_any_positive"], column
                    ].nunique()
                ),
            }
            for endpoint, column in [("payer", "src"), ("payee", "dst")]
        }
        for group, mask in groups.items()
    }
    result = clean(result)
    (output / "train_label_link.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    emit(
        {
            "done": True,
            "out": str(output / "train_label_link.json"),
            "groups": {
                group: {
                    key: value
                    for key, value in summary.items()
                    if key in [
                        "n", "payer_train_any_positive", "payee_train_any_positive",
                        "either_endpoint_train_any_positive", "both_endpoints_train_any_positive",
                    ]
                }
                for group, summary in result["groups_and_slices"]["all"].items()
            },
            "fn_receiver_top12": receivers[:12],
        }
    )


if __name__ == "__main__":
    main()
