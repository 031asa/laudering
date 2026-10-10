"""Export labeled discovery transactions and complete, deliberately selected account cases.

This is retrospective EDA only. Case selection uses discovery labels and must not
be used as a prediction feature or as evidence that all activity is illicit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/discovery_cases"
BASE_COLUMNS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
                "amount_received", "receiving_currency", "amount_paid", "payment_currency",
                "payment_format", "is_laundering", "transaction_id", "source_file", "source_record_number"]


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(name, rows):
    assert rows
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def serialize(raw, date):
    row = dict(raw)
    row["timestamp"] = raw["timestamp"].isoformat(timespec="seconds")
    row["transaction_date"] = date
    row["amount_paid"] = format(raw["amount_paid"], "f")
    row["amount_received"] = format(raw["amount_received"], "f")
    row["source_record_number"] = str(raw["source_record_number"])
    row["from_account_key"] = raw["from_bank"] + "|" + raw["from_account"]
    row["to_account_key"] = raw["to_bank"] + "|" + raw["to_account"]
    return row


def ordered(rows):
    return sorted(rows, key=lambda row: (row["timestamp"], int(row["source_record_number"])))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config/dataset_split.json").read_text(encoding="utf-8"))
    source = ROOT / config["source"]
    assert hashlib.sha256((source / "_manifest.json").read_bytes()).hexdigest() == config["source_manifest_sha256"]
    discovery = next(split for split in config["splits"] if split["name"] == "discovery")
    dates = discovery["dates"]
    assert dates == [f"2022-09-{day:02}" for day in range(1, 8)]
    files = [(date, path) for date in dates for path in sorted((source / f"transaction_date={date}").glob("*.parquet"))]
    assert all(any(d == date for d, _ in files) for date in dates)

    accounts = pd.concat([pq.ParquetFile(path).read(columns=["bank_id", "account_number", "entity_id", "entity_name"]).to_pandas()
                          for path in sorted((ROOT / "data/silver/accounts").glob("*.parquet"))], ignore_index=True)
    account_keys = pd.Index(accounts.bank_id + "|" + accounts.account_number)
    assert account_keys.is_unique
    size = len(account_keys)
    counts = np.zeros(size, dtype=np.uint32)
    all_in = np.zeros(size, dtype=np.uint32)
    all_out = np.zeros(size, dtype=np.uint32)
    pos_counts = np.zeros(size, dtype=np.uint32)
    pos_in = np.zeros(size, dtype=np.uint32)
    pos_out = np.zeros(size, dtype=np.uint32)
    non_ach = np.zeros(size, dtype=np.uint32)
    bitcoin = np.zeros(size, dtype=np.uint32)
    positive_rows, positive_raw = [], []
    daily_stats = {date: Counter() for date in dates}
    format_stats = defaultdict(Counter)
    first_schema = None

    for date, path in files:
        parquet = pq.ParquetFile(path)
        first_schema = first_schema or parquet.schema_arrow
        for batch in parquet.iter_batches(batch_size=150000, columns=BASE_COLUMNS):
            frame = batch.select(["timestamp", "from_bank", "from_account", "to_bank", "to_account", "is_laundering", "payment_format"]).to_pandas()
            assert frame.timestamp.dt.strftime("%Y-%m-%d").eq(date).all()
            y = frame.is_laundering.to_numpy()
            assert np.isin(y, [0, 1]).all()
            src = account_keys.get_indexer(frame.from_bank + "|" + frame.from_account)
            dst = account_keys.get_indexer(frame.to_bank + "|" + frame.to_account)
            assert (src >= 0).all() and (dst >= 0).all()
            distinct = src != dst
            np.add.at(all_out, src, 1)
            np.add.at(all_in, dst, 1)
            np.add.at(counts, src, 1)
            np.add.at(counts, dst[distinct], 1)
            positive = y == 1
            np.add.at(pos_out, src[positive], 1)
            np.add.at(pos_in, dst[positive], 1)
            np.add.at(pos_counts, src[positive], 1)
            np.add.at(pos_counts, dst[positive & distinct], 1)
            mask = positive & frame.payment_format.ne("ACH").to_numpy()
            np.add.at(non_ach, src[mask], 1)
            np.add.at(non_ach, dst[mask & distinct], 1)
            raw_positive = pa.Table.from_batches([batch]).filter(pa.array(positive)).to_pylist()
            for raw in raw_positive:
                row = serialize(raw, date)
                positive_rows.append(row)
                positive_raw.append(raw)
            daily_stats[date]["total"] += len(frame)
            daily_stats[date]["positive"] += int(positive.sum())
            grouped = frame.groupby("payment_format").is_laundering.agg(["size", "sum"])
            for fmt, stat in grouped.iterrows():
                format_stats[fmt]["total"] += int(stat["size"])
                format_stats[fmt]["positive"] += int(stat["sum"])
        print(f"pass1 {date} rows={daily_stats[date]['total']:,} positives={daily_stats[date]['positive']:,}", flush=True)

    total = sum(stat["total"] for stat in daily_stats.values())
    positive_total = len(positive_rows)
    assert total == discovery["rows"] == 14935508
    assert positive_total == discovery["positive_rows"] == 11360
    assert len({row["transaction_id"] for row in positive_rows}) == positive_total
    assert len({row["source_record_number"] for row in positive_rows}) == positive_total
    currencies = defaultdict(set)
    positive_counterparties = defaultdict(set)
    positive_two_way = defaultdict(list)
    for row in positive_rows:
        src, dst = account_keys.get_indexer([row["from_account_key"], row["to_account_key"]])
        currencies[int(src)].add(row["payment_currency"])
        currencies[int(dst)].add(row["receiving_currency"])
        positive_counterparties[int(src)].add(int(dst))
        positive_counterparties[int(dst)].add(int(src))
        if row["payment_currency"] == "Bitcoin":
            bitcoin[src] += 1
        if dst != src and row["receiving_currency"] == "Bitcoin":
            bitcoin[dst] += 1
        positive_two_way[int(src)].append((row["timestamp"], "OUT"))
        if dst != src:
            positive_two_way[int(dst)].append((row["timestamp"], "IN"))

    eligible = np.flatnonzero((pos_counts > 0) & (counts <= 300))
    chosen = []
    choices = []

    def choose(label, reason, score, predicate=lambda idx: True):
        candidates = [int(idx) for idx in eligible if int(idx) not in chosen and predicate(int(idx))]
        assert candidates, label
        best = max(candidates, key=lambda idx: (score(idx), -idx))
        chosen.append(best)
        choices.append((label, reason, best))

    choose("C01", "在总参与流水不超过300条的正例关联账户中，按正例收款次数最高选择。", lambda i: int(pos_in[i]))
    choose("C02", "在未选且总参与流水不超过300条的账户中，按正例付款次数最高选择。", lambda i: int(pos_out[i]))
    choose("C03", "在未选账户中，按正例收付较小一侧次数最高选择，观察双向活动。", lambda i: (min(int(pos_in[i]), int(pos_out[i])), int(pos_counts[i])))
    choose("C04", "在未选账户中，按非ACH正例参与次数最高选择，避免案例只覆盖ACH。", lambda i: int(non_ach[i]), lambda i: non_ach[i] > 0)
    choose("C05", "在未选且有Bitcoin正例的账户中，按Bitcoin正例参与次数最高选择。", lambda i: int(bitcoin[i]), lambda i: bitcoin[i] > 0)
    choose("C06", "在未选账户中，优先选择正例账户侧币种种类多的账户，并按双向程度打破并列。", lambda i: (len(currencies[i]), min(int(pos_in[i]), int(pos_out[i])), int(pos_counts[i])))
    choose("C07", "在未选且至少有2条正例、发现期同时收付款的账户中，按标签0流水条数最高选择，查看完整背景。", lambda i: int(counts[i] - pos_counts[i]), lambda i: pos_counts[i] >= 2 and all_in[i] > 0 and all_out[i] > 0)
    choose("C08", "在未选且正例有收有付的账户中，按正例外部对手数最多选择，观察网络连接。", lambda i: (len(positive_counterparties[i] - {i}), min(int(pos_in[i]), int(pos_out[i]))), lambda i: pos_in[i] > 0 and pos_out[i] > 0)
    cases = []
    index_to_case = {}
    for case_id, reason, idx in choices:
        metadata = accounts.iloc[idx]
        case = dict(case_id=case_id, account_key=str(account_keys[idx]), bank_id=str(metadata.bank_id),
                    account=str(metadata.account_number), entity_id=str(metadata.entity_id), entity_name=str(metadata.entity_name),
                    reason=reason, total_count=int(counts[idx]), positive_count=int(pos_counts[idx]),
                    normal_count=int(counts[idx] - pos_counts[idx]), in_count=int(all_in[idx]), out_count=int(all_out[idx]),
                    positive_in_count=int(pos_in[idx]), positive_out_count=int(pos_out[idx]), rows=[])
        cases.append(case)
        index_to_case[idx] = case
    print("chosen " + ", ".join(f"{case['case_id']}={case['account_key']} ({case['total_count']} rows)" for case in cases), flush=True)
    membership = np.zeros(size, dtype=bool)
    membership[chosen] = True
    second_rows = second_positive = 0
    for date, path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=150000, columns=BASE_COLUMNS):
            frame = batch.select(["from_bank", "from_account", "to_bank", "to_account", "is_laundering"]).to_pandas()
            src = account_keys.get_indexer(frame.from_bank + "|" + frame.from_account)
            dst = account_keys.get_indexer(frame.to_bank + "|" + frame.to_account)
            assert (src >= 0).all() and (dst >= 0).all()
            second_rows += len(frame)
            second_positive += int(frame.is_laundering.sum())
            matched = membership[src] | membership[dst]
            selected_rows = pa.Table.from_batches([batch]).filter(pa.array(matched)).to_pylist()
            for raw in selected_rows:
                row = serialize(raw, date)
                s, d = account_keys.get_indexer([row["from_account_key"], row["to_account_key"]])
                for idx in {int(s), int(d)}:
                    if idx not in index_to_case:
                        continue
                    is_self = s == d
                    outgoing = idx == s
                    case_row = dict(row, case_id=index_to_case[idx]["case_id"], observed_account_key=str(account_keys[idx]),
                                    direction="SELF" if is_self else "OUT" if outgoing else "IN",
                                    counterparty_key=row["to_account_key"] if outgoing else row["from_account_key"],
                                    account_amount=row["amount_paid"] if outgoing else row["amount_received"],
                                    account_currency=row["payment_currency"] if outgoing else row["receiving_currency"])
                    index_to_case[idx]["rows"].append(case_row)
        print(f"pass2 {date} scanned={second_rows:,}", flush=True)
    assert (second_rows, second_positive) == (total, positive_total)

    clues = []
    for case in cases:
        rows = case["rows"] = ordered(case["rows"])
        assert len(rows) == case["total_count"]
        assert len({r["transaction_id"] for r in rows}) == len(rows)
        assert sum(r["is_laundering"] for r in rows) == case["positive_count"]
        assert sum(r["direction"] in ("IN", "SELF") for r in rows) == case["in_count"]
        assert sum(r["direction"] in ("OUT", "SELF") for r in rows) == case["out_count"]
        case["currencies"] = sorted({r["account_currency"] for r in rows})
        case["payment_formats"] = dict(Counter(r["payment_format"] for r in rows))
        case["dates"] = sorted({r["transaction_date"] for r in rows})
        case["self_count"] = sum(r["direction"] == "SELF" for r in rows)
        for row in rows:
            if row["direction"] != "OUT":
                continue
            t = datetime.fromisoformat(row["timestamp"])
            history = [r for r in rows if r["direction"] == "IN" and r["account_currency"] == row["account_currency"]
                       and t - timedelta(hours=24) <= datetime.fromisoformat(r["timestamp"]) < t]
            inflow = sum((Decimal(r["account_amount"]) for r in history), Decimal(0))
            if len(history) < 2 or inflow <= 0:
                continue
            ratio = Decimal(row["account_amount"]) / inflow
            if not Decimal("0.8") <= ratio <= Decimal("1.2"):
                continue
            intervening_out = [r for r in rows if r["direction"] == "OUT" and r["account_currency"] == row["account_currency"]
                               and datetime.fromisoformat(history[0]["timestamp"]) <= datetime.fromisoformat(r["timestamp"]) < t]
            clues.append(dict(case_id=case["case_id"], account_key=case["account_key"], timestamp=row["timestamp"],
                              transaction_id=row["transaction_id"], source_record_number=row["source_record_number"],
                              payment_format=row["payment_format"], is_laundering=row["is_laundering"], currency=row["account_currency"],
                              outgoing_amount=row["account_amount"], historical_in_amount_24h=format(inflow, "f"),
                              outgoing_to_incoming_ratio=float(ratio), incoming_count=len(history),
                              incoming_counterparties=len({r["counterparty_key"] for r in history}),
                              first_incoming_timestamp=history[0]["timestamp"], latest_incoming_timestamp=history[-1]["timestamp"],
                              span_hours=(t - datetime.fromisoformat(history[0]["timestamp"])).total_seconds()/3600,
                              history_window_complete_24h=t >= datetime.fromisoformat(dates[0]) + timedelta(hours=24),
                              observed_history_hours=min(24.0, max(0.0, (t - datetime.fromisoformat(dates[0])).total_seconds()/3600)),
                              intervening_outgoing_count=len(intervening_out),
                              intervening_outgoing_amount=format(sum((Decimal(r["account_amount"]) for r in intervening_out), Decimal(0)), "f"),
                              incoming_transaction_ids=[r["transaction_id"] for r in history],
                              warning="仅为发现期同币种金额接近候选，80%–120%是本次人工查看筛选带；不证明这些入账实际支付了该笔出账，也不代表诈骗证据。"))
    clues.sort(key=lambda c: (-int(c["span_hours"] > 1), c["intervening_outgoing_count"], abs(c["outgoing_to_incoming_ratio"]-1), -c["incoming_count"]))

    positive_rows = ordered(positive_rows)
    daily = [dict(date=date, total=stat["total"], positive=stat["positive"], normal=stat["total"]-stat["positive"],
                  positive_rate=stat["positive"]/stat["total"]) for date, stat in daily_stats.items()]
    formats = [dict(payment_format=fmt, total=stat["total"], positive=stat["positive"], normal=stat["total"]-stat["positive"],
                    positive_rate=stat["positive"]/stat["total"], lift=(stat["positive"]/stat["total"])/(positive_total/total),
                    share_all=stat["total"]/total, share_positive=stat["positive"]/positive_total)
               for fmt, stat in sorted(format_stats.items(), key=lambda pair: -pair[1]["positive"])]
    case_rows = [row for case in cases for row in case["rows"]]
    unique_case_ids = len({row["transaction_id"] for row in case_rows})
    source_info = dict(dataset="IBM AML HI-Medium（合成交易数据）", split="discovery", dates=dates,
                       transaction_source=config["source"], manifest_sha256=config["source_manifest_sha256"],
                       label_definition="is_laundering=1 表示合成数据的洗钱交易标签；数据没有单独诈骗标签，不能称为已确认诈骗交易。",
                       selection="全量标签1交易完整导出；8个账户为确定性目的抽样，均选择总参与流水<=300条的账户。案例不是随机样本，也不代表全体账户。",
                       case_history="每个选中账户收录整个发现期2022-09-01至09-07所有参与交易（含标签0与1）；SELF仅一行。同一交易若连接两个案例账户可在合表出现两行。",
                       direction="OUT=付款方；IN=收款方；SELF=两侧为同一银行+账户。SELF的account_amount/currency展示付款侧，原始两侧金额币种均保留。",
                       amount="金额以6位小数精确字符串保存，不跨币种相加，不做汇率换算。", 
                       timestamp="无时区本地时间字符串，原数据仅分钟精度。同分钟按来源序号便于稳定展示，不能当作真实先后顺序。候选历史严格排除同分钟。",
                       metadata="entity_id/name来自静态账户元数据，仅用于回溯查看；本导出不判断其历史时点可用性。",
                       no_leakage="仅扫描发现期交易分区；未读取验证期、测试期、尾段交易或Patterns说明。案例选择使用标签，仅供回溯EDA，不能成为模型输入或评价样本。",
                       selection_clue="24h金额接近候选使用当前付款与严格更早同币种入账，至少2笔入账，比例80%–120%；此阈值是查看筛选条件，不是已验证规则。",
                       history_coverage="发现期从2022-09-01 00:00开始，首日不足24h历史；候选附history_window_complete_24h和observed_history_hours，后者最多24h。",
                       files=[str(path.relative_to(ROOT)).replace("\\", "/") for _, path in files])
    counts_summary = dict(scanned_rows=total, positive_rows=positive_total, normal_rows=total-positive_total,
                          active_accounts=int((counts>0).sum()), positive_associated_accounts=int((pos_counts>0).sum()),
                          eligible_case_accounts=len(eligible), positive_associated_accounts_over_300_rows=int(((pos_counts>0)&(counts>300)).sum()),
                          selected_accounts=len(cases), selected_account_rows=len(case_rows), unique_case_transactions=unique_case_ids,
                          candidate_clues=len(clues))
    verification = dict(status="passed", discovery_only=True, expected_scanned_rows=discovery["rows"], actual_scanned_rows=total,
                        expected_positive_rows=discovery["positive_rows"], actual_positive_rows=positive_total,
                        positive_transaction_id_duplicates=0, positive_source_record_number_duplicates=0,
                        second_scan_rows=second_rows, second_scan_positives=second_positive,
                        all_selected_accounts_complete=True, per_case_transaction_id_duplicates=0,
                        case_counts_reconciled=True, daily_rows_reconcile=sum(d["total"] for d in daily)==total,
                        formats_positive_reconcile=sum(f["positive"] for f in formats)==positive_total,
                        elapsed_seconds=round(time.perf_counter()-started,2))
    payload = dict(source=source_info, counts=counts_summary, daily=daily, formats=formats, positives=positive_rows, cases=cases, candidate_clues=clues)
    save_json("payload.json", payload)
    save_json("summary.json", dict(source=source_info, counts=counts_summary, daily=daily, formats=formats,
                                   cases=[{key: value for key, value in case.items() if key != "rows"} for case in cases], candidate_clues=clues))
    save_json("verification.json", verification)
    save_csv("发现期_全量标签1交易.csv", positive_rows)
    save_csv("筛选样例_账户完整发现期流水.csv", case_rows)
    save_csv("发现期_支付方式统计.csv", formats)
    save_csv("发现期_每日统计.csv", daily)
    raw_ordered = sorted(positive_raw, key=lambda row: (row["timestamp"], row["source_record_number"]))
    pq.write_table(pa.Table.from_pylist(raw_ordered, schema=first_schema).append_column("transaction_date", pa.array([r["timestamp"].date() for r in raw_ordered])),
                   OUT / "发现期_全量标签1交易.parquet", compression="snappy")
    with (OUT / "发现期_全量标签1交易.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    parquet_rows = pq.ParquetFile(OUT / "发现期_全量标签1交易.parquet").read().to_pylist()
    assert len(csv_rows) == len(parquet_rows) == len(positive_rows)
    for csv_row, parquet_row, row in zip(csv_rows, parquet_rows, positive_rows):
        assert csv_row["transaction_id"] == parquet_row["transaction_id"] == row["transaction_id"]
        assert csv_row["timestamp"] == parquet_row["timestamp"].isoformat(timespec="seconds") == row["timestamp"]
        for amount in ["amount_paid", "amount_received"]:
            assert csv_row[amount] == row[amount]
            assert parquet_row[amount] == Decimal(row[amount])
    verification.update(positive_csv_roundtrip_exact=True, positive_parquet_roundtrip_exact=True, positive_amount_precision_preserved=True)
    save_json("verification.json", verification)
    print(json.dumps(dict(counts=counts_summary, verification=verification), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
