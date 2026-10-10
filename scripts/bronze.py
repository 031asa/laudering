"""Run conversion or independent verification from any working directory."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.bronze import TABLES, convert, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["convert", "verify"])
    parser.add_argument("--table", choices=["all", *TABLES], default="all")
    args = parser.parse_args()
    downloaded = json.loads((ROOT / "data/raw/download_manifest.json").read_text(encoding="utf-8"))
    hashes = {item["name"]: item["sha256"] for item in downloaded["files"]}
    summary = {}
    for kind in TABLES if args.table == "all" else [args.table]:
        source = ROOT / "data/raw" / TABLES[kind][0]
        destination = ROOT / "data/bronze" / kind
        if args.action == "convert":
            result = convert(source, destination, kind, hashes[source.name])
            summary[kind] = {key: result[key] for key in ["rows", "parquet_bytes", "conversion_seconds"]}
        else:
            summary[kind] = verify(source, destination, kind)
    report = ROOT / "result" / f"bronze_{args.action}.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
