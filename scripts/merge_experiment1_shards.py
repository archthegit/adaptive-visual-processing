#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard-specific Experiment 1 records into records.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--merged-records", default="records.jsonl")
    parser.add_argument("--summary-json", default="summary.json")
    return parser.parse_args()


def shard_records_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / f"records_shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_records(output_dir: Path, num_shards: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing = []
    for shard_index in range(num_shards):
        path = shard_records_path(output_dir, shard_index, num_shards)
        if not path.exists():
            missing.append(str(path))
            continue
        records.extend(load_records(path))
    if missing:
        raise FileNotFoundError(f"Missing shard record files: {missing}")
    records.sort(key=lambda item: (item.get("question_id", ""), item.get("status", "")))
    return records


def summarize(records: list[dict[str, Any]], num_shards: int) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "complete"]
    failed = [record for record in records if record.get("status") == "failed"]
    skipped = [record for record in records if record.get("status") == "skipped_complete"]
    return {
        "num_shards": num_shards,
        "total_records": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "skipped_complete": len(skipped),
        "correct": sum(1 for record in completed if record.get("correct")),
        "accuracy": sum(1 for record in completed if record.get("correct")) / len(completed) if completed else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    records = merge_records(output_dir, args.num_shards)
    merged_path = output_dir / args.merged_records
    with merged_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = summarize(records, args.num_shards)
    write_json(output_dir / args.summary_json, summary)
    print(json.dumps({"merged_records": str(merged_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
