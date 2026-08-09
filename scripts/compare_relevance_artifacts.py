#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json


FIELDS = (
    "raw_token_scores",
    "normalized_token_scores",
    "absolute_visual_mass_by_layer",
    "raw_frame_scores",
    "normalized_frame_scores",
    "raw_spatial_scores_by_input",
    "normalized_spatial_scores_by_input",
    "aggregate_frame_scores",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Experiment 1 relevance artifacts from two runs.")
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def load_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with (output_dir / "records.jsonl").open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "complete":
                records[record["question_id"]] = record
    return records


def load_artifact(record: dict[str, Any]) -> dict[str, Any]:
    with Path(record["artifact"]).open("r") as handle:
        return json.load(handle)


def max_abs_diff(left: Any, right: Any) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    return float(np.max(np.abs(left_arr - right_arr))) if left_arr.size else 0.0


def compare_dirs(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left_records = load_records(left_dir)
    right_records = load_records(right_dir)
    common_ids = sorted(set(left_records) & set(right_records))
    comparisons = []
    for question_id in common_ids:
        left = load_artifact(left_records[question_id])
        right = load_artifact(right_records[question_id])
        field_diffs = {
            field: max_abs_diff(left["relevance"].get(field, []), right["relevance"].get(field, []))
            for field in FIELDS
        }
        comparisons.append(
            {
                "question_id": question_id,
                "left_correct": left["correct"],
                "right_correct": right["correct"],
                "left_extraction": left["metadata"].get("attention_extraction", "unknown"),
                "right_extraction": right["metadata"].get("attention_extraction", "unknown"),
                "max_abs_diff_by_field": field_diffs,
                "max_abs_diff": max(field_diffs.values()),
            }
        )
    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "left_completed": len(left_records),
        "right_completed": len(right_records),
        "common": len(common_ids),
        "missing_from_left": sorted(set(right_records) - set(left_records)),
        "missing_from_right": sorted(set(left_records) - set(right_records)),
        "comparisons": comparisons,
        "max_abs_diff": max((item["max_abs_diff"] for item in comparisons), default=0.0),
    }


def main() -> None:
    args = parse_args()
    summary = compare_dirs(Path(args.left_dir), Path(args.right_dir))
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
