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
    "raw_temporal_bin_scores",
    "normalized_temporal_bin_scores",
    "absolute_question_to_visual_attention_mass",
)

LEGACY_FIELDS = (
    "raw_token_scores",
    "normalized_token_scores",
    "absolute_visual_mass_by_layer",
    "raw_frame_scores",
    "normalized_frame_scores",
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


def compare_topk_logits(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_topk = left.get("metadata", {}).get("prefill_next_token_topk", [])
    right_topk = right.get("metadata", {}).get("prefill_next_token_topk", [])
    left_ids = [item.get("token_id") for item in left_topk]
    right_ids = [item.get("token_id") for item in right_topk]
    paired = list(zip(left_topk, right_topk))
    logit_diffs = [
        abs(float(left_item.get("logit", 0.0)) - float(right_item.get("logit", 0.0)))
        for left_item, right_item in paired
        if left_item.get("token_id") == right_item.get("token_id")
    ]
    return {
        "left_token_ids": left_ids,
        "right_token_ids": right_ids,
        "token_ids_match": left_ids == right_ids,
        "max_abs_logit_diff_matching_positions": max(logit_diffs, default=None),
    }


def relevance_payload(artifact: dict[str, Any]) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    if "temporal_relevance" in artifact:
        return "temporal_relevance", artifact["temporal_relevance"], FIELDS
    return "relevance", artifact["relevance"], LEGACY_FIELDS


def compare_dirs(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left_records = load_records(left_dir)
    right_records = load_records(right_dir)
    common_ids = sorted(set(left_records) & set(right_records))
    comparisons = []
    for question_id in common_ids:
        left = load_artifact(left_records[question_id])
        right = load_artifact(right_records[question_id])
        left_name, left_relevance, fields = relevance_payload(left)
        right_name, right_relevance, right_fields = relevance_payload(right)
        if fields != right_fields:
            fields = tuple(sorted(set(fields) & set(right_fields)))
        field_diffs = {
            field: max_abs_diff(left_relevance.get(field, []), right_relevance.get(field, []))
            for field in fields
        }
        comparisons.append(
            {
                "question_id": question_id,
                "left_correct": left["correct"],
                "right_correct": right["correct"],
                "left_relevance_payload": left_name,
                "right_relevance_payload": right_name,
                "left_extraction": left["metadata"].get("attention_extraction", "unknown"),
                "right_extraction": right["metadata"].get("attention_extraction", "unknown"),
                "max_abs_diff_by_field": field_diffs,
                "max_abs_diff": max(field_diffs.values()),
                "prefill_next_token_topk": compare_topk_logits(left, right),
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
