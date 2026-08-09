#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 1 JSONL records and relevance artifacts.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def load_records(output_dir: Path) -> list[dict[str, Any]]:
    records_path = output_dir / "records.jsonl"
    with records_path.open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_artifacts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts = []
    for record in records:
        if record.get("status") == "complete" and record.get("artifact"):
            with Path(record["artifact"]).open("r") as handle:
                artifacts.append(json.load(handle))
    return artifacts


def accuracy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "complete"]
    failed = [record for record in records if record.get("status") == "failed"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed:
        by_category[record["category"]].append(record)
    return {
        "total_records": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "status_counts": dict(sorted((status, sum(1 for record in records if record.get("status") == status)) for status in {record.get("status") for record in records})),
        "correct": sum(1 for record in completed if record.get("correct")),
        "accuracy": sum(1 for record in completed if record.get("correct")) / len(completed) if completed else 0.0,
        "by_category": {
            category: {
                "completed": len(items),
                "correct": sum(1 for item in items if item.get("correct")),
                "accuracy": sum(1 for item in items if item.get("correct")) / len(items) if items else 0.0,
            }
            for category, items in sorted(by_category.items())
        },
        "visual_token_counts": sorted({record.get("num_visual_tokens") for record in completed}),
        "failures": [
            {
                "question_id": record.get("question_id"),
                "category": record.get("category"),
                "error": record.get("error"),
            }
            for record in failed
        ],
    }


def layer_fusion_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    if not artifacts:
        return {}
    num_layers = len(artifacts[0]["relevance"]["normalized_frame_scores"])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_category[artifact["category"]].append(artifact)

    def layer_stats(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stats = []
        for layer_idx in range(num_layers):
            top1_values = [
                max(artifact["relevance"]["normalized_frame_scores"][layer_idx])
                for artifact in items
            ]
            entropy_values = [
                artifact["relevance"]["concentration_by_layer"][layer_idx]["normalized_entropy"]
                for artifact in items
            ]
            stats.append(
                {
                    "layer": layer_idx,
                    "mean_top1_frame_mass": statistics.mean(top1_values),
                    "mean_normalized_entropy": statistics.mean(entropy_values),
                }
            )
        return stats

    overall = layer_stats(artifacts)
    return {
        "num_artifacts": len(artifacts),
        "num_layers": num_layers,
        "overall": overall,
        "by_category": {category: layer_stats(items) for category, items in sorted(by_category.items())},
        "peak_overall_top1_layer": max(overall, key=lambda item: item["mean_top1_frame_mass"]),
        "lowest_overall_entropy_layer": min(overall, key=lambda item: item["mean_normalized_entropy"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    records = load_records(output_dir)
    artifacts = load_artifacts(records)
    summary = {
        "output_dir": str(output_dir),
        "accuracy": accuracy_summary(records),
        "layer_fusion": layer_fusion_summary(artifacts),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "analysis_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
