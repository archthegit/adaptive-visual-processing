#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Experiment 1 temporal intervention manifests from baseline artifacts."
    )
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--strategy", required=True, choices=["top", "bottom", "random"])
    parser.add_argument(
        "--intervention-type",
        required=True,
        choices=["decoder_direct_access", "pre_encoder"],
        help="decoder_direct_access blocks decoder attention columns; pre_encoder masks frames before Qwen vision.",
    )
    parser.add_argument("--removal-fraction", type=float, required=True)
    parser.add_argument("--ranking-layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def read_records(output_dir: Path) -> list[dict[str, Any]]:
    records_path = output_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing baseline records file: {records_path}")
    records = []
    with records_path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "complete":
                records.append(record)
    if not records:
        raise ValueError(f"No complete baseline records found in {records_path}")
    return records


def load_artifact(record: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(record["artifact"])
    if not artifact_path.exists():
        raise FileNotFoundError(f"Baseline artifact does not exist: {artifact_path}")
    return json.loads(artifact_path.read_text())


def layer_scores(artifact: dict[str, Any], ranking_layer: int) -> list[float]:
    by_layer = artifact["temporal_relevance"]["by_layer"]
    layer_index = ranking_layer if ranking_layer >= 0 else len(by_layer) + ranking_layer
    if not 0 <= layer_index < len(by_layer):
        raise ValueError(
            f"ranking_layer={ranking_layer} resolves to {layer_index}, outside available layers 0..{len(by_layer) - 1}."
        )
    scores = by_layer[layer_index]["normalized_temporal_bin_scores"]
    if not scores:
        raise ValueError(f"Artifact {artifact.get('question_id')} has no temporal-bin scores at layer {layer_index}.")
    return [float(score) for score in scores]


def stable_random(seed: int, question_id: str) -> random.Random:
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    offset = int(digest[:16], 16)
    return random.Random(seed + offset)


def select_bins(scores: list[float], strategy: str, fraction: float, seed: int, question_id: str) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("--removal-fraction must satisfy 0 < fraction <= 1.")
    count = max(1, math.ceil(len(scores) * fraction))
    indices = list(range(len(scores)))
    if strategy == "top":
        indices.sort(key=lambda idx: (-scores[idx], idx))
    elif strategy == "bottom":
        indices.sort(key=lambda idx: (scores[idx], idx))
    elif strategy == "random":
        rng = stable_random(seed, question_id)
        rng.shuffle(indices)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return sorted(indices[:count])


def manifest_record(
    artifact: dict[str, Any],
    baseline_artifact: str,
    args: argparse.Namespace,
    selected_bins: list[int],
    resolved_ranking_layer: int,
) -> dict[str, Any]:
    keep_keys = (
        "question_id",
        "question_type",
        "category",
        "question",
        "choices",
        "correct_idx",
        "correct_answer",
        "video_clip",
        "raw_metadata",
        "experiment",
    )
    record = {key: artifact[key] for key in keep_keys if key in artifact}
    intervention = {
        "type": args.intervention_type,
        "strategy": args.strategy,
        "ranking_layer": resolved_ranking_layer,
        "removal_fraction": args.removal_fraction,
        "selected_temporal_bins": selected_bins,
        "seed": args.seed,
        "baseline_artifact": baseline_artifact,
    }
    record["intervention"] = intervention
    if args.intervention_type == "decoder_direct_access":
        record["decoder_direct_access_mask_temporal_bins"] = selected_bins
    else:
        record["pre_encoder_mask_temporal_bins"] = selected_bins
    return record


def main() -> None:
    args = parse_args()
    output_dir = Path(args.baseline_output_dir)
    records = read_records(output_dir)
    intervention_records = []
    for record in records:
        artifact = load_artifact(record)
        by_layer = artifact["temporal_relevance"]["by_layer"]
        resolved_layer = args.ranking_layer if args.ranking_layer >= 0 else len(by_layer) + args.ranking_layer
        scores = layer_scores(artifact, args.ranking_layer)
        selected = select_bins(scores, args.strategy, args.removal_fraction, args.seed, artifact["question_id"])
        intervention_records.append(
            manifest_record(artifact, record["artifact"], args, selected, resolved_layer)
        )
    write_jsonl(args.output_jsonl, intervention_records)
    summary = {
        "baseline_output_dir": str(output_dir),
        "output_jsonl": args.output_jsonl,
        "num_records": len(intervention_records),
        "strategy": args.strategy,
        "intervention_type": args.intervention_type,
        "ranking_layer": args.ranking_layer,
        "removal_fraction": args.removal_fraction,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
