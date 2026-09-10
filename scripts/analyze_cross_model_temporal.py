#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen/VILA temporal-attention replication artifacts.")
    parser.add_argument("--qwen-root", required=True, help="Root containing Qwen condition output directories.")
    parser.add_argument("--vila-root", required=True, help="Root containing VILA condition output directories.")
    parser.add_argument(
        "--condition",
        action="append",
        default=["baseline", "repeated_frame", "reversed_video"],
        help="Condition subdirectory to compare. Can be repeated.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def load_artifacts(path: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return artifacts
    for item in sorted(path.glob("*.json")):
        if item.name in {"summary.json", "profile.json"}:
            continue
        payload = json.loads(item.read_text())
        question_id = payload.get("question_id")
        if question_id:
            artifacts[str(question_id)] = payload
    return artifacts


def source_video_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    if clips:
        return str(clips[0].get("video_id") or artifact.get("question_id"))
    return str(artifact.get("question_id"))


def temporal_distribution(artifact: dict[str, Any]) -> np.ndarray:
    values = artifact["temporal_relevance"]["normalized_temporal_bin_scores"]
    return np.asarray(values, dtype=np.float64)


def absolute_mass(artifact: dict[str, Any]) -> np.ndarray:
    values = artifact["temporal_relevance"]["absolute_question_to_visual_attention_mass"]
    return np.asarray(values, dtype=np.float64)


def layer_depths(num_layers: int) -> np.ndarray:
    if num_layers <= 1:
        return np.zeros((num_layers,), dtype=np.float64)
    return np.arange(num_layers, dtype=np.float64) / float(num_layers - 1)


def align_layers(reference: np.ndarray, target: np.ndarray) -> list[tuple[int, int, float]]:
    left = layer_depths(reference.shape[0])
    right = layer_depths(target.shape[0])
    aligned = []
    for left_idx, depth in enumerate(left):
        right_idx = int(np.argmin(np.abs(right - depth)))
        aligned.append((left_idx, right_idx, float(depth)))
    return aligned


def interpolate_distribution(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == size:
        return values
    if values.size == 1:
        return np.full((size,), float(values[0]) / float(size))
    source_x = (np.arange(values.size, dtype=np.float64) + 0.5) / float(values.size)
    target_x = (np.arange(size, dtype=np.float64) + 0.5) / float(size)
    interp = np.interp(target_x, source_x, values)
    total = float(interp.sum())
    return interp / total if total > 0 else interp


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        right = interpolate_distribution(right, left.size)
    if left.size <= 1:
        return 1.0
    left_order = np.argsort(np.argsort(left))
    right_order = np.argsort(np.argsort(right))
    left_centered = left_order.astype(np.float64) - float(left_order.mean())
    right_centered = right_order.astype(np.float64) - float(right_order.mean())
    denom = math.sqrt(float((left_centered * left_centered).sum() * (right_centered * right_centered).sum()))
    return float((left_centered * right_centered).sum() / denom) if denom > 0 else 0.0


def entropy(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    total = float(values.sum())
    if total <= 0:
        return 0.0
    norm = values / total
    nonzero = norm[norm > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / math.log(values.size))


def top_lift(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.size * float(values.max()) - 1.0)


def paired_bootstrap(values: list[dict[str, Any]], key: str, replicates: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    by_video: dict[str, list[float]] = {}
    for item in values:
        by_video.setdefault(str(item["source_video_id"]), []).append(float(item[key]))
    video_ids = sorted(by_video)
    per_video = [float(np.mean(by_video[video_id])) for video_id in video_ids]
    rng = random.Random(seed)
    boot = []
    for _ in range(replicates):
        sample = [per_video[rng.randrange(len(per_video))] for _ in per_video]
        boot.append(float(np.mean(sample)))
    return {
        "mean": float(np.mean(per_video)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": len(per_video),
    }


def compare_condition(qwen_dir: Path, vila_dir: Path, replicates: int, seed: int) -> dict[str, Any]:
    qwen = load_artifacts(qwen_dir)
    vila = load_artifacts(vila_dir)
    common = sorted(set(qwen) & set(vila))
    rows = []
    for question_id in common:
        q_art = qwen[question_id]
        v_art = vila[question_id]
        q_dist = temporal_distribution(q_art)
        v_dist = temporal_distribution(v_art)
        q_abs = absolute_mass(q_art)
        v_abs = absolute_mass(v_art)
        aligned = align_layers(q_dist, v_dist)
        for q_layer, v_layer, depth in aligned:
            q_values = q_dist[q_layer]
            v_values = v_dist[v_layer]
            rows.append(
                {
                    "question_id": question_id,
                    "source_video_id": source_video_id(q_art),
                    "normalized_depth": depth,
                    "qwen_layer": q_layer,
                    "vila_layer": v_layer,
                    "spearman": spearman(q_values, v_values),
                    "qwen_temporal_lift": top_lift(q_values),
                    "vila_temporal_lift": top_lift(v_values),
                    "qwen_entropy": entropy(q_values),
                    "vila_entropy": entropy(v_values),
                    "qwen_first_bin_mass": float(q_values[0]) if q_values.size else 0.0,
                    "vila_first_bin_mass": float(v_values[0]) if v_values.size else 0.0,
                    "qwen_last_bin_mass": float(q_values[-1]) if q_values.size else 0.0,
                    "vila_last_bin_mass": float(v_values[-1]) if v_values.size else 0.0,
                    "qwen_absolute_visual_mass": float(q_abs[q_layer]) if q_layer < q_abs.size else 0.0,
                    "vila_absolute_visual_mass": float(v_abs[v_layer]) if v_layer < v_abs.size else 0.0,
                }
            )
    metrics = {}
    for key in (
        "spearman",
        "qwen_temporal_lift",
        "vila_temporal_lift",
        "qwen_entropy",
        "vila_entropy",
        "qwen_first_bin_mass",
        "vila_first_bin_mass",
        "qwen_last_bin_mass",
        "vila_last_bin_mass",
        "qwen_absolute_visual_mass",
        "vila_absolute_visual_mass",
    ):
        metrics[key] = paired_bootstrap(rows, key, replicates, seed)
    return {
        "qwen_dir": str(qwen_dir),
        "vila_dir": str(vila_dir),
        "qwen_artifacts": len(qwen),
        "vila_artifacts": len(vila),
        "common_examples": len(common),
        "per_example_layer_rows": rows,
        "aggregate": metrics,
    }


def main() -> None:
    args = parse_args()
    qwen_root = Path(args.qwen_root)
    vila_root = Path(args.vila_root)
    report = {
        "qwen_root": str(qwen_root),
        "vila_root": str(vila_root),
        "conditions": {},
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    for condition in args.condition:
        report["conditions"][condition] = compare_condition(
            qwen_root / condition,
            vila_root / condition,
            args.bootstrap_replicates,
            args.seed,
        )
    write_json_atomic(Path(args.output_json), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
