#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json


STAGE_ORDER = (
    "vision_block_early",
    "vision_block_middle",
    "vision_block_late",
    "vision_merger_pre_reverse",
    "vision_final",
)

ENCODER_METRICS = (
    "raw_adjacent_mean",
    "raw_nonadjacent_mean",
    "raw_far_mean",
    "raw_adjacent_advantage_vs_nonadjacent",
    "raw_adjacent_advantage_vs_far",
    "centered_adjacent_mean",
    "centered_nonadjacent_mean",
    "centered_far_mean",
    "centered_adjacent_advantage_vs_nonadjacent",
    "centered_adjacent_advantage_vs_far",
    "mean_adjacent_unit_l2",
    "mean_nonadjacent_unit_l2",
    "adjacent_to_nonadjacent_l2_ratio",
    "nearest_neighbor_is_adjacent_fraction",
    "effective_rank",
    "stable_rank",
    "pc1_variance_fraction",
    "temporal_variance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal redundancy in captured Qwen vision-encoder representations."
    )
    parser.add_argument("--output-dir", required=True, help="Experiment directory containing records.jsonl.")
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.mean(vals)) if vals else None


def _median(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(vals)) if vals else None


def _sample_std(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0 if vals else None


def _unit_rows(representations: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(representations, axis=1, keepdims=True)
    return np.divide(
        representations,
        norms,
        out=np.zeros_like(representations, dtype=np.float64),
        where=norms > 0,
    )


def cosine_matrix(representations: np.ndarray, centered: bool = False) -> np.ndarray:
    reps = np.asarray(representations, dtype=np.float64)
    if reps.ndim != 2:
        raise ValueError(f"Expected [temporal_bins, embedding_dim], got {reps.shape}.")
    if centered:
        reps = reps - reps.mean(axis=0, keepdims=True)
    unit = _unit_rows(reps)
    return np.clip(unit @ unit.T, -1.0, 1.0)


def pair_values(matrix: np.ndarray, minimum_lag: int, maximum_lag: int | None = None) -> list[float]:
    values: list[float] = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            lag = right - left
            if lag < minimum_lag:
                continue
            if maximum_lag is not None and lag > maximum_lag:
                continue
            values.append(float(matrix[left, right]))
    return values


def lag_profile(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lag in range(1, matrix.shape[0]):
        values = [float(matrix[index, index + lag]) for index in range(matrix.shape[0] - lag)]
        rows.append(
            {
                "lag": lag,
                "n_pairs": len(values),
                "mean": _mean(values),
                "median": _median(values),
                "std": _sample_std(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        )
    return rows


def spectral_metrics(representations: np.ndarray) -> dict[str, float]:
    centered = representations - representations.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 0:
        return {
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "pc1_variance_fraction": 0.0,
            "temporal_variance": 0.0,
        }
    probabilities = energy[energy > 0] / total
    effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    stable_rank = float(total / energy.max())
    return {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "pc1_variance_fraction": float(energy.max() / total),
        "temporal_variance": float(np.mean(np.sum(centered**2, axis=1))),
    }


def nearest_neighbor_is_adjacent_fraction(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    work = matrix.copy()
    np.fill_diagonal(work, -np.inf)
    neighbors = np.argmax(work, axis=1)
    return float(np.mean([abs(index - int(neighbor)) == 1 for index, neighbor in enumerate(neighbors)]))


def analyze_stage(stage_name: str, stage: dict[str, Any]) -> dict[str, Any]:
    reps = np.asarray(stage["temporal_representations"], dtype=np.float64)
    raw = cosine_matrix(reps)
    centered = cosine_matrix(reps, centered=True)
    num_bins = int(reps.shape[0])
    far_lag = max(2, math.ceil(num_bins / 2))

    raw_adjacent = pair_values(raw, 1, 1)
    raw_nonadjacent = pair_values(raw, 2)
    raw_far = pair_values(raw, far_lag)
    centered_adjacent = pair_values(centered, 1, 1)
    centered_nonadjacent = pair_values(centered, 2)
    centered_far = pair_values(centered, far_lag)

    unit = _unit_rows(reps)
    unit_l2 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(unit @ unit.T, -1.0, 1.0)))
    adjacent_l2 = pair_values(unit_l2, 1, 1)
    nonadjacent_l2 = pair_values(unit_l2, 2)
    mean_adjacent_l2 = _mean(adjacent_l2)
    mean_nonadjacent_l2 = _mean(nonadjacent_l2)

    raw_adjacent_mean = _mean(raw_adjacent)
    raw_nonadjacent_mean = _mean(raw_nonadjacent)
    raw_far_mean = _mean(raw_far)
    centered_adjacent_mean = _mean(centered_adjacent)
    centered_nonadjacent_mean = _mean(centered_nonadjacent)
    centered_far_mean = _mean(centered_far)

    return {
        "stage": stage_name,
        "source_order": stage.get("source_order"),
        "canonical_order_recovered": stage.get("canonical_order_recovered"),
        "num_temporal_bins": num_bins,
        "embedding_dim": int(reps.shape[1]),
        "raw_adjacent_mean": raw_adjacent_mean,
        "raw_adjacent_median": _median(raw_adjacent),
        "raw_adjacent_std": _sample_std(raw_adjacent),
        "raw_adjacent_min": min(raw_adjacent) if raw_adjacent else None,
        "raw_adjacent_max": max(raw_adjacent) if raw_adjacent else None,
        "raw_nonadjacent_mean": raw_nonadjacent_mean,
        "raw_far_mean": raw_far_mean,
        "raw_adjacent_advantage_vs_nonadjacent": raw_adjacent_mean - raw_nonadjacent_mean,
        "raw_adjacent_advantage_vs_far": raw_adjacent_mean - raw_far_mean,
        "centered_adjacent_mean": centered_adjacent_mean,
        "centered_nonadjacent_mean": centered_nonadjacent_mean,
        "centered_far_mean": centered_far_mean,
        "centered_adjacent_advantage_vs_nonadjacent": centered_adjacent_mean - centered_nonadjacent_mean,
        "centered_adjacent_advantage_vs_far": centered_adjacent_mean - centered_far_mean,
        "mean_adjacent_unit_l2": mean_adjacent_l2,
        "mean_nonadjacent_unit_l2": mean_nonadjacent_l2,
        "adjacent_to_nonadjacent_l2_ratio": (
            mean_adjacent_l2 / mean_nonadjacent_l2 if mean_nonadjacent_l2 and mean_nonadjacent_l2 > 0 else None
        ),
        "nearest_neighbor_is_adjacent_fraction": nearest_neighbor_is_adjacent_fraction(raw),
        "raw_lag_profile": lag_profile(raw),
        "centered_lag_profile": lag_profile(centered),
        **spectral_metrics(reps),
    }


def _resolve_artifact_path(raw_path: str, output_dir: Path) -> Path:
    path = Path(raw_path)
    candidates = (path, output_dir / path.name, output_dir.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve artifact {raw_path!r} from {output_dir}.")


def load_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    latest_complete: dict[str, dict[str, Any]] = {}
    for line in (output_dir / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "complete" and record.get("artifact"):
            latest_complete[str(record["question_id"])] = record
    artifacts = []
    for question_id in sorted(latest_complete):
        path = _resolve_artifact_path(latest_complete[question_id]["artifact"], output_dir)
        artifacts.append(json.loads(path.read_text()))
    return artifacts


def _duration_metadata(artifact: dict[str, Any]) -> tuple[float | None, str]:
    clips = artifact.get("video_clip") or []
    if not clips:
        return None, "unknown"
    start = clips[0].get("start_seconds")
    end = clips[0].get("end_seconds")
    if start is None or end is None:
        return None, "unbounded"
    duration = max(0.0, float(end) - float(start))
    bucket = "short" if duration < 15 else "medium" if duration < 120 else "long"
    return duration, bucket


def _source_video_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    return str(clips[0].get("video_id", "unknown")) if clips else "unknown"


def decoder_context(artifact: dict[str, Any]) -> dict[str, Any]:
    relevance = artifact.get("temporal_relevance") or {}
    metrics = relevance.get("layer_metrics") or []
    if not metrics:
        return {}
    final = metrics[-1]
    order = [int(item) for item in final["temporal_bin_rank_order"]]
    num_bins = int(relevance.get("metadata", {}).get("num_temporal_bins", len(order)))
    absolute = relevance.get("absolute_question_to_visual_attention_mass") or []
    return {
        "final_top_bin": order[0],
        "final_top1_temporal_bin_mass": float(final["top1_temporal_bin_mass"]),
        "final_normalized_temporal_entropy": float(final["normalized_temporal_entropy"]),
        "final_bins_to_80pct_mass": int(final["bins_to_80pct_mass"]),
        "final_absolute_visual_mass": float(absolute[-1]) if absolute else None,
        "top_bin_is_first": order[0] == 0,
        "top_bin_is_boundary": order[0] in {0, num_bins - 1},
        "spearman_with_final_by_layer": [float(item["spearman_with_final_layer_ordering"]) for item in metrics],
        "topk_overlap_fraction_with_final_by_layer": [
            float(item["topk_overlap_fraction_with_final_layer"]) for item in metrics
        ],
    }


def analyze_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        duration, bucket = _duration_metadata(artifact)
        base = {
            "question_id": artifact.get("question_id"),
            "question_type": artifact.get("question_type"),
            "category": artifact.get("category"),
            "source_video_id": _source_video_id(artifact),
            "duration_seconds": duration,
            "duration_bucket": bucket,
            **decoder_context(artifact),
        }
        encoder = artifact.get("encoder_temporal") or {}
        for stage_name, stage in (encoder.get("stages") or {}).items():
            if stage.get("available") is False or "temporal_representations" not in stage:
                continue
            rows.append({**base, **analyze_stage(stage_name, stage)})
    return rows


def cluster_bootstrap_interval(
    rows: list[dict[str, Any]], field: str, samples: int, seed: int
) -> tuple[float | None, float | None]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            by_video[str(row["source_video_id"])].append(row)
    clusters = sorted(by_video)
    if not clusters:
        return None, None
    rng = random.Random(seed)
    boot = []
    for _ in range(max(1, samples)):
        sampled_rows = []
        for _index in clusters:
            sampled_rows.extend(by_video[rng.choice(clusters)])
        boot.append(statistics.mean(float(row[field]) for row in sampled_rows))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_rows(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_examples": len({row["question_id"] for row in rows}),
        "n_source_videos": len({row["source_video_id"] for row in rows}),
    }
    for offset, field in enumerate(ENCODER_METRICS):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        low, high = cluster_bootstrap_interval(rows, field, samples, seed + offset)
        summary[field] = {
            "mean": _mean(values),
            "median": _median(values),
            "sample_std": _sample_std(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
        }
    return summary


def grouped_summaries(
    rows: list[dict[str, Any]], keys: tuple[str, ...], samples: int, seed: int
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
    return {
        "|".join(group): summarize_rows(items, samples, seed + index * 101)
        for index, (group, items) in enumerate(sorted(groups.items()))
    }


def aggregate_lag_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        centered_by_lag = {int(item["lag"]): item for item in row["centered_lag_profile"]}
        for item in row["raw_lag_profile"]:
            key = (str(row["stage"]), int(item["lag"]))
            grouped[key]["raw"].append(float(item["mean"]))
            grouped[key]["centered"].append(float(centered_by_lag[int(item["lag"])]["mean"]))
    return [
        {
            "stage": stage,
            "lag": lag,
            "n_examples": len(values["raw"]),
            "raw_mean": _mean(values["raw"]),
            "raw_sample_std": _sample_std(values["raw"]),
            "centered_mean": _mean(values["centered"]),
            "centered_sample_std": _sample_std(values["centered"]),
        }
        for (stage, lag), values in sorted(grouped.items(), key=lambda item: (stage_index(item[0][0]), item[0][1]))
    ]


def _rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: list[float], right: list[float], ranks: bool = False) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x = _rankdata(left) if ranks else np.asarray(left, dtype=np.float64)
    y = _rankdata(right) if ranks else np.asarray(right, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0 else None


def encoder_decoder_associations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    targets = ("final_top1_temporal_bin_mass", "final_normalized_temporal_entropy")
    metrics = (
        "raw_adjacent_mean",
        "raw_adjacent_advantage_vs_nonadjacent",
        "centered_adjacent_advantage_vs_nonadjacent",
        "effective_rank",
        "pc1_variance_fraction",
    )
    for stage in sorted({str(row["stage"]) for row in rows}, key=stage_index):
        stage_rows = [row for row in rows if row["stage"] == stage]
        for metric in metrics:
            for target in targets:
                pairs = [
                    (float(row[metric]), float(row[target]))
                    for row in stage_rows
                    if row.get(metric) is not None and row.get(target) is not None
                ]
                left = [pair[0] for pair in pairs]
                right = [pair[1] for pair in pairs]
                outputs.append(
                    {
                        "stage": stage,
                        "encoder_metric": metric,
                        "decoder_metric": target,
                        "n_examples": len(pairs),
                        "pearson_r": correlation(left, right),
                        "spearman_rho": correlation(left, right, ranks=True),
                        "note": "Descriptive only at engineering-set sample size.",
                    }
                )
    return outputs


def decoder_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    one_per_example: dict[str, dict[str, Any]] = {}
    for row in rows:
        one_per_example[str(row["question_id"])] = row
    examples = list(one_per_example.values())
    top_bins = [int(row["final_top_bin"]) for row in examples if row.get("final_top_bin") is not None]
    return {
        "n_examples": len(examples),
        "final_top_bin_histogram": dict(sorted(Counter(top_bins).items())),
        "fraction_final_top_bin_is_first": _mean(float(row.get("top_bin_is_first", False)) for row in examples),
        "fraction_final_top_bin_is_boundary": _mean(float(row.get("top_bin_is_boundary", False)) for row in examples),
        "mean_final_top1_temporal_bin_mass": _mean(row.get("final_top1_temporal_bin_mass") for row in examples),
        "mean_final_normalized_temporal_entropy": _mean(row.get("final_normalized_temporal_entropy") for row in examples),
        "mean_final_bins_to_80pct_mass": _mean(row.get("final_bins_to_80pct_mass") for row in examples),
    }


def stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return len(STAGE_ORDER)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_plots(rows: list[dict[str, Any]], lag_rows: list[dict[str, Any]], analysis_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or pass --no-plots.") from exc

    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stages = sorted({str(row["stage"]) for row in rows}, key=stage_index)
    labels = [stage.replace("vision_", "").replace("_pre_reverse", "") for stage in stages]
    paths = []

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stages))
    width = 0.25
    for offset, (field, label) in enumerate(
        (("raw_adjacent_mean", "Adjacent"), ("raw_nonadjacent_mean", "Non-adjacent"), ("raw_far_mean", "Far"))
    ):
        means = [_mean(row[field] for row in rows if row["stage"] == stage) for stage in stages]
        ax.bar(x + (offset - 1) * width, means, width, label=label)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean raw cosine similarity")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "encoder_pairwise_cosine_by_stage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    for stage in stages:
        selected = [row for row in lag_rows if row["stage"] == stage]
        ax.plot([row["lag"] for row in selected], [row["raw_mean"] for row in selected], marker="o", label=stage)
    ax.set_xlabel("Temporal-bin lag")
    ax.set_ylabel("Mean raw cosine similarity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "encoder_cosine_by_temporal_lag.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    raw = [_mean(row["raw_adjacent_advantage_vs_nonadjacent"] for row in rows if row["stage"] == stage) for stage in stages]
    centered = [
        _mean(row["centered_adjacent_advantage_vs_nonadjacent"] for row in rows if row["stage"] == stage)
        for stage in stages
    ]
    ax.bar(x - width / 2, raw, width, label="Raw")
    ax.bar(x + width / 2, centered, width, label="Mean-centered")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Adjacent minus non-adjacent cosine")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "encoder_local_temporal_advantage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    effective = [_mean(row["effective_rank"] for row in rows if row["stage"] == stage) for stage in stages]
    pc1 = [_mean(row["pc1_variance_fraction"] for row in rows if row["stage"] == stage) for stage in stages]
    ax.bar(x - width / 2, effective, width, label="Effective rank")
    ax.set_ylabel("Effective temporal rank")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    twin = ax.twinx()
    twin.plot(x, pc1, color="tab:red", marker="o", label="PC1 variance fraction")
    twin.set_ylabel("PC1 variance fraction")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    path = plot_dir / "encoder_temporal_effective_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_dir / "encoder_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts(output_dir)
    rows = analyze_artifacts(artifacts)
    if not rows:
        raise RuntimeError("No encoder temporal representations were found in complete artifacts.")
    lag_rows = aggregate_lag_profiles(rows)
    report = {
        "source_output_dir": str(output_dir),
        "num_artifacts": len(artifacts),
        "num_encoder_stage_rows": len(rows),
        "stages": sorted({str(row["stage"]) for row in rows}, key=stage_index),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "aggregate_by_stage": grouped_summaries(rows, ("stage",), args.bootstrap_samples, args.seed),
        "aggregate_by_category_and_stage": grouped_summaries(
            rows, ("category", "stage"), args.bootstrap_samples, args.seed + 10000
        ),
        "aggregate_by_duration_bucket_and_stage": grouped_summaries(
            rows, ("duration_bucket", "stage"), args.bootstrap_samples, args.seed + 20000
        ),
        "decoder_context_summary": decoder_summary(rows),
        "encoder_decoder_associations": encoder_decoder_associations(rows),
        "lag_profiles": lag_rows,
        "interpretation_guardrails": [
            "Raw adjacent cosine alone does not prove temporal redundancy; compare it with non-adjacent and far pairs.",
            "Mean-centered cosine reduces the effect of a shared representation direction but is not a causal test.",
            "Engineering-set category and correlation results are descriptive because the sample is small.",
            "Masking top, bottom and random bins with the same intervention mechanism is required for causal evidence.",
        ],
    }
    write_json(analysis_dir / "encoder_analysis.json", report)

    scalar_fields = [
        "question_id",
        "question_type",
        "category",
        "source_video_id",
        "duration_seconds",
        "duration_bucket",
        "stage",
        "source_order",
        "canonical_order_recovered",
        "num_temporal_bins",
        "embedding_dim",
        *ENCODER_METRICS,
        "final_top_bin",
        "final_top1_temporal_bin_mass",
        "final_normalized_temporal_entropy",
        "final_bins_to_80pct_mass",
        "final_absolute_visual_mass",
        "top_bin_is_first",
        "top_bin_is_boundary",
    ]
    write_csv(analysis_dir / "encoder_metrics_per_example.csv", rows, scalar_fields)
    write_csv(
        analysis_dir / "encoder_lag_profiles.csv",
        lag_rows,
        ["stage", "lag", "n_examples", "raw_mean", "raw_sample_std", "centered_mean", "centered_sample_std"],
    )
    plot_paths = [] if args.no_plots else create_plots(rows, lag_rows, analysis_dir)
    stage_headlines = {
        stage: {
            "n_examples": report["aggregate_by_stage"][stage]["n_examples"],
            "raw_adjacent_mean": report["aggregate_by_stage"][stage]["raw_adjacent_mean"],
            "raw_nonadjacent_mean": report["aggregate_by_stage"][stage]["raw_nonadjacent_mean"],
            "raw_far_mean": report["aggregate_by_stage"][stage]["raw_far_mean"],
            "raw_adjacent_advantage_vs_nonadjacent": report["aggregate_by_stage"][stage][
                "raw_adjacent_advantage_vs_nonadjacent"
            ],
            "centered_adjacent_advantage_vs_nonadjacent": report["aggregate_by_stage"][stage][
                "centered_adjacent_advantage_vs_nonadjacent"
            ],
            "effective_rank": report["aggregate_by_stage"][stage]["effective_rank"],
            "pc1_variance_fraction": report["aggregate_by_stage"][stage]["pc1_variance_fraction"],
        }
        for stage in report["stages"]
    }
    print(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "num_artifacts": len(artifacts),
                "num_encoder_stage_rows": len(rows),
                "stage_headlines": stage_headlines,
                "decoder_context_summary": report["decoder_context_summary"],
                "report": str(analysis_dir / "encoder_analysis.json"),
                "per_example_csv": str(analysis_dir / "encoder_metrics_per_example.csv"),
                "lag_csv": str(analysis_dir / "encoder_lag_profiles.csv"),
                "plots": plot_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
