#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot temporal-only Experiment 1 relevance artifacts.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory containing records.jsonl.")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--artifact", action="append", default=None, help="Specific artifact JSON to plot. Can repeat.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to create temporal Experiment 1 plots.") from exc
    return plt


def load_records(output_dir: Path) -> list[dict[str, Any]]:
    with (output_dir / "records.jsonl").open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_artifacts(output_dir: Path, artifact_paths: list[str] | None) -> list[dict[str, Any]]:
    if artifact_paths:
        paths = [Path(path) for path in artifact_paths]
    else:
        paths = [
            Path(record["artifact"])
            for record in load_records(output_dir)
            if record.get("status") == "complete" and record.get("artifact")
        ]
    artifacts = []
    for path in paths:
        with path.open("r") as handle:
            artifact = json.load(handle)
        if "temporal_relevance" not in artifact:
            continue
        artifacts.append(artifact)
    return artifacts


def source_video_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or artifact.get("raw", {}).get("video_clip") or []
    if clips:
        return str(clips[0].get("video_id", "unknown"))
    paths = artifact.get("metadata", {}).get("source_video_paths", [])
    return str(paths[0]) if paths else "unknown"


def save_example_heatmap(artifact: dict[str, Any], plot_dir: Path) -> None:
    plt = _plt()
    scores = np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(5, scores.shape[1] * 0.7), 6))
    image = ax.imshow(scores, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(f"{artifact['question_id']} temporal relevance")
    ax.set_xlabel("Temporal bin")
    ax.set_ylabel("Decoder layer")
    fig.colorbar(image, ax=ax, label="Normalized attention mass")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{artifact['question_id']}_temporal_heatmap.png", dpi=180)
    plt.close(fig)


def save_rank_trajectory(artifact: dict[str, Any], plot_dir: Path) -> None:
    plt = _plt()
    metrics = artifact["temporal_relevance"]["layer_metrics"]
    num_bins = artifact["temporal_relevance"]["metadata"]["num_temporal_bins"]
    positions = np.zeros((len(metrics), num_bins), dtype=np.float64)
    for layer_idx, metric in enumerate(metrics):
        for rank, temporal_bin in enumerate(metric["temporal_bin_rank_order"], start=1):
            positions[layer_idx, int(temporal_bin)] = rank
    fig, ax = plt.subplots(figsize=(7, 5))
    layers = np.arange(len(metrics))
    for temporal_bin in range(num_bins):
        ax.plot(layers, positions[:, temporal_bin], marker="o", linewidth=1.2, label=f"bin {temporal_bin}")
    ax.invert_yaxis()
    ax.set_title(f"{artifact['question_id']} temporal-bin rank trajectory")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Rank, 1 = most attended")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{artifact['question_id']}_rank_trajectory.png", dpi=180)
    plt.close(fig)


def metric_array(artifact: dict[str, Any], field: str) -> np.ndarray:
    return np.asarray([layer[field] for layer in artifact["temporal_relevance"]["layer_metrics"]], dtype=np.float64)


def clustered_bootstrap_curve(
    artifacts: list[dict[str, Any]],
    field: str,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_video[source_video_id(artifact)].append(artifact)
    clusters = sorted(by_video)
    if not clusters:
        return np.array([]), np.array([]), np.array([])
    rng = random.Random(seed)
    curves = []
    for _ in range(samples):
        selected = []
        for _cluster in clusters:
            selected.extend(by_video[rng.choice(clusters)])
        curves.append(np.mean([metric_array(item, field) for item in selected], axis=0))
    stacked = np.stack(curves, axis=0)
    return (
        np.mean(stacked, axis=0),
        np.percentile(stacked, 2.5, axis=0),
        np.percentile(stacked, 97.5, axis=0),
    )


def save_aggregate_curves(artifacts: list[dict[str, Any]], plot_dir: Path, samples: int, seed: int) -> None:
    plt = _plt()
    fields = [
        ("normalized_temporal_entropy", "Normalized entropy", "aggregate_entropy_by_category.png"),
        ("top1_temporal_bin_mass", "Top-1 temporal-bin mass", "aggregate_top1_mass_by_category.png"),
        ("bins_to_80pct_mass", "Bins needed for 80% mass", "aggregate_bins_to_80_by_category.png"),
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_category[artifact["category"]].append(artifact)
    for field, ylabel, filename in fields:
        fig, ax = plt.subplots(figsize=(7, 5))
        for offset, (category, items) in enumerate(sorted(by_category.items())):
            mean, low, high = clustered_bootstrap_curve(items, field, samples, seed + offset)
            if mean.size == 0:
                continue
            layers = np.arange(mean.shape[0])
            ax.plot(layers, mean, label=category)
            ax.fill_between(layers, low, high, alpha=0.18)
        ax.set_xlabel("Decoder layer")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "temporal_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts(output_dir, args.artifact)
    for artifact in artifacts:
        save_example_heatmap(artifact, plot_dir)
        save_rank_trajectory(artifact, plot_dir)
    save_aggregate_curves(artifacts, plot_dir, args.bootstrap_samples, args.seed)
    print(json.dumps({"plot_dir": str(plot_dir), "num_artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
