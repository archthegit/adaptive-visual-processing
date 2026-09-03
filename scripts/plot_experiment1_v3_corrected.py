#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


QUESTION_TYPE_LABELS = {
    "fine_grained_action_recognition": "action recognition",
    "fine_grained_action_localization": "action localization",
    "gaze_interaction_anticipation": "gaze anticipation",
    "ingredient_ingredient_adding_localization": "ingredient localization",
    "ingredient_ingredient_weight": "ingredient weight",
    "object_motion_object_movement_itinerary": "object itinerary",
    "fine_grained_why_recognition": "why recognition",
}

STAGE_ORDER = (
    "vision_block_early",
    "vision_block_middle",
    "vision_block_late",
    "vision_merger_pre_reverse",
    "vision_final",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create defensible aggregate plots for Experiment 1 v3.")
    parser.add_argument("--output-dir", required=True, help="Baseline run directory containing records.jsonl.")
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--grid-bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--expected-artifacts", type=int, default=77)
    return parser.parse_args()


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _artifact_path(raw: str, output_dir: Path) -> Path:
    path = Path(raw)
    candidates = (path, output_dir / path.name, output_dir.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve artifact {raw!r}")


def load_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    records_path = output_dir / "records.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "complete" and record.get("artifact"):
            latest[str(record["question_id"])] = record
    return [json.loads(_artifact_path(latest[q]["artifact"], output_dir).read_text()) for q in sorted(latest)]


def source_video_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    return str(clips[0].get("video_id", "unknown")) if clips else "unknown"


def participant_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    if clips and clips[0].get("participant_id"):
        return str(clips[0]["participant_id"])
    return source_video_id(artifact).split("-", 1)[0]


def normalize(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid temporal distribution with shape {arr.shape}")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Temporal distribution has non-positive mass")
    return arr / total


def resample_lift(values: Any, target_bins: int) -> np.ndarray:
    """Convert p(bin) to lift B*p(bin), then interpolate at relative-time bin centers."""
    distribution = normalize(values)
    bins = distribution.size
    lift = bins * distribution
    source_x = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    target_x = (np.arange(target_bins, dtype=np.float64) + 0.5) / target_bins
    return np.interp(target_x, source_x, lift, left=lift[0], right=lift[-1])


def validate_artifacts(artifacts: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    if len(artifacts) != expected:
        raise RuntimeError(f"Expected {expected} artifacts, found {len(artifacts)}")
    videos: set[str] = set()
    bin_counts = []
    for artifact in artifacts:
        question_id = str(artifact.get("question_id"))
        video = source_video_id(artifact)
        if video in videos:
            raise RuntimeError(f"Duplicate source video: {video}")
        videos.add(video)
        decoder = (artifact.get("temporal_relevance") or {}).get("normalized_temporal_bin_scores") or []
        encoder = (artifact.get("encoder_attention_temporal") or {}).get("normalized_incoming_temporal_attention") or []
        if len(decoder) != 28 or len(encoder) != 32:
            raise RuntimeError(f"{question_id}: expected 28 decoder and 32 encoder layers")
        bins = int((artifact["temporal_relevance"].get("metadata") or {}).get("num_temporal_bins", len(decoder[0])))
        if not 8 <= bins <= 64:
            raise RuntimeError(f"{question_id}: invalid bin count {bins}")
        bin_counts.append(bins)
        for layer in decoder:
            if len(layer) != bins or not np.isclose(normalize(layer).sum(), 1.0, atol=1e-8):
                raise RuntimeError(f"{question_id}: invalid decoder distribution")
        for layer in encoder:
            arr = np.asarray(layer, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != bins or not np.all(np.isfinite(arr)):
                raise RuntimeError(f"{question_id}: invalid encoder distribution")
    return {
        "num_artifacts": len(artifacts),
        "num_source_videos": len(videos),
        "min_bins": min(bin_counts),
        "median_bins": float(np.median(bin_counts)),
        "max_bins": max(bin_counts),
    }


def decoder_lift_matrix(artifact: dict[str, Any], grid_bins: int) -> np.ndarray:
    layers = artifact["temporal_relevance"]["normalized_temporal_bin_scores"]
    return np.stack([resample_lift(layer, grid_bins) for layer in layers])


def encoder_lift_matrix(artifact: dict[str, Any], grid_bins: int) -> np.ndarray:
    layers = artifact["encoder_attention_temporal"]["normalized_incoming_temporal_attention"]
    outputs = []
    for layer in layers:
        heads = np.asarray(layer, dtype=np.float64)
        outputs.append(resample_lift(heads.mean(axis=0), grid_bins))
    return np.stack(outputs)


def save_lift_heatmap(matrix: np.ndarray, path: Path, title: str, n: int) -> dict[str, float]:
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    deviation = matrix - 1.0
    observed = float(np.max(np.abs(deviation)))
    limit = max(0.05, observed)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(
        deviation,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    ax.set_title(f"{title} (n={n})")
    ax.set_xlabel("Normalized temporal position")
    ax.set_ylabel("Layer")
    width = matrix.shape[1]
    ticks = np.linspace(0, width - 1, 5)
    ax.set_xticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
    fig.colorbar(image, ax=ax, label="Attention lift minus uniform (B p - 1)")
    ax.text(
        0.01,
        -0.22,
        f"Equal video weight; each example normalized before interpolation. Max |B p - 1| = {observed:.3g}.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"max_absolute_lift_deviation": observed, "color_limit": limit}


def metric_curve(artifact: dict[str, Any], field: str) -> np.ndarray:
    if field == "absolute_visual_mass":
        return np.asarray(artifact["temporal_relevance"]["absolute_question_to_visual_attention_mass"], dtype=np.float64)
    return np.asarray([row[field] for row in artifact["temporal_relevance"]["layer_metrics"]], dtype=np.float64)


def hierarchical_curve_ci(
    artifacts: list[dict[str, Any]],
    extractor: Callable[[dict[str, Any]], np.ndarray],
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.stack([extractor(item) for item in artifacts])
    mean = observed.mean(axis=0)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        grouped[participant_id(artifact)].append(artifact)
    participants = sorted(grouped)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        selected = []
        for index in rng.integers(0, len(participants), size=len(participants)):
            group = grouped[participants[int(index)]]
            for item_index in rng.integers(0, len(group), size=len(group)):
                selected.append(group[int(item_index)])
        boot.append(np.stack([extractor(item) for item in selected]).mean(axis=0))
    stacked = np.stack(boot)
    return mean, np.percentile(stacked, 2.5, axis=0), np.percentile(stacked, 97.5, axis=0)


def save_curve_groups(
    artifacts: list[dict[str, Any]],
    field: str,
    ylabel: str,
    title: str,
    path: Path,
    samples: int,
    seed: int,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all", artifacts)]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_type[str(artifact.get("question_type", "unknown"))].append(artifact)
    groups.extend((key, items) for key, items in sorted(by_type.items()) if len(items) >= 5)
    for offset, (key, items) in enumerate(groups):
        mean, low, high = hierarchical_curve_ci(items, lambda a: metric_curve(a, field), samples, seed + offset * 97)
        layers = np.arange(mean.size)
        label = f"{QUESTION_TYPE_LABELS.get(key, key)} (n={len(items)})"
        kwargs = {"color": "black", "linewidth": 2.8} if key == "all" else {"linewidth": 1.4, "alpha": 0.9}
        ax.plot(layers, mean, label=label, **kwargs)
        ax.fill_between(layers, low, high, alpha=0.10 if key != "all" else 0.16, color=kwargs.get("color"))
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if field == "normalized_temporal_entropy":
        ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1, label="uniform")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def cosine_matrix(representations: np.ndarray, centered: bool) -> np.ndarray:
    values = np.asarray(representations, dtype=np.float64)
    if centered:
        values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return np.clip(unit @ unit.T, -1.0, 1.0)


def adjacent_far_advantage(representations: Any, centered: bool) -> float:
    matrix = cosine_matrix(np.asarray(representations, dtype=np.float64), centered=centered)
    bins = matrix.shape[0]
    adjacent = [matrix[i, i + 1] for i in range(bins - 1)]
    far_lag = max(2, math.ceil(bins / 2))
    far = [matrix[i, j] for i in range(bins) for j in range(i + far_lag, bins)]
    return float(np.mean(adjacent) - np.mean(far))


def save_encoder_advantage(artifacts: list[dict[str, Any]], path: Path, samples: int, seed: int) -> dict[str, Any]:
    plt = _plt()
    stages = [
        stage
        for stage in STAGE_ORDER
        if all(
            (a.get("encoder_temporal") or {}).get("stages", {}).get(stage, {}).get("temporal_representations")
            for a in artifacts
        )
    ]
    if not stages:
        raise RuntimeError("No encoder representation stage is available in every artifact")
    raw_by_stage: dict[str, list[float]] = {}
    centered_by_stage: dict[str, list[float]] = {}
    for stage in stages:
        raw_by_stage[stage] = [adjacent_far_advantage(a["encoder_temporal"]["stages"][stage]["temporal_representations"], False) for a in artifacts]
        centered_by_stage[stage] = [adjacent_far_advantage(a["encoder_temporal"]["stages"][stage]["temporal_representations"], True) for a in artifacts]

    def bootstrap_scalar(field: dict[str, list[float]], stage: str, local_seed: int) -> tuple[float, float, float]:
        rows = list(zip(artifacts, field[stage]))
        grouped: dict[str, list[float]] = defaultdict(list)
        for artifact, value in rows:
            grouped[participant_id(artifact)].append(value)
        participants = sorted(grouped)
        rng = np.random.default_rng(local_seed)
        estimates = []
        for _ in range(samples):
            values = []
            for index in rng.integers(0, len(participants), size=len(participants)):
                group = grouped[participants[int(index)]]
                values.extend(group[int(i)] for i in rng.integers(0, len(group), size=len(group)))
            estimates.append(float(np.mean(values)))
        return float(np.mean(field[stage])), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    results: dict[str, Any] = {}
    for offset, (name, values, color) in enumerate((("raw", raw_by_stage, "tab:blue"), ("mean-centered", centered_by_stage, "tab:orange"))):
        stats = [bootstrap_scalar(values, stage, seed + offset * 1000 + index) for index, stage in enumerate(stages)]
        means = np.asarray([item[0] for item in stats])
        lows = np.asarray([item[1] for item in stats])
        highs = np.asarray([item[2] for item in stats])
        shift = -0.07 if offset == 0 else 0.07
        ax.errorbar(x + shift, means, yerr=np.vstack([means - lows, highs - means]), marker="o", capsize=3, label=name, color=color)
        results[name] = {stage: {"mean": stats[i][0], "ci95": [stats[i][1], stats[i][2]]} for i, stage in enumerate(stages)}
    labels = [stage.replace("vision_", "").replace("_pre_reverse", "") for stage in stages]
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Adjacent minus far-bin cosine similarity")
    ax.set_title(f"Encoder local temporal advantage (n={len(artifacts)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return results


def layer14_diagnostics(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        groups[str(artifact.get("question_type", "unknown"))].append(artifact)
    for key, items in sorted(groups.items()):
        rows = []
        for layer in (13, 14, 15):
            entropy = np.asarray([metric_curve(a, "normalized_temporal_entropy")[layer] for a in items])
            mass = np.asarray([metric_curve(a, "absolute_visual_mass")[layer] for a in items])
            rows.append({
                "layer": layer,
                "entropy_mean": float(entropy.mean()),
                "entropy_median": float(np.median(entropy)),
                "entropy_iqr": [float(np.percentile(entropy, 25)), float(np.percentile(entropy, 75))],
                "absolute_visual_mass_mean": float(mass.mean()),
                "absolute_visual_mass_median": float(np.median(mass)),
            })
        output[key] = {"n": len(items), "layers": rows}
    return output


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts(output_dir)
    validation = validate_artifacts(artifacts, args.expected_artifacts)

    encoder = np.mean(np.stack([encoder_lift_matrix(a, args.grid_bins) for a in artifacts]), axis=0)
    decoder = np.mean(np.stack([decoder_lift_matrix(a, args.grid_bins) for a in artifacts]), axis=0)
    encoder_scale = save_lift_heatmap(encoder, plot_dir / "encoder_attention_lift_heatmap.png", "Encoder temporal attention relative to uniform", len(artifacts))
    decoder_scale = save_lift_heatmap(decoder, plot_dir / "decoder_attention_lift_heatmap.png", "Decoder temporal attention relative to uniform", len(artifacts))
    save_curve_groups(artifacts, "normalized_temporal_entropy", "Normalized temporal entropy", "Decoder temporal entropy by question type", plot_dir / "decoder_entropy_by_question_type.png", args.bootstrap_samples, args.seed)
    save_curve_groups(artifacts, "absolute_visual_mass", "Absolute question-to-visual attention mass", "Decoder visual access by question type", plot_dir / "decoder_absolute_visual_mass.png", args.bootstrap_samples, args.seed + 10000)
    encoder_advantage = save_encoder_advantage(artifacts, plot_dir / "encoder_local_temporal_advantage_ci.png", args.bootstrap_samples, args.seed + 20000)

    diagnostics = {
        "validation": validation,
        "normalization": "For each example and layer: lift = B * p(bin); plotted heatmaps show lift - 1 after interpolation at relative-time bin centers. Uniform attention is zero.",
        "aggregation": "Each source video receives equal weight. Curve intervals use hierarchical participant-then-video bootstrap.",
        "encoder_heatmap": encoder_scale,
        "decoder_heatmap": decoder_scale,
        "encoder_adjacent_vs_far": encoder_advantage,
        "layer_14_check": layer14_diagnostics(artifacts),
        "guardrails": [
            "Absolute duration remains confounded with question type in this 77-video manifest.",
            "Question types with fewer than five examples are excluded from stratified curves.",
            "Attention concentration is descriptive until positional, query-mismatch, and causal controls are complete.",
            "Encoder representation similarity is available only at the five captured stages, not all 32 blocks.",
        ],
    }
    (plot_dir / "corrected_plot_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({"plot_dir": str(plot_dir), **validation, "files": sorted(p.name for p in plot_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
