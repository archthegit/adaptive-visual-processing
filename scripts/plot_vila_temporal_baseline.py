#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
}

EXPECTED_CATEGORY_COUNTS = {
    "fine_grained": 33,
    "gaze": 13,
    "ingredient": 13,
    "object_motion": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VILA temporal baseline diagnostics for Experiment 1 v3.")
    parser.add_argument("--input-dir", default="outputs/experiment1_v3_cross_model/runs/vila_baseline")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_cross_model/preliminary/vila_baseline")
    parser.add_argument("--docs-output", default="docs/experiment1_v3_vila_baseline.md")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--expected-complete", type=int, default=71)
    parser.add_argument("--expected-failed", type=int, default=6)
    return parser.parse_args()


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _artifact_path(raw: str, input_dir: Path) -> Path:
    path = Path(raw)
    candidates = (path, input_dir / path.name, input_dir.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve artifact {raw!r} from {input_dir}")


def _category(question_type: str, artifact: dict[str, Any] | None = None) -> str:
    if artifact and artifact.get("category"):
        return str(artifact["category"])
    if question_type.startswith("fine_grained_"):
        return "fine_grained"
    if question_type.startswith("gaze_"):
        return "gaze"
    if question_type.startswith("ingredient_"):
        return "ingredient"
    if question_type.startswith("object_motion_"):
        return "object_motion"
    return "unknown"


def source_video_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    return str(clips[0].get("video_id", "unknown")) if clips else "unknown"


def participant_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    if clips and clips[0].get("participant_id"):
        return str(clips[0]["participant_id"])
    return source_video_id(artifact).split("-", 1)[0]


def load_latest_records(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records_path = input_dir / "records.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"Missing records file: {records_path}")
    latest_complete: dict[str, dict[str, Any]] = {}
    latest_any: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = str(record.get("question_id"))
        latest_any[qid] = record
        if record.get("status") == "complete" and record.get("artifact"):
            latest_complete[qid] = record
    failed = [
        record
        for qid, record in sorted(latest_any.items())
        if qid not in latest_complete and record.get("status") == "failed"
    ]
    artifacts = [json.loads(_artifact_path(latest_complete[qid]["artifact"], input_dir).read_text()) for qid in sorted(latest_complete)]
    return artifacts, failed, [latest_complete[qid] for qid in sorted(latest_complete)]


def _as_2d_scores(artifact: dict[str, Any]) -> np.ndarray:
    scores = np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"], dtype=np.float64)
    if scores.shape != (32, 8):
        raise RuntimeError(f"{artifact.get('question_id')}: expected normalized_temporal_bin_scores shape [32,8], got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{artifact.get('question_id')}: non-finite normalized temporal distribution")
    if not np.allclose(scores.sum(axis=1), np.ones(32), atol=1e-6):
        raise RuntimeError(f"{artifact.get('question_id')}: normalized temporal distributions do not sum to one")
    return scores


def _absolute_mass(artifact: dict[str, Any]) -> np.ndarray:
    mass = np.asarray(artifact["temporal_relevance"]["absolute_question_to_visual_attention_mass"], dtype=np.float64)
    if mass.shape != (32,) or not np.all(np.isfinite(mass)):
        raise RuntimeError(f"{artifact.get('question_id')}: invalid absolute visual mass shape {mass.shape}")
    return mass


def _sampled_entries(values: Any, label: str, question_id: str) -> list[Any]:
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        entries = list(values[0])
    else:
        entries = list(values)
    if len(entries) != 8:
        raise RuntimeError(f"{question_id}: expected 8 sampled {label}, found {len(entries)}")
    if any(entries[i] > entries[i + 1] for i in range(len(entries) - 1)):
        raise RuntimeError(f"{question_id}: sampled {label} are not ordered")
    return entries


def validate_artifacts(
    artifacts: list[dict[str, Any]],
    failed_records: list[dict[str, Any]],
    *,
    expected_complete: int = 71,
    expected_failed: int = 6,
    expected_category_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    expected_category_counts = expected_category_counts or EXPECTED_CATEGORY_COUNTS
    if len(artifacts) != expected_complete:
        raise RuntimeError(f"Expected {expected_complete} complete artifacts, found {len(artifacts)}")
    if len(failed_records) != expected_failed:
        raise RuntimeError(f"Expected {expected_failed} failed records, found {len(failed_records)}")

    videos: set[str] = set()
    completed_question_ids: list[str] = []
    categories = Counter()
    qtypes = Counter()
    correct = 0
    validation_rows = []
    for artifact in artifacts:
        qid = str(artifact.get("question_id"))
        metadata = artifact.get("metadata") or {}
        if metadata.get("model_backend") != "vila_llama3":
            raise RuntimeError(f"{qid}: model_backend is not vila_llama3")
        if int(metadata.get("num_decoder_layers", -1)) != 32:
            raise RuntimeError(f"{qid}: expected 32 decoder layers")
        bins = int((artifact.get("temporal_relevance", {}).get("metadata") or {}).get("num_temporal_bins", -1))
        if bins != 8:
            raise RuntimeError(f"{qid}: expected exactly 8 temporal bins, found {bins}")
        scores = _as_2d_scores(artifact)
        mass = _absolute_mass(artifact)
        _sampled_entries(artifact.get("sampled_frame_indices") or [], "frame indices", qid)
        _sampled_entries(artifact.get("sampled_timestamps") or [], "timestamps", qid)
        video = source_video_id(artifact)
        if video in videos:
            raise RuntimeError(f"Duplicate source video: {video}")
        videos.add(video)
        question_type = str(artifact.get("question_type", "unknown"))
        category = _category(question_type, artifact)
        categories[category] += 1
        qtypes[question_type] += 1
        completed_question_ids.append(qid)
        correct += int(bool(artifact.get("correct")))
        validation_rows.append(
            {
                "question_id": qid,
                "source_video_id": video,
                "category": category,
                "question_type": question_type,
                "decoder_layers": scores.shape[0],
                "temporal_bins": scores.shape[1],
                "absolute_mass_layers": mass.shape[0],
            }
        )
    if dict(categories) != expected_category_counts:
        raise RuntimeError(f"Category counts differ from expected: observed={dict(categories)}, expected={expected_category_counts}")
    return {
        "num_complete_artifacts": len(artifacts),
        "num_failed_records": len(failed_records),
        "num_unique_source_videos": len(videos),
        "completed_question_ids": sorted(completed_question_ids),
        "failed_question_ids": sorted(str(item.get("question_id")) for item in failed_records),
        "failure_reasons": {str(item.get("question_id")): str(item.get("error") or item.get("failure_reason") or "unknown") for item in failed_records},
        "category_counts": dict(sorted(categories.items())),
        "question_type_counts": dict(sorted(qtypes.items())),
        "overall_accuracy": {"correct": correct, "total": len(artifacts), "accuracy": correct / len(artifacts)},
        "validation_rows": validation_rows,
    }


def normalized_entropy(distribution: np.ndarray) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / np.log(values.size))


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
        for participant_index in rng.integers(0, len(participants), size=len(participants)):
            group = grouped[participants[int(participant_index)]]
            for video_index in rng.integers(0, len(group), size=len(group)):
                selected.append(group[int(video_index)])
        boot.append(np.stack([extractor(item) for item in selected]).mean(axis=0))
    stacked = np.stack(boot)
    return mean, np.percentile(stacked, 2.5, axis=0), np.percentile(stacked, 97.5, axis=0)


def entropy_curve(artifact: dict[str, Any]) -> np.ndarray:
    return np.asarray([normalized_entropy(layer) for layer in _as_2d_scores(artifact)], dtype=np.float64)


def top_bin_mass_curve(artifact: dict[str, Any]) -> np.ndarray:
    return np.max(_as_2d_scores(artifact), axis=1)


def first_bin_curve(artifact: dict[str, Any]) -> np.ndarray:
    return _as_2d_scores(artifact)[:, 0]


def last_bin_curve(artifact: dict[str, Any]) -> np.ndarray:
    return _as_2d_scores(artifact)[:, -1]


def group_curves_by_question_type(artifacts: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        grouped[str(artifact.get("question_type", "unknown"))].append(artifact)
    return [("all", artifacts)] + [(key, items) for key, items in sorted(grouped.items()) if len(items) >= 5]


def save_lift_heatmap(mean_distribution: np.ndarray, path: Path, n: int) -> dict[str, Any]:
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm

    lift = 8.0 * mean_distribution
    deviation = lift - 1.0
    observed = float(np.max(np.abs(deviation)))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    image = ax.imshow(
        deviation,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max(0.05, observed), vcenter=0.0, vmax=max(0.05, observed)),
    )
    ax.set_title(f"VILA decoder temporal attention relative to uniform (n={n})")
    ax.set_xlabel("Temporal bin")
    ax.set_ylabel("Decoder layer")
    ax.set_xticks(np.arange(8), [str(index) for index in range(8)])
    fig.colorbar(image, ax=ax, label="Attention lift minus uniform (8 p - 1)")
    ax.text(0.0, -0.18, f"Uniform attention is zero. Max |8p - 1| = {observed:.3g}.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"max_absolute_lift_minus_uniform_deviation": observed}


def save_curve_by_question_type(
    artifacts: list[dict[str, Any]],
    extractor: Callable[[dict[str, Any]], np.ndarray],
    path: Path,
    *,
    ylabel: str,
    title: str,
    samples: int,
    seed: int,
    reference: float | None = None,
) -> dict[str, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    output: dict[str, Any] = {}
    for offset, (key, items) in enumerate(group_curves_by_question_type(artifacts)):
        mean, low, high = hierarchical_curve_ci(items, extractor, samples, seed + offset * 101)
        layers = np.arange(mean.size)
        label = f"{QUESTION_TYPE_LABELS.get(key, key)} (n={len(items)})"
        kwargs = {"color": "black", "linewidth": 2.7} if key == "all" else {"linewidth": 1.4, "alpha": 0.9}
        ax.plot(layers, mean, label=label, **kwargs)
        ax.fill_between(layers, low, high, alpha=0.16 if key == "all" else 0.10, color=kwargs.get("color"))
        output[key] = {
            "n": len(items),
            "mean": mean.tolist(),
            "ci95_low": low.tolist(),
            "ci95_high": high.tolist(),
        }
    if reference is not None:
        ax.axhline(reference, color="0.5", linestyle="--", linewidth=1.0, label=f"uniform={reference:g}")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_first_last_bin_mass(artifacts: list[dict[str, Any]], path: Path, samples: int, seed: int) -> dict[str, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    output: dict[str, Any] = {}
    for offset, (label, extractor, color) in enumerate(
        (("first bin", first_bin_curve, "tab:blue"), ("last bin", last_bin_curve, "tab:orange"))
    ):
        mean, low, high = hierarchical_curve_ci(artifacts, extractor, samples, seed + offset * 103)
        layers = np.arange(mean.size)
        ax.plot(layers, mean, label=label, color=color, linewidth=2.0)
        ax.fill_between(layers, low, high, color=color, alpha=0.16)
        output[label.replace(" ", "_")] = {"mean": mean.tolist(), "ci95_low": low.tolist(), "ci95_high": high.tolist()}
    ax.axhline(1 / 8, color="0.5", linestyle="--", linewidth=1.0, label="uniform=0.125")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Temporal-bin mass")
    ax.set_title(f"VILA first- and last-bin mass (n={len(artifacts)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_top_bin_position(artifacts: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    plt = _plt()
    counts = np.zeros((32, 8), dtype=np.float64)
    for artifact in artifacts:
        top_bins = np.argmax(_as_2d_scores(artifact), axis=1)
        for layer, bin_index in enumerate(top_bins):
            counts[layer, int(bin_index)] += 1.0
    fractions = counts / len(artifacts)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    image = ax.imshow(fractions, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("VILA top temporal-bin position by decoder layer")
    ax.set_xlabel("Top-ranked temporal bin")
    ax.set_ylabel("Decoder layer")
    ax.set_xticks(np.arange(8), [str(index) for index in range(8)])
    fig.colorbar(image, ax=ax, label="Fraction of examples")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"fraction_by_layer_bin": fractions.tolist()}


def save_accuracy_by_category(artifacts: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    plt = _plt()
    rows: list[tuple[str, int, int, float]] = []
    rows.append(("overall", sum(int(bool(a.get("correct"))) for a in artifacts), len(artifacts), 0.0))
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_category[_category(str(artifact.get("question_type", "unknown")), artifact)].append(artifact)
    for category, items in sorted(by_category.items()):
        rows.append((category, sum(int(bool(a.get("correct"))) for a in items), len(items), 0.0))
    rows = [(label, correct, total, correct / total if total else 0.0) for label, correct, total, _ in rows]
    labels = [row[0].replace("_", " ") for row in rows]
    values = [row[3] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=["black"] + ["tab:blue"] * (len(rows) - 1), alpha=0.82)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("VILA baseline VQA accuracy (descriptive)")
    for bar, (_, correct, total, accuracy) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{correct}/{total}\n{accuracy:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        label: {"correct": correct, "total": total, "accuracy": accuracy}
        for label, correct, total, accuracy in rows
    }


def aggregate_diagnostics(artifacts: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    scores = np.stack([_as_2d_scores(artifact) for artifact in artifacts])
    mean_distribution = scores.mean(axis=0)
    entropy_mean, entropy_low, entropy_high = hierarchical_curve_ci(artifacts, entropy_curve, samples, seed)
    abs_mean, abs_low, abs_high = hierarchical_curve_ci(artifacts, _absolute_mass, samples, seed + 1000)
    top_mass = np.stack([top_bin_mass_curve(artifact) for artifact in artifacts]).mean(axis=0)
    top_lift = 8.0 * top_mass
    first_mass = np.stack([first_bin_curve(artifact) for artifact in artifacts]).mean(axis=0)
    last_mass = np.stack([last_bin_curve(artifact) for artifact in artifacts]).mean(axis=0)
    deviation = 8.0 * mean_distribution - 1.0
    return {
        "per_layer_mean_eight_bin_distributions": mean_distribution.tolist(),
        "per_layer_entropy_mean": entropy_mean.tolist(),
        "per_layer_entropy_ci95_low": entropy_low.tolist(),
        "per_layer_entropy_ci95_high": entropy_high.tolist(),
        "per_layer_top_bin_mass": top_mass.tolist(),
        "per_layer_top_bin_lift": top_lift.tolist(),
        "per_layer_first_bin_mass": first_mass.tolist(),
        "per_layer_last_bin_mass": last_mass.tolist(),
        "per_layer_absolute_visual_mass_mean": abs_mean.tolist(),
        "per_layer_absolute_visual_mass_ci95_low": abs_low.tolist(),
        "per_layer_absolute_visual_mass_ci95_high": abs_high.tolist(),
        "layer_with_lowest_entropy": int(np.argmin(entropy_mean)),
        "layer_with_maximum_top_bin_lift": int(np.argmax(top_lift)),
        "layer_with_strongest_first_bin_preference": int(np.argmax(first_mass)),
        "layer_with_strongest_last_bin_preference": int(np.argmax(last_mass)),
        "maximum_absolute_lift_minus_uniform_deviation": float(np.max(np.abs(deviation))),
    }


def write_report(path: Path, diagnostics: dict[str, Any]) -> None:
    max_dev = diagnostics["maximum_absolute_lift_minus_uniform_deviation"]
    max_lift = 1.0 + max_dev
    lowest_entropy = diagnostics["layer_with_lowest_entropy"]
    max_top = diagnostics["layer_with_maximum_top_bin_lift"]
    first_layer = diagnostics["layer_with_strongest_first_bin_preference"]
    last_layer = diagnostics["layer_with_strongest_last_bin_preference"]
    validation = diagnostics["validation"]
    accuracy = validation["overall_accuracy"]
    path.write_text(
        f"""# Experiment 1 v3: VILA Baseline Decoder Analysis

This document summarizes the VILA-Llama3 baseline generated by
`scripts/plot_vila_temporal_baseline.py`.

Input run: `outputs/experiment1_v3_cross_model/runs/vila_baseline`

Output figures: `outputs/experiment1_v3_cross_model/preliminary/vila_baseline`

## Scope

This is a decoder-only cross-architecture baseline for
`Efficient-Large-Model/Llama-3-VILA1.5-8B`. It uses the frozen Experiment 1 v3
question/video set where VILA can accept the input: {validation['num_complete_artifacts']}
complete artifacts from {validation['num_unique_source_videos']} unique source
videos. The six sampling-ineligible failures are recorded in
`vila_baseline_diagnostics.json` and are not included in denominators.

VILA uses exactly eight temporal bins and one deterministic center frame per bin.
No VILA encoder measurements are produced or implied.

## Decoder Temporal Allocation

![Decoder attention lift heatmap](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_attention_lift_heatmap.png)

The VILA decoder attention is nonuniform if the heatmap departs from zero.
For this run, the maximum absolute mean lift-minus-uniform deviation is
`{max_dev:.6g}`. Since lift is `8 p(bin)`, the most emphasized averaged
position reaches approximately `{max_lift:.6g}x` the uniform expectation.

The temporal allocation changes with depth. The lowest mean entropy occurs at
decoder layer `{lowest_entropy}`, while the maximum top-bin lift occurs at
decoder layer `{max_top}`. This should be read as a descriptive temporal
allocation pattern, not content relevance.

## Entropy By Question Type

![Decoder entropy by question type](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_entropy_by_question_type.png)

The entropy curves show whether temporal concentration changes
non-monotonically with decoder depth. Question-type curves are shown only for
types with at least five examples. Differences across question types are
descriptive because task type, duration, and answerability may be confounded.

## Absolute Visual Access

![Decoder absolute visual mass](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_absolute_visual_mass.png)

Absolute question-token-to-visual-token attention mass is kept separate from the
conditional temporal distribution. It is not normalized across temporal bins.
If this curve differs from entropy or first/last-bin curves, then total visual
access and conditional temporal allocation are not following the same depth
profile.

## Early Versus Late Bins

![First and last bin mass](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_first_last_bin_mass.png)

The strongest first-bin preference occurs at layer `{first_layer}`. The
strongest last-bin preference occurs at layer `{last_layer}`. These curves
answer whether early layers prefer early bins and whether later layers shift
toward later bins, but only descriptively.

![Top bin position](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_top_bin_position.png)

The top-bin position heatmap shows the fraction of examples whose top-ranked bin
is each of bins 0 through 7 at every decoder layer. It is the clearest plot for
detecting early-to-late movement of preferred positions.

## Accuracy

![Accuracy by category](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/accuracy_by_category.png)

Overall descriptive accuracy is `{accuracy['correct']}/{accuracy['total']}`
(`{accuracy['accuracy']:.2%}`). Category-level numerators and denominators are
shown in the figure and diagnostics.

## Relationship To Qwen

This baseline can qualitatively reproduce the Qwen pattern only if it shows
depth-dependent, nonuniform decoder temporal allocation with separable absolute
visual-access dynamics. The generated diagnostics provide the numerical basis
for that comparison. This document does not claim that VILA and Qwen match
quantitatively, and it does not claim that attention is content relevance.

## Required Controls

Repeated-frame, reversed-video, and matched-Qwen controls are still required.
This baseline alone does not establish content relevance, causal importance, or
evidence for pruning.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts, failed_records, _complete_records = load_latest_records(input_dir)
    validation = validate_artifacts(
        artifacts,
        failed_records,
        expected_complete=args.expected_complete,
        expected_failed=args.expected_failed,
    )
    mean_distribution = np.stack([_as_2d_scores(artifact) for artifact in artifacts]).mean(axis=0)
    heatmap = save_lift_heatmap(mean_distribution, output_dir / "decoder_attention_lift_heatmap.png", len(artifacts))
    entropy_curves = save_curve_by_question_type(
        artifacts,
        entropy_curve,
        output_dir / "decoder_entropy_by_question_type.png",
        ylabel="Normalized temporal entropy",
        title="VILA decoder temporal entropy by question type",
        samples=args.bootstrap_samples,
        seed=args.seed,
        reference=1.0,
    )
    absolute_curves = save_curve_by_question_type(
        artifacts,
        _absolute_mass,
        output_dir / "decoder_absolute_visual_mass.png",
        ylabel="Absolute question-to-visual attention mass",
        title="VILA decoder absolute visual access by question type",
        samples=args.bootstrap_samples,
        seed=args.seed + 10000,
    )
    first_last = save_first_last_bin_mass(artifacts, output_dir / "decoder_first_last_bin_mass.png", args.bootstrap_samples, args.seed + 20000)
    top_positions = save_top_bin_position(artifacts, output_dir / "decoder_top_bin_position.png")
    accuracy = save_accuracy_by_category(artifacts, output_dir / "accuracy_by_category.png")

    diagnostics = {
        "validation": validation,
        "normalization": "Fixed-bin VILA baseline: lift = 8 * p(bin); heatmap displays lift - 1. Uniform attention is zero.",
        "aggregation": "Each source video receives equal weight. Confidence intervals use participant-then-video hierarchical bootstrap.",
        **aggregate_diagnostics(artifacts, args.bootstrap_samples, args.seed + 30000),
        "decoder_attention_lift_heatmap": heatmap,
        "entropy_by_question_type": entropy_curves,
        "absolute_visual_mass_by_question_type": absolute_curves,
        "first_last_bin_mass": first_last,
        "top_bin_position": top_positions,
        "accuracy": accuracy,
        "guardrails": [
            "Failed records are excluded from denominators.",
            "Attention is descriptive and is not evidence of content relevance, causal importance, or pruning utility.",
            "Repeated-frame, reversed-video, and matched-Qwen controls remain required.",
            "VILA encoder measurements are not available in this analysis.",
        ],
    }
    (output_dir / "vila_baseline_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.docs_output), diagnostics)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "docs_output": args.docs_output,
                "complete": validation["num_complete_artifacts"],
                "failed": validation["num_failed_records"],
                "files": sorted(path.name for path in output_dir.iterdir()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
