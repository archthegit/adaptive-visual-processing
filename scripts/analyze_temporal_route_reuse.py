#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json_atomic


LAYER_GAPS = (1, 2, 4, 8)
RETENTION_RATIOS = (0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal route reuse across decoder layers.")
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--vila-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--expected-matched-examples", type=int, default=71)
    parser.add_argument("--expected-dev-examples", type=int, default=15)
    return parser.parse_args()


def _plt():
    import matplotlib.pyplot as plt

    return plt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_video_id(artifact: dict[str, Any]) -> str:
    clips = artifact.get("video_clip") or []
    if clips:
        return str(clips[0].get("video_id") or artifact.get("question_id"))
    return str(artifact.get("question_id"))


def participant_id_from_video(video_id: str) -> str:
    return str(video_id).split("-", 1)[0]


def _artifact_path(raw: str, output_dir: Path) -> Path:
    path = Path(raw)
    candidates = (path, output_dir / path.name, output_dir.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve artifact {raw!r} from {output_dir}")


def load_latest_complete_artifacts(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    output_path = Path(output_dir)
    records_path = output_path / "records.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if records_path.is_file():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "complete" and record.get("artifact"):
                latest[str(record["question_id"])] = record
        return {
            question_id: json.loads(_artifact_path(record["artifact"], output_path).read_text())
            for question_id, record in sorted(latest.items())
        }
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output_path.glob("*.json")):
        if path.name in {"summary.json", "profile.json"}:
            continue
        payload = json.loads(path.read_text())
        if payload.get("question_id"):
            artifacts[str(payload["question_id"])] = payload
    return artifacts


def category_from_question_type(question_type: str) -> str:
    if question_type.startswith("fine_grained_"):
        return "fine_grained"
    if question_type.startswith("gaze_"):
        return "gaze"
    if question_type.startswith("ingredient_"):
        return "ingredient"
    if question_type.startswith("object_motion_"):
        return "object_motion"
    return "unknown"


def manifest_by_question(path: str | Path) -> dict[str, dict[str, Any]]:
    output = {}
    for record in read_jsonl(path):
        output[str(record["question_id"])] = record
    return output


def dev_question_ids(path: str | Path) -> set[str]:
    return {str(record["question_id"]) for record in read_jsonl(path)}


def normalized_distributions(artifact: dict[str, Any]) -> np.ndarray:
    values = np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"{artifact.get('question_id')}: expected normalized temporal scores with shape [layers,8], got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{artifact.get('question_id')}: non-finite normalized temporal scores.")
    if not np.allclose(values.sum(axis=1), np.ones(values.shape[0]), atol=1e-6):
        raise ValueError(f"{artifact.get('question_id')}: temporal distributions do not sum to one.")
    return values


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        ranks[order[index:end]] = rank
        index = end
    return ranks


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Spearman inputs must have matching shapes.")
    if a.size <= 1:
        return 1.0
    ar = rankdata(a)
    br = rankdata(b)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = math.sqrt(float(np.sum(ar * ar) * np.sum(br * br)))
    return float(np.sum(ar * br) / denom) if denom > 0 else 0.0


def jensen_shannon_divergence(left: Iterable[float], right: Iterable[float]) -> float:
    p = np.asarray(list(left), dtype=np.float64)
    q = np.asarray(list(right), dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("JSD inputs must have matching shapes.")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def topk_indices(values: Iterable[float], k: int) -> tuple[int, ...]:
    arr = np.asarray(list(values), dtype=np.float64)
    if not 1 <= k <= arr.size:
        raise ValueError(f"k must be in [1,{arr.size}], got {k}.")
    # Stable tie-break toward lower bin index.
    return tuple(int(index) for index in np.lexsort((np.arange(arr.size), -arr))[:k])


def topk_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a = set(int(item) for item in left)
    b = set(int(item) for item in right)
    return float(len(a & b) / len(a | b)) if a or b else 1.0


def captured_mass(distribution: Iterable[float], indices: Iterable[int]) -> float:
    arr = np.asarray(list(distribution), dtype=np.float64)
    return float(np.sum(arr[list(indices)]))


def retention_to_k(ratio: float, num_bins: int = 8) -> int:
    return int(round(float(ratio) * num_bins))


def comparable_target_layers(num_layers: int) -> tuple[int, ...]:
    if num_layers <= max(LAYER_GAPS):
        raise ValueError(f"Need more than {max(LAYER_GAPS)} decoder layers for comparable layer-gap analysis, got {num_layers}.")
    return tuple(range(max(LAYER_GAPS), num_layers))


def build_static_priors(
    artifacts_by_model: dict[str, dict[str, dict[str, Any]]],
    dev_ids: set[str],
    matched_ids: set[str],
) -> dict[str, np.ndarray]:
    priors: dict[str, np.ndarray] = {}
    for model, artifacts in artifacts_by_model.items():
        eligible = sorted(dev_ids & matched_ids & set(artifacts))
        if not eligible:
            raise ValueError(f"No development examples are available to build the {model} static prior.")
        arrays = [normalized_distributions(artifacts[question_id]) for question_id in eligible]
        layer_counts = {array.shape[0] for array in arrays}
        if len(layer_counts) != 1:
            raise ValueError(f"{model} development artifacts have inconsistent decoder layer counts: {sorted(layer_counts)}")
        priors[model] = np.mean(np.stack(arrays), axis=0)
    return priors


def _record_metadata(question_id: str, artifact: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = manifest.get(question_id, {})
    question_type = str(record.get("question_type") or artifact.get("question_type") or "unknown")
    video_id = str(record.get("source_video_id") or source_video_id(artifact))
    return {
        "question_id": question_id,
        "source_video_id": video_id,
        "participant_id": str(record.get("participant_id") or participant_id_from_video(video_id)),
        "category": str(record.get("category") or artifact.get("category") or category_from_question_type(question_type)),
        "question_type": question_type,
        "duration_group": str(record.get("duration_group") or record.get("duration_bucket") or "unknown"),
        "manifest_split": str(record.get("split") or "unknown"),
    }


def dynamic_route_rows(
    model: str,
    question_id: str,
    artifact: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    scores = normalized_distributions(artifact)
    target_layers = comparable_target_layers(scores.shape[0])
    rows = []
    for gap in LAYER_GAPS:
        for target_layer in target_layers:
            source_layer = target_layer - gap
            source = scores[source_layer]
            target = scores[target_layer]
            source_top_by_ratio = {ratio: topk_indices(source, retention_to_k(ratio)) for ratio in RETENTION_RATIOS}
            target_top_by_ratio = {ratio: topk_indices(target, retention_to_k(ratio)) for ratio in RETENTION_RATIOS}
            for ratio in RETENTION_RATIOS:
                source_top = source_top_by_ratio[ratio]
                target_top = target_top_by_ratio[ratio]
                reused = captured_mass(target, source_top)
                oracle = captured_mass(target, target_top)
                rows.append(
                    {
                        **metadata,
                        "model": model,
                        "selection_method": "dynamic_source_layer_topk",
                        "source_layer": source_layer,
                        "target_layer": target_layer,
                        "layer_gap": gap,
                        "retention_ratio": ratio,
                        "k": retention_to_k(ratio),
                        "spearman": spearman_correlation(source, target),
                        "jensen_shannon_divergence": jensen_shannon_divergence(source, target),
                        "topk_jaccard": topk_jaccard(source_top, target_top),
                        "reused_captured_mass": reused,
                        "oracle_captured_mass": oracle,
                        "random_expected_captured_mass": ratio,
                        "reuse_efficiency": reused / oracle if oracle > 0 else 0.0,
                        "selected_bins": list(source_top),
                        "oracle_bins": list(target_top),
                        "static_prior_training_question_ids": [],
                        "static_prior_training_participant_ids": [],
                        "static_prior_training_source_video_ids": [],
                    }
                )
    return rows


def static_prior_rows(
    model: str,
    question_id: str,
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    prior: np.ndarray,
    training_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores = normalized_distributions(artifact)
    if prior.shape != scores.shape:
        raise ValueError(f"{model} static prior shape {prior.shape} does not match target scores shape {scores.shape}.")
    target_layers = comparable_target_layers(scores.shape[0])
    training_question_ids = sorted(item["question_id"] for item in training_metadata)
    training_participant_ids = sorted({item["participant_id"] for item in training_metadata})
    training_source_video_ids = sorted({item["source_video_id"] for item in training_metadata})
    rows = []
    for target_layer in target_layers:
        target = scores[target_layer]
        prior_distribution = prior[target_layer]
        for ratio in RETENTION_RATIOS:
            prior_top = topk_indices(prior_distribution, retention_to_k(ratio))
            target_top = topk_indices(target, retention_to_k(ratio))
            reused = captured_mass(target, prior_top)
            oracle = captured_mass(target, target_top)
            rows.append(
                {
                    **metadata,
                    "model": model,
                    "selection_method": "static_dev_prior_lopo",
                    "source_layer": None,
                    "target_layer": target_layer,
                    "layer_gap": None,
                    "retention_ratio": ratio,
                    "k": retention_to_k(ratio),
                    "spearman": spearman_correlation(prior_distribution, target),
                    "jensen_shannon_divergence": jensen_shannon_divergence(prior_distribution, target),
                    "topk_jaccard": topk_jaccard(prior_top, target_top),
                    "reused_captured_mass": reused,
                    "oracle_captured_mass": oracle,
                    "random_expected_captured_mass": ratio,
                    "reuse_efficiency": reused / oracle if oracle > 0 else 0.0,
                    "selected_bins": list(prior_top),
                    "oracle_bins": list(target_top),
                    "static_prior_training_question_ids": training_question_ids,
                    "static_prior_training_participant_ids": training_participant_ids,
                    "static_prior_training_source_video_ids": training_source_video_ids,
                }
            )
    return rows


def build_lopo_static_prior(
    model: str,
    artifacts: dict[str, dict[str, Any]],
    dev_ids: set[str],
    manifest: dict[str, dict[str, Any]],
    held_out_metadata: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    training_metadata = []
    arrays = []
    for question_id in sorted(dev_ids & set(artifacts)):
        metadata = _record_metadata(question_id, artifacts[question_id], manifest)
        if metadata["participant_id"] == held_out_metadata["participant_id"]:
            continue
        if metadata["source_video_id"] == held_out_metadata["source_video_id"]:
            continue
        training_metadata.append(metadata)
        arrays.append(normalized_distributions(artifacts[question_id]))
    if not arrays:
        raise ValueError(
            f"No development examples remain to build {model} static prior with participant "
            f"{held_out_metadata['participant_id']} held out."
        )
    layer_counts = {array.shape[0] for array in arrays}
    if len(layer_counts) != 1:
        raise ValueError(f"{model} LOP training artifacts have inconsistent decoder layer counts: {sorted(layer_counts)}")
    return np.mean(np.stack(arrays), axis=0), training_metadata


def route_reuse_rows(
    qwen_artifacts: dict[str, dict[str, Any]],
    vila_artifacts: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    dev_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matched_ids = set(qwen_artifacts) & set(vila_artifacts) & set(manifest)
    artifacts_by_model = {"qwen": qwen_artifacts, "vila": vila_artifacts}
    matched_dev_ids = dev_ids & matched_ids
    matched_non_dev_ids = matched_ids - dev_ids
    if not matched_dev_ids:
        raise ValueError("No matched development examples are available for route-reuse analysis.")

    rows: list[dict[str, Any]] = []
    for model, artifacts in artifacts_by_model.items():
        for question_id in sorted(matched_dev_ids & set(artifacts)):
            metadata = _record_metadata(question_id, artifacts[question_id], manifest)
            metadata["is_static_prior_dev_example"] = True
            rows.extend(dynamic_route_rows(model, question_id, artifacts[question_id], metadata))
            prior, training_metadata = build_lopo_static_prior(model, artifacts, matched_dev_ids, manifest, metadata)
            rows.extend(static_prior_rows(model, question_id, artifacts[question_id], metadata, prior, training_metadata))
    participants = {
        _record_metadata(question_id, qwen_artifacts[question_id], manifest)["participant_id"]
        for question_id in matched_dev_ids & set(qwen_artifacts)
    }
    diagnostics = {
        "matched_complete_examples": len(matched_ids),
        "development_examples_analyzed": sorted(matched_dev_ids),
        "non_development_examples_held_out": sorted(matched_non_dev_ids),
        "non_development_examples_held_out_count": len(matched_non_dev_ids),
        "development_participants": sorted(participants),
        "models": {
            "qwen": {
                "complete_artifacts": len(qwen_artifacts),
                "layers": int(normalized_distributions(qwen_artifacts[sorted(matched_dev_ids & set(qwen_artifacts))[0]]).shape[0]),
            },
            "vila": {
                "complete_artifacts": len(vila_artifacts),
                "layers": int(normalized_distributions(vila_artifacts[sorted(matched_dev_ids & set(vila_artifacts))[0]]).shape[0]),
            },
        },
        "layer_gaps": list(LAYER_GAPS),
        "comparable_target_layers_start": max(LAYER_GAPS),
        "retention_ratios": list(RETENTION_RATIOS),
        "dynamic_route_rule": "Development examples only; compare every layer gap on identical target layers from 8 through the final decoder layer.",
        "static_prior_rule": "Leave-one-participant-out within the 15-example development set; per model and target layer, average normalized 8-bin distributions over other development participants only, excluding the evaluated source video, then select top-k prior bins.",
    }
    return rows, diagnostics


def bootstrap_ci(rows: list[dict[str, Any]], field: str, samples: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"mean": None, "ci95": [None, None], "n": 0}
    by_participant: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_participant[str(row["participant_id"])][str(row["source_video_id"])].append(float(row[field]))
    participants = sorted(by_participant)
    per_video = [
        float(np.mean(values))
        for participant in participants
        for values in by_participant[participant].values()
    ]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = []
        for participant_index in rng.integers(0, len(participants), size=len(participants)):
            participant = participants[int(participant_index)]
            videos = sorted(by_participant[participant])
            for video_index in rng.integers(0, len(videos), size=len(videos)):
                selected.append(float(np.mean(by_participant[participant][videos[int(video_index)]])))
        estimates.append(float(np.mean(selected)))
    return {
        "mean": float(np.mean(per_video)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": len(per_video),
    }


def aggregate_summary(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    group_fields = ("model", "selection_method", "layer_gap", "retention_ratio", "category", "duration_group")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    metrics = (
        "spearman",
        "jensen_shannon_divergence",
        "topk_jaccard",
        "reused_captured_mass",
        "oracle_captured_mass",
        "random_expected_captured_mass",
        "reuse_efficiency",
    )
    output = []
    for index, (key, items) in enumerate(sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0]))):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["num_rows"] = len(items)
        summary["num_examples"] = len({item["question_id"] for item in items})
        summary["metrics"] = {
            metric: bootstrap_ci(items, metric, samples=samples, seed=seed + index * 997 + offset)
            for offset, metric in enumerate(metrics)
        }
        output.append(summary)
    return {"group_fields": list(group_fields), "groups": output}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _filter_summary(summary: dict[str, Any], *, method: str, ratio: float, category: str = "all", duration_group: str = "all") -> list[dict[str, Any]]:
    return [
        group
        for group in summary["groups"]
        if group["selection_method"] == method
        and abs(float(group["retention_ratio"]) - float(ratio)) < 1e-12
        and group["category"] == category
        and group["duration_group"] == duration_group
    ]


def _aggregate_all_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = list(rows)
    for row in rows:
        all_category = dict(row)
        all_category["category"] = "all"
        expanded.append(all_category)
        all_duration = dict(row)
        all_duration["duration_group"] = "all"
        expanded.append(all_duration)
        both = dict(row)
        both["category"] = "all"
        both["duration_group"] = "all"
        expanded.append(both)
    return expanded


def save_metric_plot(summary: dict[str, Any], metric: str, path: Path, *, ylabel: str, title: str, ratio: float = 0.5) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    markers = {"qwen": "o", "vila": "s"}
    for model in ("qwen", "vila"):
        groups = [
            group
            for group in _filter_summary(summary, method="dynamic_source_layer_topk", ratio=ratio)
            if group["model"] == model
        ]
        groups.sort(key=lambda group: int(group["layer_gap"]))
        if not groups:
            continue
        x = np.asarray([int(group["layer_gap"]) for group in groups], dtype=np.float64)
        mean = np.asarray([group["metrics"][metric]["mean"] for group in groups], dtype=np.float64)
        low = np.asarray([group["metrics"][metric]["ci95"][0] for group in groups], dtype=np.float64)
        high = np.asarray([group["metrics"][metric]["ci95"][1] for group in groups], dtype=np.float64)
        ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), marker=markers[model], capsize=3, label=model)
    ax.set_xlabel("Layer gap")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (retention={ratio:.0%})")
    ax.set_xticks(list(LAYER_GAPS))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_static_vs_dynamic_plot(summary: dict[str, Any], path: Path, *, ratio: float = 0.5) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    dynamic_styles = {
        "qwen": ("tab:blue", "o", "Qwen dynamic"),
        "vila": ("tab:orange", "s", "VILA dynamic"),
    }
    static_styles = {
        "qwen": ("tab:blue", "Qwen LOP static prior"),
        "vila": ("tab:orange", "VILA LOP static prior"),
    }
    for model, (color, marker, label) in dynamic_styles.items():
        groups = [
            group
            for group in _filter_summary(summary, method="dynamic_source_layer_topk", ratio=ratio)
            if group["model"] == model
        ]
        groups.sort(key=lambda group: int(group["layer_gap"]))
        if not groups:
            continue
        x = np.asarray([int(group["layer_gap"]) for group in groups], dtype=np.float64)
        mean = np.asarray([group["metrics"]["reused_captured_mass"]["mean"] for group in groups], dtype=np.float64)
        low = np.asarray([group["metrics"]["reused_captured_mass"]["ci95"][0] for group in groups], dtype=np.float64)
        high = np.asarray([group["metrics"]["reused_captured_mass"]["ci95"][1] for group in groups], dtype=np.float64)
        ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), marker=marker, capsize=3, color=color, label=label)
    for model, (color, label) in static_styles.items():
        groups = [
            group
            for group in _filter_summary(summary, method="static_dev_prior_lopo", ratio=ratio)
            if group["model"] == model
        ]
        if not groups:
            continue
        group = groups[0]
        mean = group["metrics"]["reused_captured_mass"]["mean"]
        low, high = group["metrics"]["reused_captured_mass"]["ci95"]
        ax.axhline(mean, color=color, linestyle=":", linewidth=1.6, label=label)
        ax.fill_between([min(LAYER_GAPS), max(LAYER_GAPS)], [low, low], [high, high], color=color, alpha=0.10)
    ax.axhline(ratio, color="0.5", linestyle="--", linewidth=1.0, label="random expectation")
    ax.set_xlabel("Layer gap")
    ax.set_ylabel("Target-layer captured mass")
    ax.set_title(f"LOP static prior vs dynamic route reuse (retention={ratio:.0%})")
    ax.set_xticks(list(LAYER_GAPS))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plots(summary: dict[str, Any], output_dir: Path) -> None:
    save_metric_plot(summary, "reused_captured_mass", output_dir / "captured_mass_vs_layer_gap.png", ylabel="Captured target-layer mass", title="Dynamic route reuse captured mass")
    save_metric_plot(summary, "topk_jaccard", output_dir / "topk_jaccard_vs_layer_gap.png", ylabel="Top-k Jaccard overlap", title="Top-k route overlap")
    save_metric_plot(summary, "reuse_efficiency", output_dir / "reuse_efficiency_vs_layer_gap.png", ylabel="Reuse efficiency", title="Dynamic route reuse efficiency")
    save_static_vs_dynamic_plot(summary, output_dir / "static_prior_vs_dynamic_reuse.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen = load_latest_complete_artifacts(args.qwen_dir)
    vila = load_latest_complete_artifacts(args.vila_dir)
    manifest = manifest_by_question(args.manifest)
    dev_ids = dev_question_ids(args.dev_manifest)
    rows, diagnostics = route_reuse_rows(qwen, vila, manifest, dev_ids)
    if diagnostics["matched_complete_examples"] != args.expected_matched_examples:
        raise ValueError(
            f"Expected {args.expected_matched_examples} matched complete examples, "
            f"found {diagnostics['matched_complete_examples']}."
        )
    if len(diagnostics["development_examples_analyzed"]) != args.expected_dev_examples:
        raise ValueError(
            f"Expected {args.expected_dev_examples} matched development examples for method-selection analysis, "
            f"found {len(diagnostics['development_examples_analyzed'])}."
        )
    write_jsonl(output_dir / "route_reuse_metrics.jsonl", rows)
    summary = {
        "inputs": {
            "qwen_dir": args.qwen_dir,
            "vila_dir": args.vila_dir,
            "manifest": args.manifest,
            "dev_manifest": args.dev_manifest,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "diagnostics": diagnostics,
        "aggregate": aggregate_summary(_aggregate_all_rows(rows), args.bootstrap_samples, args.seed),
    }
    write_json_atomic(output_dir / "route_reuse_summary.json", summary)
    save_plots(summary["aggregate"], output_dir)
    print(json.dumps({"output_dir": str(output_dir), "num_rows": len(rows), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
