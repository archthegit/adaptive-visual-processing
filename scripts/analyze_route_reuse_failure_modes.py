#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_route_reuse_pilot import (  # noqa: E402
    CONDITIONS,
    EXPECTED_ROUTED_LAYERS,
    answer_metric,
    artifact_checkpoint,
    artifact_model_id,
    category_for,
    load_latest_complete_artifacts,
    participant_id,
    predicted_idx,
    read_jsonl,
    route_summary,
    sampling_signature,
    source_video_id,
    validate_route_summary,
)
from src.io import write_json_atomic  # noqa: E402


EXPECTED_COUNTS = {"development": 15, "heldout": 56}
CONDITION_DIR_KEYS = {
    "adaptive": "route_reuse",
    "random": "random",
    "uniform": "uniform",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen route-replay failure modes without running inference.")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--dev-route-reuse-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/route_reuse_gap4_top50_dev")
    parser.add_argument("--dev-random-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/random_reuse_gap4_top50_dev")
    parser.add_argument("--dev-uniform-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/uniform_reuse_gap4_top50_dev")
    parser.add_argument("--heldout-route-reuse-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/route_reuse_gap4_top50_test")
    parser.add_argument("--heldout-random-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/random_reuse_gap4_top50_test")
    parser.add_argument("--heldout-uniform-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/uniform_reuse_gap4_top50_test")
    parser.add_argument("--dev-manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--heldout-manifest", default="outputs/experiment1_v3_cross_model/manifests/heldout_eligible_8frame.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_cross_model/analysis/qwen_route_failure_modes")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def _plt():
    import matplotlib.pyplot as plt

    return plt


def manifest_by_id(path: str | Path, expected_count: int, label: str) -> dict[str, dict[str, Any]]:
    records = {str(record["question_id"]): record for record in read_jsonl(path)}
    if len(records) != expected_count:
        raise RuntimeError(f"Expected {expected_count} {label} manifest records, found {len(records)} in {path}.")
    return records


def temporal_scores(artifact: dict[str, Any]) -> list[list[float]]:
    scores = (artifact.get("temporal_relevance") or {}).get("normalized_temporal_bin_scores")
    if not scores:
        scores = artifact.get("normalized_temporal_bin_scores")
    if not scores:
        raise RuntimeError(f"{artifact.get('question_id')}: missing normalized temporal distributions.")
    rows = [[float(value) for value in row] for row in scores]
    if any(len(row) != 8 for row in rows):
        raise RuntimeError(f"{artifact.get('question_id')}: expected 8-bin temporal distributions.")
    return rows


def normalized_entropy(distribution: Sequence[float]) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-(values * np.log(values)).sum() / math.log(8))


def topk_indices(distribution: Sequence[float], k: int = 4) -> tuple[int, ...]:
    values = np.asarray(distribution, dtype=np.float64)
    return tuple(int(index) for index in np.argsort(-values, kind="mergesort")[:k])


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    lx = np.asarray(left, dtype=np.float64)
    rx = np.asarray(right, dtype=np.float64)
    if lx.size != rx.size:
        raise ValueError("Spearman inputs must have matching lengths.")
    lrank = np.argsort(np.argsort(lx, kind="mergesort"), kind="mergesort").astype(np.float64)
    rrank = np.argsort(np.argsort(rx, kind="mergesort"), kind="mergesort").astype(np.float64)
    if np.std(lrank) == 0 or np.std(rrank) == 0:
        return 0.0
    return float(np.corrcoef(lrank, rrank)[0, 1])


def unit_analysis_bins(route: dict[str, Any], layer_route: dict[str, Any], unit_id: int) -> set[int]:
    for unit in route.get("native_routing_units") or []:
        if int(unit.get("unit_id")) != int(unit_id):
            continue
        if unit.get("analysis_bins") is not None:
            return {int(item) for item in unit["analysis_bins"]}
    selected_units = {int(item) for item in layer_route.get("selected_native_unit_ids", ())}
    omitted_units = {int(item) for item in layer_route.get("omitted_native_unit_ids", ())}
    if unit_id in selected_units and layer_route.get("selected_analysis_bins") is not None:
        return {int(item) for item in layer_route["selected_analysis_bins"] if 0 <= int(item) <= 7}
    if unit_id in omitted_units and layer_route.get("omitted_analysis_bins") is not None:
        return {int(item) for item in layer_route["omitted_analysis_bins"] if 0 <= int(item) <= 7}
    # Qwen route replay uses four native temporal cells over eight analysis bins.
    return {2 * int(unit_id), 2 * int(unit_id) + 1}


def selected_analysis_bins(route: dict[str, Any], layer_route: dict[str, Any]) -> tuple[int, ...]:
    bins: set[int] = set()
    for unit_id in layer_route.get("selected_native_unit_ids", ()):
        bins |= unit_analysis_bins(route, layer_route, int(unit_id))
    return tuple(sorted(index for index in bins if 0 <= index <= 7))


def route_layer_metrics(
    qid: str,
    cohort: str,
    manifest_record: dict[str, Any],
    baseline_artifact: dict[str, Any],
    adaptive_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    route = validate_route_summary(adaptive_artifact, CONDITIONS["adaptive"])
    scores = temporal_scores(baseline_artifact)
    rows: list[dict[str, Any]] = []
    for layer_text, layer_route in sorted((route.get("layer_routes") or {}).items(), key=lambda item: int(item[0])):
        target_layer = int(layer_text)
        source_layer = int(layer_route["source_anchor_layer"])
        selected_units = tuple(int(item) for item in layer_route.get("selected_native_unit_ids", ()))
        selected_bins = selected_analysis_bins(route, layer_route)
        source_distribution = scores[source_layer]
        target_distribution = scores[target_layer]
        oracle_bins = topk_indices(target_distribution, 4)
        retained_mass = float(sum(target_distribution[index] for index in selected_bins))
        oracle_mass = float(sum(target_distribution[index] for index in oracle_bins))
        source_top = set(topk_indices(source_distribution, 4))
        target_top = set(oracle_bins)
        jaccard = len(source_top & target_top) / len(source_top | target_top)
        coverage_span = ((max(selected_units) - min(selected_units) + 1) / 4.0) if selected_units else 0.0
        rows.append(
            {
                "cohort": cohort,
                "question_id": qid,
                "participant_id": participant_id(baseline_artifact, manifest_record),
                "source_video_id": source_video_id(baseline_artifact, manifest_record),
                "category": category_for(baseline_artifact, manifest_record),
                "question_type": str(manifest_record.get("question_type") or baseline_artifact.get("question_type")),
                "anchor_layer": source_layer,
                "target_layer": target_layer,
                "distance_from_anchor": target_layer - source_layer,
                "anchor_temporal_entropy": normalized_entropy(source_distribution),
                "retained_target_layer_mass": retained_mass,
                "oracle_top50_target_layer_mass": oracle_mass,
                "reuse_efficiency": retained_mass / oracle_mass if oracle_mass > 0 else 0.0,
                "source_target_topk_jaccard": jaccard,
                "source_target_spearman": spearman_correlation(source_distribution, target_distribution),
                "selected_native_unit_ids": " ".join(str(item) for item in selected_units),
                "selected_analysis_bins": " ".join(str(item) for item in selected_bins),
                "selected_units_adjacent": bool(len(selected_units) > 1 and max(selected_units) - min(selected_units) + 1 == len(selected_units)),
                "normalized_temporal_coverage_span": coverage_span,
                "contains_first_unit": 0 in selected_units,
                "contains_last_unit": 3 in selected_units,
            }
        )
    return rows


def score_pair(
    baseline_artifact: dict[str, Any],
    adaptive_artifact: dict[str, Any],
    random_artifact: dict[str, Any],
    uniform_artifact: dict[str, Any],
) -> dict[str, Any]:
    dense_scores = baseline_artifact.get("answer_choice_scores") or {}
    adaptive_scores = adaptive_artifact.get("intervention_answer_choice_scores") or {}
    random_scores = random_artifact.get("intervention_answer_choice_scores") or {}
    uniform_scores = uniform_artifact.get("intervention_answer_choice_scores") or {}
    dense_logp = answer_metric(dense_scores, "correct_choice_log_probability")
    adaptive_logp = answer_metric(adaptive_scores, "correct_choice_log_probability")
    random_logp = answer_metric(random_scores, "correct_choice_log_probability")
    uniform_logp = answer_metric(uniform_scores, "correct_choice_log_probability")
    dense_margin = answer_metric(dense_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    adaptive_margin = answer_metric(adaptive_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    random_margin = answer_metric(random_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    uniform_margin = answer_metric(uniform_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    return {
        "dense_correct_answer_log_probability": dense_logp,
        "adaptive_correct_answer_log_probability": adaptive_logp,
        "random_correct_answer_log_probability": random_logp,
        "uniform_correct_answer_log_probability": uniform_logp,
        "adaptive_minus_dense_correct_answer_log_probability": adaptive_logp - dense_logp,
        "adaptive_minus_random_correct_answer_log_probability": adaptive_logp - random_logp,
        "adaptive_minus_uniform_correct_answer_log_probability": adaptive_logp - uniform_logp,
        "dense_answer_margin": dense_margin,
        "adaptive_answer_margin": adaptive_margin,
        "random_answer_margin": random_margin,
        "uniform_answer_margin": uniform_margin,
        "adaptive_minus_dense_answer_margin": adaptive_margin - dense_margin,
        "adaptive_minus_random_answer_margin": adaptive_margin - random_margin,
        "adaptive_minus_uniform_answer_margin": adaptive_margin - uniform_margin,
        "dense_predicted_idx": predicted_idx(baseline_artifact),
        "adaptive_predicted_idx": predicted_idx(adaptive_artifact),
        "random_predicted_idx": predicted_idx(random_artifact),
        "uniform_predicted_idx": predicted_idx(uniform_artifact),
        "adaptive_prediction_changed": predicted_idx(baseline_artifact) != predicted_idx(adaptive_artifact),
        "adaptive_correct": bool(adaptive_artifact.get("correct")),
        "dense_correct": bool(baseline_artifact.get("correct")),
        "correctness_change": int(bool(adaptive_artifact.get("correct"))) - int(bool(baseline_artifact.get("correct"))),
    }


def validate_condition_artifacts(
    cohort: str,
    baseline: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    ids = set(manifest)
    if not ids <= set(baseline):
        raise RuntimeError(f"Baseline missing {cohort} IDs: {sorted(ids - set(baseline))}")
    execution_commits: set[str] = set()
    manifest_commits: set[str] = set()
    for label, artifacts in conditions.items():
        qids = set(artifacts)
        if qids != ids:
            raise RuntimeError(f"{cohort}/{label}: expected exactly manifest IDs; extra={sorted(qids - ids)}, missing={sorted(ids - qids)}")
        for qid, artifact in artifacts.items():
            base = baseline[qid]
            if artifact_model_id(artifact) != artifact_model_id(base):
                raise RuntimeError(f"{cohort}/{qid}/{label}: model differs from baseline.")
            if artifact_checkpoint(artifact) != artifact_checkpoint(base):
                raise RuntimeError(f"{cohort}/{qid}/{label}: checkpoint differs from baseline.")
            if sampling_signature(artifact) != sampling_signature(base):
                raise RuntimeError(f"{cohort}/{qid}/{label}: prompt/frame/sampling signature differs from baseline.")
            route = validate_route_summary(artifact, CONDITIONS[label])
            execution_commit = (artifact.get("run_config") or {}).get("git_commit")
            if not execution_commit:
                raise RuntimeError(f"{cohort}/{qid}/{label}: missing execution commit.")
            execution_commits.add(str(execution_commit))
            if route.get("git_commit"):
                manifest_commits.add(str(route["git_commit"]))
            if not artifact.get("intervention_answer_choice_scores"):
                raise RuntimeError(f"{cohort}/{qid}/{label}: missing intervention_answer_choice_scores.")
    return {
        "num_examples": len(ids),
        "question_ids": sorted(ids),
        "execution_commits": sorted(execution_commits),
        "route_manifest_commits": sorted(manifest_commits),
    }


def build_rows_for_cohort(
    cohort: str,
    manifest: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_example: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    for qid in sorted(manifest):
        base = baseline[qid]
        adaptive = conditions["adaptive"][qid]
        random = conditions["random"][qid]
        uniform = conditions["uniform"][qid]
        manifest_record = manifest[qid]
        layer_rows = route_layer_metrics(qid, cohort, manifest_record, base, adaptive)
        per_layer.extend(layer_rows)
        aggregates = aggregate_layer_diagnostics(layer_rows)
        per_example.append(
            {
                "cohort": cohort,
                "question_id": qid,
                "participant_id": participant_id(base, manifest_record),
                "source_video_id": source_video_id(base, manifest_record),
                "category": category_for(base, manifest_record),
                "question_type": str(manifest_record.get("question_type") or base.get("question_type")),
                **score_pair(base, adaptive, random, uniform),
                **aggregates,
            }
        )
    return per_example, per_layer


def aggregate_layer_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    fields = {
        "mean_anchor_temporal_entropy": "anchor_temporal_entropy",
        "mean_retained_target_layer_mass": "retained_target_layer_mass",
        "mean_oracle_top50_target_layer_mass": "oracle_top50_target_layer_mass",
        "mean_reuse_efficiency": "reuse_efficiency",
        "mean_source_target_topk_jaccard": "source_target_topk_jaccard",
        "mean_source_target_spearman": "source_target_spearman",
        "mean_normalized_temporal_coverage_span": "normalized_temporal_coverage_span",
        "mean_distance_from_anchor": "distance_from_anchor",
    }
    output = {}
    for out_field, row_field in fields.items():
        output[out_field] = float(np.mean([float(row[row_field]) for row in rows])) if rows else 0.0
    output["fraction_adjacent_selections"] = float(np.mean([bool(row["selected_units_adjacent"]) for row in rows])) if rows else 0.0
    output["fraction_contains_first_unit"] = float(np.mean([bool(row["contains_first_unit"]) for row in rows])) if rows else 0.0
    output["fraction_contains_last_unit"] = float(np.mean([bool(row["contains_last_unit"]) for row in rows])) if rows else 0.0
    return output


def clustered_bootstrap_ci(rows: Sequence[dict[str, Any]], field: str, samples: int, seed: int) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    if not values:
        return {"mean": None, "median": None, "ci95": [None, None], "n": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["participant_id"])].append(float(row[field]))
    participants = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = []
        for idx in rng.integers(0, len(participants), size=len(participants)):
            participant_values = grouped[participants[int(idx)]]
            draw = rng.integers(0, len(participant_values), size=len(participant_values))
            selected.extend(participant_values[int(item)] for item in draw)
        estimates.append(float(np.mean(selected)))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": len(values),
    }


def median_split_effect(rows: Sequence[dict[str, Any]], feature: str, outcome: str) -> dict[str, Any]:
    values = [float(row[feature]) for row in rows]
    if not values:
        return {"threshold": None, "high_mean": None, "low_mean": None, "high_minus_low": None}
    threshold = float(np.median(values))
    high = [float(row[outcome]) for row in rows if float(row[feature]) >= threshold]
    low = [float(row[outcome]) for row in rows if float(row[feature]) < threshold]
    high_mean = float(np.mean(high)) if high else None
    low_mean = float(np.mean(low)) if low else None
    return {
        "threshold": threshold,
        "high_mean": high_mean,
        "low_mean": low_mean,
        "high_minus_low": (high_mean - low_mean) if high_mean is not None and low_mean is not None else None,
    }


def correlation(rows: Sequence[dict[str, Any]], feature: str, outcome: str) -> float | None:
    if len(rows) < 2:
        return None
    x = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stable_route_harm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_stable_examples": 0, "mean_logp_delta": None, "mean_margin_delta": None}
    thresholds = {
        "mean_anchor_temporal_entropy": float(np.median([row["mean_anchor_temporal_entropy"] for row in rows])),
        "mean_retained_target_layer_mass": float(np.median([row["mean_retained_target_layer_mass"] for row in rows])),
        "mean_normalized_temporal_coverage_span": float(np.median([row["mean_normalized_temporal_coverage_span"] for row in rows])),
        "mean_source_target_spearman": float(np.median([row["mean_source_target_spearman"] for row in rows])),
    }
    stable = [
        row
        for row in rows
        if row["mean_anchor_temporal_entropy"] < thresholds["mean_anchor_temporal_entropy"]
        and row["mean_retained_target_layer_mass"] >= thresholds["mean_retained_target_layer_mass"]
        and row["mean_normalized_temporal_coverage_span"] >= thresholds["mean_normalized_temporal_coverage_span"]
        and row["mean_source_target_spearman"] >= thresholds["mean_source_target_spearman"]
    ]
    return {
        "thresholds": thresholds,
        "num_stable_examples": len(stable),
        "mean_logp_delta": float(np.mean([row["adaptive_minus_dense_correct_answer_log_probability"] for row in stable])) if stable else None,
        "mean_margin_delta": float(np.mean([row["adaptive_minus_dense_answer_margin"] for row in stable])) if stable else None,
    }


def hypothesis_summary(rows: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    outcomes = ("adaptive_minus_dense_correct_answer_log_probability", "adaptive_minus_dense_answer_margin")
    output: dict[str, Any] = {}
    for outcome in outcomes:
        output[outcome] = {
            "overall": clustered_bootstrap_ci(rows, outcome, samples, seed),
            "H1_entropy": {
                "median_split": median_split_effect(rows, "mean_anchor_temporal_entropy", outcome),
                "correlation": correlation(rows, "mean_anchor_temporal_entropy", outcome),
            },
            "H1_retained_mass": {
                "median_split": median_split_effect(rows, "mean_retained_target_layer_mass", outcome),
                "correlation": correlation(rows, "mean_retained_target_layer_mass", outcome),
            },
            "H2_adjacent_selection": {
                "median_split": median_split_effect(rows, "fraction_adjacent_selections", outcome),
                "correlation": correlation(rows, "fraction_adjacent_selections", outcome),
            },
            "H2_temporal_coverage": {
                "median_split": median_split_effect(rows, "mean_normalized_temporal_coverage_span", outcome),
                "correlation": correlation(rows, "mean_normalized_temporal_coverage_span", outcome),
            },
            "H3_anchor_distance": {
                "median_split": median_split_effect(rows, "mean_distance_from_anchor", outcome),
                "correlation": correlation(rows, "mean_distance_from_anchor", outcome),
            },
            "H3_route_agreement": {
                "median_split": median_split_effect(rows, "mean_source_target_spearman", outcome),
                "correlation": correlation(rows, "mean_source_target_spearman", outcome),
            },
        }
    output["H4_stable_high_mass_well_covered_routes"] = stable_route_harm(rows)
    return output


def choose_recommendation(summary: dict[str, Any]) -> str:
    heldout = summary["cohorts"].get("heldout", {})
    hypotheses = heldout.get("hypotheses") or summary["cohorts"].get("development", {}).get("hypotheses") or {}
    logp = hypotheses.get("adaptive_minus_dense_correct_answer_log_probability") or {}
    h4 = hypotheses.get("H4_stable_high_mass_well_covered_routes") or {}
    h1_entropy = (logp.get("H1_entropy") or {}).get("correlation")
    h1_mass = (logp.get("H1_retained_mass") or {}).get("correlation")
    h2_coverage = (logp.get("H2_temporal_coverage") or {}).get("correlation")
    h3_agreement = (logp.get("H3_route_agreement") or {}).get("correlation")
    h3_distance = (logp.get("H3_anchor_distance") or {}).get("correlation")
    if h4.get("mean_logp_delta") is not None and h4["mean_logp_delta"] < 0:
        return "compression instead of deletion"
    if h2_coverage is not None and h2_coverage > 0:
        return "coverage-constrained routing"
    if h3_distance is not None and h3_distance < 0 or h3_agreement is not None and h3_agreement > 0:
        return "shorter/dynamic refresh"
    if h1_entropy is not None and h1_entropy < 0 or h1_mass is not None and h1_mass > 0:
        return "variable-budget routing"
    return "abandon temporal routing"


def summarize_rows(per_example: Sequence[dict[str, Any]], per_layer: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    cohorts = {}
    for offset, cohort in enumerate(("development", "heldout")):
        example_rows = [row for row in per_example if row["cohort"] == cohort]
        layer_rows = [row for row in per_layer if row["cohort"] == cohort]
        cohorts[cohort] = {
            "num_examples": len(example_rows),
            "num_layer_rows": len(layer_rows),
            "metrics": {
                field: clustered_bootstrap_ci(example_rows, field, samples, seed + offset * 1000 + index * 100)
                for index, field in enumerate(
                    (
                        "adaptive_minus_dense_correct_answer_log_probability",
                        "adaptive_minus_dense_answer_margin",
                        "adaptive_minus_random_correct_answer_log_probability",
                        "adaptive_minus_random_answer_margin",
                        "adaptive_minus_uniform_correct_answer_log_probability",
                        "adaptive_minus_uniform_answer_margin",
                    )
                )
            },
            "hypotheses": hypothesis_summary(example_rows, samples, seed + 10000 + offset * 1000),
        }
    return {"cohorts": cohorts}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_scatter(rows: Sequence[dict[str, Any]], xfield: str, yfield: str, output: Path, title: str, xlabel: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for cohort, color in (("development", "#4477AA"), ("heldout", "#CC6677")):
        subset = [row for row in rows if row["cohort"] == cohort]
        ax.scatter([row[xfield] for row in subset], [row[yfield] for row in subset], label=cohort, alpha=0.75, color=color)
    ax.axhline(0.0, color="0.3", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("adaptive - dense log probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_bar(rows: Sequence[dict[str, Any]], group_field: str, output: Path, title: str, xlabel: str) -> None:
    plt = _plt()
    groups = sorted({str(row[group_field]) for row in rows})
    labels = []
    means = []
    for cohort in ("development", "heldout"):
        for group in groups:
            subset = [row for row in rows if row["cohort"] == cohort and str(row[group_field]) == group]
            if subset:
                labels.append(f"{cohort}\n{group}")
                means.append(float(np.mean([row["adaptive_minus_dense_correct_answer_log_probability"] for row in subset])))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0.0, color="0.3", linewidth=1.0)
    ax.bar(labels, means, color="#66A61E")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("adaptive - dense log probability")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_adaptive_controls(rows: Sequence[dict[str, Any]], output: Path) -> None:
    plt = _plt()
    fields = [
        "adaptive_minus_dense_correct_answer_log_probability",
        "adaptive_minus_random_correct_answer_log_probability",
        "adaptive_minus_uniform_correct_answer_log_probability",
    ]
    labels = ["adaptive-dense", "adaptive-random", "adaptive-uniform"]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(len(labels))
    width = 0.35
    for idx, cohort in enumerate(("development", "heldout")):
        subset = [row for row in rows if row["cohort"] == cohort]
        means = [float(np.mean([row[field] for row in subset])) for field in fields]
        ax.bar(x + (idx - 0.5) * width, means, width=width, label=cohort)
    ax.axhline(0.0, color="0.3", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("correct-answer log probability delta")
    ax.set_title("Adaptive route replay versus dense and controls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    recommendation = summary["recommendation"]
    lines = [
        "# Qwen Route-Replay Failure-Mode Analysis",
        "",
        "This CPU-only analysis explains the completed route-replay results. It does not change inference, routing, manifests, artifacts, or causal outcomes.",
        "Development and held-out cohorts are reported separately; pooled 71-example statistics are not treated as confirmatory.",
        "",
    ]
    for cohort, payload in summary["cohorts"].items():
        logp = payload["metrics"]["adaptive_minus_dense_correct_answer_log_probability"]
        margin = payload["metrics"]["adaptive_minus_dense_answer_margin"]
        lines.extend(
            [
                f"## {cohort.title()}",
                "",
                f"Examples: {payload['num_examples']}; routed-layer rows: {payload['num_layer_rows']}.",
                f"Mean adaptive-minus-dense log probability: {logp['mean']:.4f} [{logp['ci95'][0]:.4f}, {logp['ci95'][1]:.4f}].",
                f"Mean adaptive-minus-dense margin: {margin['mean']:.4f} [{margin['ci95'][0]:.4f}, {margin['ci95'][1]:.4f}].",
                "",
            ]
        )
    lines.extend(
        [
            "## Failure Hypotheses",
            "",
            "H1 tests whether harm tracks high anchor entropy or low retained target-layer mass.",
            "H2 tests whether harm tracks adjacent selections or low temporal coverage.",
            "H3 tests whether harm grows with anchor distance or declining route agreement.",
            "H4 tests whether apparently stable, high-mass, well-covered routes still cause harm, which would implicate hard deletion itself.",
            "",
            f"## Recommendation: {recommendation}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def analyze(
    *,
    baseline_dir: str | Path,
    dev_condition_dirs: dict[str, str | Path],
    heldout_condition_dirs: dict[str, str | Path],
    dev_manifest: str | Path,
    heldout_manifest: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_latest_complete_artifacts(baseline_dir)
    manifests = {
        "development": manifest_by_id(dev_manifest, EXPECTED_COUNTS["development"], "development"),
        "heldout": manifest_by_id(heldout_manifest, EXPECTED_COUNTS["heldout"], "heldout"),
    }
    dev_ids = set(manifests["development"])
    heldout_ids = set(manifests["heldout"])
    if dev_ids & heldout_ids:
        raise RuntimeError(f"Development and held-out manifests overlap: {sorted(dev_ids & heldout_ids)}")
    if len(dev_ids | heldout_ids) != 71:
        raise RuntimeError(f"Expected 71 unique examples total, found {len(dev_ids | heldout_ids)}.")

    condition_dirs = {"development": dev_condition_dirs, "heldout": heldout_condition_dirs}
    validation: dict[str, Any] = {"cohorts": {}}
    all_example_rows: list[dict[str, Any]] = []
    all_layer_rows: list[dict[str, Any]] = []
    for cohort in ("development", "heldout"):
        conditions = {
            label: load_latest_complete_artifacts(condition_dirs[cohort][CONDITION_DIR_KEYS[label]])
            for label in ("adaptive", "random", "uniform")
        }
        validation["cohorts"][cohort] = validate_condition_artifacts(cohort, baseline, manifests[cohort], conditions)
        example_rows, layer_rows = build_rows_for_cohort(cohort, manifests[cohort], baseline, conditions)
        all_example_rows.extend(example_rows)
        all_layer_rows.extend(layer_rows)

    write_csv(output / "per_example.csv", all_example_rows)
    write_csv(output / "per_layer.csv", all_layer_rows)
    summary = {
        "inputs": {
            "baseline_dir": str(baseline_dir),
            "dev_condition_dirs": {key: str(value) for key, value in dev_condition_dirs.items()},
            "heldout_condition_dirs": {key: str(value) for key, value in heldout_condition_dirs.items()},
            "dev_manifest": str(dev_manifest),
            "heldout_manifest": str(heldout_manifest),
        },
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "validation": validation,
        **summarize_rows(all_example_rows, all_layer_rows, bootstrap_samples, seed),
    }
    summary["recommendation"] = choose_recommendation(summary)
    write_json_atomic(output / "summary.json", summary)
    save_scatter(all_example_rows, "mean_anchor_temporal_entropy", "adaptive_minus_dense_correct_answer_log_probability", output / "harm_vs_entropy.png", "Harm versus anchor entropy", "mean anchor entropy")
    save_scatter(all_example_rows, "mean_retained_target_layer_mass", "adaptive_minus_dense_correct_answer_log_probability", output / "harm_vs_retained_mass.png", "Harm versus retained target-layer mass", "mean retained mass")
    save_bar(all_example_rows, "fraction_adjacent_selections", output / "harm_by_temporal_coverage.png", "Harm by adjacent-selection frequency", "fraction adjacent")
    save_scatter(all_example_rows, "mean_distance_from_anchor", "adaptive_minus_dense_correct_answer_log_probability", output / "harm_by_anchor_distance.png", "Harm versus anchor distance", "mean distance from anchor")
    save_adaptive_controls(all_example_rows, output / "adaptive_vs_controls.png")
    write_report(output / "report.md", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(
        baseline_dir=args.baseline_dir,
        dev_condition_dirs={
            "route_reuse": args.dev_route_reuse_dir,
            "random": args.dev_random_dir,
            "uniform": args.dev_uniform_dir,
        },
        heldout_condition_dirs={
            "route_reuse": args.heldout_route_reuse_dir,
            "random": args.heldout_random_dir,
            "uniform": args.heldout_uniform_dir,
        },
        dev_manifest=args.dev_manifest,
        heldout_manifest=args.heldout_manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": args.output_dir, "recommendation": summary["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
