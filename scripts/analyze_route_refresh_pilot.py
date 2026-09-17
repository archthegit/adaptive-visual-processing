#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_route_reuse_pilot import (  # noqa: E402
    answer_metric,
    category_for,
    dev_manifest_by_id,
    load_latest_complete_artifacts,
    participant_id,
    predicted_idx,
    route_summary,
    sampling_signature,
    source_video_id,
)
from src.io import write_json_atomic  # noqa: E402


CONDITIONS = {
    "gap2": "route_reuse_gap2_top50",
    "gap3": "route_reuse_gap3_top50",
    "gap4": "route_reuse_gap4_top50",
}
EXPECTED_DEV_EXAMPLES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen route-refresh-frequency development pilot.")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--gap2-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/route_reuse_gap2_top50_dev")
    parser.add_argument("--gap3-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/route_reuse_gap3_top50_dev")
    parser.add_argument("--gap4-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/route_reuse_gap4_top50_dev")
    parser.add_argument("--dev-manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_cross_model/analysis/qwen_route_refresh_dev")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def validate_route_spec(route: dict[str, Any], condition: str) -> None:
    if route.get("type") != "baseline_derived_causal_route_replay":
        raise RuntimeError(f"{condition}: route spec is not baseline-derived replay.")
    if route.get("condition") != condition:
        raise RuntimeError(f"Expected {condition}, got {route.get('condition')}.")
    if int(route.get("retained_native_units", -1)) != 2:
        raise RuntimeError(f"{condition}: expected two retained native units.")
    units = route.get("native_routing_units") or []
    if len(units) != 4:
        raise RuntimeError(f"{condition}: expected four native temporal units.")
    for layer, layer_route in (route.get("layer_routes") or {}).items():
        selected = set(int(item) for item in layer_route.get("selected_native_unit_ids", ()))
        omitted = set(int(item) for item in layer_route.get("omitted_native_unit_ids", ()))
        if selected & omitted or selected | omitted != {0, 1, 2, 3}:
            raise RuntimeError(f"{condition}: selected/omitted native units do not partition layer {layer}.")
        allowed = set(int(item) for item in layer_route.get("allowed_visual_token_indices", ()))
        blocked = set(int(item) for item in layer_route.get("blocked_visual_token_indices", ()))
        if allowed & blocked:
            raise RuntimeError(f"{condition}: allowed/blocked visual tokens overlap at layer {layer}.")
        if abs(float(layer_route.get("actual_retained_visual_token_fraction", -1)) - 0.5) > 1e-9:
            raise RuntimeError(f"{condition}: routed layer {layer} does not retain 50% of visual tokens.")


def theoretical_edge_savings(route: dict[str, Any], num_decoder_layers: int = 28) -> dict[str, Any]:
    routed_layers = sorted(int(layer) for layer in route.get("layer_routes", {}))
    retained_fraction = float(route.get("mean_actual_retained_visual_token_fraction", 0.5))
    blocked_fraction = 1.0 - retained_fraction
    routed_layer_fraction = len(routed_layers) / float(num_decoder_layers)
    return {
        "num_decoder_layers": num_decoder_layers,
        "num_routed_layers": len(routed_layers),
        "routed_layers": routed_layers,
        "retained_visual_token_fraction_at_routed_layers": retained_fraction,
        "blocked_visual_token_fraction_at_routed_layers": blocked_fraction,
        "theoretical_question_to_visual_edge_savings_fraction_all_layers": routed_layer_fraction * blocked_fraction,
        "note": "Theoretical attention-edge savings only; this is not measured latency or FLOP savings.",
    }


def validate_inputs(
    baseline: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
    dev_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dev_ids = set(dev_records)
    if not dev_ids <= set(baseline):
        raise RuntimeError(f"Baseline missing dev IDs: {sorted(dev_ids - set(baseline))}")
    validation = {"conditions": {}}
    for label, artifacts in conditions.items():
        qids = set(artifacts)
        if qids != dev_ids:
            raise RuntimeError(f"{label}: expected exactly dev IDs; extra={sorted(qids - dev_ids)}, missing={sorted(dev_ids - qids)}")
        commits = set()
        route_commits = set()
        for qid, artifact in artifacts.items():
            base = baseline[qid]
            if sampling_signature(artifact) != sampling_signature(base):
                raise RuntimeError(f"{qid}/{label}: prompt/frame/sampling signature differs from dense baseline.")
            route = route_summary(artifact)
            validate_route_spec(route, CONDITIONS[label])
            commit = (artifact.get("run_config") or {}).get("git_commit")
            if not commit:
                raise RuntimeError(f"{qid}/{label}: missing execution git commit.")
            commits.add(str(commit))
            if route.get("git_commit"):
                route_commits.add(str(route["git_commit"]))
            if not artifact.get("intervention_answer_choice_scores"):
                raise RuntimeError(f"{qid}/{label}: missing intervention_answer_choice_scores.")
        if len(commits) != 1:
            raise RuntimeError(f"{label}: expected one execution git commit, got {sorted(commits)}")
        validation["conditions"][label] = {
            "num_examples": len(artifacts),
            "condition": CONDITIONS[label],
            "execution_commits": sorted(commits),
            "route_manifest_commits": sorted(route_commits),
            "theoretical_edge_savings": theoretical_edge_savings(route_summary(next(iter(artifacts.values())))),
        }
    return validation


def metric_row(
    qid: str,
    label: str,
    baseline_artifact: dict[str, Any],
    artifact: dict[str, Any],
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    dense_scores = baseline_artifact.get("answer_choice_scores") or {}
    routed_scores = artifact.get("intervention_answer_choice_scores") or {}
    dense_logp = answer_metric(dense_scores, "correct_choice_log_probability")
    routed_logp = answer_metric(routed_scores, "correct_choice_log_probability")
    dense_margin = answer_metric(dense_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    routed_margin = answer_metric(routed_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    return {
        "question_id": qid,
        "condition_label": label,
        "condition": CONDITIONS[label],
        "participant_id": participant_id(baseline_artifact, manifest_record),
        "source_video_id": source_video_id(baseline_artifact, manifest_record),
        "category": category_for(baseline_artifact, manifest_record),
        "dense_correct_answer_log_probability": dense_logp,
        "routed_correct_answer_log_probability": routed_logp,
        "delta_correct_answer_log_probability": routed_logp - dense_logp,
        "dense_answer_margin": dense_margin,
        "routed_answer_margin": routed_margin,
        "delta_answer_margin": routed_margin - dense_margin,
        "dense_predicted_idx": predicted_idx(baseline_artifact),
        "routed_predicted_idx": predicted_idx(artifact),
        "prediction_changed": predicted_idx(baseline_artifact) != predicted_idx(artifact),
        "dense_correct": bool(baseline_artifact.get("correct")),
        "routed_correct": bool(artifact.get("correct")),
    }


def per_example_rows(
    baseline: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
    dev_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for qid in sorted(dev_records):
        for label in ("gap2", "gap3", "gap4"):
            rows.append(metric_row(qid, label, baseline[qid], conditions[label][qid], dev_records[qid]))
    return rows


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
            vals = grouped[participants[int(idx)]]
            draw = rng.integers(0, len(vals), size=len(vals))
            selected.extend(vals[int(item)] for item in draw)
        estimates.append(float(np.mean(selected)))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": len(values),
    }


def paired_rows(rows: Sequence[dict[str, Any]], left: str, right: str, field: str) -> list[dict[str, Any]]:
    by_key = {(row["question_id"], row["condition_label"]): row for row in rows}
    output = []
    for qid in sorted({row["question_id"] for row in rows}):
        lrow = by_key[(qid, left)]
        rrow = by_key[(qid, right)]
        output.append(
            {
                "question_id": qid,
                "participant_id": lrow["participant_id"],
                "source_video_id": lrow["source_video_id"],
                field: float(lrow[field]) - float(rrow[field]),
            }
        )
    return output


def summarize(rows: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    condition_summary = {}
    for index, label in enumerate(("gap2", "gap3", "gap4")):
        subset = [row for row in rows if row["condition_label"] == label]
        condition_summary[label] = {
            "num_examples": len(subset),
            "log_probability_delta": clustered_bootstrap_ci(subset, "delta_correct_answer_log_probability", samples, seed + index * 1000),
            "answer_margin_delta": clustered_bootstrap_ci(subset, "delta_answer_margin", samples, seed + 100 + index * 1000),
            "accuracy": sum(int(row["routed_correct"]) for row in subset) / len(subset),
            "prediction_flip_rate": sum(int(row["prediction_changed"]) for row in subset) / len(subset),
        }
    paired = {}
    for index, (left, right) in enumerate((("gap2", "gap4"), ("gap3", "gap4"), ("gap2", "gap3"))):
        paired[f"{left}_minus_{right}"] = {
            "log_probability_delta": clustered_bootstrap_ci(
                paired_rows(rows, left, right, "delta_correct_answer_log_probability"),
                "delta_correct_answer_log_probability",
                samples,
                seed + 5000 + index * 1000,
            ),
            "answer_margin_delta": clustered_bootstrap_ci(
                paired_rows(rows, left, right, "delta_answer_margin"),
                "delta_answer_margin",
                samples,
                seed + 5500 + index * 1000,
            ),
        }
    means = {
        label: condition_summary[label]["log_probability_delta"]["mean"]
        for label in ("gap2", "gap3", "gap4")
    }
    return {
        "conditions": condition_summary,
        "paired_differences": paired,
        "preregistered_ordering_gap2_gt_gap3_gt_gap4": bool(means["gap2"] > means["gap3"] > means["gap4"]),
        "ordering_means": means,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    baseline_dir: str | Path,
    condition_dirs: dict[str, str | Path],
    dev_manifest: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dev_records = dev_manifest_by_id(dev_manifest, expected_examples=EXPECTED_DEV_EXAMPLES, cohort_label="development")
    baseline = load_latest_complete_artifacts(baseline_dir)
    conditions = {label: load_latest_complete_artifacts(path) for label, path in condition_dirs.items()}
    validation = validate_inputs(baseline, conditions, dev_records)
    rows = per_example_rows(baseline, conditions, dev_records)
    write_csv(output / "per_example.csv", rows)
    summary = {
        "inputs": {
            "baseline_dir": str(baseline_dir),
            "condition_dirs": {label: str(path) for label, path in condition_dirs.items()},
            "dev_manifest": str(dev_manifest),
        },
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "validation": validation,
        "metrics": summarize(rows, bootstrap_samples, seed),
        "latency_claim": "No measured latency or FLOP savings are claimed; edge savings are theoretical question-to-visual attention-edge savings only.",
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(
        baseline_dir=args.baseline_dir,
        condition_dirs={"gap2": args.gap2_dir, "gap3": args.gap3_dir, "gap4": args.gap4_dir},
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": args.output_dir, "ordering": summary["metrics"]["preregistered_ordering_gap2_gt_gap3_gt_gap4"]}, indent=2))


if __name__ == "__main__":
    main()
