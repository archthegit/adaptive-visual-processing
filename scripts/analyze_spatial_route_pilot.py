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
    artifact_checkpoint,
    artifact_model_id,
    category_for,
    dev_manifest_by_id,
    load_latest_complete_artifacts,
    participant_id,
    predicted_idx,
    sampling_signature,
    source_video_id,
)
from src.experiment1.spatial import validate_spatial_route_spec  # noqa: E402
from src.io import write_json_atomic  # noqa: E402


CONDITIONS = {
    "adaptive": "spatial_route_gap4_top50",
    "random": "spatial_random_gap4_top50",
    "uniform": "spatial_uniform_gap4_top50",
}
EXPECTED_DEV_EXAMPLES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen spatial-route development kill test.")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline_spatial")
    parser.add_argument("--adaptive-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/spatial_route_gap4_top50_dev")
    parser.add_argument("--random-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/spatial_random_gap4_top50_dev")
    parser.add_argument("--uniform-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/spatial_uniform_gap4_top50_dev")
    parser.add_argument("--dev-manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_cross_model/analysis/qwen_spatial_route_dev")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def spatial_route_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    route = artifact.get("spatial_route") or (artifact.get("metadata") or {}).get("spatial_route")
    if not isinstance(route, dict):
        raise RuntimeError(f"{artifact.get('question_id')}: missing spatial_route metadata.")
    return route


def validate_inputs(
    baseline: dict[str, dict[str, Any]],
    conditions: dict[str, dict[str, dict[str, Any]]],
    dev_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dev_ids = set(dev_records)
    if not dev_ids <= set(baseline):
        raise RuntimeError(f"Baseline missing dev IDs: {sorted(dev_ids - set(baseline))}")
    validation: dict[str, Any] = {"conditions": {}}
    reference_ids: set[str] | None = None
    for label, artifacts in conditions.items():
        qids = set(artifacts)
        if qids != dev_ids:
            raise RuntimeError(f"{label}: expected exactly dev IDs; extra={sorted(qids - dev_ids)}, missing={sorted(dev_ids - qids)}")
        if reference_ids is None:
            reference_ids = qids
        elif qids != reference_ids:
            raise RuntimeError(f"{label}: condition question IDs differ from other conditions.")
        commits = set()
        route_commits = set()
        budget_signatures = set()
        for qid, artifact in artifacts.items():
            base = baseline[qid]
            if artifact.get("model_backend") != "qwen" or base.get("model_backend") != "qwen":
                raise RuntimeError(f"{qid}/{label}: expected Qwen artifacts.")
            if artifact_model_id(artifact) != artifact_model_id(base) or artifact_checkpoint(artifact) != artifact_checkpoint(base):
                raise RuntimeError(f"{qid}/{label}: model/checkpoint differs from dense baseline.")
            if sampling_signature(artifact) != sampling_signature(base):
                raise RuntimeError(f"{qid}/{label}: prompt/frame/sampling signature differs from dense baseline.")
            route = spatial_route_summary(artifact)
            if route.get("condition") != CONDITIONS[label]:
                raise RuntimeError(f"{qid}/{label}: expected condition {CONDITIONS[label]}, got {route.get('condition')}.")
            validate_spatial_route_spec(route)
            budget_signatures.add(
                json.dumps(
                    {
                        "frames": route.get("temporal_frames_preserved"),
                        "per_frame": route.get("per_frame_retained_visual_tokens"),
                        "routed_layers": sorted(int(layer) for layer in route.get("layer_routes", {})),
                        "retained": route.get("mean_actual_retained_visual_token_fraction"),
                    },
                    sort_keys=True,
                )
            )
            commit = (artifact.get("run_config") or {}).get("git_commit")
            if not commit:
                raise RuntimeError(f"{qid}/{label}: missing execution git commit.")
            commits.add(str(commit))
            if route.get("git_commit"):
                route_commits.add(str(route["git_commit"]))
            if not artifact.get("intervention_answer_choice_scores"):
                raise RuntimeError(f"{qid}/{label}: missing intervention_answer_choice_scores.")
        if len(commits) != 1:
            raise RuntimeError(f"{label}: expected one execution commit, got {sorted(commits)}")
        if len(budget_signatures) != len(dev_ids):
            validation["conditions"].setdefault(label, {})["budget_varies_by_example"] = True
        validation["conditions"][label] = {
            **validation["conditions"].get(label, {}),
            "num_examples": len(artifacts),
            "condition": CONDITIONS[label],
            "execution_commits": sorted(commits),
            "route_manifest_commits": sorted(route_commits),
            "question_ids": sorted(qids),
        }
    for qid in sorted(dev_ids):
        signatures = []
        for label, artifacts in conditions.items():
            route = spatial_route_summary(artifacts[qid])
            signatures.append(
                json.dumps(
                    {
                        "frames": route.get("temporal_frames_preserved"),
                        "per_frame": route.get("per_frame_retained_visual_tokens"),
                        "routed_layers": sorted(int(layer) for layer in route.get("layer_routes", {})),
                        "retained": route.get("mean_actual_retained_visual_token_fraction"),
                    },
                    sort_keys=True,
                )
            )
        if len(set(signatures)) != 1:
            raise RuntimeError(f"{qid}: spatial token budgets differ across conditions.")
    return validation


def metric_row(qid: str, label: str, baseline_artifact: dict[str, Any], artifact: dict[str, Any], manifest_record: dict[str, Any]) -> dict[str, Any]:
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
        for label in ("adaptive", "random", "uniform"):
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


def _accuracy(rows: Sequence[dict[str, Any]], field: str) -> float:
    return sum(int(row[field]) for row in rows) / len(rows) if rows else 0.0


def gate(metrics: dict[str, Any]) -> dict[str, Any]:
    paired = metrics["paired_differences"]
    conditions = metrics["conditions"]
    ar_logp = float(paired["adaptive_minus_random"]["log_probability_delta"]["mean"])
    au_logp = float(paired["adaptive_minus_uniform"]["log_probability_delta"]["mean"])
    ar_margin = float(paired["adaptive_minus_random"]["answer_margin_delta"]["mean"])
    au_margin = float(paired["adaptive_minus_uniform"]["answer_margin_delta"]["mean"])
    adaptive_accuracy = float(conditions["adaptive"]["routed_accuracy"])
    random_accuracy = float(conditions["random"]["routed_accuracy"])
    uniform_accuracy = float(conditions["uniform"]["routed_accuracy"])
    promising = ar_logp > 0 and au_logp > 0 and ar_margin > 0 and au_margin > 0 and adaptive_accuracy >= random_accuracy and adaptive_accuracy >= uniform_accuracy
    reject = (ar_logp <= 0 and ar_margin <= 0) or (au_logp <= 0 and au_margin <= 0)
    status = "PROMISING" if promising else "REJECT" if reject else "INCONCLUSIVE"
    return {
        "status": status,
        "scope": "development kill test only; this is not held-out confirmatory evidence.",
        "inputs": {
            "adaptive_minus_random_mean_logp": ar_logp,
            "adaptive_minus_uniform_mean_logp": au_logp,
            "adaptive_minus_random_mean_margin": ar_margin,
            "adaptive_minus_uniform_mean_margin": au_margin,
            "adaptive_routed_accuracy": adaptive_accuracy,
            "random_routed_accuracy": random_accuracy,
            "uniform_routed_accuracy": uniform_accuracy,
        },
    }


def summarize(rows: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    condition_summary: dict[str, Any] = {}
    for index, label in enumerate(("adaptive", "random", "uniform")):
        subset = [row for row in rows if row["condition_label"] == label]
        condition_summary[label] = {
            "num_examples": len(subset),
            "log_probability_delta": clustered_bootstrap_ci(subset, "delta_correct_answer_log_probability", samples, seed + index * 1000),
            "answer_margin_delta": clustered_bootstrap_ci(subset, "delta_answer_margin", samples, seed + 100 + index * 1000),
            "dense_accuracy": _accuracy(subset, "dense_correct"),
            "routed_accuracy": _accuracy(subset, "routed_correct"),
            "prediction_flip_rate": sum(int(row["prediction_changed"]) for row in subset) / len(subset),
            "by_category": {
                category: {
                    "num_examples": len(cat_rows),
                    "log_probability_delta_mean": float(np.mean([row["delta_correct_answer_log_probability"] for row in cat_rows])),
                    "answer_margin_delta_mean": float(np.mean([row["delta_answer_margin"] for row in cat_rows])),
                    "routed_accuracy": _accuracy(cat_rows, "routed_correct"),
                }
                for category, cat_rows in sorted(_group_by(subset, "category").items())
            },
        }
    paired = {}
    for index, (left, right) in enumerate((("adaptive", "random"), ("adaptive", "uniform"))):
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
    output = {"conditions": condition_summary, "paired_differences": paired}
    output["development_gate"] = gate(output)
    return output


def _group_by(rows: Sequence[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return grouped


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    gate_info = summary["metrics"]["development_gate"]
    lines = [
        "# Qwen spatial-route development kill test",
        "",
        f"Gate: **{gate_info['status']}**",
        "",
        "This is a 15-example development kill test. Routes are replayed from dense baseline artifacts; this does not establish online routing or measured latency/FLOP savings.",
        "",
        "The gate tests whether dense decoder question-to-visual attention identifies spatial visual tokens that preserve answer quality better than equal-budget random and uniform spatial controls while preserving every temporal frame.",
    ]
    path.write_text("\n".join(lines) + "\n")


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
    }
    write_json_atomic(output / "summary.json", summary)
    write_report(output / "report.md", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(
        baseline_dir=args.baseline_dir,
        condition_dirs={"adaptive": args.adaptive_dir, "random": args.random_dir, "uniform": args.uniform_dir},
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": args.output_dir, "gate": summary["metrics"]["development_gate"]}, indent=2))


if __name__ == "__main__":
    main()
