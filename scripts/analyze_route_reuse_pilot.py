#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import write_json_atomic


CONDITIONS = {
    "adaptive": "route_reuse_gap4_top50",
    "random": "random_reuse_gap4_top50",
    "uniform": "uniform_reuse_gap4_top50",
}
EXPECTED_ROUTED_LAYERS = tuple(list(range(9, 12)) + list(range(13, 16)) + list(range(17, 20)) + list(range(21, 24)) + list(range(25, 28)))
DEFAULT_EXPECTED_EXAMPLES = 15
DEFAULT_COHORT_LABEL = "development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen development route-replay causal pilot.")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--route-reuse-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/route_reuse_gap4_top50_dev")
    parser.add_argument("--random-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/random_reuse_gap4_top50_dev")
    parser.add_argument("--uniform-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/uniform_reuse_gap4_top50_dev")
    parser.add_argument("--dev-manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_cross_model/analysis/qwen_route_reuse_dev")
    parser.add_argument("--expected-examples", type=int, default=DEFAULT_EXPECTED_EXAMPLES)
    parser.add_argument("--cohort-label", default=DEFAULT_COHORT_LABEL)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def _plt():
    import matplotlib.pyplot as plt

    return plt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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


def dev_manifest_by_id(
    path: str | Path,
    *,
    expected_examples: int = DEFAULT_EXPECTED_EXAMPLES,
    cohort_label: str = DEFAULT_COHORT_LABEL,
) -> dict[str, dict[str, Any]]:
    records = {str(record["question_id"]): record for record in read_jsonl(path)}
    if len(records) != expected_examples:
        raise RuntimeError(f"Expected {expected_examples} {cohort_label} records, found {len(records)} in {path}.")
    return records


def source_video_id(artifact: dict[str, Any], manifest_record: dict[str, Any] | None = None) -> str:
    if manifest_record and manifest_record.get("source_video_id"):
        return str(manifest_record["source_video_id"])
    clips = artifact.get("video_clip") or []
    if clips:
        return str(clips[0].get("video_id") or "unknown")
    return "unknown"


def participant_id(artifact: dict[str, Any], manifest_record: dict[str, Any] | None = None) -> str:
    if manifest_record and manifest_record.get("participant_id"):
        return str(manifest_record["participant_id"])
    clips = artifact.get("video_clip") or []
    if clips and clips[0].get("participant_id"):
        return str(clips[0]["participant_id"])
    return source_video_id(artifact, manifest_record).split("-", 1)[0]


def category_for(artifact: dict[str, Any], manifest_record: dict[str, Any] | None = None) -> str:
    if manifest_record and manifest_record.get("category"):
        return str(manifest_record["category"])
    if artifact.get("category"):
        return str(artifact["category"])
    question_type = str(artifact.get("question_type", "unknown"))
    if question_type.startswith("fine_grained_"):
        return "fine_grained"
    if question_type.startswith("gaze_"):
        return "gaze"
    if question_type.startswith("ingredient_"):
        return "ingredient"
    if question_type.startswith("object_motion_"):
        return "object_motion"
    return "unknown"


def answer_metric(scores: dict[str, Any], primary: str, fallback: str | None = None) -> float:
    value = scores.get(primary)
    if value is None and fallback is not None:
        value = scores.get(fallback)
    if value is None:
        raise RuntimeError(f"Missing answer score field {primary!r}.")
    return float(value)


def predicted_idx(artifact: dict[str, Any]) -> int | None:
    value = artifact.get("predicted_idx")
    return None if value is None else int(value)


def artifact_model_id(artifact: dict[str, Any]) -> str:
    metadata = artifact.get("metadata") or {}
    return str(artifact.get("model_checkpoint") or metadata.get("checkpoint") or metadata.get("model_id") or artifact.get("model_backend") or "unknown")


def artifact_checkpoint(artifact: dict[str, Any]) -> str:
    metadata = artifact.get("metadata") or {}
    return str(artifact.get("model_checkpoint") or metadata.get("checkpoint") or metadata.get("model_id") or "unknown")


def flat_sampled(values: Any) -> list[Any]:
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], (list, tuple)):
        return list(values[0])
    return list(values or [])


def sampling_signature(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": artifact.get("question"),
        "choices": artifact.get("choices"),
        "prompt": artifact.get("prompt"),
        "rendered_prompt": artifact.get("rendered_prompt"),
        "messages": artifact.get("messages"),
        "sampled_frame_indices": flat_sampled(artifact.get("sampled_frame_indices")),
        "sampled_timestamps": [round(float(x), 6) for x in flat_sampled(artifact.get("sampled_timestamps"))],
        "frame_bin_mappings": artifact.get("frame_bin_mappings"),
        "sampling_metadata": artifact.get("sampling_metadata"),
    }


def route_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    route = artifact.get("route_reuse") or (artifact.get("metadata") or {}).get("route_reuse")
    if not isinstance(route, dict):
        raise RuntimeError(f"{artifact.get('question_id')}: missing route_reuse metadata.")
    return route


def validate_route_summary(artifact: dict[str, Any], expected_condition: str) -> dict[str, Any]:
    route = route_summary(artifact)
    if route.get("type") != "baseline_derived_causal_route_replay":
        raise RuntimeError(f"{artifact.get('question_id')}: route type is not baseline_derived_causal_route_replay.")
    if route.get("condition") != expected_condition:
        raise RuntimeError(f"{artifact.get('question_id')}: expected condition {expected_condition}, got {route.get('condition')}.")
    if route.get("routing_unit_type") != "qwen_native_temporal_cell":
        raise RuntimeError(f"{artifact.get('question_id')}: expected qwen native routing units.")
    units = route.get("native_routing_units") or []
    if len(units) != 4:
        raise RuntimeError(f"{artifact.get('question_id')}: expected four Qwen native routing units, found {len(units)}.")
    if int(route.get("retained_native_units", -1)) != 2:
        raise RuntimeError(f"{artifact.get('question_id')}: expected two retained native routing units.")
    native_unit_ids = {int(unit.get("unit_id")) for unit in units}
    if native_unit_ids != {0, 1, 2, 3}:
        raise RuntimeError(f"{artifact.get('question_id')}: native unit IDs must be exactly 0,1,2,3.")
    all_visual_tokens = {
        int(token)
        for unit in units
        for token in unit.get("visual_token_indices", ())
    }
    layers = sorted(int(layer) for layer in (route.get("layer_routes") or {}))
    if tuple(layers) != EXPECTED_ROUTED_LAYERS:
        raise RuntimeError(f"{artifact.get('question_id')}: routed layers differ from expected: {layers}")
    for layer, layer_route in (route.get("layer_routes") or {}).items():
        selected_units = {int(item) for item in layer_route.get("selected_native_unit_ids", ())}
        omitted_units = {int(item) for item in layer_route.get("omitted_native_unit_ids", ())}
        if selected_units & omitted_units:
            raise RuntimeError(f"{artifact.get('question_id')}: selected/omitted native unit overlap at layer {layer}.")
        if selected_units | omitted_units != native_unit_ids:
            raise RuntimeError(f"{artifact.get('question_id')}: selected/omitted native units do not partition all units at layer {layer}.")
        allowed = set(int(item) for item in layer_route.get("allowed_visual_token_indices", ()))
        blocked = set(int(item) for item in layer_route.get("blocked_visual_token_indices", ()))
        if allowed & blocked:
            raise RuntimeError(f"{artifact.get('question_id')}: allowed/blocked token overlap at layer {layer}.")
        if allowed | blocked != all_visual_tokens:
            raise RuntimeError(f"{artifact.get('question_id')}: allowed/blocked tokens do not partition all visual tokens at layer {layer}.")
        if "num_allowed_visual_tokens" in layer_route and int(layer_route["num_allowed_visual_tokens"]) != len(allowed):
            raise RuntimeError(f"{artifact.get('question_id')}: recorded allowed token count is wrong at layer {layer}.")
        if "num_blocked_visual_tokens" in layer_route and int(layer_route["num_blocked_visual_tokens"]) != len(blocked):
            raise RuntimeError(f"{artifact.get('question_id')}: recorded blocked token count is wrong at layer {layer}.")
        if abs(float(layer_route.get("actual_retained_visual_token_fraction")) - 0.5) > 1e-9:
            raise RuntimeError(f"{artifact.get('question_id')}: retained token fraction is not 0.5 at layer {layer}.")
    return route


def validate_inputs(
    baseline: dict[str, dict[str, Any]],
    condition_artifacts: dict[str, dict[str, dict[str, Any]]],
    dev_records: dict[str, dict[str, Any]],
    *,
    cohort_label: str,
) -> dict[str, Any]:
    dev_ids = set(dev_records)
    if not dev_ids <= set(baseline):
        raise RuntimeError(f"Baseline is missing development IDs: {sorted(dev_ids - set(baseline))}")
    execution_commits = set()
    manifest_commits = set()
    validation = {"conditions": {}, "execution_code_commits": [], "route_manifest_commits": []}
    for label, artifacts in condition_artifacts.items():
        qids = set(artifacts)
        if qids != dev_ids:
            extra = sorted(qids - dev_ids)
            missing = sorted(dev_ids - qids)
            raise RuntimeError(f"{label}: expected exactly the {cohort_label} IDs; extra={extra}, missing={missing}")
        validation["conditions"][label] = {"num_examples": len(artifacts), "question_ids": sorted(qids)}
        expected_condition = CONDITIONS[label]
        for qid, artifact in artifacts.items():
            base = baseline[qid]
            if artifact_model_id(artifact) != artifact_model_id(base):
                raise RuntimeError(f"{qid}/{label}: model differs from baseline.")
            if artifact_checkpoint(artifact) != artifact_checkpoint(base):
                raise RuntimeError(f"{qid}/{label}: checkpoint differs from baseline.")
            if sampling_signature(artifact) != sampling_signature(base):
                raise RuntimeError(f"{qid}/{label}: prompt/frame/sampling signature differs from baseline.")
            route = validate_route_summary(artifact, expected_condition)
            execution_commit = (artifact.get("run_config") or {}).get("git_commit")
            if not execution_commit:
                raise RuntimeError(f"{qid}/{label}: missing run_config.git_commit execution commit.")
            execution_commits.add(str(execution_commit))
            manifest_commit = route.get("git_commit")
            if manifest_commit:
                manifest_commits.add(str(manifest_commit))
            if not artifact.get("intervention_answer_choice_scores"):
                raise RuntimeError(f"{qid}/{label}: missing intervention_answer_choice_scores.")
    if len(execution_commits) != 1:
        raise RuntimeError(f"Intervention runs must use one execution commit, got {sorted(execution_commits)}")
    validation["execution_code_commits"] = sorted(execution_commits)
    validation["route_manifest_commits"] = sorted(manifest_commits)
    validation["code_commits"] = sorted(execution_commits)
    return validation


def instrumentation_runtime_seconds(artifact: dict[str, Any]) -> float | None:
    metadata = artifact.get("metadata") or {}
    values = [
        metadata.get("prefill_runtime_seconds"),
        metadata.get("answer_scoring_runtime_seconds"),
        metadata.get("generation_runtime_seconds"),
    ]
    if all(value is None for value in values):
        profiling = metadata.get("profiling") or {}
        stages = profiling.get("stages") or {}
        stage_items = stages.values() if isinstance(stages, dict) else stages
        total = sum(float(stage.get("elapsed_seconds", 0.0)) for stage in stage_items if isinstance(stage, dict))
        return total if total > 0 else None
    return float(sum(float(value or 0.0) for value in values))


def per_example_rows(
    baseline: dict[str, dict[str, Any]],
    condition_artifacts: dict[str, dict[str, dict[str, Any]]],
    dev_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for qid in sorted(dev_records):
        base = baseline[qid]
        base_scores = base.get("answer_choice_scores") or {}
        dense_logp = answer_metric(base_scores, "correct_choice_log_probability")
        dense_margin = answer_metric(
            base_scores,
            "correct_vs_best_incorrect_margin",
            fallback="correct_vs_strongest_incorrect_margin",
        )
        manifest_record = dev_records[qid]
        for label, artifacts in condition_artifacts.items():
            routed = artifacts[qid]
            routed_scores = routed.get("intervention_answer_choice_scores") or {}
            routed_logp = answer_metric(routed_scores, "correct_choice_log_probability")
            routed_margin = answer_metric(
                routed_scores,
                "correct_vs_best_incorrect_margin",
                fallback="correct_vs_strongest_incorrect_margin",
            )
            route = route_summary(routed)
            row = {
                "question_id": qid,
                "condition_label": label,
                "condition": CONDITIONS[label],
                "category": category_for(base, manifest_record),
                "question_type": str(manifest_record.get("question_type") or base.get("question_type")),
                "source_video_id": source_video_id(base, manifest_record),
                "participant_id": participant_id(base, manifest_record),
                "dense_correct_choice_log_probability": dense_logp,
                "routed_correct_choice_log_probability": routed_logp,
                "delta_correct_choice_log_probability": routed_logp - dense_logp,
                "dense_correct_vs_best_incorrect_margin": dense_margin,
                "routed_correct_vs_best_incorrect_margin": routed_margin,
                "delta_answer_margin": routed_margin - dense_margin,
                "dense_predicted_idx": predicted_idx(base),
                "routed_predicted_idx": predicted_idx(routed),
                "dense_correct": bool(base.get("correct")),
                "routed_correct": bool(routed.get("correct")),
                "prediction_changed": predicted_idx(base) != predicted_idx(routed),
                "generated_answer_correctness": bool(routed.get("correct")),
                "retained_token_fraction": float(route.get("mean_actual_retained_visual_token_fraction", 0.5)),
                "instrumentation_runtime_seconds": instrumentation_runtime_seconds(routed),
                "runtime_label": "instrumentation_runtime_not_optimized_kernel_speed",
            }
            rows.append(row)
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


def summarize_condition_rows(rows: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    metrics = {}
    for offset, field in enumerate(("delta_correct_choice_log_probability", "delta_answer_margin")):
        metrics[field] = clustered_bootstrap_ci(rows, field, samples, seed + offset * 1000)
    accuracy = sum(int(row["routed_correct"]) for row in rows) / len(rows) if rows else None
    dense_accuracy = sum(int(row["dense_correct"]) for row in rows) / len(rows) if rows else None
    flip_rate = sum(int(row["prediction_changed"]) for row in rows) / len(rows) if rows else None
    return {
        "num_examples": len(rows),
        "dense_accuracy": dense_accuracy,
        "routed_accuracy": accuracy,
        "prediction_flip_rate": flip_rate,
        "metrics": metrics,
    }


def paired_difference_rows(rows: Sequence[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    by_key = {(row["question_id"], row["condition_label"]): row for row in rows}
    output = []
    for qid in sorted({row["question_id"] for row in rows}):
        lrow = by_key[(qid, left)]
        rrow = by_key[(qid, right)]
        base = {
            "question_id": qid,
            "participant_id": lrow["participant_id"],
            "source_video_id": lrow["source_video_id"],
            "category": lrow["category"],
        }
        for field in ("delta_correct_choice_log_probability", "delta_answer_margin"):
            output.append({**base, "comparison": f"{left}_minus_{right}", "metric": field, "value": float(lrow[field]) - float(rrow[field])})
    return output


def summarize_pairwise(rows: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    output = {}
    for offset, control in enumerate(("random", "uniform")):
        diffs = paired_difference_rows(rows, "adaptive", control)
        output[f"adaptive_minus_{control}"] = {}
        for metric in ("delta_correct_choice_log_probability", "delta_answer_margin"):
            items = [row for row in diffs if row["metric"] == metric]
            normalized = [{**row, metric: row["value"]} for row in items]
            output[f"adaptive_minus_{control}"][metric] = clustered_bootstrap_ci(
                normalized,
                metric,
                samples,
                seed + 5000 + offset * 1000,
            )
    return output


def prediction_transition_counts(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output = {}
    for label in CONDITIONS:
        counts = Counter()
        for row in rows:
            if row["condition_label"] != label:
                continue
            key = f"{row['dense_predicted_idx']}->{row['routed_predicted_idx']}"
            counts[key] += 1
        output[label] = dict(sorted(counts.items()))
    return output


def exploratory_breakdowns(rows: Sequence[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    output = {}
    categories = sorted({row["category"] for row in rows})
    for category in categories:
        output[category] = {}
        for label in CONDITIONS:
            items = [row for row in rows if row["category"] == category and row["condition_label"] == label]
            if items:
                output[category][label] = summarize_condition_rows(items, samples, seed + len(output) * 100)
    return output


def causal_gate(summary: dict[str, Any]) -> dict[str, Any]:
    pairwise = summary["paired_adaptive_control_differences"]
    checks = []
    for comparison in ("adaptive_minus_random", "adaptive_minus_uniform"):
        for metric in ("delta_correct_choice_log_probability", "delta_answer_margin"):
            stats = pairwise[comparison][metric]
            checks.append({"comparison": comparison, "metric": metric, "mean": stats["mean"], "ci_low": stats["ci95"][0]})
    if all(item["mean"] > 0 and item["ci_low"] > 0 for item in checks):
        status = "PASS"
    elif any(item["mean"] <= 0 for item in checks):
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "rule": (
            "PASS if adaptive routing preserves correct-answer log probability and answer margin better than both "
            "equal-budget random and uniform controls; FAIL if adaptive is no better than either control; "
            "INCONCLUSIVE if point estimates favor adaptive but uncertainty is too large."
        ),
        "checks": checks,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_bar_plot(rows: Sequence[dict[str, Any]], field: str, output: Path, ylabel: str, title: str) -> None:
    plt = _plt()
    labels = list(CONDITIONS)
    means = [float(np.mean([row[field] for row in rows if row["condition_label"] == label])) for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.bar(labels, means, color=["#4477AA", "#CC6677", "#DDCC77"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_flip_plot(rows: Sequence[dict[str, Any]], output: Path) -> None:
    plt = _plt()
    labels = list(CONDITIONS)
    accuracy = [np.mean([row["routed_correct"] for row in rows if row["condition_label"] == label]) for label in labels]
    flips = [np.mean([row["prediction_changed"] for row in rows if row["condition_label"] == label]) for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.36
    ax.bar(x - width / 2, accuracy, width=width, label="routed accuracy")
    ax.bar(x + width / 2, flips, width=width, label="prediction flip rate")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Accuracy and prediction flips")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["causal_gate"]
    cohort_label = str(summary["cohort_label"])
    expected_examples = int(summary["expected_examples"])
    lines = [
        f"# Qwen {cohort_label.replace('_', ' ').title()} Route-Replay Analysis",
        "",
        f"This is a {expected_examples}-example {cohort_label} route-replay analysis. Routes were replayed from dense baseline artifacts.",
        "It does not establish online routing and does not measure actual latency savings; runtimes are instrumentation runtimes.",
        "Category breakdowns are exploratory because the cells are small.",
        "",
        f"Causal gate: **{gate['status']}**.",
        "",
        gate["rule"],
        "",
        "## Condition Summary",
        "",
    ]
    for label, stats in summary["conditions"].items():
        logp = stats["metrics"]["delta_correct_choice_log_probability"]
        margin = stats["metrics"]["delta_answer_margin"]
        lines.append(
            f"- {label}: mean Δ logp {logp['mean']:.4f} [{logp['ci95'][0]:.4f}, {logp['ci95'][1]:.4f}], "
            f"mean Δ margin {margin['mean']:.4f} [{margin['ci95'][0]:.4f}, {margin['ci95'][1]:.4f}], "
            f"accuracy {stats['routed_accuracy']:.3f}, flip rate {stats['prediction_flip_rate']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `per_example.csv`: exact paired per-example table.",
            "- `summary.json`: validation, aggregate statistics, pairwise adaptive-control differences, and gate decision.",
            "- `quality_delta.png`, `margin_delta.png`, `accuracy_and_flip_rates.png`: descriptive development-set plots.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def analyze(
    *,
    baseline_dir: str | Path,
    condition_dirs: dict[str, str | Path],
    dev_manifest: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int,
    seed: int,
    expected_examples: int = DEFAULT_EXPECTED_EXAMPLES,
    cohort_label: str = DEFAULT_COHORT_LABEL,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dev_records = dev_manifest_by_id(
        dev_manifest,
        expected_examples=expected_examples,
        cohort_label=cohort_label,
    )
    baseline = load_latest_complete_artifacts(baseline_dir)
    condition_artifacts = {
        label: load_latest_complete_artifacts(path)
        for label, path in condition_dirs.items()
    }
    validation = validate_inputs(baseline, condition_artifacts, dev_records, cohort_label=cohort_label)
    rows = per_example_rows(baseline, condition_artifacts, dev_records)
    write_csv(output / "per_example.csv", rows)
    summary = {
        "inputs": {
            "baseline_dir": str(baseline_dir),
            "condition_dirs": {label: str(path) for label, path in condition_dirs.items()},
            "dev_manifest": str(dev_manifest),
        },
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "expected_examples": expected_examples,
        "cohort_label": cohort_label,
        "validation": validation,
        "conditions": {
            label: summarize_condition_rows([row for row in rows if row["condition_label"] == label], bootstrap_samples, seed + index * 1000)
            for index, label in enumerate(CONDITIONS)
        },
        "paired_adaptive_control_differences": summarize_pairwise(rows, bootstrap_samples, seed + 10000),
        "exploratory_category_breakdowns": exploratory_breakdowns(rows, bootstrap_samples, seed + 20000),
        "prediction_transition_counts": prediction_transition_counts(rows),
    }
    summary["causal_gate"] = causal_gate(summary)
    write_json_atomic(output / "summary.json", summary)
    save_bar_plot(rows, "delta_correct_choice_log_probability", output / "quality_delta.png", "Δ correct-answer log probability", "Correct-answer log probability delta")
    save_bar_plot(rows, "delta_answer_margin", output / "margin_delta.png", "Δ answer margin", "Answer margin delta")
    save_accuracy_flip_plot(rows, output / "accuracy_and_flip_rates.png")
    write_report(output / "report.md", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(
        baseline_dir=args.baseline_dir,
        condition_dirs={
            "adaptive": args.route_reuse_dir,
            "random": args.random_dir,
            "uniform": args.uniform_dir,
        },
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        expected_examples=args.expected_examples,
        cohort_label=args.cohort_label,
    )
    print(json.dumps({"output_dir": args.output_dir, "causal_gate": summary["causal_gate"]}, indent=2))


if __name__ == "__main__":
    main()
