#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.temporal_handoff import SCHEMA_VERSION, append_jsonl, write_json


CONDITIONS = ("dense_custom", "handoff_mean", "hard_evict", "random_handoff")
EXPECTED_DEV_EXAMPLES = 15
DEFAULT_SEED = 20260818
IMMUTABLE_RUN_CONFIG_FIELDS = (
    "schema_version",
    "model_id",
    "resolution_config",
    "sampling_mode",
    "handoff_layer",
    "retain_regions",
    "memory_tokens_per_region",
    "seed",
    "warmup",
    "repeats",
    "conditions",
    "git_commit",
)
EXPERIMENT_OUTPUT_FILES = (
    "records.jsonl",
    "run_summary.json",
    "per_example.csv",
    "analysis_summary.json",
    "report.md",
    "latency_speedup.png",
    "quality_delta.png",
    "margin_delta.png",
    "accuracy_and_flip_rate.png",
    "sequence_and_flops_reduction.png",
    "run_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and analyze the 15-example Qwen temporal-handoff development pilot.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_temporal_handoff/dev_qwen")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resolution-config", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--handoff-layer", type=int, default=8)
    parser.add_argument("--retain-regions", type=int, default=2)
    parser.add_argument("--memory-tokens-per-region", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260925)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def artifact_path(output_dir: Path, question_id: str, condition: str) -> Path:
    return output_dir / "artifacts" / question_id / f"{condition}.json"


def equivalence_path(output_dir: Path, question_id: str) -> Path:
    return output_dir / "artifacts" / question_id / "dense_equivalence_report.json"


def completed_conditions(output_dir: Path) -> set[tuple[str, str]]:
    records = output_dir / "records.jsonl"
    completed: set[tuple[str, str]] = set()
    if not records.exists():
        return completed
    for record in read_jsonl(records):
        if record.get("status") == "complete":
            completed.add((str(record.get("question_id")), str(record.get("condition"))))
    return completed


def completed_condition_records(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = output_dir / "records.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not records.exists():
        return completed
    for record in read_jsonl(records):
        if record.get("status") == "complete":
            completed[(str(record.get("question_id")), str(record.get("condition")))] = record
    return completed


def immutable_config_subset(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in IMMUTABLE_RUN_CONFIG_FIELDS}


def immutable_config_mismatches(saved: dict[str, Any], requested: dict[str, Any]) -> dict[str, dict[str, Any]]:
    saved_subset = immutable_config_subset(saved)
    requested_subset = immutable_config_subset(requested)
    return {
        field: {"saved": saved_subset.get(field), "requested": requested_subset.get(field)}
        for field in IMMUTABLE_RUN_CONFIG_FIELDS
        if saved_subset.get(field) != requested_subset.get(field)
    }


def experiment_outputs_present(output_dir: Path) -> bool:
    if (output_dir / "artifacts").exists():
        return True
    return any((output_dir / name).exists() for name in EXPERIMENT_OUTPUT_FILES)


def clean_experiment_outputs(output_dir: Path) -> None:
    for name in EXPERIMENT_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    artifacts = output_dir / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)


def prepare_run_config(output_dir: Path, requested_config: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if overwrite:
        clean_experiment_outputs(output_dir)
        write_json(config_path, requested_config)
        return requested_config
    if experiment_outputs_present(output_dir):
        if not config_path.exists():
            raise RuntimeError(
                f"Existing temporal-handoff outputs are present in {output_dir}, but run_config.json is missing. "
                "Use a new output directory or explicitly use --overwrite."
            )
        saved_config = read_json(config_path)
        mismatches = immutable_config_mismatches(saved_config, requested_config)
        if mismatches:
            details = "; ".join(
                f"{field}: saved={values['saved']!r}, requested={values['requested']!r}"
                for field, values in sorted(mismatches.items())
            )
            raise RuntimeError(
                "Refusing to resume temporal-handoff development run because immutable run configuration differs: "
                f"{details}. Use a new output directory or explicitly use --overwrite."
            )
        return saved_config
    write_json(config_path, requested_config)
    return requested_config


def resolve_record_artifact_path(record: dict[str, Any], output_dir: Path) -> Path:
    raw = record.get("artifact")
    if not raw:
        raise RuntimeError("Complete record is missing artifact path.")
    path = Path(raw)
    candidates = (path, output_dir / path.name, output_dir / "artifacts" / str(record.get("question_id")) / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def validate_resumed_condition_artifact(
    artifact_path_: Path,
    *,
    question_id: str,
    condition: str,
    saved_run_config: dict[str, Any],
) -> dict[str, Any]:
    if not artifact_path_.exists() or artifact_path_.stat().st_size <= 0:
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact path is missing or empty: {artifact_path_}")
    artifact = read_json(artifact_path_)
    if artifact.get("question_id") != question_id:
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact question_id mismatch.")
    if artifact.get("condition") != condition:
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact condition mismatch.")
    if artifact.get("status") != "complete":
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact is not complete.")
    metadata = artifact.get("metadata") or {}
    if metadata.get("git_commit") != saved_run_config.get("git_commit"):
        raise RuntimeError(
            f"{question_id}/{condition}: artifact git_commit {metadata.get('git_commit')!r} "
            f"does not match run_config git_commit {saved_run_config.get('git_commit')!r}."
        )
    artifact_config = metadata.get("run_config")
    if not isinstance(artifact_config, dict):
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact lacks metadata.run_config.")
    mismatches = immutable_config_mismatches(artifact_config, saved_run_config)
    if mismatches:
        details = "; ".join(
            f"{field}: artifact={values['saved']!r}, run_config={values['requested']!r}"
            for field, values in sorted(mismatches.items())
        )
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact run_config mismatch: {details}")
    validate_condition_artifact(artifact)
    return artifact


def validate_existing_equivalence_report(path: Path, *, question_id: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{question_id}: dense-equivalence report is missing or empty: {path}")
    report = read_json(path)
    if report.get("question_id") != question_id:
        raise RuntimeError(f"{question_id}: dense-equivalence report question_id mismatch.")
    if report.get("passed") is not True:
        raise RuntimeError(f"{question_id}: dense-equivalence report did not pass.")
    return report


def latest_artifacts(output_dir: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    root = Path(output_dir)
    latest: dict[str, dict[str, Path]] = defaultdict(dict)
    records = root / "records.jsonl"
    if records.exists():
        for record in read_jsonl(records):
            if record.get("status") == "complete" and record.get("artifact"):
                latest[str(record["question_id"])][str(record["condition"])] = Path(record["artifact"])
    else:
        for path in root.glob("artifacts/*/*.json"):
            if path.name == "dense_equivalence_report.json":
                continue
            payload = read_json(path)
            latest[str(payload["question_id"])][str(payload["condition"])] = path
    return {
        qid: {condition: read_json(path) for condition, path in sorted(paths.items())}
        for qid, paths in sorted(latest.items())
    }


def load_equivalence_reports(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(output_dir)
    reports: dict[str, dict[str, Any]] = {}
    for path in root.glob("artifacts/*/dense_equivalence_report.json"):
        reports[path.parent.name] = read_json(path)
    return reports


def manifest_by_id(path: str | Path, expected_examples: int = EXPECTED_DEV_EXAMPLES) -> dict[str, dict[str, Any]]:
    records = {str(record["question_id"]): record for record in read_jsonl(path)}
    if len(records) != expected_examples:
        raise RuntimeError(f"Expected {expected_examples} development records, found {len(records)} in {path}.")
    return records


def _metric(scores: dict[str, Any], primary: str, fallback: str | None = None) -> float:
    value = scores.get(primary)
    if value is None and fallback:
        value = scores.get(fallback)
    if value is None:
        raise RuntimeError(f"Missing score field {primary!r}.")
    return float(value)


def _prediction(scores: dict[str, Any]) -> int:
    logits = scores.get("choice_logits")
    if not logits:
        raise RuntimeError("Missing choice_logits.")
    return int(max(range(len(logits)), key=lambda idx: float(logits[idx])))


def _profile_metric(profile: dict[str, Any], name: str) -> float:
    value = profile.get(name)
    return float("nan") if value is None else float(value)


def _combined_profile(artifact: dict[str, Any]) -> dict[str, Any]:
    return (((artifact.get("metadata") or {}).get("cuda_profile") or {}).get("combined_prefill") or {})


def _stack_profile(artifact: dict[str, Any]) -> dict[str, Any]:
    return (((artifact.get("metadata") or {}).get("cuda_profile") or {}).get("decoder_stack_excluding_lm_head") or {})


def _lm_head_profile(artifact: dict[str, Any]) -> dict[str, Any]:
    return (((artifact.get("metadata") or {}).get("cuda_profile") or {}).get("final_token_lm_head") or {})


def _handoff_meta(artifact: dict[str, Any]) -> dict[str, Any]:
    meta = artifact.get("temporal_handoff")
    if not isinstance(meta, dict):
        raise RuntimeError(f"{artifact.get('question_id')}/{artifact.get('condition')}: missing temporal_handoff metadata.")
    return meta


def _instrumentation(artifact: dict[str, Any]) -> dict[str, Any]:
    return _handoff_meta(artifact).get("instrumentation") or {}


def _layers(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_instrumentation(artifact).get("layers") or [])


def validate_condition_artifact(artifact: dict[str, Any]) -> None:
    condition = str(artifact.get("condition"))
    if condition not in CONDITIONS:
        raise RuntimeError(f"Unsupported condition {condition!r}.")
    if artifact.get("status") != "complete":
        raise RuntimeError(f"{artifact.get('question_id')}/{condition}: artifact is not complete.")
    if artifact.get("model_backend") != "qwen":
        raise RuntimeError(f"{artifact.get('question_id')}/{condition}: expected Qwen backend.")
    if artifact.get("generation_supported") is not False:
        raise RuntimeError(f"{artifact.get('question_id')}/{condition}: handoff phase must be prefill-only.")
    layers = _layers(artifact)
    if not layers:
        raise RuntimeError(f"{artifact.get('question_id')}/{condition}: missing layer instrumentation.")
    for layer in layers:
        if layer.get("layer_type") == "full_attention":
            if layer.get("native_sdpa_is_causal_used") is not True:
                raise RuntimeError(f"{artifact.get('question_id')}/{condition}: full layer did not use native causal SDPA.")
            if layer.get("explicit_mask_materialized"):
                raise RuntimeError(f"{artifact.get('question_id')}/{condition}: full layer materialized an explicit mask.")
            if layer.get("mask_shape") is not None:
                raise RuntimeError(f"{artifact.get('question_id')}/{condition}: full layer recorded a mask shape.")
    scores = artifact.get("answer_choice_scores") or {}
    _metric(scores, "correct_choice_log_probability")
    _metric(scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    _prediction(scores)


def validate_example_group(qid: str, artifacts: dict[str, dict[str, Any]], equivalence: dict[str, Any]) -> None:
    missing = [condition for condition in CONDITIONS if condition not in artifacts]
    if missing:
        raise RuntimeError(f"{qid}: missing conditions {missing}.")
    for artifact in artifacts.values():
        validate_condition_artifact(artifact)
    if equivalence.get("passed") is not True:
        raise RuntimeError(f"{qid}: dense equivalence gate failed.")
    dense = artifacts["dense_custom"]
    original_len = int((dense.get("metadata") or {}).get("original_sequence_length"))
    dense_final_len = int((dense.get("metadata") or {}).get("final_sequence_length"))
    if original_len != dense_final_len:
        raise RuntimeError(f"{qid}: dense_custom sequence length differs from original.")
    stock_len = equivalence.get("stock_sequence_length")
    if stock_len is not None and int(stock_len) != dense_final_len:
        raise RuntimeError(f"{qid}: dense_custom sequence length differs from stock.")
    handoff_len = int((artifacts["handoff_mean"].get("metadata") or {}).get("final_sequence_length"))
    random_len = int((artifacts["random_handoff"].get("metadata") or {}).get("final_sequence_length"))
    hard_len = int((artifacts["hard_evict"].get("metadata") or {}).get("final_sequence_length"))
    if handoff_len != random_len:
        raise RuntimeError(f"{qid}: handoff_mean and random_handoff final lengths differ.")
    memory = int(_handoff_meta(artifacts["handoff_mean"]).get("memory_tokens_per_region"))
    handed_off = len(((_handoff_meta(artifacts["handoff_mean"]).get("compaction_plan") or {}).get("handed_off_temporal_regions") or []))
    if handoff_len - hard_len != memory * handed_off:
        raise RuntimeError(f"{qid}: handoff_mean and hard_evict length difference does not match memory-token budget.")


def validate_all_outputs(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    equivalence_reports: dict[str, dict[str, Any]],
    manifest_records: dict[str, dict[str, Any]],
) -> None:
    if set(artifacts) != set(manifest_records):
        raise RuntimeError(f"Artifact IDs do not match manifest: extra={sorted(set(artifacts)-set(manifest_records))}, missing={sorted(set(manifest_records)-set(artifacts))}")
    if set(equivalence_reports) != set(manifest_records):
        raise RuntimeError("Dense equivalence reports do not match manifest IDs.")
    for qid in sorted(manifest_records):
        validate_example_group(qid, artifacts[qid], equivalence_reports[qid])


def bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if clean.size == 0:
        return {"mean": None, "median": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        draw = rng.integers(0, clean.size, size=clean.size)
        estimates.append(float(np.mean(clean[draw])))
    return {
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": int(clean.size),
    }


def per_example_rows(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    manifest_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(manifest_records):
        dense = artifacts[qid]["dense_custom"]
        dense_scores = dense["answer_choice_scores"]
        dense_logp = _metric(dense_scores, "correct_choice_log_probability")
        dense_margin = _metric(dense_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
        dense_pred = _prediction(dense_scores)
        dense_stack_latency = _profile_metric(_stack_profile(dense), "prefill_latency_seconds_median")
        dense_combined_latency = _profile_metric(_combined_profile(dense), "prefill_latency_seconds_median")
        dense_lm_head_latency = _profile_metric(_lm_head_profile(dense), "prefill_latency_seconds_median")
        dense_flops = float(_instrumentation(dense).get("total_estimated_attention_flops"))
        dense_seq = int((dense.get("metadata") or {}).get("final_sequence_length"))
        for condition in CONDITIONS:
            artifact = artifacts[qid][condition]
            scores = artifact["answer_choice_scores"]
            logp = _metric(scores, "correct_choice_log_probability")
            margin = _metric(scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
            stack_latency = _profile_metric(_stack_profile(artifact), "prefill_latency_seconds_median")
            combined_latency = _profile_metric(_combined_profile(artifact), "prefill_latency_seconds_median")
            lm_head_latency = _profile_metric(_lm_head_profile(artifact), "prefill_latency_seconds_median")
            flops = float(_instrumentation(artifact).get("total_estimated_attention_flops"))
            final_seq = int((artifact.get("metadata") or {}).get("final_sequence_length"))
            selected = _handoff_meta(artifact).get("selected_retained_native_cells") or []
            rows.append(
                {
                    "question_id": qid,
                    "condition": condition,
                    "category": manifest_records[qid].get("category"),
                    "question_type": manifest_records[qid].get("question_type"),
                    "participant_id": manifest_records[qid].get("participant_id"),
                    "source_video_id": manifest_records[qid].get("source_video_id"),
                    "dense_correct_choice_log_probability": dense_logp,
                    "condition_correct_choice_log_probability": logp,
                    "delta_correct_choice_log_probability": logp - dense_logp,
                    "dense_answer_margin": dense_margin,
                    "condition_answer_margin": margin,
                    "delta_answer_margin": margin - dense_margin,
                    "dense_predicted_idx": dense_pred,
                    "condition_predicted_idx": _prediction(scores),
                    "prediction_changed": dense_pred != _prediction(scores),
                    "dense_correct": bool(dense.get("correct")),
                    "condition_correct": bool(artifact.get("correct")),
                    "dense_stack_latency_seconds_median": dense_stack_latency,
                    "condition_stack_latency_seconds_median": stack_latency,
                    "stack_latency_reduction_fraction": (dense_stack_latency - stack_latency) / dense_stack_latency if dense_stack_latency else float("nan"),
                    "stack_latency_speedup": dense_stack_latency / stack_latency if stack_latency else float("nan"),
                    "dense_combined_latency_seconds_median": dense_combined_latency,
                    "condition_combined_latency_seconds_median": combined_latency,
                    "dense_lm_head_latency_seconds_median": dense_lm_head_latency,
                    "condition_lm_head_latency_seconds_median": lm_head_latency,
                    "dense_sequence_length": dense_seq,
                    "condition_sequence_length": final_seq,
                    "sequence_reduction_fraction": (dense_seq - final_seq) / dense_seq,
                    "dense_attention_flops": dense_flops,
                    "condition_attention_flops": flops,
                    "attention_flop_reduction_fraction": (dense_flops - flops) / dense_flops if dense_flops else float("nan"),
                    "dense_incremental_peak_allocated_bytes": _combined_profile(dense).get("incremental_peak_allocated_bytes"),
                    "condition_incremental_peak_allocated_bytes": _combined_profile(artifact).get("incremental_peak_allocated_bytes"),
                    "dense_absolute_peak_allocated_bytes": _combined_profile(dense).get("absolute_peak_allocated_bytes"),
                    "condition_absolute_peak_allocated_bytes": _combined_profile(artifact).get("absolute_peak_allocated_bytes"),
                    "selected_retained_temporal_cells": json.dumps(selected),
                    "final_visual_token_count": len((_instrumentation(artifact).get("final_visual_token_indices") or [])),
                    "final_memory_token_count": len((_instrumentation(artifact).get("final_memory_token_indices") or [])),
                }
            )
    return rows


def condition_stats(rows: Sequence[dict[str, Any]], condition: str, samples: int, seed: int) -> dict[str, Any]:
    subset = [row for row in rows if row["condition"] == condition]
    if not subset:
        raise RuntimeError(f"No rows for {condition}.")
    return {
        "num_examples": len(subset),
        "correct_choice_log_probability_delta": bootstrap_ci([row["delta_correct_choice_log_probability"] for row in subset], samples=samples, seed=seed),
        "answer_margin_delta": bootstrap_ci([row["delta_answer_margin"] for row in subset], samples=samples, seed=seed + 11),
        "dense_accuracy": float(np.mean([bool(row["dense_correct"]) for row in subset])),
        "condition_accuracy": float(np.mean([bool(row["condition_correct"]) for row in subset])),
        "prediction_flip_rate": float(np.mean([bool(row["prediction_changed"]) for row in subset])),
        "median_stack_latency_seconds": float(np.median([row["condition_stack_latency_seconds_median"] for row in subset])),
        "mean_stack_latency_speedup": float(np.mean([row["stack_latency_speedup"] for row in subset])),
        "median_stack_latency_speedup": float(np.median([row["stack_latency_speedup"] for row in subset])),
        "median_sequence_reduction_fraction": float(np.median([row["sequence_reduction_fraction"] for row in subset])),
        "median_attention_flop_reduction_fraction": float(np.median([row["attention_flop_reduction_fraction"] for row in subset])),
        "peak_memory_difference_incremental_allocated_bytes": bootstrap_ci(
            [
                float(row["condition_incremental_peak_allocated_bytes"] or 0.0)
                - float(row["dense_incremental_peak_allocated_bytes"] or 0.0)
                for row in subset
            ],
            samples=samples,
            seed=seed + 23,
        ),
    }


def paired_condition_difference(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_key = {(row["question_id"], row["condition"]): row for row in rows}
    values = []
    for qid in sorted({row["question_id"] for row in rows}):
        values.append(float(by_key[(qid, left)][metric]) - float(by_key[(qid, right)][metric]))
    return bootstrap_ci(values, samples=samples, seed=seed)


def gate_decision(summary: dict[str, Any], equivalence_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    equivalence_passed = all(report.get("passed") is True for report in equivalence_reports.values())
    handoff = summary["conditions"]["handoff_mean"]
    hard = summary["conditions"]["hard_evict"]
    random = summary["conditions"]["random_handoff"]
    latency_reduction = handoff["median_stack_latency_speedup"]
    latency_reduction_fraction = 1.0 - (1.0 / latency_reduction) if latency_reduction and math.isfinite(latency_reduction) else float("nan")
    handoff_logp = handoff["correct_choice_log_probability_delta"]["mean"]
    hard_logp = hard["correct_choice_log_probability_delta"]["mean"]
    random_logp = random["correct_choice_log_probability_delta"]["mean"]
    handoff_margin = handoff["answer_margin_delta"]["mean"]
    hard_margin = hard["answer_margin_delta"]["mean"]
    random_margin = random["answer_margin_delta"]["mean"]
    dense_acc = handoff["dense_accuracy"]
    handoff_acc = handoff["condition_accuracy"]
    materially_reduces_accuracy = handoff_acc < dense_acc
    promising = (
        equivalence_passed
        and latency_reduction_fraction >= 0.20
        and not materially_reduces_accuracy
        and handoff_logp > hard_logp
        and handoff_logp > random_logp
        and handoff_margin > hard_margin
        and handoff_margin > random_margin
    )
    reject = (
        (not equivalence_passed)
        or latency_reduction_fraction < 0.10
        or (
            handoff_logp <= hard_logp
            and handoff_logp <= random_logp
            and handoff_margin <= hard_margin
            and handoff_margin <= random_margin
        )
    )
    status = "PROMISING" if promising else "REJECT" if reject else "INCONCLUSIVE"
    return {
        "status": status,
        "scope": "15-example development kill test; not held-out confirmatory evidence.",
        "equivalence_passed": equivalence_passed,
        "handoff_median_stack_latency_reduction_fraction": latency_reduction_fraction,
        "handoff_accuracy": handoff_acc,
        "dense_accuracy": dense_acc,
        "materially_reduces_accuracy": materially_reduces_accuracy,
        "rules": {
            "promising": [
                "every dense equivalence check passes",
                "handoff_mean provides at least 20% median measured decoder-stack latency reduction",
                "handoff_mean does not materially reduce development accuracy relative to dense_custom",
                "handoff_mean mean log-probability delta is better than hard_evict and random_handoff",
                "handoff_mean mean answer-margin delta is better than hard_evict and random_handoff",
            ],
            "reject": [
                "dense equivalence fails",
                "measured latency reduction is below 10%",
                "handoff_mean is no better than hard_evict or random_handoff on both continuous quality metrics",
            ],
        },
    }


def summarize_analysis(
    rows: Sequence[dict[str, Any]],
    equivalence_reports: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    conditions = {
        condition: condition_stats(rows, condition, bootstrap_samples, seed + idx * 1000)
        for idx, condition in enumerate(CONDITIONS)
    }
    paired = {
        "handoff_mean_minus_hard_evict": {
            "correct_choice_log_probability_delta": paired_condition_difference(rows, "handoff_mean", "hard_evict", "delta_correct_choice_log_probability", bootstrap_samples, seed + 5000),
            "answer_margin_delta": paired_condition_difference(rows, "handoff_mean", "hard_evict", "delta_answer_margin", bootstrap_samples, seed + 5011),
        },
        "handoff_mean_minus_random_handoff": {
            "correct_choice_log_probability_delta": paired_condition_difference(rows, "handoff_mean", "random_handoff", "delta_correct_choice_log_probability", bootstrap_samples, seed + 6000),
            "answer_margin_delta": paired_condition_difference(rows, "handoff_mean", "random_handoff", "delta_answer_margin", bootstrap_samples, seed + 6011),
        },
    }
    summary = {
        "conditions": conditions,
        "paired_condition_differences": paired,
        "dense_equivalence": {
            "num_examples": len(equivalence_reports),
            "all_passed": all(report.get("passed") is True for report in equivalence_reports.values()),
            "failed_question_ids": [qid for qid, report in sorted(equivalence_reports.items()) if report.get("passed") is not True],
        },
        "timing_scope": "Profiling covers custom decoder prefill calls only; it excludes video decoding, preprocessing, model loading, artifact serialization and route selection.",
    }
    summary["development_kill_test_gate"] = gate_decision(summary, equivalence_reports)
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _bar_plot(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_plots(output_dir: Path, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    labels = list(CONDITIONS)
    _bar_plot(
        output_dir / "quality_delta.png",
        labels,
        [summary["conditions"][label]["correct_choice_log_probability_delta"]["mean"] for label in labels],
        "Correct-answer log-probability delta",
        "Δ log probability vs dense_custom",
    )
    _bar_plot(
        output_dir / "margin_delta.png",
        labels,
        [summary["conditions"][label]["answer_margin_delta"]["mean"] for label in labels],
        "Answer-margin delta",
        "Δ margin vs dense_custom",
    )
    _bar_plot(
        output_dir / "latency_speedup.png",
        labels,
        [summary["conditions"][label]["median_stack_latency_speedup"] for label in labels],
        "Decoder-stack prefill speedup",
        "Median paired speedup",
    )
    _bar_plot(
        output_dir / "accuracy_and_flip_rate.png",
        labels,
        [summary["conditions"][label]["condition_accuracy"] for label in labels],
        "Routed accuracy",
        "Accuracy",
    )
    _bar_plot(
        output_dir / "sequence_and_flops_reduction.png",
        labels,
        [summary["conditions"][label]["median_attention_flop_reduction_fraction"] for label in labels],
        "Estimated attention-FLOP reduction",
        "Median reduction fraction",
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["development_kill_test_gate"]
    lines = [
        "# Qwen Temporal-Handoff Development Pilot",
        "",
        "This is a 15-example development kill test, not held-out confirmatory evidence.",
        "The intervention physically compacts the decoder sequence after layer 8; it is not attention masking or KV-cache reuse.",
        "",
        f"Gate decision: **{gate['status']}**.",
        "",
        "| Condition | Accuracy | Mean Δ logp | Mean Δ margin | Median stack speedup | Median FLOP reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        stats = summary["conditions"][condition]
        lines.append(
            "| {condition} | {acc:.3f} | {logp:.6f} | {margin:.6f} | {speedup:.3f} | {flops:.3f} |".format(
                condition=condition,
                acc=stats["condition_accuracy"],
                logp=stats["correct_choice_log_probability_delta"]["mean"],
                margin=stats["answer_margin_delta"]["mean"],
                speedup=stats["median_stack_latency_speedup"],
                flops=stats["median_attention_flop_reduction_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "Profiling excludes video decoding, preprocessing, model loading, artifact serialization and route selection.",
            "All intervention comparisons use `dense_custom` as the paired dense control.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def analyze_outputs(
    *,
    output_dir: str | Path,
    manifest: str | Path,
    bootstrap_samples: int,
    seed: int,
    write_outputs: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_records = manifest_by_id(manifest)
    artifacts = latest_artifacts(output)
    equivalence_reports = load_equivalence_reports(output)
    validate_all_outputs(artifacts, equivalence_reports, manifest_records)
    rows = per_example_rows(artifacts, manifest_records)
    summary = summarize_analysis(rows, equivalence_reports, bootstrap_samples=bootstrap_samples, seed=seed)
    if write_outputs:
        write_csv(output / "per_example.csv", rows)
        write_json(output / "analysis_summary.json", summary)
        write_plots(output, rows, summary)
        write_report(output / "report.md", summary)
    return summary


def run_development_pilot(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.run_experiment1 import current_git_commit, frame_batches_for_example, load_examples_by_id, load_manifest
    from scripts.run_qwen_temporal_handoff_smoke import (
        _build_layout,
        _load_artifact_from_record,
        _latest_complete_record,
        _prediction_from_scores,
        _prepare_inputs,
        _stock_logits,
        _temporal_scores_from_artifact,
    )
    from src.experiment1.answer_scoring import score_answer_choice_logits
    from src.experiment1.resolution import get_resolution_config
    from src.experiment1.temporal_handoff import (
        aggregate_analysis_scores_to_native_cells,
        condition_from_baseline_scores,
        cuda_profile_prefill,
        dense_equivalence_report,
        qwen_decoder_stack,
        qwen_multimodal_decoder_inputs,
        run_custom_decoder_prefill,
    )
    from src.models.qwen import Qwen25VLWrapper, QwenConfig

    output = Path(args.output_dir)
    manifest_records_list = load_manifest(args.manifest)
    if len(manifest_records_list) != EXPECTED_DEV_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_DEV_EXAMPLES} development examples, found {len(manifest_records_list)}.")
    manifest_records = {str(record["question_id"]): record for record in manifest_records_list}
    examples = load_examples_by_id(args.questions_dir, manifest_records_list)
    resolution = get_resolution_config(args.resolution_config)
    git_commit = current_git_commit()
    run_config = {
        "schema_version": "qwen_temporal_handoff_dev_pilot_v1",
        "model_id": args.model_id,
        "resolution_config": args.resolution_config,
        "sampling_mode": "cross_model_8",
        "handoff_layer": args.handoff_layer,
        "retain_regions": args.retain_regions,
        "memory_tokens_per_region": args.memory_tokens_per_region,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "git_commit": git_commit,
        "conditions": list(CONDITIONS),
        "timing_scope": "Profiles include only custom decoder prefill calls; video decoding, preprocessing, model loading, serialization and route selection are outside timing.",
    }
    saved_run_config = prepare_run_config(output, run_config, overwrite=args.overwrite)
    records_path = output / "records.jsonl"

    model = Qwen25VLWrapper(QwenConfig(model_id=args.model_id, max_new_tokens=1, attn_implementation="sdpa"))
    model._load()
    assert model._model is not None
    assert model._processor is not None
    stack = qwen_decoder_stack(model._model)
    completed = completed_condition_records(output) if not args.overwrite else {}

    for qid in sorted(manifest_records):
        example = examples[qid]
        manifest_record = manifest_records[qid]
        frame_batches = frame_batches_for_example(
            example,
            args.mp4_dir,
            num_frames=8,
            sampling_mode="cross_model_8",
            manifest_record=manifest_record,
            frames_per_bin_override=1,
        )
        if len(frame_batches[0].frame_indices) != 8:
            raise RuntimeError(f"{qid}: expected exactly eight sampled frames.")
        baseline_record = _latest_complete_record(Path(args.baseline_dir) / "records.jsonl", qid)
        baseline_artifact = _load_artifact_from_record(baseline_record)
        native_aggregation = aggregate_analysis_scores_to_native_cells(_temporal_scores_from_artifact(baseline_artifact), baseline_artifact)
        baseline_scores = [list(layer) for layer in native_aggregation.native_temporal_scores]
        if native_aggregation.native_temporal_cell_count != 4:
            raise RuntimeError(f"{qid}: expected four Qwen-native temporal cells, got {native_aggregation.native_temporal_cell_count}.")
        inputs, rendered, prompt, _video_kwargs = _prepare_inputs(model, example, frame_batches, resolution)
        layout = _build_layout(model, example, inputs, rendered, frame_batches)
        decoder_inputs, position_ids = qwen_multimodal_decoder_inputs(model._model, inputs)
        stock_logits = _stock_logits(model, inputs)
        dense_custom_logits = None
        dense_scores = None
        stock_scores = score_answer_choice_logits(
            stock_logits[0, -1],
            model._processor.tokenizer,
            example.correct_idx,
            len(example.choices),
        ).to_json_dict()

        for condition in CONDITIONS:
            out_path = artifact_path(output, qid, condition)
            if not args.overwrite and (qid, condition) in completed:
                resumed_path = resolve_record_artifact_path(completed[(qid, condition)], output)
                resumed_artifact = validate_resumed_condition_artifact(
                    resumed_path,
                    question_id=qid,
                    condition=condition,
                    saved_run_config=saved_run_config,
                )
                if condition == "dense_custom":
                    dense_scores = resumed_artifact["answer_choice_scores"]
                continue
            config = condition_from_baseline_scores(
                condition=condition,
                baseline_temporal_scores=baseline_scores,
                handoff_layer=args.handoff_layer,
                retain_count=args.retain_regions,
                num_regions=native_aggregation.native_temporal_cell_count,
                memory_tokens_per_region=args.memory_tokens_per_region,
                seed=args.seed,
                question_id=qid,
            )

            def execute_condition() -> Any:
                return run_custom_decoder_prefill(
                    layers=stack["layers"],
                    hidden_states=decoder_inputs.clone(),
                    position_ids=position_ids.clone() if position_ids is not None else None,
                    layout=layout,
                    config=config,
                    lm_head=stack["lm_head"],
                    norm=stack["norm"],
                    num_attention_heads=stack["num_attention_heads"],
                    head_dim=stack["head_dim"],
                    rotary_emb=stack["rotary_emb"],
                    layer_types=stack["layer_types"],
                    sliding_window=stack["sliding_window"],
                )

            def execute_stack_only() -> Any:
                return run_custom_decoder_prefill(
                    layers=stack["layers"],
                    hidden_states=decoder_inputs.clone(),
                    position_ids=position_ids.clone() if position_ids is not None else None,
                    layout=layout,
                    config=config,
                    lm_head=None,
                    norm=stack["norm"],
                    num_attention_heads=stack["num_attention_heads"],
                    head_dim=stack["head_dim"],
                    rotary_emb=stack["rotary_emb"],
                    layer_types=stack["layer_types"],
                    sliding_window=stack["sliding_window"],
                )

            result, profile = cuda_profile_prefill(execute_condition, warmup=args.warmup, repeats=args.repeats)
            stack_result, stack_profile = cuda_profile_prefill(execute_stack_only, warmup=args.warmup, repeats=args.repeats)
            lm_logits, lm_profile = cuda_profile_prefill(
                lambda: stack["lm_head"](stack_result.final_token_hidden_state),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            scores = score_answer_choice_logits(
                result.logits[0, -1],
                model._processor.tokenizer,
                example.correct_idx,
                len(example.choices),
            ).to_json_dict()
            artifact = {
                "question_id": qid,
                "condition": condition,
                "status": "complete",
                "category": manifest_record.get("category"),
                "question_type": manifest_record.get("question_type"),
                "participant_id": manifest_record.get("participant_id"),
                "source_video_id": manifest_record.get("source_video_id"),
                "model_backend": "qwen",
                "model_checkpoint": args.model_id,
                "generation_supported": False,
                "generation_unsupported_reason": "Phase-1 handoff prototype is prefill-only; layer-specific decoding cache is not implemented.",
                "question": example.question,
                "prompt": prompt,
                "rendered_prompt": rendered,
                "choices": list(example.choices),
                "correct_idx": example.correct_idx,
                "predicted_idx": _prediction_from_scores(scores),
                "correct": _prediction_from_scores(scores) == example.correct_idx,
                "answer_choice_scores": scores,
                "sampled_frame_indices": [list(frame_batches[0].frame_indices)],
                "sampled_timestamps": [list(frame_batches[0].timestamps)],
                "frame_bin_mappings": [frame_batches[0].metadata.get("frame_bin_mapping", [])],
                "temporal_handoff": {
                    "schema_version": SCHEMA_VERSION,
                    "handoff_layer": args.handoff_layer,
                    "retained_temporal_regions": list(config.retained_temporal_regions),
                    "memory_tokens_per_region": config.memory_tokens_per_region,
                    "condition": condition,
                    "baseline_artifact": baseline_record.get("artifact"),
                    "baseline_selection_layer": args.handoff_layer,
                    "native_temporal_aggregation": native_aggregation.to_metadata(),
                    "per_layer_native_cell_scores_used_for_selection": [list(layer) for layer in baseline_scores],
                    "selected_retained_native_cells": list(config.retained_temporal_regions),
                    "compaction_plan": result.compaction_plan.to_metadata() if result.compaction_plan else None,
                    "instrumentation": result.instrumentation_metadata(),
                },
                "metadata": {
                    "git_commit": git_commit,
                    "run_config": saved_run_config,
                    "resolution": resolution.to_metadata(),
                    "sampling_mode": "cross_model_8",
                    "query_scope": "question",
                    "input_token_count": int(inputs["input_ids"].shape[1]),
                    "original_sequence_length": int(decoder_inputs.shape[1]),
                    "final_sequence_length": int(result.final_hidden_states.shape[1]),
                    "cuda_profile": {
                        "combined_prefill": profile,
                        "decoder_stack_excluding_lm_head": stack_profile,
                        "final_token_lm_head": lm_profile,
                    },
                    "final_token_lm_head_shape": list(lm_logits.shape),
                },
            }
            write_json(out_path, artifact)
            validate_resumed_condition_artifact(
                out_path,
                question_id=qid,
                condition=condition,
                saved_run_config=saved_run_config,
            )
            append_jsonl(records_path, {"question_id": qid, "condition": condition, "status": "complete", "artifact": str(out_path)})
            if condition == "dense_custom":
                dense_custom_logits = result.logits
                dense_scores = scores

        eq_path = equivalence_path(output, qid)
        if args.overwrite or not eq_path.exists():
            if dense_custom_logits is None:
                # Resume can skip dense logits. In that case force the user to rerun this example;
                # stock-vs-dense logits are intentionally not reconstructed from artifacts.
                raise RuntimeError(f"{qid}: dense equivalence report missing but dense_custom was resumed from {artifact_path(output, qid, 'dense_custom')}.")
            if dense_scores is None:
                raise RuntimeError(f"{qid}: dense scores unavailable for equivalence.")
            equivalence = dense_equivalence_report(
                stock_logits,
                dense_custom_logits,
                stock_scores=stock_scores,
                dense_scores=dense_scores,
                stock_sequence_length=int(inputs["input_ids"].shape[1]),
                dense_sequence_length=int(read_json(artifact_path(output, qid, "dense_custom"))["metadata"]["final_sequence_length"]),
                legacy_dense_equivalence_atol=None,
            )
            equivalence["question_id"] = qid
            equivalence["stock_answer_choice_scores"] = stock_scores
            equivalence["dense_custom_answer_choice_scores"] = dense_scores
            write_json(eq_path, equivalence)
            if not equivalence["passed"]:
                raise RuntimeError(f"{qid}: dense_custom failed BF16-aware equivalence gate.")
        else:
            validate_existing_equivalence_report(eq_path, question_id=qid)

        group = {condition: read_json(artifact_path(output, qid, condition)) for condition in CONDITIONS}
        validate_example_group(qid, group, validate_existing_equivalence_report(eq_path, question_id=qid))

    summary = analyze_outputs(
        output_dir=output,
        manifest=args.manifest,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.analysis_seed,
        write_outputs=True,
    )
    write_json(
        output / "run_summary.json",
        {
            "run_config": saved_run_config,
            "num_examples": EXPECTED_DEV_EXAMPLES,
            "conditions": list(CONDITIONS),
            "analysis_summary": summary,
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_development_pilot(args)
    print(json.dumps({"output_dir": args.output_dir, "gate": summary["development_kill_test_gate"]}, indent=2))


if __name__ == "__main__":
    main()
