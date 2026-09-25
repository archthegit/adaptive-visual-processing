#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_qwen_temporal_handoff_dev import (
    bootstrap_ci,
    read_json,
    read_jsonl,
)
from src.experiment1.temporal_handoff import (
    SCHEMA_VERSION,
    TemporalHandoffConfig,
    append_jsonl,
    write_json,
)


CONDITIONS = ("dense_custom", "hard_evict", "handoff_mean")
EXPECTED_HELDOUT_EXAMPLES = 56
FROZEN_RETAINED_CELLS = (1, 3)
FROZEN_HANDOFF_LAYER = 8
FROZEN_MEMORY_TOKENS_PER_REGION = 2
DEFAULT_SEED = 20260818
DEFAULT_PROFILE_COUNT = 8
IMMUTABLE_RUN_CONFIG_FIELDS = (
    "schema_version",
    "model_id",
    "resolution_config",
    "sampling_mode",
    "handoff_layer",
    "retained_native_temporal_cells",
    "retention_ratio",
    "primary_condition",
    "secondary_condition",
    "memory_tokens_per_region",
    "seed",
    "conditions",
    "quality_forward_repetitions",
    "profiling_subset_question_ids",
    "warmup",
    "repeats",
    "git_commit",
)
EXPERIMENT_OUTPUT_FILES = (
    "records.jsonl",
    "run_config.json",
    "per_example.csv",
    "summary.json",
    "report.md",
    "quality_delta.png",
    "accuracy_comparison.png",
    "latency_and_flops.png",
    "degradation_distribution.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/analyze frozen held-out Qwen temporal compaction.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", default="outputs/experiment1_v3_cross_model/manifests/heldout_eligible_8frame.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_temporal_compaction/heldout_qwen_pair_1_3")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resolution-config", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile-count", type=int, default=DEFAULT_PROFILE_COUNT)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260928)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def artifact_path(output_dir: Path, question_id: str, condition: str) -> Path:
    return output_dir / "artifacts" / question_id / f"{condition}.json"


def equivalence_path(output_dir: Path, question_id: str) -> Path:
    return output_dir / "artifacts" / question_id / "dense_equivalence_report.json"


def manifest_by_id(path: str | Path, expected_examples: int = EXPECTED_HELDOUT_EXAMPLES) -> dict[str, dict[str, Any]]:
    records = {str(record["question_id"]): record for record in read_jsonl(path)}
    if len(records) != expected_examples:
        raise RuntimeError(f"Expected {expected_examples} held-out records, found {len(records)} in {path}.")
    return records


def select_profiling_subset(question_ids: Sequence[str], *, count: int, seed: int) -> list[str]:
    if count < 0:
        raise ValueError("profile_count must be non-negative.")
    ids = sorted(str(qid) for qid in question_ids)
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[: min(count, len(shuffled))])


def requested_run_config(*, args: argparse.Namespace, git_commit: str, manifest_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    profile_ids = select_profiling_subset(
        sorted(manifest_records),
        count=args.profile_count,
        seed=args.seed,
    )
    return {
        "schema_version": "qwen_temporal_compaction_heldout_pair_1_3_v1",
        "model_id": args.model_id,
        "resolution_config": args.resolution_config,
        "sampling_mode": "cross_model_8",
        "handoff_layer": FROZEN_HANDOFF_LAYER,
        "retained_native_temporal_cells": list(FROZEN_RETAINED_CELLS),
        "retention_ratio": 0.5,
        "primary_condition": "hard_evict",
        "secondary_condition": "handoff_mean",
        "memory_tokens_per_region": FROZEN_MEMORY_TOKENS_PER_REGION,
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "quality_forward_repetitions": 1,
        "profiling_subset_question_ids": profile_ids,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "git_commit": git_commit,
        "timing_scope": (
            "Quality inference is one deterministic forward per condition. Repeated timing is run only "
            "for the frozen profiling subset and excludes video decoding, preprocessing, model loading, "
            "route selection and artifact serialization."
        ),
        "baseline_dir": args.baseline_dir,
        "manifest": args.manifest,
        "output_dir": args.output_dir,
    }


def immutable_subset(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in IMMUTABLE_RUN_CONFIG_FIELDS}


def config_mismatches(saved: dict[str, Any], requested: dict[str, Any]) -> dict[str, dict[str, Any]]:
    saved_subset = immutable_subset(saved)
    requested_subset = immutable_subset(requested)
    return {
        field: {"saved": saved_subset.get(field), "requested": requested_subset.get(field)}
        for field in IMMUTABLE_RUN_CONFIG_FIELDS
        if saved_subset.get(field) != requested_subset.get(field)
    }


def clean_outputs(output_dir: Path) -> None:
    for name in EXPERIMENT_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    artifacts = output_dir / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)


def outputs_present(output_dir: Path) -> bool:
    return (output_dir / "artifacts").exists() or any((output_dir / name).exists() for name in EXPERIMENT_OUTPUT_FILES)


def prepare_run_config(output_dir: Path, requested: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    if overwrite:
        clean_outputs(output_dir)
        write_json(path, requested)
        return requested
    if outputs_present(output_dir):
        if not path.exists():
            raise RuntimeError("Existing held-out compaction outputs are present but run_config.json is missing. Use a new output directory or --overwrite.")
        saved = read_json(path)
        mismatches = config_mismatches(saved, requested)
        if mismatches:
            details = "; ".join(
                f"{field}: saved={values['saved']!r}, requested={values['requested']!r}"
                for field, values in sorted(mismatches.items())
            )
            raise RuntimeError(f"Refusing to resume held-out compaction run because immutable run configuration differs: {details}.")
        return saved
    write_json(path, requested)
    return requested


def completed_records(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = output_dir / "records.jsonl"
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for record in read_jsonl(path):
        if record.get("status") == "complete":
            records[(str(record.get("question_id")), str(record.get("condition")))] = record
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


def _handoff_meta(artifact: dict[str, Any]) -> dict[str, Any]:
    meta = artifact.get("temporal_handoff")
    if not isinstance(meta, dict):
        raise RuntimeError(f"{artifact.get('question_id')}/{artifact.get('condition')}: missing temporal_handoff metadata.")
    return meta


def _instrumentation(artifact: dict[str, Any]) -> dict[str, Any]:
    data = _handoff_meta(artifact).get("instrumentation")
    if not isinstance(data, dict):
        raise RuntimeError(f"{artifact.get('question_id')}/{artifact.get('condition')}: missing instrumentation.")
    return data


def _profile(artifact: dict[str, Any], stage: str = "decoder_stack_excluding_lm_head") -> dict[str, Any]:
    return (((artifact.get("metadata") or {}).get("cuda_profile") or {}).get(stage) or {})


def validate_native_causal_layers(artifact: dict[str, Any]) -> None:
    for layer in _instrumentation(artifact).get("layers") or []:
        if layer.get("layer_type") == "full_attention":
            if layer.get("native_sdpa_is_causal_used") is not True:
                raise RuntimeError(f"{artifact.get('question_id')}/{artifact.get('condition')}: full layer did not use native causal SDPA.")
            if layer.get("explicit_mask_materialized"):
                raise RuntimeError(f"{artifact.get('question_id')}/{artifact.get('condition')}: full layer materialized an explicit mask.")


def validate_condition_artifact(
    artifact: dict[str, Any],
    *,
    question_id: str,
    condition: str,
    run_config: dict[str, Any],
) -> None:
    if artifact.get("question_id") != question_id:
        raise RuntimeError(f"{question_id}/{condition}: artifact question_id mismatch.")
    if artifact.get("condition") != condition:
        raise RuntimeError(f"{question_id}/{condition}: artifact condition mismatch.")
    if artifact.get("status") != "complete":
        raise RuntimeError(f"{question_id}/{condition}: artifact is not complete.")
    if condition not in CONDITIONS:
        raise RuntimeError(f"{question_id}/{condition}: unsupported held-out condition.")
    if artifact.get("model_backend") != "qwen":
        raise RuntimeError(f"{question_id}/{condition}: expected Qwen backend.")
    if artifact.get("model_checkpoint") != run_config.get("model_id"):
        raise RuntimeError(f"{question_id}/{condition}: checkpoint mismatch.")
    metadata = artifact.get("metadata") or {}
    if metadata.get("git_commit") != run_config.get("git_commit"):
        raise RuntimeError(f"{question_id}/{condition}: git commit mismatch.")
    artifact_config = metadata.get("run_config")
    if not isinstance(artifact_config, dict):
        raise RuntimeError(f"{question_id}/{condition}: missing artifact run_config.")
    mismatches = config_mismatches(artifact_config, run_config)
    if mismatches:
        raise RuntimeError(f"{question_id}/{condition}: artifact run_config mismatch: {mismatches}")
    if metadata.get("sampling_mode") != "cross_model_8":
        raise RuntimeError(f"{question_id}/{condition}: expected cross_model_8 sampling.")
    if (metadata.get("resolution") or {}).get("name") not in {None, "medium"}:
        raise RuntimeError(f"{question_id}/{condition}: expected medium resolution metadata.")
    scores = artifact.get("answer_choice_scores") or {}
    _metric(scores, "correct_choice_log_probability")
    _metric(scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    _prediction(scores)
    validate_native_causal_layers(artifact)
    meta = _handoff_meta(artifact)
    if int(meta.get("handoff_layer", FROZEN_HANDOFF_LAYER)) != FROZEN_HANDOFF_LAYER:
        raise RuntimeError(f"{question_id}/{condition}: handoff layer drifted.")
    retained = tuple(int(item) for item in meta.get("selected_retained_native_cells") or meta.get("retained_temporal_regions") or [])
    if condition == "dense_custom":
        if retained:
            raise RuntimeError(f"{question_id}/{condition}: dense_custom must not record retained cells.")
    else:
        if retained != FROZEN_RETAINED_CELLS:
            raise RuntimeError(f"{question_id}/{condition}: retained cells must be {FROZEN_RETAINED_CELLS}, got {retained}.")
    memory = int(meta.get("memory_tokens_per_region", 0))
    expected_memory = FROZEN_MEMORY_TOKENS_PER_REGION if condition == "handoff_mean" else 0
    if memory != expected_memory:
        raise RuntimeError(f"{question_id}/{condition}: memory-token count mismatch.")


def validate_resumed_artifact(path: Path, *, question_id: str, condition: str, run_config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact missing or empty: {path}")
    artifact = read_json(path)
    validate_condition_artifact(artifact, question_id=question_id, condition=condition, run_config=run_config)
    return artifact


def validate_equivalence(path: Path, question_id: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{question_id}: dense-equivalence report missing or empty.")
    report = read_json(path)
    if report.get("question_id") != question_id or report.get("passed") is not True:
        raise RuntimeError(f"{question_id}: dense equivalence gate failed.")
    return report


def resolve_artifact(record: dict[str, Any], output_dir: Path) -> Path:
    raw = record.get("artifact")
    if not raw:
        raise RuntimeError("Complete record is missing artifact path.")
    path = Path(raw)
    if path.exists():
        return path
    return artifact_path(output_dir, str(record.get("question_id")), str(record.get("condition")))


def latest_artifacts(output_dir: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    output = Path(output_dir)
    records = output / "records.jsonl"
    artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    if records.exists():
        for record in read_jsonl(records):
            if record.get("status") == "complete":
                qid = str(record["question_id"])
                condition = str(record["condition"])
                artifacts.setdefault(qid, {})[condition] = read_json(resolve_artifact(record, output))
        return artifacts
    for path in output.glob("artifacts/*/*.json"):
        if path.name == "dense_equivalence_report.json":
            continue
        artifact = read_json(path)
        artifacts.setdefault(str(artifact["question_id"]), {})[str(artifact["condition"])] = artifact
    return artifacts


def load_equivalence_reports(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    output = Path(output_dir)
    return {path.parent.name: read_json(path) for path in output.glob("artifacts/*/dense_equivalence_report.json")}


def validate_all_outputs(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    equivalence_reports: dict[str, dict[str, Any]],
    manifest_records: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    if set(artifacts) != set(manifest_records):
        raise RuntimeError(f"Artifact IDs do not match manifest: extra={sorted(set(artifacts)-set(manifest_records))}, missing={sorted(set(manifest_records)-set(artifacts))}")
    if set(equivalence_reports) != set(manifest_records):
        raise RuntimeError("Dense equivalence reports do not match manifest IDs.")
    for qid in sorted(manifest_records):
        missing = [condition for condition in CONDITIONS if condition not in artifacts[qid]]
        if missing:
            raise RuntimeError(f"{qid}: missing conditions {missing}.")
        for condition in CONDITIONS:
            validate_condition_artifact(artifacts[qid][condition], question_id=qid, condition=condition, run_config=run_config)
        if equivalence_reports[qid].get("passed") is not True:
            raise RuntimeError(f"{qid}: dense equivalence did not pass.")


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
        dense_stack = _profile(dense)
        dense_latency = dense_stack.get("prefill_latency_seconds_median")
        dense_flops = float(_instrumentation(dense).get("total_estimated_attention_flops"))
        dense_seq = int((dense.get("metadata") or {}).get("final_sequence_length"))
        for condition in CONDITIONS:
            artifact = artifacts[qid][condition]
            scores = artifact["answer_choice_scores"]
            logp = _metric(scores, "correct_choice_log_probability")
            margin = _metric(scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
            stack = _profile(artifact)
            latency = stack.get("prefill_latency_seconds_median")
            flops = float(_instrumentation(artifact).get("total_estimated_attention_flops"))
            final_seq = int((artifact.get("metadata") or {}).get("final_sequence_length"))
            latency_reduction = None
            latency_speedup = None
            if dense_latency is not None and latency is not None and float(latency) > 0:
                latency_reduction = (float(dense_latency) - float(latency)) / float(dense_latency)
                latency_speedup = float(dense_latency) / float(latency)
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
                    "dense_stack_latency_seconds_median": dense_latency,
                    "condition_stack_latency_seconds_median": latency,
                    "stack_latency_reduction_fraction": latency_reduction,
                    "stack_latency_speedup": latency_speedup,
                    "timing_profiled": bool(stack.get("profiled", latency is not None)),
                    "dense_sequence_length": dense_seq,
                    "condition_sequence_length": final_seq,
                    "sequence_reduction_fraction": (dense_seq - final_seq) / dense_seq if dense_seq else float("nan"),
                    "dense_attention_flops": dense_flops,
                    "condition_attention_flops": flops,
                    "attention_flop_reduction_fraction": (dense_flops - flops) / dense_flops if dense_flops else float("nan"),
                    "dense_incremental_peak_allocated_bytes": dense_stack.get("incremental_peak_allocated_bytes"),
                    "condition_incremental_peak_allocated_bytes": stack.get("incremental_peak_allocated_bytes"),
                    "selected_retained_temporal_cells": json.dumps((_handoff_meta(artifact).get("selected_retained_native_cells") or [])),
                }
            )
    return rows


def finite_values(values: Sequence[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def condition_rows(rows: Sequence[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    subset = [row for row in rows if row["condition"] == condition]
    if not subset:
        raise RuntimeError(f"No rows for {condition}.")
    return subset


def paired_condition_difference(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    field: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_key = {(row["question_id"], row["condition"]): row for row in rows}
    values = []
    for qid in sorted({row["question_id"] for row in rows}):
        values.append(float(by_key[(qid, left)][field]) - float(by_key[(qid, right)][field]))
    return bootstrap_ci(values, samples=samples, seed=seed)


def paired_accuracy_difference(rows: Sequence[dict[str, Any]], left: str, right: str, *, samples: int, seed: int) -> dict[str, Any]:
    by_key = {(row["question_id"], row["condition"]): row for row in rows}
    values = []
    for qid in sorted({row["question_id"] for row in rows}):
        values.append(float(bool(by_key[(qid, left)]["condition_correct"])) - float(bool(by_key[(qid, right)]["condition_correct"])))
    return bootstrap_ci(values, samples=samples, seed=seed)


def condition_stats(rows: Sequence[dict[str, Any]], condition: str, *, samples: int, seed: int) -> dict[str, Any]:
    subset = condition_rows(rows, condition)
    latency_reductions = finite_values([row["stack_latency_reduction_fraction"] for row in subset])
    speedups = finite_values([row["stack_latency_speedup"] for row in subset])
    memory_deltas = finite_values(
        [
            (float(row["condition_incremental_peak_allocated_bytes"]) - float(row["dense_incremental_peak_allocated_bytes"]))
            for row in subset
            if row["condition_incremental_peak_allocated_bytes"] is not None and row["dense_incremental_peak_allocated_bytes"] is not None
        ]
    )
    return {
        "num_examples": len(subset),
        "num_profiled_examples": int(sum(bool(row["timing_profiled"]) for row in subset)),
        "correct_choice_log_probability_delta": bootstrap_ci([row["delta_correct_choice_log_probability"] for row in subset], samples=samples, seed=seed),
        "answer_margin_delta": bootstrap_ci([row["delta_answer_margin"] for row in subset], samples=samples, seed=seed + 11),
        "dense_accuracy": float(np.mean([bool(row["dense_correct"]) for row in subset])),
        "condition_accuracy": float(np.mean([bool(row["condition_correct"]) for row in subset])),
        "accuracy_difference": paired_accuracy_difference(rows, condition, "dense_custom", samples=samples, seed=seed + 13),
        "prediction_flip_rate": float(np.mean([bool(row["prediction_changed"]) for row in subset])),
        "fraction_logp_degradation_below_minus_0_10": float(np.mean([float(row["delta_correct_choice_log_probability"]) < -0.10 for row in subset])),
        "fraction_logp_degradation_below_minus_0_50": float(np.mean([float(row["delta_correct_choice_log_probability"]) < -0.50 for row in subset])),
        "median_stack_latency_reduction_fraction": float(np.median(latency_reductions)) if latency_reductions else None,
        "median_stack_latency_speedup": float(np.median(speedups)) if speedups else None,
        "median_attention_flop_reduction_fraction": float(np.median([row["attention_flop_reduction_fraction"] for row in subset])),
        "median_sequence_reduction_fraction": float(np.median([row["sequence_reduction_fraction"] for row in subset])),
        "peak_memory_difference_incremental_allocated_bytes": bootstrap_ci(memory_deltas, samples=samples, seed=seed + 23),
    }


def primary_gate(summary: dict[str, Any]) -> dict[str, Any]:
    hard = summary["conditions"]["hard_evict"]
    lower = hard["correct_choice_log_probability_delta"]["ci95"][0]
    latency = hard["median_stack_latency_reduction_fraction"]
    noninferior = lower is not None and lower > -0.10
    efficient = latency is not None and latency >= 0.20
    status = "PASS" if noninferior and efficient else "FAIL"
    return {
        "status": status,
        "quality_noninferiority_pass": noninferior,
        "hard_minus_dense_logp_lower_ci95": lower,
        "efficiency_pass": efficient,
        "hard_evict_median_decoder_stack_latency_reduction_fraction": latency,
        "rules": [
            "lower 95% paired-bootstrap bound for hard_evict minus dense log probability must be greater than -0.10",
            "median decoder-stack latency reduction must be at least 20%",
        ],
    }


def summarize_analysis(rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    conditions = {
        condition: condition_stats(rows, condition, samples=bootstrap_samples, seed=seed + idx * 1000)
        for idx, condition in enumerate(CONDITIONS)
    }
    paired = {
        "handoff_mean_minus_hard_evict": {
            "correct_choice_log_probability_delta": paired_condition_difference(rows, "handoff_mean", "hard_evict", "delta_correct_choice_log_probability", samples=bootstrap_samples, seed=seed + 5000),
            "answer_margin_delta": paired_condition_difference(rows, "handoff_mean", "hard_evict", "delta_answer_margin", samples=bootstrap_samples, seed=seed + 5011),
            "accuracy_difference": paired_accuracy_difference(rows, "handoff_mean", "hard_evict", samples=bootstrap_samples, seed=seed + 5022),
        }
    }
    memory_logp_ci = paired["handoff_mean_minus_hard_evict"]["correct_choice_log_probability_delta"]["ci95"]
    memory_margin_ci = paired["handoff_mean_minus_hard_evict"]["answer_margin_delta"]["ci95"]
    summary = {
        "schema_version": "qwen_temporal_compaction_heldout_analysis_v1",
        "num_examples": len({row["question_id"] for row in rows}),
        "conditions": conditions,
        "paired_condition_differences": paired,
        "primary_condition": "hard_evict",
        "secondary_condition": "handoff_mean",
        "retained_native_temporal_cells": list(FROZEN_RETAINED_CELLS),
        "timing_scope": "Repeated profiling is limited to the frozen profiling subset; quality rows use deterministic single forwards.",
        "primary_heldout_gate": {},
        "secondary_memory_comparison": {
            "memory_benefit_supported_logp": memory_logp_ci[0] is not None and memory_logp_ci[0] > 0,
            "memory_benefit_supported_margin": memory_margin_ci[0] is not None and memory_margin_ci[0] > 0,
            "interpretation": "Do not claim memory benefit if the paired confidence interval crosses zero.",
        },
    }
    summary["primary_heldout_gate"] = primary_gate(summary)
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
    ax.bar(labels, values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
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
        "Δ logp vs dense_custom",
    )
    _bar_plot(
        output_dir / "accuracy_comparison.png",
        labels,
        [summary["conditions"][label]["condition_accuracy"] for label in labels],
        "Condition accuracy",
        "Accuracy",
    )
    _bar_plot(
        output_dir / "latency_and_flops.png",
        labels,
        [summary["conditions"][label]["median_attention_flop_reduction_fraction"] for label in labels],
        "Estimated attention-FLOP reduction",
        "Median reduction fraction",
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for condition in ("hard_evict", "handoff_mean"):
        subset = [row["delta_correct_choice_log_probability"] for row in rows if row["condition"] == condition]
        ax.hist(subset, bins=12, alpha=0.5, label=condition)
    ax.axvline(-0.10, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-0.50, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Δ correct-answer log probability vs dense")
    ax.set_ylabel("Examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "degradation_distribution.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["primary_heldout_gate"]
    lines = [
        "# Qwen Temporal Compaction Held-Out Evaluation",
        "",
        "This is the frozen 56-example held-out confirmatory evaluation for retained native temporal cells `[1, 3]`.",
        "The primary condition is `hard_evict`; `handoff_mean` is a secondary mechanistic comparison on identical examples.",
        "",
        f"Primary gate: **{gate['status']}**.",
        "",
        "| Condition | Accuracy | Mean Δ logp | Mean Δ margin | Median FLOP reduction | Profiled examples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        stats = summary["conditions"][condition]
        lines.append(
            "| {condition} | {acc:.3f} | {logp:.6f} | {margin:.6f} | {flops:.3f} | {profiled} |".format(
                condition=condition,
                acc=stats["condition_accuracy"],
                logp=stats["correct_choice_log_probability_delta"]["mean"],
                margin=stats["answer_margin_delta"]["mean"],
                flops=stats["median_attention_flop_reduction_fraction"],
                profiled=stats["num_profiled_examples"],
            )
        )
    lines.extend(
        [
            "",
            "Latency profiling is limited to the frozen profiling subset and excludes video decoding, preprocessing, model loading, route selection and serialization.",
            "The example is the independent statistical unit; pair rows are not used in this held-out evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def analyze_outputs(
    output_dir: str | Path,
    manifest: str | Path,
    *,
    bootstrap_samples: int,
    seed: int,
    write_outputs: bool = True,
    expected_examples: int = EXPECTED_HELDOUT_EXAMPLES,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_records = manifest_by_id(manifest, expected_examples=expected_examples)
    run_config = read_json(output / "run_config.json")
    artifacts = latest_artifacts(output)
    equivalence = load_equivalence_reports(output)
    validate_all_outputs(artifacts, equivalence, manifest_records, run_config)
    rows = per_example_rows(artifacts, manifest_records)
    summary = summarize_analysis(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    if write_outputs:
        write_csv(output / "per_example.csv", rows)
        write_json(output / "summary.json", summary)
        write_plots(output, rows, summary)
        write_report(output / "report.md", summary)
    return summary


def make_artifact(
    *,
    qid: str,
    condition: str,
    result: Any,
    scores: dict[str, Any],
    example: Any,
    manifest_record: dict[str, Any],
    model_id: str,
    prompt: str,
    rendered: str,
    frame_batches: Sequence[Any],
    run_config: dict[str, Any],
    git_commit: str,
    resolution: Any,
    baseline_record: dict[str, Any],
    native: Any,
    baseline_scores: Sequence[Sequence[float]],
    original_sequence_length: int,
    cuda_profile: dict[str, Any],
) -> dict[str, Any]:
    from scripts.run_qwen_temporal_handoff_smoke import _prediction_from_scores

    return {
        "question_id": qid,
        "condition": condition,
        "status": "complete",
        "category": manifest_record.get("category"),
        "question_type": manifest_record.get("question_type"),
        "participant_id": manifest_record.get("participant_id"),
        "source_video_id": manifest_record.get("source_video_id"),
        "model_backend": "qwen",
        "model_checkpoint": model_id,
        "generation_supported": False,
        "generation_unsupported_reason": "Temporal compaction held-out evaluation is prefill-only.",
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
            "handoff_layer": FROZEN_HANDOFF_LAYER,
            "retained_temporal_regions": list(FROZEN_RETAINED_CELLS) if condition != "dense_custom" else [],
            "selected_retained_native_cells": list(FROZEN_RETAINED_CELLS) if condition != "dense_custom" else [],
            "memory_tokens_per_region": FROZEN_MEMORY_TOKENS_PER_REGION if condition == "handoff_mean" else 0,
            "condition": condition,
            "baseline_artifact": baseline_record.get("artifact"),
            "native_temporal_aggregation": native.to_metadata(),
            "per_layer_native_cell_scores_used_for_selection": [list(layer) for layer in baseline_scores],
            "compaction_plan": result.compaction_plan.to_metadata() if result.compaction_plan else None,
            "instrumentation": result.instrumentation_metadata(),
        },
        "metadata": {
            "git_commit": git_commit,
            "run_config": run_config,
            "resolution": resolution.to_metadata(),
            "sampling_mode": "cross_model_8",
            "query_scope": "question",
            "original_sequence_length": original_sequence_length,
            "final_sequence_length": int(result.final_hidden_states.shape[1]),
            "cuda_profile": cuda_profile,
        },
    }


def run_condition(
    *,
    condition: str,
    execute: Any,
    stack: dict[str, Any],
    tokenizer: Any,
    correct_idx: int,
    num_choices: int,
    profile: bool,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from src.experiment1.answer_scoring import score_answer_choice_logits
    from src.experiment1.temporal_handoff import cuda_profile_prefill

    result = execute()
    scores = score_answer_choice_logits(result.logits[0, -1], tokenizer, correct_idx, num_choices).to_json_dict()
    if not profile:
        return result, scores, {
            "combined_prefill": {"profiled": False},
            "decoder_stack_excluding_lm_head": {"profiled": False},
            "final_token_lm_head": {"profiled": False},
        }

    profiled_result, combined = cuda_profile_prefill(execute, warmup=warmup, repeats=repeats)

    def execute_stack_only() -> Any:
        return execute(lm_head_override=None)

    stack_result, stack_profile = cuda_profile_prefill(execute_stack_only, warmup=warmup, repeats=repeats)
    lm_logits, lm_profile = cuda_profile_prefill(
        lambda: stack["lm_head"](stack_result.final_token_hidden_state),
        warmup=warmup,
        repeats=repeats,
    )
    combined["profiled"] = True
    stack_profile["profiled"] = True
    lm_profile["profiled"] = True
    _ = profiled_result, lm_logits
    return result, scores, {
        "combined_prefill": combined,
        "decoder_stack_excluding_lm_head": stack_profile,
        "final_token_lm_head": lm_profile,
    }


def run_heldout(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.run_experiment1 import current_git_commit, frame_batches_for_example, load_examples_by_id, load_manifest
    from scripts.run_qwen_temporal_handoff_smoke import (
        _build_layout,
        _load_artifact_from_record,
        _latest_complete_record,
        _prepare_inputs,
        _stock_logits,
        _temporal_scores_from_artifact,
    )
    from src.experiment1.answer_scoring import score_answer_choice_logits
    from src.experiment1.resolution import get_resolution_config
    from src.experiment1.temporal_handoff import (
        aggregate_analysis_scores_to_native_cells,
        dense_equivalence_report,
        qwen_decoder_stack,
        qwen_multimodal_decoder_inputs,
        run_custom_decoder_prefill,
    )
    from src.models.qwen import Qwen25VLWrapper, QwenConfig

    output = Path(args.output_dir)
    manifest_records_list = load_manifest(args.manifest)
    if len(manifest_records_list) != EXPECTED_HELDOUT_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_HELDOUT_EXAMPLES} held-out examples, found {len(manifest_records_list)}.")
    manifest_records = {str(record["question_id"]): record for record in manifest_records_list}
    examples = load_examples_by_id(args.questions_dir, manifest_records_list)
    resolution = get_resolution_config(args.resolution_config)
    git_commit = current_git_commit()
    requested = requested_run_config(args=args, git_commit=git_commit, manifest_records=manifest_records)
    run_config = prepare_run_config(output, requested, overwrite=args.overwrite)
    records_path = output / "records.jsonl"
    completed = completed_records(output) if not args.overwrite else {}

    model = Qwen25VLWrapper(QwenConfig(model_id=args.model_id, max_new_tokens=1, attn_implementation="sdpa"))
    model._load()
    assert model._model is not None
    assert model._processor is not None
    stack = qwen_decoder_stack(model._model)

    for qid in sorted(manifest_records):
        example = examples[qid]
        manifest_record = manifest_records[qid]
        should_profile = qid in set(run_config["profiling_subset_question_ids"])
        frame_batches = frame_batches_for_example(
            example,
            args.mp4_dir,
            num_frames=8,
            sampling_mode="cross_model_8",
            manifest_record=manifest_record,
            frames_per_bin_override=1,
        )
        baseline_record = _latest_complete_record(Path(args.baseline_dir) / "records.jsonl", qid)
        baseline_artifact = _load_artifact_from_record(baseline_record)
        native = aggregate_analysis_scores_to_native_cells(_temporal_scores_from_artifact(baseline_artifact), baseline_artifact)
        baseline_scores = [list(layer) for layer in native.native_temporal_scores]
        if native.native_temporal_cell_count != 4:
            raise RuntimeError(f"{qid}: expected four Qwen native temporal cells.")
        inputs, rendered, prompt, _video_kwargs = _prepare_inputs(model, example, frame_batches, resolution)
        layout = _build_layout(model, example, inputs, rendered, frame_batches)
        decoder_inputs, position_ids = qwen_multimodal_decoder_inputs(model._model, inputs)
        stock_logits = _stock_logits(model, inputs)
        stock_scores = score_answer_choice_logits(stock_logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()

        def build_execute(config: TemporalHandoffConfig) -> Any:
            def execute(*, lm_head_override: Any = stack["lm_head"]) -> Any:
                return run_custom_decoder_prefill(
                    layers=stack["layers"],
                    hidden_states=decoder_inputs.clone(),
                    position_ids=position_ids.clone() if position_ids is not None else None,
                    layout=layout,
                    config=config,
                    lm_head=lm_head_override,
                    norm=stack["norm"],
                    num_attention_heads=stack["num_attention_heads"],
                    head_dim=stack["head_dim"],
                    rotary_emb=stack["rotary_emb"],
                    layer_types=stack["layer_types"],
                    sliding_window=stack["sliding_window"],
                )
            return execute

        dense_logits = None
        dense_scores = None
        for condition in CONDITIONS:
            out_path = artifact_path(output, qid, condition)
            if (qid, condition) in completed:
                artifact = validate_resumed_artifact(resolve_artifact(completed[(qid, condition)], output), question_id=qid, condition=condition, run_config=run_config)
                if condition == "dense_custom":
                    dense_scores = artifact["answer_choice_scores"]
                continue
            config = TemporalHandoffConfig(
                condition=condition,
                handoff_layer=FROZEN_HANDOFF_LAYER,
                retained_temporal_regions=FROZEN_RETAINED_CELLS if condition != "dense_custom" else (),
                memory_tokens_per_region=FROZEN_MEMORY_TOKENS_PER_REGION if condition == "handoff_mean" else 0,
                random_seed=args.seed,
            )
            execute = build_execute(config)
            result, scores, cuda_profile = run_condition(
                condition=condition,
                execute=execute,
                stack=stack,
                tokenizer=model._processor.tokenizer,
                correct_idx=example.correct_idx,
                num_choices=len(example.choices),
                profile=should_profile,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            artifact = make_artifact(
                qid=qid,
                condition=condition,
                result=result,
                scores=scores,
                example=example,
                manifest_record=manifest_record,
                model_id=args.model_id,
                prompt=prompt,
                rendered=rendered,
                frame_batches=frame_batches,
                run_config=run_config,
                git_commit=git_commit,
                resolution=resolution,
                baseline_record=baseline_record,
                native=native,
                baseline_scores=baseline_scores,
                original_sequence_length=int(decoder_inputs.shape[1]),
                cuda_profile=cuda_profile,
            )
            write_json(out_path, artifact)
            validate_resumed_artifact(out_path, question_id=qid, condition=condition, run_config=run_config)
            append_jsonl(records_path, {"question_id": qid, "condition": condition, "status": "complete", "artifact": str(out_path)})
            if condition == "dense_custom":
                dense_logits = result.logits
                dense_scores = scores

        eq_path = equivalence_path(output, qid)
        if args.overwrite or not eq_path.exists():
            if dense_logits is None or dense_scores is None:
                raise RuntimeError(f"{qid}: dense equivalence report missing but dense_custom was resumed.")
            equivalence = dense_equivalence_report(
                stock_logits,
                dense_logits,
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
            if equivalence.get("passed") is not True:
                raise RuntimeError(f"{qid}: dense_custom failed BF16-aware equivalence gate.")
        validate_equivalence(eq_path, qid)

    return analyze_outputs(output, args.manifest, bootstrap_samples=args.bootstrap_samples, seed=args.analysis_seed, write_outputs=True)


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        summary = analyze_outputs(args.output_dir, args.manifest, bootstrap_samples=args.bootstrap_samples, seed=args.analysis_seed, write_outputs=True)
    else:
        summary = run_heldout(args)
    print(json.dumps({"output_dir": args.output_dir, "primary_gate": summary["primary_heldout_gate"]}, indent=2))


if __name__ == "__main__":
    main()
