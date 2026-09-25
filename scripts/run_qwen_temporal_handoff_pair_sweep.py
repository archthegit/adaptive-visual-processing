#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_qwen_temporal_handoff_dev import (
    DEFAULT_SEED,
    EXPECTED_DEV_EXAMPLES,
    bootstrap_ci,
    read_json,
    read_jsonl,
)
from src.experiment1.temporal_handoff import SCHEMA_VERSION, TemporalHandoffConfig, append_jsonl, write_json


PAIR_SWEEP_CONDITIONS = ("handoff_mean", "hard_evict")
RETAINED_PAIRS = tuple(tuple(pair) for pair in itertools.combinations(range(4), 2))
IMMUTABLE_RUN_CONFIG_FIELDS = (
    "schema_version",
    "model_id",
    "resolution_config",
    "sampling_mode",
    "handoff_layer",
    "memory_tokens_per_region",
    "seed",
    "conditions",
    "retained_pairs",
    "git_commit",
)
EXPERIMENT_OUTPUT_FILES = (
    "records.jsonl",
    "run_config.json",
    "per_pair.csv",
    "per_example_summary.csv",
    "analysis_summary.json",
    "report.md",
    "attention_selection_rank.png",
    "attention_mass_vs_quality.png",
    "memory_vs_eviction.png",
    "route_quality_range.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exhaustive Qwen temporal-handoff retained-pair sweep.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--pilot-dir", default="outputs/experiment1_v3_temporal_handoff/dev_qwen")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_temporal_handoff/dev_qwen_pair_sweep")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resolution-config", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--handoff-layer", type=int, default=8)
    parser.add_argument("--memory-tokens-per-region", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260926)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pair_key(pair: Sequence[int]) -> str:
    return "_".join(str(int(item)) for item in pair)


def artifact_path(output_dir: Path, question_id: str, pair: Sequence[int] | None, condition: str) -> Path:
    if condition == "dense_custom":
        return output_dir / "artifacts" / question_id / "dense_custom.json"
    return output_dir / "artifacts" / question_id / f"pair_{pair_key(pair or ())}" / f"{condition}.json"


def equivalence_path(output_dir: Path, question_id: str) -> Path:
    return output_dir / "artifacts" / question_id / "dense_equivalence_report.json"


def manifest_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    records = {str(record["question_id"]): record for record in read_jsonl(path)}
    if len(records) != EXPECTED_DEV_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_DEV_EXAMPLES} development records, found {len(records)} in {path}.")
    return records


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
            raise RuntimeError("Existing pair-sweep outputs are present but run_config.json is missing. Use a new output directory or --overwrite.")
        saved = read_json(path)
        mismatches = config_mismatches(saved, requested)
        if mismatches:
            details = "; ".join(
                f"{field}: saved={values['saved']!r}, requested={values['requested']!r}"
                for field, values in sorted(mismatches.items())
            )
            raise RuntimeError(f"Refusing to resume pair sweep because immutable run configuration differs: {details}.")
        return saved
    write_json(path, requested)
    return requested


def completed_records(output_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = output_dir / "records.jsonl"
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for record in read_jsonl(path):
        if record.get("status") == "complete":
            qid = str(record.get("question_id"))
            condition = str(record.get("condition"))
            pair = "dense" if condition == "dense_custom" else pair_key(record.get("retained_pair") or ())
            records[(qid, pair, condition)] = record
    return records


def resolve_artifact(record: dict[str, Any], output_dir: Path) -> Path:
    raw = record.get("artifact")
    if not raw:
        raise RuntimeError("Complete record is missing artifact path.")
    path = Path(raw)
    if path.exists():
        return path
    qid = str(record.get("question_id"))
    condition = str(record.get("condition"))
    pair = record.get("retained_pair")
    fallback = artifact_path(output_dir, qid, pair, condition)
    return fallback


def answer_metric(scores: dict[str, Any], primary: str, fallback: str | None = None) -> float:
    value = scores.get(primary)
    if value is None and fallback:
        value = scores.get(fallback)
    if value is None:
        raise RuntimeError(f"Missing score field {primary!r}.")
    return float(value)


def predicted_idx(scores: dict[str, Any]) -> int:
    logits = scores.get("choice_logits")
    if not logits:
        raise RuntimeError("Missing choice_logits.")
    return int(max(range(len(logits)), key=lambda idx: float(logits[idx])))


def instrumentation(artifact: dict[str, Any]) -> dict[str, Any]:
    handoff = artifact.get("temporal_handoff") or {}
    data = handoff.get("instrumentation")
    if not isinstance(data, dict):
        raise RuntimeError(f"{artifact.get('question_id')}/{artifact.get('condition')}: missing instrumentation.")
    return data


def validate_artifact(artifact: dict[str, Any], *, question_id: str, condition: str, pair: Sequence[int] | None, run_config: dict[str, Any]) -> None:
    if artifact.get("question_id") != question_id:
        raise RuntimeError(f"{question_id}/{condition}: artifact question_id mismatch.")
    if artifact.get("condition") != condition:
        raise RuntimeError(f"{question_id}/{condition}: artifact condition mismatch.")
    if artifact.get("status") != "complete":
        raise RuntimeError(f"{question_id}/{condition}: artifact is not complete.")
    if artifact.get("model_backend") != "qwen":
        raise RuntimeError(f"{question_id}/{condition}: expected Qwen backend.")
    metadata = artifact.get("metadata") or {}
    if metadata.get("git_commit") != run_config.get("git_commit"):
        raise RuntimeError(f"{question_id}/{condition}: artifact git commit differs from run config.")
    artifact_config = metadata.get("run_config")
    if not isinstance(artifact_config, dict):
        raise RuntimeError(f"{question_id}/{condition}: missing artifact run_config.")
    mismatches = config_mismatches(artifact_config, run_config)
    if mismatches:
        raise RuntimeError(f"{question_id}/{condition}: artifact run_config differs from run_config: {mismatches}")
    if condition != "dense_custom":
        retained = tuple(int(item) for item in artifact.get("retained_pair") or [])
        if retained != tuple(pair or ()):
            raise RuntimeError(f"{question_id}/{condition}: retained pair mismatch.")
    for layer in instrumentation(artifact).get("layers") or []:
        if layer.get("layer_type") == "full_attention":
            if layer.get("native_sdpa_is_causal_used") is not True or layer.get("explicit_mask_materialized"):
                raise RuntimeError(f"{question_id}/{condition}: full attention did not use native causal SDPA.")
    scores = artifact.get("answer_choice_scores") or {}
    answer_metric(scores, "correct_choice_log_probability")
    answer_metric(scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
    predicted_idx(scores)


def validate_resumed_artifact(path: Path, *, question_id: str, condition: str, pair: Sequence[int] | None, run_config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{question_id}/{condition}: resumed artifact missing or empty: {path}")
    artifact = read_json(path)
    validate_artifact(artifact, question_id=question_id, condition=condition, pair=pair, run_config=run_config)
    return artifact


def validate_equivalence(path: Path, question_id: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{question_id}: dense-equivalence report missing or empty.")
    report = read_json(path)
    if report.get("question_id") != question_id or report.get("passed") is not True:
        raise RuntimeError(f"{question_id}: dense-equivalence report is invalid.")
    return report


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][0] == indexed[start][0]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for _, idx in indexed[start : end + 1]:
            ranks[idx] = rank
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = np.asarray(average_ranks(x), dtype=float)
    ry = np.asarray(average_ranks(y), dtype=float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def latest_artifacts(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    output = Path(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    for path in output.glob("artifacts/**/*.json"):
        if path.name == "dense_equivalence_report.json":
            continue
        payload = read_json(path)
        qid = str(payload["question_id"])
        condition = str(payload["condition"])
        if condition == "dense_custom":
            key = f"{qid}|dense|dense_custom"
        else:
            key = f"{qid}|{pair_key(payload.get('retained_pair') or [])}|{condition}"
        artifacts[key] = payload
    return artifacts


def pair_rows_from_artifacts(artifacts: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(manifest):
        dense_key = f"{qid}|dense|dense_custom"
        if dense_key not in artifacts:
            raise RuntimeError(f"{qid}: missing dense_custom artifact.")
        dense = artifacts[dense_key]
        dense_scores = dense.get("answer_choice_scores") or {}
        dense_logp = answer_metric(dense_scores, "correct_choice_log_probability")
        dense_margin = answer_metric(dense_scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
        for pair in RETAINED_PAIRS:
            for condition in PAIR_SWEEP_CONDITIONS:
                key = f"{qid}|{pair_key(pair)}|{condition}"
                if key not in artifacts:
                    raise RuntimeError(f"{qid}: missing {condition} pair {pair}.")
                artifact = artifacts[key]
                scores = artifact.get("answer_choice_scores") or {}
                logp = answer_metric(scores, "correct_choice_log_probability")
                margin = answer_metric(scores, "correct_vs_best_incorrect_margin", "correct_vs_strongest_incorrect_margin")
                instr = instrumentation(artifact)
                rows.append(
                    {
                        "question_id": qid,
                        "condition": condition,
                        "retained_pair": pair_key(pair),
                        "retained_cell_a": pair[0],
                        "retained_cell_b": pair[1],
                        "retained_attention_mass": float(artifact.get("retained_baseline_attention_mass")),
                        "correct_choice_log_probability": logp,
                        "answer_margin": margin,
                        "delta_logp_vs_dense": logp - dense_logp,
                        "delta_margin_vs_dense": margin - dense_margin,
                        "predicted_idx": predicted_idx(scores),
                        "correct": bool(artifact.get("correct")),
                        "dense_correct_choice_log_probability": dense_logp,
                        "dense_answer_margin": dense_margin,
                        "original_sequence_length": int((artifact.get("metadata") or {}).get("original_sequence_length")),
                        "final_sequence_length": int((artifact.get("metadata") or {}).get("final_sequence_length")),
                        "estimated_attention_flops": float(instr.get("total_estimated_attention_flops")),
                        "visual_token_count": len(instr.get("final_visual_token_indices") or []),
                        "memory_token_count": len(instr.get("final_memory_token_indices") or []),
                        "category": manifest[qid].get("category"),
                        "participant_id": manifest[qid].get("participant_id"),
                        "source_video_id": manifest[qid].get("source_video_id"),
                    }
                )
    return rows


def _rank_desc(values: Sequence[float], selected_idx: int) -> int:
    selected = float(values[selected_idx])
    return 1 + sum(1 for value in values if float(value) > selected)


def per_example_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    output = []
    for qid, all_rows in sorted(grouped.items()):
        handoff = [row for row in all_rows if row["condition"] == "handoff_mean"]
        hard = [row for row in all_rows if row["condition"] == "hard_evict"]
        handoff_by_pair = {row["retained_pair"]: row for row in handoff}
        hard_by_pair = {row["retained_pair"]: row for row in hard}
        selected_pair = max(handoff, key=lambda row: (row["retained_attention_mass"], row["retained_pair"]))["retained_pair"]
        selected = handoff_by_pair[selected_pair]
        others = [row for row in handoff if row["retained_pair"] != selected_pair]
        handoff_logps = [row["correct_choice_log_probability"] for row in handoff]
        handoff_margins = [row["answer_margin"] for row in handoff]
        selected_idx = [row["retained_pair"] for row in handoff].index(selected_pair)
        memory_diffs_logp = [
            handoff_by_pair[pair]["correct_choice_log_probability"] - hard_by_pair[pair]["correct_choice_log_probability"]
            for pair in handoff_by_pair
        ]
        memory_diffs_margin = [
            handoff_by_pair[pair]["answer_margin"] - hard_by_pair[pair]["answer_margin"]
            for pair in handoff_by_pair
        ]
        logp_threshold_degradation = [
            row["delta_logp_vs_dense"] for row in handoff
        ]
        row = {
            "question_id": qid,
            "attention_selected_pair": selected_pair,
            "attention_selected_logp": selected["correct_choice_log_probability"],
            "attention_selected_margin": selected["answer_margin"],
            "mean_other_five_logp": float(np.mean([row["correct_choice_log_probability"] for row in others])),
            "median_other_five_logp": float(np.median([row["correct_choice_log_probability"] for row in others])),
            "attention_selected_minus_mean_other_five_logp": selected["correct_choice_log_probability"] - float(np.mean([row["correct_choice_log_probability"] for row in others])),
            "attention_selected_minus_mean_other_five_margin": selected["answer_margin"] - float(np.mean([row["answer_margin"] for row in others])),
            "attention_selected_rank_logp": _rank_desc(handoff_logps, selected_idx),
            "attention_selected_is_best": _rank_desc(handoff_logps, selected_idx) == 1,
            "attention_selected_is_top_two": _rank_desc(handoff_logps, selected_idx) <= 2,
            "attention_selected_is_worst": _rank_desc(handoff_logps, selected_idx) == 6,
            "mean_handoff_minus_hard_evict_logp": float(np.mean(memory_diffs_logp)),
            "mean_handoff_minus_hard_evict_margin": float(np.mean(memory_diffs_margin)),
            "spearman_attention_mass_vs_logp": spearman([row["retained_attention_mass"] for row in handoff], handoff_logps),
            "spearman_attention_mass_vs_margin": spearman([row["retained_attention_mass"] for row in handoff], handoff_margins),
            "best_pair_logp": float(np.max(handoff_logps)),
            "worst_pair_logp": float(np.min(handoff_logps)),
            "pair_logp_range": float(np.max(handoff_logps) - np.min(handoff_logps)),
            "fraction_pairs_logp_degradation_below_minus_0_1": float(np.mean([value < -0.1 for value in logp_threshold_degradation])),
            "fraction_pairs_logp_degradation_below_minus_0_5": float(np.mean([value < -0.5 for value in logp_threshold_degradation])),
            "attention_selection_avoids_catastrophic_minus_0_5": not (selected["delta_logp_vs_dense"] < -0.5),
        }
        output.append(row)
    return output


def summarize(rows: Sequence[dict[str, Any]], example_rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    def ci(field: str, offset: int = 0) -> dict[str, Any]:
        return bootstrap_ci([float(row[field]) for row in example_rows], samples=bootstrap_samples, seed=seed + offset)

    pooled_handoff = [row for row in rows if row["condition"] == "handoff_mean"]
    pooled_spearman_logp = spearman(
        [row["retained_attention_mass"] for row in pooled_handoff],
        [row["correct_choice_log_probability"] for row in pooled_handoff],
    )
    pooled_spearman_margin = spearman(
        [row["retained_attention_mass"] for row in pooled_handoff],
        [row["answer_margin"] for row in pooled_handoff],
    )
    return {
        "num_examples": len(example_rows),
        "num_pair_condition_rows": len(rows),
        "selection_benefit": {
            "selected_minus_mean_other_five_logp": ci("attention_selected_minus_mean_other_five_logp", 1),
            "selected_minus_mean_other_five_margin": ci("attention_selected_minus_mean_other_five_margin", 2),
            "fraction_best": float(np.mean([bool(row["attention_selected_is_best"]) for row in example_rows])),
            "fraction_top_two": float(np.mean([bool(row["attention_selected_is_top_two"]) for row in example_rows])),
            "fraction_worst": float(np.mean([bool(row["attention_selected_is_worst"]) for row in example_rows])),
        },
        "memory_benefit": {
            "mean_handoff_minus_hard_evict_logp": ci("mean_handoff_minus_hard_evict_logp", 10),
            "mean_handoff_minus_hard_evict_margin": ci("mean_handoff_minus_hard_evict_margin", 11),
        },
        "attention_predictiveness": {
            "per_example_spearman_mass_vs_logp": ci("spearman_attention_mass_vs_logp", 20),
            "per_example_spearman_mass_vs_margin": ci("spearman_attention_mass_vs_margin", 21),
            "pooled_spearman_mass_vs_logp_descriptive_only": pooled_spearman_logp,
            "pooled_spearman_mass_vs_margin_descriptive_only": pooled_spearman_margin,
        },
        "robustness": {
            "pair_logp_range": ci("pair_logp_range", 30),
            "fraction_pairs_logp_degradation_below_minus_0_1": ci("fraction_pairs_logp_degradation_below_minus_0_1", 31),
            "fraction_pairs_logp_degradation_below_minus_0_5": ci("fraction_pairs_logp_degradation_below_minus_0_5", 32),
            "fraction_attention_selection_avoids_catastrophic_minus_0_5": float(np.mean([bool(row["attention_selection_avoids_catastrophic_minus_0_5"]) for row in example_rows])),
        },
        "conclusions": {
            "compact_memory_vs_hard_deletion": "development descriptive; see memory_benefit CIs",
            "attention_ranked_vs_arbitrary_routes": "development descriptive; compare selected_minus_mean_other_five metrics",
            "retained_attention_mass_predictiveness": "development descriptive; see per-example Spearman summaries",
            "extreme_random_route_wins": "inspect route_quality_range and selection ranks; no held-out claim",
        },
    }


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


def write_plots(output_dir: Path, rows: Sequence[dict[str, Any]], example_rows: Sequence[dict[str, Any]]) -> None:
    plt = _plt()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ranks = [row["attention_selected_rank_logp"] for row in example_rows]
    ax.hist(ranks, bins=np.arange(0.5, 7.5, 1), rwidth=0.8)
    ax.set_xlabel("Attention-selected pair rank by log probability")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(output_dir / "attention_selection_rank.png", dpi=180)
    plt.close(fig)

    handoff = [row for row in rows if row["condition"] == "handoff_mean"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([row["retained_attention_mass"] for row in handoff], [row["correct_choice_log_probability"] for row in handoff], alpha=0.7)
    ax.set_xlabel("Retained layer-8 attention mass")
    ax.set_ylabel("Correct-answer log probability")
    fig.tight_layout()
    fig.savefig(output_dir / "attention_mass_vs_quality.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([row["mean_handoff_minus_hard_evict_logp"] for row in example_rows], bins=10)
    ax.set_xlabel("Mean handoff - hard eviction logp")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(output_dir / "memory_vs_eviction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([row["pair_logp_range"] for row in example_rows], bins=10)
    ax.set_xlabel("Best - worst pair logp range")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(output_dir / "route_quality_range.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen Temporal-Handoff Retained-Pair Sweep",
        "",
        "This is a 15-example development analysis. It is not held-out evidence and does not claim end-to-end generation speedup.",
        "",
        "The sweep evaluates all six retained native-cell pairs for `handoff_mean` and `hard_evict`, using the example as the independent unit.",
        "",
        "## Development Conclusions",
        "",
        "1. Compact memory versus hard deletion: see `memory_benefit` in `analysis_summary.json`.",
        "2. Attention-ranked selection versus arbitrary routes: see `selection_benefit`.",
        "3. Retained attention mass predictiveness: see per-example Spearman summaries.",
        "4. Extreme random-route wins: inspect `route_quality_range` and per-example selected-pair ranks.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze_outputs(output_dir: str | Path, manifest: str | Path, *, bootstrap_samples: int, seed: int, write_outputs: bool = True) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_records = manifest_by_id(manifest)
    artifacts = latest_artifacts(output)
    rows = pair_rows_from_artifacts(artifacts, manifest_records)
    example_rows = per_example_summary(rows)
    summary = summarize(rows, example_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    if write_outputs:
        write_csv(output / "per_pair.csv", rows)
        write_csv(output / "per_example_summary.csv", example_rows)
        write_json(output / "analysis_summary.json", summary)
        write_plots(output, rows, example_rows)
        write_report(output / "report.md", summary)
    return summary


def run_pair_sweep(args: argparse.Namespace) -> dict[str, Any]:
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
    if not Path(args.pilot_dir).exists():
        raise FileNotFoundError(f"Completed handoff pilot directory not found: {args.pilot_dir}")
    run_config = {
        "schema_version": "qwen_temporal_handoff_pair_sweep_v1",
        "model_id": args.model_id,
        "resolution_config": args.resolution_config,
        "sampling_mode": "cross_model_8",
        "handoff_layer": args.handoff_layer,
        "memory_tokens_per_region": args.memory_tokens_per_region,
        "seed": args.seed,
        "conditions": list(PAIR_SWEEP_CONDITIONS),
        "retained_pairs": [list(pair) for pair in RETAINED_PAIRS],
        "git_commit": git_commit,
        "completed_handoff_pilot_dir": str(args.pilot_dir),
        "quality_forward_repetitions": 1,
        "timing_scope": "No repeated latency profiling is run in this exhaustive quality sweep.",
    }
    saved_config = prepare_run_config(output, run_config, overwrite=args.overwrite)
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
            raise RuntimeError(f"{qid}: expected four native temporal cells.")
        layer_scores = baseline_scores[args.handoff_layer]
        inputs, rendered, prompt, _video_kwargs = _prepare_inputs(model, example, frame_batches, resolution)
        layout = _build_layout(model, example, inputs, rendered, frame_batches)
        decoder_inputs, position_ids = qwen_multimodal_decoder_inputs(model._model, inputs)
        stock_logits = _stock_logits(model, inputs)

        def execute(config: TemporalHandoffConfig) -> Any:
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

        dense_key = (qid, "dense", "dense_custom")
        dense_logits = None
        dense_scores = None
        dense_path = artifact_path(output, qid, None, "dense_custom")
        if dense_key in completed:
            validate_resumed_artifact(resolve_artifact(completed[dense_key], output), question_id=qid, condition="dense_custom", pair=None, run_config=saved_config)
            dense_scores = read_json(dense_path)["answer_choice_scores"]
        else:
            dense_config = TemporalHandoffConfig(condition="dense_custom", handoff_layer=args.handoff_layer, memory_tokens_per_region=0, random_seed=args.seed)
            result = execute(dense_config)
            dense_logits = result.logits
            scores = score_answer_choice_logits(result.logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()
            dense_scores = scores
            artifact = make_artifact(
                qid=qid,
                condition="dense_custom",
                pair=None,
                result=result,
                scores=scores,
                example=example,
                manifest_record=manifest_record,
                model_id=args.model_id,
                prompt=prompt,
                rendered=rendered,
                frame_batches=frame_batches,
                run_config=saved_config,
                git_commit=git_commit,
                resolution=resolution,
                retained_mass=None,
                baseline_record=baseline_record,
                native=native,
                baseline_scores=baseline_scores,
                original_sequence_length=int(decoder_inputs.shape[1]),
            )
            write_json(dense_path, artifact)
            validate_resumed_artifact(dense_path, question_id=qid, condition="dense_custom", pair=None, run_config=saved_config)
            append_jsonl(records_path, {"question_id": qid, "condition": "dense_custom", "status": "complete", "artifact": str(dense_path)})

        eq_path = equivalence_path(output, qid)
        if not eq_path.exists():
            if dense_logits is None or dense_scores is None:
                raise RuntimeError(f"{qid}: dense equivalence missing but dense_custom was resumed.")
            stock_scores = score_answer_choice_logits(stock_logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()
            equivalence = dense_equivalence_report(
                stock_logits,
                dense_logits,
                stock_scores=stock_scores,
                dense_scores=dense_scores,
                stock_sequence_length=int(inputs["input_ids"].shape[1]),
                dense_sequence_length=int(read_json(dense_path)["metadata"]["final_sequence_length"]),
                legacy_dense_equivalence_atol=None,
            )
            equivalence["question_id"] = qid
            write_json(eq_path, equivalence)
            if equivalence.get("passed") is not True:
                raise RuntimeError(f"{qid}: dense equivalence failed.")
        validate_equivalence(eq_path, qid)

        for pair in RETAINED_PAIRS:
            retained_mass = float(sum(layer_scores[idx] for idx in pair))
            for condition in PAIR_SWEEP_CONDITIONS:
                key = (qid, pair_key(pair), condition)
                out_path = artifact_path(output, qid, pair, condition)
                if key in completed:
                    validate_resumed_artifact(resolve_artifact(completed[key], output), question_id=qid, condition=condition, pair=pair, run_config=saved_config)
                    continue
                config = TemporalHandoffConfig(
                    condition=condition,
                    handoff_layer=args.handoff_layer,
                    retained_temporal_regions=tuple(pair),
                    memory_tokens_per_region=0 if condition == "hard_evict" else args.memory_tokens_per_region,
                    random_seed=args.seed,
                )
                result = execute(config)
                scores = score_answer_choice_logits(result.logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()
                artifact = make_artifact(
                    qid=qid,
                    condition=condition,
                    pair=pair,
                    result=result,
                    scores=scores,
                    example=example,
                    manifest_record=manifest_record,
                    model_id=args.model_id,
                    prompt=prompt,
                    rendered=rendered,
                    frame_batches=frame_batches,
                    run_config=saved_config,
                    git_commit=git_commit,
                    resolution=resolution,
                    retained_mass=retained_mass,
                    baseline_record=baseline_record,
                    native=native,
                    baseline_scores=baseline_scores,
                    original_sequence_length=int(decoder_inputs.shape[1]),
                )
                write_json(out_path, artifact)
                validate_resumed_artifact(out_path, question_id=qid, condition=condition, pair=pair, run_config=saved_config)
                append_jsonl(records_path, {"question_id": qid, "condition": condition, "retained_pair": list(pair), "status": "complete", "artifact": str(out_path)})

    summary = analyze_outputs(output, args.manifest, bootstrap_samples=args.bootstrap_samples, seed=args.analysis_seed, write_outputs=True)
    return summary


def make_artifact(
    *,
    qid: str,
    condition: str,
    pair: Sequence[int] | None,
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
    retained_mass: float | None,
    baseline_record: dict[str, Any],
    native: Any,
    baseline_scores: Sequence[Sequence[float]],
    original_sequence_length: int,
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "condition": condition,
        "retained_pair": list(pair) if pair is not None else None,
        "status": "complete",
        "category": manifest_record.get("category"),
        "question_type": manifest_record.get("question_type"),
        "participant_id": manifest_record.get("participant_id"),
        "source_video_id": manifest_record.get("source_video_id"),
        "model_backend": "qwen",
        "model_checkpoint": model_id,
        "generation_supported": False,
        "question": example.question,
        "prompt": prompt,
        "rendered_prompt": rendered,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "predicted_idx": predicted_idx(scores),
        "correct": predicted_idx(scores) == example.correct_idx,
        "answer_choice_scores": scores,
        "retained_baseline_attention_mass": retained_mass,
        "sampled_frame_indices": [list(frame_batches[0].frame_indices)],
        "sampled_timestamps": [list(frame_batches[0].timestamps)],
        "frame_bin_mappings": [frame_batches[0].metadata.get("frame_bin_mapping", [])],
        "temporal_handoff": {
            "schema_version": SCHEMA_VERSION,
            "handoff_layer": run_config["handoff_layer"],
            "retained_temporal_regions": list(pair) if pair is not None else [],
            "memory_tokens_per_region": 0 if condition in {"dense_custom", "hard_evict"} else run_config["memory_tokens_per_region"],
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
        },
    }


def main() -> None:
    args = parse_args()
    summary = run_pair_sweep(args)
    print(json.dumps({"output_dir": args.output_dir, "summary": summary["conclusions"]}, indent=2))


if __name__ == "__main__":
    main()
