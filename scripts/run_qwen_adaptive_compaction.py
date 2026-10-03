#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.adaptive_compaction import (
    ADAPTIVE_SCHEMA_VERSION,
    COMPACTION_LAYERS,
    FRAME_COUNTS,
    RETENTION_FRACTIONS,
    candidate_routes,
    extract_router_features,
    router_feature_metadata,
    shard_records,
    validate_compact_action_artifact,
)
from src.experiment1.temporal_handoff import (
    TemporalHandoffConfig,
    append_jsonl,
    dense_equivalence_report,
    qwen_decoder_stack,
    qwen_multimodal_decoder_inputs,
    run_compacted_decoder_from_prefix_cache,
    run_custom_decoder_prefill,
    run_dense_decoder_with_prefix_cache,
    temporal_regions_from_layout,
    write_json,
)


CONDITIONS = ("dense_custom", "hard_evict")
IMMUTABLE_CONFIG_FIELDS = (
    "schema_version",
    "model_id",
    "resolution_config",
    "sampling_mode",
    "frame_counts",
    "sampling_policy",
    "compaction_layers",
    "retention_fractions",
    "condition",
    "seed",
    "num_shards",
    "git_commit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive temporal compaction labels with prefix caching.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True, choices=["train", "development"])
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_adaptive_compaction/train_labels")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resolution-config", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--frame-counts", type=int, nargs="+", default=list(FRAME_COUNTS))
    parser.add_argument("--compaction-layers", type=int, nargs="+", default=list(COMPACTION_LAYERS))
    parser.add_argument("--retention-fractions", type=float, nargs="+", default=list(RETENTION_FRACTIONS))
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def action_artifact_path(output_dir: Path, question_id: str, frame_count: int, action_id: str) -> Path:
    return output_dir / "artifacts" / question_id / f"frames_{frame_count}" / f"{action_id}.json"


def dense_artifact_path(output_dir: Path, question_id: str, frame_count: int) -> Path:
    return output_dir / "artifacts" / question_id / f"frames_{frame_count}" / "dense_custom.json"


def feature_path(output_dir: Path, question_id: str, frame_count: int, layer: int) -> Path:
    return output_dir / "features" / question_id / f"frames_{frame_count}" / f"layer_{layer}.pt"


def records_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / f"records_shard_{shard_index}_of_{num_shards}.jsonl"


def summary_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / f"summary_shard_{shard_index}_of_{num_shards}.json"


def shard_metadata_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / f"shard_config_{shard_index}_of_{num_shards}.json"


def immutable_subset(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in IMMUTABLE_CONFIG_FIELDS}


def config_mismatches(saved: dict[str, Any], requested: dict[str, Any]) -> dict[str, dict[str, Any]]:
    saved_subset = immutable_subset(saved)
    requested_subset = immutable_subset(requested)
    return {
        field: {"saved": saved_subset.get(field), "requested": requested_subset.get(field)}
        for field in IMMUTABLE_CONFIG_FIELDS
        if saved_subset.get(field) != requested_subset.get(field)
    }


def clean_outputs(output_dir: Path, shard_index: int, num_shards: int) -> None:
    for path in (
        records_path(output_dir, shard_index, num_shards),
        summary_path(output_dir, shard_index, num_shards),
        shard_metadata_path(output_dir, shard_index, num_shards),
    ):
        if path.exists():
            path.unlink()


def outputs_present(output_dir: Path, shard_index: int, num_shards: int) -> bool:
    return (
        (output_dir / "artifacts").exists()
        or (output_dir / "features").exists()
        or records_path(output_dir, shard_index, num_shards).exists()
        or summary_path(output_dir, shard_index, num_shards).exists()
        or shard_metadata_path(output_dir, shard_index, num_shards).exists()
    )


def write_json_if_absent_or_validate(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return json.loads(path.read_text())
    with os.fdopen(fd, "w") as handle:
        handle.write(encoded)
    return payload


def prepare_run_config(output_dir: Path, requested: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    shard_index = int(requested["shard_index"])
    num_shards = int(requested["num_shards"])
    global_requested = dict(requested)
    global_requested.pop("shard_index", None)
    shard_payload = {
        "schema_version": "adaptive_compaction_shard_config_v1",
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    if overwrite:
        if (output_dir / "artifacts").exists() or (output_dir / "features").exists():
            raise RuntimeError(
                "--overwrite refuses to clean adaptive-compaction artifacts/features in place. "
                "Use a new output directory to avoid mixing stale action or feature artifacts."
            )
        clean_outputs(output_dir, shard_index, num_shards)
        write_json(path, global_requested)
        write_json(shard_metadata_path(output_dir, shard_index, num_shards), shard_payload)
        return global_requested
    if outputs_present(output_dir, requested["shard_index"], requested["num_shards"]) and not path.exists():
        raise RuntimeError(
            "Existing adaptive-compaction outputs are present but run_config.json is missing. "
            "Use a new output directory or explicitly use --overwrite."
        )
    if path.exists():
        saved = json.loads(path.read_text())
        mismatches = config_mismatches(saved, global_requested)
        if mismatches:
            raise RuntimeError(f"Refusing to resume adaptive compaction run; immutable configuration differs: {mismatches}")
        write_json(shard_metadata_path(output_dir, shard_index, num_shards), shard_payload)
        return saved
    saved = write_json_if_absent_or_validate(path, global_requested)
    mismatches = config_mismatches(saved, global_requested)
    if mismatches:
        raise RuntimeError(f"Refusing to resume adaptive compaction run; immutable configuration differs: {mismatches}")
    write_json(shard_metadata_path(output_dir, shard_index, num_shards), shard_payload)
    return saved


def current_completed_records(output_dir: Path, shard_index: int, num_shards: int) -> set[tuple[str, int, str]]:
    path = records_path(output_dir, shard_index, num_shards)
    completed: set[tuple[str, int, str]] = set()
    if not path.exists():
        return completed
    for record in read_jsonl(path):
        if record.get("status") == "complete":
            completed.add((str(record["question_id"]), int(record["frame_count"]), str(record["action_id"])))
    return completed


def validate_resumed_action_artifact(path: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Completed action artifact is missing or empty: {path}")
    artifact = json.loads(path.read_text())
    validate_compact_action_artifact(artifact)
    mismatches = config_mismatches(artifact["run_config"], run_config)
    if mismatches:
        raise RuntimeError(f"Action artifact run_config differs from current run: {mismatches}")
    return artifact


def validate_router_feature_file(
    path: Path,
    *,
    expected_layer: int,
    expected_cell_count: int,
    expected_frame_count: int,
    expected_dtype: str | None = None,
) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Router feature file is missing or empty: {path}")
    import torch

    payload = torch.load(path, map_location="cpu")
    if int(payload.get("layer")) != int(expected_layer):
        raise RuntimeError(f"Router feature layer mismatch in {path}.")
    if int(payload.get("frame_count")) != int(expected_frame_count):
        raise RuntimeError(f"Router feature frame_count mismatch in {path}.")
    cell_ids = payload.get("native_temporal_cell_ids") or []
    if len(cell_ids) != expected_cell_count:
        raise RuntimeError(f"Router feature native-cell count mismatch in {path}.")
    cell_mean = payload.get("cell_mean_residual")
    question_mean = payload.get("question_mean_residual")
    if cell_mean is None or question_mean is None:
        raise RuntimeError(f"Router feature tensors are missing in {path}.")
    if list(cell_mean.shape)[0] != expected_cell_count:
        raise RuntimeError(f"Router cell feature shape mismatch in {path}: {tuple(cell_mean.shape)}")
    if question_mean.ndim != 1 or int(question_mean.shape[0]) != int(cell_mean.shape[1]):
        raise RuntimeError(f"Router question feature shape mismatch in {path}: {tuple(question_mean.shape)}")
    loaded_dtype = str(cell_mean.dtype)
    if expected_dtype is not None and loaded_dtype != expected_dtype:
        raise RuntimeError(f"Router feature dtype mismatch in {path}: saved={expected_dtype}, loaded={loaded_dtype}")
    if not torch.isfinite(cell_mean).all() or not torch.isfinite(question_mean).all():
        raise RuntimeError(f"Router features contain non-finite values in {path}.")
    return payload


def validate_resumed_dense_artifact(
    path: Path,
    *,
    run_config: dict[str, Any],
    output_dir: Path,
    expected_feature_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Completed dense artifact is missing or empty: {path}")
    artifact = json.loads(path.read_text())
    required = {
        "question_id",
        "frame_count",
        "condition",
        "native_temporal_cell_count",
        "correct_choice_log_probability",
        "answer_margin",
        "router_features",
        "status",
        "git_commit",
        "run_config",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise RuntimeError(f"Dense artifact is missing required fields: {missing}")
    if artifact["condition"] != "dense_custom" or artifact["status"] != "complete":
        raise RuntimeError("Dense artifact is not a complete dense_custom artifact.")
    mismatches = config_mismatches(artifact["run_config"], run_config)
    if mismatches:
        raise RuntimeError(f"Dense artifact run_config differs from current run: {mismatches}")
    eq_path = path.parent / "dense_equivalence_report.json"
    if not eq_path.exists() or eq_path.stat().st_size <= 0:
        raise RuntimeError(f"Dense equivalence report is missing or empty: {eq_path}")
    eq = json.loads(eq_path.read_text())
    if eq.get("passed") is not True:
        raise RuntimeError(f"Dense equivalence report did not pass: {eq_path}")
    expected_cells = int(artifact["native_temporal_cell_count"])
    expected_layers = tuple(int(layer) for layer in (expected_feature_layers or run_config.get("compaction_layers") or ()))
    features = artifact.get("router_features") or []
    feature_layers = [int(item.get("layer")) for item in features]
    duplicate_layers = sorted({layer for layer in feature_layers if feature_layers.count(layer) > 1})
    if duplicate_layers:
        raise RuntimeError(f"Dense artifact has duplicate router feature layers: {duplicate_layers}")
    if expected_layers and sorted(feature_layers) != sorted(expected_layers):
        raise RuntimeError(
            "Dense artifact router feature layers do not match configured compaction layers: "
            f"expected={sorted(expected_layers)}, actual={sorted(feature_layers)}."
        )
    for feature_meta in features:
        layer = int(feature_meta["layer"])
        if int(feature_meta.get("frame_count")) != int(artifact["frame_count"]):
            raise RuntimeError(f"Router feature metadata frame_count mismatch for layer {layer}.")
        feature_file = Path(feature_meta["feature_file"])
        if not feature_file.exists() and not feature_file.is_absolute():
            feature_file = output_dir / feature_file
        expected_name = f"layer_{layer}.pt"
        if feature_file.name != expected_name:
            raise RuntimeError(f"Router feature filename mismatch: expected {expected_name}, got {feature_file.name}.")
        payload = validate_router_feature_file(
            feature_file,
            expected_layer=layer,
            expected_cell_count=expected_cells,
            expected_frame_count=int(artifact["frame_count"]),
            expected_dtype=str(feature_meta.get("dtype")) if feature_meta.get("dtype") is not None else None,
        )
        if list(payload["cell_mean_residual"].shape) != list(feature_meta.get("cell_mean_residual_shape", [])):
            raise RuntimeError(f"Router feature cell shape does not match metadata for {feature_file}.")
        if list(payload["question_mean_residual"].shape) != list(feature_meta.get("question_mean_residual_shape", [])):
            raise RuntimeError(f"Router feature question shape does not match metadata for {feature_file}.")
    return artifact


def make_run_config(args: argparse.Namespace, git_commit: str) -> dict[str, Any]:
    return {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "model_id": args.model_id,
        "resolution_config": args.resolution_config,
        "sampling_mode": "adaptive_fixed_count",
        "sampling_policy": "adaptive_fixed_center_frames_exact_count",
        "frame_counts": list(args.frame_counts),
        "compaction_layers": list(args.compaction_layers),
        "retention_fractions": list(args.retention_fractions),
        "condition": "physical_hard_deletion",
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "git_commit": git_commit,
        "quality_forward_repetitions": 1,
        "test_split_access_policy": "test split must not be passed before router and thresholds are frozen",
    }


def _score_from_logits(logits: Any, tokenizer: Any, correct_idx: int, num_choices: int) -> dict[str, Any]:
    from src.experiment1.answer_scoring import score_answer_choice_logits

    return score_answer_choice_logits(logits[0, -1], tokenizer, correct_idx, num_choices).to_json_dict()


def _pred(scores: dict[str, Any]) -> int:
    return int(max(range(len(scores["choice_logits"])), key=lambda idx: float(scores["choice_logits"][idx])))


def _logit_max_abs(a: Any, b: Any) -> float:
    import torch

    return float(torch.max(torch.abs(a.detach().float() - b.detach().float())).item())


def validate_manifest_leakage_summary(manifest_path: str | Path) -> dict[str, Any]:
    summary_path = Path(manifest_path).with_name("summary.json")
    if not summary_path.exists():
        return {"passed": False, "reason": f"summary.json not found next to {manifest_path}"}
    payload = json.loads(summary_path.read_text())
    source = payload.get("source_video_overlap_counts") or {}
    questions = payload.get("question_id_overlap_counts") or {}
    passed = bool(source) and all(int(value) == 0 for value in source.values()) and all(int(value) == 0 for value in questions.values())
    return {
        "passed": passed,
        "source_video_overlap_counts": source,
        "question_id_overlap_counts": questions,
    }


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_completed_smoke_resume(
    *,
    output_dir: Path,
    run_config: dict[str, Any],
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    records_file = records_path(output_dir, shard_index, num_shards)
    if not records_file.exists():
        return {"passed": False, "reason": "records file missing"}
    records = [record for record in read_jsonl(records_file) if record.get("status") == "complete"]
    artifact_paths = [Path(record["artifact"]) for record in records]
    before = {str(path): {"sha256": _file_sha256(path), "mtime_ns": path.stat().st_mtime_ns} for path in artifact_paths}
    dense_count = 0
    action_count = 0
    for record in records:
        path = Path(record["artifact"])
        if record.get("action_id") == "dense_custom":
            validate_resumed_dense_artifact(path, run_config=run_config, output_dir=output_dir)
            dense_count += 1
        else:
            validate_resumed_action_artifact(path, run_config)
            action_count += 1
    after = {str(path): {"sha256": _file_sha256(path), "mtime_ns": path.stat().st_mtime_ns} for path in artifact_paths}
    mismatch = config_mismatches(run_config, {**run_config, "frame_counts": [999]})
    unchanged = before == after
    return {
        "passed": bool(records) and dense_count > 0 and action_count > 0 and unchanged and "frame_counts" in mismatch,
        "validated_without_model_execution": True,
        "stock_inference_skipped": True,
        "dense_artifacts_validated": dense_count,
        "action_artifacts_validated": action_count,
        "artifact_hashes_unchanged": unchanged,
        "immutable_configuration_drift_rejected": "frame_counts" in mismatch,
    }


def active_lengths(result: Any) -> list[dict[str, int]]:
    return [
        {
            "layer": int(item.layer),
            "sequence_length_in": int(item.sequence_length_in),
            "sequence_length_out": int(item.sequence_length_out),
        }
        for item in result.instrumentation
    ]


def expected_routes_for_frame(
    *,
    question_id: str,
    frame_count: int,
    native_temporal_cell_count: int,
    compaction_layers: Sequence[int],
    retention_fractions: Sequence[float],
    seed: int,
) -> list[Any]:
    routes = []
    for layer in compaction_layers:
        for fraction in retention_fractions:
            routes.extend(
                candidate_routes(
                    question_id=question_id,
                    frame_count=frame_count,
                    compaction_layer=layer,
                    native_temporal_cell_count=native_temporal_cell_count,
                    retention_fraction=fraction,
                    seed=seed,
                )
            )
    return routes


def make_dense_artifact(
    *,
    qid: str,
    split: str,
    frame_count: int,
    result: Any,
    scores: dict[str, Any],
    example: Any,
    manifest_record: dict[str, Any],
    run_config: dict[str, Any],
    git_commit: str,
    native_cell_count: int,
    feature_metadata: list[dict[str, Any]],
    sampling_metadata: dict[str, Any],
) -> dict[str, Any]:
    pred = int(max(range(len(scores["choice_logits"])), key=lambda idx: float(scores["choice_logits"][idx])))
    return {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "question_id": qid,
        "source_video_id": manifest_record.get("source_video_id"),
        "split": split,
        "frame_count": frame_count,
        "condition": "dense_custom",
        "native_temporal_cell_count": native_cell_count,
        "sampling_mode": "adaptive_fixed_count",
        "sampling_policy": "adaptive_fixed_center_frames_exact_count",
        "sampling_metadata": sampling_metadata,
        "sampled_frame_indices": sampling_metadata.get("sampled_frame_indices"),
        "sampled_timestamps": sampling_metadata.get("sampled_timestamps"),
        "correct_choice_log_probability": float(scores["correct_choice_log_probability"]),
        "answer_margin": float(scores.get("correct_vs_best_incorrect_margin", scores.get("correct_vs_strongest_incorrect_margin"))),
        "predicted_answer": example.choices[pred],
        "predicted_idx": pred,
        "correct": pred == example.correct_idx,
        "original_sequence_length": int(result.instrumentation[0].sequence_length_in),
        "compacted_sequence_length": int(result.final_hidden_states.shape[1]),
        "active_sequence_length_by_layer": active_lengths(result),
        "estimated_attention_flops": int(result.total_estimated_attention_flops),
        "memory_token_count": 0,
        "router_features": feature_metadata,
        "status": "complete",
        "git_commit": git_commit,
        "run_config": run_config,
    }


def make_action_artifact(
    *,
    dense: dict[str, Any],
    route: Any,
    result: Any,
    scores: dict[str, Any],
    example: Any,
    manifest_record: dict[str, Any],
    split: str,
    run_config: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    pred = int(max(range(len(scores["choice_logits"])), key=lambda idx: float(scores["choice_logits"][idx])))
    logp = float(scores["correct_choice_log_probability"])
    margin = float(scores.get("correct_vs_best_incorrect_margin", scores.get("correct_vs_strongest_incorrect_margin")))
    dense_logp = float(dense["correct_choice_log_probability"])
    dense_margin = float(dense["answer_margin"])
    dense_flops = float(dense["estimated_attention_flops"])
    return {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "question_id": route.question_id,
        "source_video_id": manifest_record.get("source_video_id"),
        "split": split,
        "frame_count": route.frame_count,
        "compaction_layer": route.compaction_layer,
        "native_temporal_cell_count": route.native_temporal_cell_count,
        "retention_fraction": route.retention_fraction,
        "retained_cell_ids": list(route.retained_cells),
        "route_family": route.route_family,
        "action_id": route.action_id,
        "condition": "hard_evict",
        "correct_choice_log_probability": logp,
        "delta_correct_choice_log_probability_from_dense": logp - dense_logp,
        "answer_margin": margin,
        "delta_answer_margin_from_dense": margin - dense_margin,
        "predicted_answer": example.choices[pred],
        "predicted_idx": pred,
        "correct": pred == example.correct_idx,
        "prediction_changed": pred != int(dense["predicted_idx"]),
        "original_sequence_length": int(dense["original_sequence_length"]),
        "compacted_sequence_length": int(result.final_hidden_states.shape[1]),
        "active_sequence_length_by_layer": active_lengths(result),
        "estimated_attention_flops": int(result.total_estimated_attention_flops),
        "paired_flop_reduction_from_dense": (dense_flops - float(result.total_estimated_attention_flops)) / dense_flops if dense_flops else 0.0,
        "memory_token_count": len(result.final_memory_token_indices),
        "status": "complete",
        "git_commit": git_commit,
        "run_config": run_config,
    }


def run_adaptive_compaction(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.run_experiment1 import current_git_commit, frame_batches_for_example, load_examples_by_id, load_manifest
    from scripts.run_qwen_temporal_handoff_smoke import _build_layout, _prepare_inputs, _stock_logits
    from src.experiment1.answer_scoring import score_answer_choice_logits
    from src.experiment1.resolution import get_resolution_config
    from src.experiment1.temporal_handoff import dense_equivalence_report
    from src.models.qwen import Qwen25VLWrapper, QwenConfig

    if args.split == "test":
        raise RuntimeError("The test split must not be used before router and thresholds are frozen.")
    output = Path(args.output_dir)
    manifest_records = load_manifest(args.manifest, args.limit)
    if args.question_id:
        wanted = set(args.question_id)
        manifest_records = [record for record in manifest_records if record["question_id"] in wanted]
    if args.smoke:
        manifest_records = manifest_records[:1]
        args.frame_counts = [8]
    sharded = shard_records(manifest_records, num_shards=args.num_shards, shard_index=args.shard_index)
    examples = load_examples_by_id(args.questions_dir, sharded)
    git_commit = current_git_commit()
    requested = make_run_config(args, git_commit)
    run_config = prepare_run_config(output, requested, overwrite=args.overwrite)
    completed = current_completed_records(output, args.shard_index, args.num_shards) if not args.overwrite else set()
    records = records_path(output, args.shard_index, args.num_shards)
    resolution = get_resolution_config(args.resolution_config)
    model = Qwen25VLWrapper(QwenConfig(model_id=args.model_id, max_new_tokens=1, attn_implementation="sdpa"))
    model._load()
    assert model._model is not None and model._processor is not None
    stack = qwen_decoder_stack(model._model)
    processed_actions = 0
    failures = 0
    smoke_validation: dict[str, Any] = {
        "schema_version": "adaptive_compaction_smoke_validation_v1",
        "required": bool(args.smoke),
        "checks": {},
        "cached_full_comparisons": [],
    }
    if args.smoke:
        smoke_validation["checks"]["manifest_zero_overlap"] = validate_manifest_leakage_summary(args.manifest)

    for record in sharded:
        qid = str(record["question_id"])
        example = examples[qid]
        for frame_count in args.frame_counts:
            dense_path = dense_artifact_path(output, qid, frame_count)
            dense_key = (qid, frame_count, "dense_custom")
            dense_already_complete = dense_key in completed
            if dense_path.exists() and not dense_already_complete and not args.overwrite:
                raise RuntimeError(
                    f"Refusing to overwrite existing dense artifact without --overwrite: {dense_path}"
                )
            existing_dense_artifact = None
            if dense_already_complete and not args.overwrite:
                existing_dense_artifact = validate_resumed_dense_artifact(
                    dense_path,
                    run_config=run_config,
                    output_dir=output,
                )
                expected_routes = expected_routes_for_frame(
                    question_id=qid,
                    frame_count=frame_count,
                    native_temporal_cell_count=int(existing_dense_artifact["native_temporal_cell_count"]),
                    compaction_layers=args.compaction_layers,
                    retention_fractions=args.retention_fractions,
                    seed=args.seed,
                )
                all_actions_complete = True
                for route in expected_routes:
                    if (qid, frame_count, route.action_id) not in completed:
                        all_actions_complete = False
                        break
                    validate_resumed_action_artifact(
                        action_artifact_path(output, qid, frame_count, route.action_id),
                        run_config,
                    )
                if all_actions_complete:
                    continue
            dense_artifact = None
            try:
                record_for_sampling = dict(record)
                record_for_sampling["_adaptive_fixed_frame_count"] = frame_count
                frame_batches = frame_batches_for_example(
                    example,
                    args.mp4_dir,
                    num_frames=frame_count,
                    sampling_mode="adaptive_fixed_count",
                    manifest_record=record_for_sampling,
                    frames_per_bin_override=frame_count,
                )
                sampled = tuple(int(index) for index in frame_batches[0].frame_indices)
                if len(sampled) != frame_count or len(set(sampled)) != frame_count:
                    raise RuntimeError(
                        f"{qid}/frames={frame_count}: adaptive_fixed_count produced {len(sampled)} frames "
                        f"and {len(set(sampled))} distinct frames: {sampled}"
                    )
                if args.smoke:
                    smoke_validation["checks"]["requested_sampled_frame_count"] = {
                        "passed": len(sampled) == frame_count,
                        "requested": frame_count,
                        "actual": len(sampled),
                        "distinct": len(set(sampled)),
                    }
                sampling_metadata = dict(frame_batches[0].metadata.get("sampling") or {})
                sampling_metadata["sampled_frame_indices"] = list(sampled)
                sampling_metadata["sampled_timestamps"] = list(frame_batches[0].timestamps)
                inputs, rendered, prompt, _video_kwargs = _prepare_inputs(model, example, frame_batches, resolution)
                layout = _build_layout(model, example, inputs, rendered, frame_batches)
                native_count = len(temporal_regions_from_layout(layout))
                if args.smoke:
                    native_ids = sorted(temporal_regions_from_layout(layout))
                    smoke_validation["checks"]["contiguous_native_temporal_cell_ids"] = {
                        "passed": native_ids == list(range(len(native_ids))),
                        "native_temporal_cell_ids": native_ids,
                        "native_temporal_cell_count": native_count,
                    }
                decoder_inputs, position_ids = qwen_multimodal_decoder_inputs(model._model, inputs)
                dense_result, caches = run_dense_decoder_with_prefix_cache(
                    layers=stack["layers"],
                    hidden_states=decoder_inputs.clone(),
                    position_ids=position_ids.clone() if position_ids is not None else None,
                    layout=layout,
                    cache_boundaries=args.compaction_layers,
                    lm_head=stack["lm_head"],
                    norm=stack["norm"],
                    num_attention_heads=stack["num_attention_heads"],
                    head_dim=stack["head_dim"],
                    rotary_emb=stack["rotary_emb"],
                    layer_types=stack["layer_types"],
                    sliding_window=stack["sliding_window"],
                )
                dense_scores = score_answer_choice_logits(dense_result.logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()
                eq_path = dense_path.parent / "dense_equivalence_report.json"
                if not dense_already_complete:
                    stock_logits = _stock_logits(model, inputs)
                    stock_scores = score_answer_choice_logits(stock_logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()
                    eq = dense_equivalence_report(
                        stock_logits,
                        dense_result.logits,
                        stock_scores=stock_scores,
                        dense_scores=dense_scores,
                        stock_sequence_length=int(inputs["input_ids"].shape[1]),
                        dense_sequence_length=int(dense_result.final_hidden_states.shape[1]),
                        legacy_dense_equivalence_atol=None,
                    )
                    eq["question_id"] = qid
                    write_json(eq_path, eq)
                    if eq.get("passed") is not True:
                        raise RuntimeError(f"{qid}/frames={frame_count}: dense equivalence failed.")
                    if args.smoke:
                        smoke_validation["checks"]["stock_vs_custom_dense_equivalence"] = {
                            "passed": bool(eq.get("passed")),
                            "checks": eq.get("checks"),
                        }
                elif not eq_path.exists():
                    raise RuntimeError(f"{qid}/frames={frame_count}: dense equivalence report missing during resume.")
                feature_metadata = []
                if dense_already_complete:
                    dense_artifact = existing_dense_artifact
                    feature_metadata = list(dense_artifact.get("router_features") or [])
                else:
                    for layer, cache in sorted(caches.items()):
                        features = extract_router_features(cache.hidden_states, layout, layer=layer, frame_count=frame_count)
                        fpath = feature_path(output, qid, frame_count, layer)
                        fpath.parent.mkdir(parents=True, exist_ok=True)
                        import torch

                        torch.save(features, fpath)
                        feature_metadata.append(router_feature_metadata(fpath, features))
                        if args.smoke:
                            validate_router_feature_file(
                                fpath,
                                expected_layer=layer,
                                expected_cell_count=native_count,
                                expected_frame_count=frame_count,
                                expected_dtype=str(features["cell_mean_residual"].dtype),
                            )
                    dense_artifact = make_dense_artifact(
                        qid=qid,
                        split=args.split,
                        frame_count=frame_count,
                        result=dense_result,
                        scores=dense_scores,
                        example=example,
                        manifest_record=record,
                        run_config=run_config,
                        git_commit=git_commit,
                        native_cell_count=native_count,
                        feature_metadata=feature_metadata,
                        sampling_metadata=sampling_metadata,
                    )
                if dense_already_complete and not args.overwrite:
                    validate_resumed_dense_artifact(dense_path, run_config=run_config, output_dir=output)
                else:
                    write_json(dense_path, dense_artifact)
                    append_jsonl(records, {"question_id": qid, "frame_count": frame_count, "action_id": "dense_custom", "status": "complete", "artifact": str(dense_path)})
                for layer in args.compaction_layers:
                    smoke_compared_layer = False
                    for fraction in args.retention_fractions:
                        routes = candidate_routes(
                            question_id=qid,
                            frame_count=frame_count,
                            compaction_layer=layer,
                            native_temporal_cell_count=native_count,
                            retention_fraction=fraction,
                            seed=args.seed,
                        )
                        for route in routes:
                            out = action_artifact_path(output, qid, frame_count, route.action_id)
                            if (qid, frame_count, route.action_id) in completed:
                                validate_resumed_action_artifact(out, run_config)
                                continue
                            if out.exists() and not args.overwrite:
                                raise RuntimeError(
                                    f"Refusing to overwrite existing action artifact without --overwrite: {out}"
                                )
                            config = TemporalHandoffConfig(
                                condition="hard_evict",
                                handoff_layer=layer,
                                retained_temporal_regions=route.retained_cells,
                                memory_tokens_per_region=0,
                                random_seed=args.seed,
                            )
                            result = run_compacted_decoder_from_prefix_cache(
                                cache_entry=caches[layer],
                                layers=stack["layers"],
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
                            scores = score_answer_choice_logits(result.logits[0, -1], model._processor.tokenizer, example.correct_idx, len(example.choices)).to_json_dict()
                            if args.smoke:
                                logp_value = float(scores["correct_choice_log_probability"])
                                margin_value = float(
                                    scores.get(
                                        "correct_vs_best_incorrect_margin",
                                        scores.get("correct_vs_strongest_incorrect_margin"),
                                    )
                                )
                                smoke_validation["checks"].setdefault("finite_answer_metrics", {"passed": True, "examples": []})
                                finite_metrics = math.isfinite(logp_value) and math.isfinite(margin_value)
                                smoke_validation["checks"]["finite_answer_metrics"]["examples"].append(
                                    {
                                        "action_id": route.action_id,
                                        "passed": finite_metrics,
                                        "correct_choice_log_probability": logp_value,
                                        "answer_margin": margin_value,
                                    }
                                )
                                smoke_validation["checks"]["finite_answer_metrics"]["passed"] = (
                                    smoke_validation["checks"]["finite_answer_metrics"]["passed"] and finite_metrics
                                )
                            if args.smoke and not smoke_compared_layer:
                                full_result = run_custom_decoder_prefill(
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
                                full_scores = _score_from_logits(
                                    full_result.logits,
                                    model._processor.tokenizer,
                                    example.correct_idx,
                                    len(example.choices),
                                )
                                comparison = {
                                    "layer": layer,
                                    "action_id": route.action_id,
                                    "retained_cells": list(route.retained_cells),
                                    "max_logit_abs_diff": _logit_max_abs(full_result.logits, result.logits),
                                    "correct_logp_diff": float(scores["correct_choice_log_probability"]) - float(full_scores["correct_choice_log_probability"]),
                                    "answer_margin_diff": float(scores.get("correct_vs_best_incorrect_margin", scores.get("correct_vs_strongest_incorrect_margin"))) - float(full_scores.get("correct_vs_best_incorrect_margin", full_scores.get("correct_vs_strongest_incorrect_margin"))),
                                    "predicted_answer_matches": _pred(scores) == _pred(full_scores),
                                    "final_sequence_length_matches": int(result.final_hidden_states.shape[1]) == int(full_result.final_hidden_states.shape[1]),
                                    "flops_match": int(result.total_estimated_attention_flops) == int(full_result.total_estimated_attention_flops),
                                }
                                comparison["passed"] = (
                                    comparison["max_logit_abs_diff"] <= 0.25
                                    and abs(comparison["correct_logp_diff"]) <= 0.10
                                    and abs(comparison["answer_margin_diff"]) <= 0.125
                                    and comparison["predicted_answer_matches"]
                                    and comparison["final_sequence_length_matches"]
                                    and comparison["flops_match"]
                                )
                                smoke_validation["cached_full_comparisons"].append(comparison)
                                smoke_compared_layer = True
                            artifact = make_action_artifact(
                                dense=dense_artifact,
                                route=route,
                                result=result,
                                scores=scores,
                                example=example,
                                manifest_record=record,
                                split=args.split,
                                run_config=run_config,
                                git_commit=git_commit,
                            )
                            validate_compact_action_artifact(artifact)
                            if args.smoke:
                                shortened = int(artifact["compacted_sequence_length"]) < int(artifact["original_sequence_length"])
                                smoke_validation["checks"].setdefault(
                                    "physical_sequence_shortening",
                                    {"passed": True, "examples": []},
                                )
                                smoke_validation["checks"]["physical_sequence_shortening"]["examples"].append(
                                    {
                                        "action_id": route.action_id,
                                        "passed": shortened,
                                        "original": artifact["original_sequence_length"],
                                        "compacted": artifact["compacted_sequence_length"],
                                    }
                                )
                                smoke_validation["checks"]["physical_sequence_shortening"]["passed"] = (
                                    smoke_validation["checks"]["physical_sequence_shortening"]["passed"] and shortened
                                )
                            write_json(out, artifact)
                            append_jsonl(records, {"question_id": qid, "frame_count": frame_count, "action_id": route.action_id, "status": "complete", "artifact": str(out)})
                            if args.smoke:
                                validate_resumed_action_artifact(out, run_config)
                                mismatch = config_mismatches(
                                    run_config,
                                    {**run_config, "frame_counts": [999]},
                                )
                                smoke_validation["checks"]["immutable_config_mismatch_rejection"] = {
                                    "passed": "frame_counts" in mismatch,
                                    "mismatch_fields": sorted(mismatch),
                                }
                            processed_actions += 1
            except Exception as exc:
                failures += 1
                append_jsonl(records, {"question_id": qid, "frame_count": frame_count, "action_id": "example_setup", "status": "failed", "error": str(exc)})
                if args.smoke:
                    raise
    summary = {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "split": args.split,
        "num_manifest_records": len(manifest_records),
        "num_shard_records": len(sharded),
        "processed_actions": processed_actions,
        "failures": failures,
        "run_config": run_config,
    }
    write_json(summary_path(output, args.shard_index, args.num_shards), summary)
    if args.smoke:
        smoke_validation["checks"]["cached_prefix_vs_full_prefix"] = {
            "passed": bool(smoke_validation["cached_full_comparisons"])
            and all(item.get("passed") for item in smoke_validation["cached_full_comparisons"])
            and {item["layer"] for item in smoke_validation["cached_full_comparisons"]} == set(args.compaction_layers),
            "num_comparisons": len(smoke_validation["cached_full_comparisons"]),
        }
        smoke_validation["checks"]["resume_validation"] = validate_completed_smoke_resume(
            output_dir=output,
            run_config=run_config,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        smoke_validation["passed"] = all(
            bool(value.get("passed")) for value in smoke_validation["checks"].values() if isinstance(value, dict)
        )
        smoke_path = output / "smoke_validation.json"
        write_json(smoke_path, smoke_validation)
        if not smoke_validation["passed"]:
            raise RuntimeError(f"Adaptive compaction smoke validation failed; see {smoke_path}")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_adaptive_compaction(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
