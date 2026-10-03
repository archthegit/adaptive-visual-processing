#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    "compaction_layers",
    "retention_fractions",
    "condition",
    "seed",
    "num_shards",
    "shard_index",
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
    for path in (records_path(output_dir, shard_index, num_shards), summary_path(output_dir, shard_index, num_shards)):
        if path.exists():
            path.unlink()


def outputs_present(output_dir: Path, shard_index: int, num_shards: int) -> bool:
    return (
        (output_dir / "artifacts").exists()
        or (output_dir / "features").exists()
        or records_path(output_dir, shard_index, num_shards).exists()
        or summary_path(output_dir, shard_index, num_shards).exists()
    )


def prepare_run_config(output_dir: Path, requested: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    if overwrite:
        clean_outputs(output_dir, requested["shard_index"], requested["num_shards"])
        write_json(path, requested)
        return requested
    if outputs_present(output_dir, requested["shard_index"], requested["num_shards"]) and not path.exists():
        raise RuntimeError(
            "Existing adaptive-compaction outputs are present but run_config.json is missing. "
            "Use a new output directory or explicitly use --overwrite."
        )
    if path.exists():
        saved = json.loads(path.read_text())
        mismatches = config_mismatches(saved, requested)
        if mismatches:
            raise RuntimeError(f"Refusing to resume adaptive compaction run; immutable configuration differs: {mismatches}")
        return saved
    write_json(path, requested)
    return requested


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


def make_run_config(args: argparse.Namespace, git_commit: str) -> dict[str, Any]:
    return {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "model_id": args.model_id,
        "resolution_config": args.resolution_config,
        "sampling_mode": "cross_model_8",
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


def active_lengths(result: Any) -> list[dict[str, int]]:
    return [
        {
            "layer": int(item.layer),
            "sequence_length_in": int(item.sequence_length_in),
            "sequence_length_out": int(item.sequence_length_out),
        }
        for item in result.instrumentation
    ]


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
            dense_artifact = None
            try:
                frame_batches = frame_batches_for_example(
                    example,
                    args.mp4_dir,
                    num_frames=frame_count,
                    sampling_mode="cross_model_8",
                    manifest_record=record,
                    frames_per_bin_override=1,
                )
                inputs, rendered, prompt, _video_kwargs = _prepare_inputs(model, example, frame_batches, resolution)
                layout = _build_layout(model, example, inputs, rendered, frame_batches)
                native_count = len(temporal_regions_from_layout(layout))
                decoder_inputs, position_ids = qwen_multimodal_decoder_inputs(model._model, inputs)
                stock_logits = _stock_logits(model, inputs)
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
                eq_path = dense_path.parent / "dense_equivalence_report.json"
                write_json(eq_path, eq)
                if eq.get("passed") is not True:
                    raise RuntimeError(f"{qid}/frames={frame_count}: dense equivalence failed.")
                feature_metadata = []
                for layer, cache in sorted(caches.items()):
                    features = extract_router_features(cache.hidden_states, layout, layer=layer, frame_count=frame_count)
                    fpath = feature_path(output, qid, frame_count, layer)
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    import torch

                    torch.save(features, fpath)
                    feature_metadata.append(router_feature_metadata(fpath, features))
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
                )
                if dense_already_complete and not args.overwrite:
                    existing_dense = json.loads(dense_path.read_text())
                    mismatches = config_mismatches(existing_dense["run_config"], run_config)
                    if mismatches:
                        raise RuntimeError(f"Existing dense artifact run_config differs from current run: {mismatches}")
                else:
                    write_json(dense_path, dense_artifact)
                    append_jsonl(records, {"question_id": qid, "frame_count": frame_count, "action_id": "dense_custom", "status": "complete", "artifact": str(dense_path)})
                for layer in args.compaction_layers:
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
                            write_json(out, artifact)
                            append_jsonl(records, {"question_id": qid, "frame_count": frame_count, "action_id": route.action_id, "status": "complete", "artifact": str(out)})
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
    return summary


def main() -> None:
    args = parse_args()
    summary = run_adaptive_compaction(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
