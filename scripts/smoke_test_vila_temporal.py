#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VILA temporal backend smoke test on one 8-frame example.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", default="outputs/experiment1_v3/primary_manifest.jsonl")
    parser.add_argument("--output-root", default="outputs/experiment1_v3_cross_model/smoke_vila")
    parser.add_argument("--checkpoint", default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--resolution-config", default="medium")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-run", action="store_true", help="Only validate existing smoke-test artifacts.")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_example(manifest: Path) -> dict[str, Any]:
    records = load_jsonl(manifest)
    if not records:
        raise RuntimeError(f"Manifest is empty: {manifest}")
    durations = sorted(float(record["analyzed_duration_seconds"]) for record in records)
    median = durations[len(durations) // 2]
    return min(
        records,
        key=lambda record: (abs(float(record["analyzed_duration_seconds"]) - median), record["question_id"]),
    )


def run_example(args: argparse.Namespace, record: dict[str, Any]) -> Path:
    output_dir = Path(args.output_root) / "bins8"
    cmd = [
        sys.executable,
        "scripts/run_experiment1.py",
        "--questions-dir",
        args.questions_dir,
        "--mp4-dir",
        args.mp4_dir,
        "--manifest",
        args.manifest,
        "--sampling-mode",
        "cross_model_8",
        "--model-backend",
        "vila_llama3",
        "--model-checkpoint",
        args.checkpoint,
        "--question-id",
        record["question_id"],
        "--resolution-config",
        args.resolution_config,
        "--attention-extraction",
        "reduced_sdpa",
        "--query-scope",
        "question",
        "--condition",
        "baseline",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-dir",
        str(output_dir),
        "--allow-7b-inference",
        "--profile-one-example",
        "--resume",
    ]
    subprocess.run(cmd, check=True)
    records_path = output_dir / "records.jsonl"
    if records_path.exists():
        with records_path.open("r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("question_id") != record["question_id"]:
                    continue
                if payload.get("status") == "failed":
                    raise RuntimeError(
                        "VILA smoke run failed inside run_experiment1.py: "
                        f"question_id={record['question_id']}, error={payload.get('error')}"
                    )
    return output_dir / f"{record['question_id']}.json"


def _finite_values(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        values: list[float] = []
        for value in payload.values():
            values.extend(_finite_values(value))
        return values
    if isinstance(payload, list):
        values = []
        for value in payload:
            values.extend(_finite_values(value))
        return values
    if isinstance(payload, (int, float)):
        return [float(payload)]
    return []


def validate_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    metadata = artifact["metadata"]
    scores = np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"], dtype=np.float64)
    raw = np.asarray(artifact["temporal_relevance"]["raw_temporal_bin_scores"], dtype=np.float64)
    frame_indices = artifact["sampled_frame_indices"][0]
    token_cells = artifact["token_layout"]["visual_token_cells"]
    failures = []
    if metadata.get("truncation_occurred"):
        failures.append("truncation_occurred")
    if len(frame_indices) != 8:
        failures.append(f"expected 8 frames, got {len(frame_indices)}")
    if len(set(frame_indices)) != len(frame_indices):
        failures.append("duplicate sampled frame indices")
    if artifact["temporal_relevance"]["metadata"]["num_temporal_bins"] != 8:
        failures.append("temporal bin count mismatch")
    if metadata.get("num_decoder_layers") != 32:
        failures.append(f"expected 32 decoder layers, got {metadata.get('num_decoder_layers')}")
    if not np.isfinite(scores).all() or not np.isfinite(raw).all():
        failures.append("non-finite temporal scores")
    if not np.allclose(scores.sum(axis=1), np.ones(scores.shape[0]), atol=1e-6):
        failures.append("normalized temporal distributions do not sum to one")
    if metadata.get("full_attention_tensors_materialized"):
        failures.append("full attention tensors were materialized")
    if metadata.get("reduced_prefill_unmodified_next_logit_max_abs_diff") is None:
        failures.append("missing reduced/unmodified output-equivalence metric")
    elif float(metadata["reduced_prefill_unmodified_next_logit_max_abs_diff"]) > 1e-5:
        failures.append("reduced/unmodified next-token logits differ by more than 1e-5")
    if not str(artifact.get("raw_response") or "").strip():
        failures.append("empty generated response")
    if not token_cells:
        failures.append("missing visual token mapping")
    cell_positions = [int(cell["sample_position"]) for cell in token_cells]
    if sorted(set(cell_positions)) != list(range(8)):
        failures.append("visual token mapping does not cover exactly frame positions 0..7")
    if min(cell_positions, default=0) < 0 or max(cell_positions, default=-1) >= len(frame_indices):
        failures.append("visual token maps outside prepared frame range")
    expanded = metadata.get("expanded_sequence_length")
    context_limit = metadata.get("context_limit")
    if expanded is None or context_limit is None:
        failures.append("missing expanded sequence/context metadata")
    elif int(expanded) > int(context_limit):
        failures.append("expanded sequence exceeds context limit")
    feature_lengths = metadata.get("image_feature_lengths") or []
    if len(feature_lengths) != len(frame_indices):
        failures.append("feature length count does not match frame count")
    if sum(int(item) for item in feature_lengths) != len(token_cells):
        failures.append("feature lengths do not reconstruct visual token count")
    counts_by_position = {position: cell_positions.count(position) for position in range(8)}
    if [counts_by_position[position] for position in range(8)] != [int(item) for item in feature_lengths]:
        failures.append("feature lengths do not match exact frame-to-token mapping")
    answer_values = _finite_values(artifact.get("answer_choice_scores", {}))
    if not answer_values or not np.isfinite(answer_values).all():
        failures.append("answer-choice scores are missing or non-finite")
    preprocessing = metadata.get("vila_preprocessing") or {}
    if not preprocessing.get("native_checkpoint_preprocessing") or preprocessing.get("native_image_size") != "384x384":
        failures.append("missing VILA native 384x384 preprocessing metadata")
    if metadata.get("cuda_max_memory_allocated_bytes") is None:
        failures.append("missing peak CUDA memory")
    if metadata.get("prefill_runtime_seconds") is None:
        failures.append("missing runtime")
    if failures:
        raise RuntimeError(f"{path} failed VILA smoke validation: {failures}")
    return {
        "artifact": str(path),
        "question_id": artifact["question_id"],
        "expected_bins": 8,
        "frames": len(frame_indices),
        "decoder_layers": metadata["num_decoder_layers"],
        "visual_tokens": metadata["actual_num_visual_tokens"],
        "expanded_sequence_length": metadata.get("expanded_sequence_length"),
        "context_limit": metadata.get("context_limit"),
        "max_equivalence_diff": metadata.get("reduced_prefill_unmodified_next_logit_max_abs_diff"),
        "generated_response": artifact.get("raw_response"),
        "peak_cuda_memory_bytes": metadata.get("cuda_max_memory_allocated_bytes"),
        "runtime_seconds": metadata.get("prefill_runtime_seconds"),
    }


def main() -> None:
    args = parse_args()
    record = select_example(Path(args.manifest))
    artifact_path = Path(args.output_root) / "bins8" / f"{record['question_id']}.json"
    if not args.skip_run:
        artifact_path = run_example(args, record)
    result = validate_artifact(artifact_path)
    print(json.dumps({"status": "ok", "result": result}, indent=2))


if __name__ == "__main__":
    main()
