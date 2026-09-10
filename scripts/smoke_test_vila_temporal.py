#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VILA temporal backend smoke tests on one 8-bin and one 64-bin example.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", default="outputs/experiment1_v3/primary_manifest.jsonl")
    parser.add_argument("--sampling-policy-json", default="outputs/experiment1_v3/split_summary.json")
    parser.add_argument("--output-root", default="outputs/experiment1_v3_cross_model/smoke_vila")
    parser.add_argument("--checkpoint", default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--resolution-config", default="medium")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-run", action="store_true", help="Only validate existing smoke-test artifacts.")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def planned_bins(record: dict[str, Any], policy: dict[str, Any]) -> int:
    duration = float(record["analyzed_duration_seconds"])
    desired = math.ceil(duration / float(policy["delta_t_seconds"]))
    return min(int(policy["max_bins"]), max(int(policy.get("min_bins", 8)), desired))


def select_examples(manifest: Path, policy_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    records = load_jsonl(manifest)
    policy_payload = json.loads(policy_json.read_text())
    policy = policy_payload.get("realtime_sampling_policy", policy_payload)
    by_bins = {int(bins): record for record in sorted(records, key=lambda item: item["question_id"]) if (bins := planned_bins(record, policy)) in {8, 64}}
    missing = [bins for bins in (8, 64) if bins not in by_bins]
    if missing:
        raise RuntimeError(f"Could not find smoke-test examples with planned bin counts: {missing}")
    return by_bins[8], by_bins[64]


def run_example(args: argparse.Namespace, record: dict[str, Any], label: str) -> Path:
    output_dir = Path(args.output_root) / label
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
        "realtime",
        "--sampling-policy-json",
        args.sampling_policy_json,
        "--frames-per-bin",
        "1",
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
    return output_dir / f"{record['question_id']}.json"


def validate_artifact(path: Path, expected_bins: int) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    metadata = artifact["metadata"]
    scores = np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"], dtype=np.float64)
    raw = np.asarray(artifact["temporal_relevance"]["raw_temporal_bin_scores"], dtype=np.float64)
    frame_indices = artifact["sampled_frame_indices"][0]
    token_cells = artifact["token_layout"]["visual_token_cells"]
    failures = []
    if metadata.get("truncation_occurred"):
        failures.append("truncation_occurred")
    if len(frame_indices) != expected_bins:
        failures.append(f"expected {expected_bins} frames, got {len(frame_indices)}")
    if len(set(frame_indices)) != len(frame_indices):
        failures.append("duplicate sampled frame indices")
    if artifact["temporal_relevance"]["metadata"]["num_temporal_bins"] != expected_bins:
        failures.append("temporal bin count mismatch")
    if not np.isfinite(scores).all() or not np.isfinite(raw).all():
        failures.append("non-finite temporal scores")
    if not np.allclose(scores.sum(axis=1), np.ones(scores.shape[0]), atol=1e-6):
        failures.append("normalized temporal distributions do not sum to one")
    if metadata.get("reduced_prefill_unmodified_next_logit_max_abs_diff") is None:
        failures.append("missing reduced/unmodified output-equivalence metric")
    if not token_cells:
        failures.append("missing visual token mapping")
    cell_positions = [int(cell["sample_position"]) for cell in token_cells]
    if min(cell_positions, default=0) < 0 or max(cell_positions, default=-1) >= len(frame_indices):
        failures.append("visual token maps outside prepared frame range")
    if failures:
        raise RuntimeError(f"{path} failed VILA smoke validation: {failures}")
    return {
        "artifact": str(path),
        "question_id": artifact["question_id"],
        "expected_bins": expected_bins,
        "frames": len(frame_indices),
        "decoder_layers": metadata["num_decoder_layers"],
        "visual_tokens": metadata["actual_num_visual_tokens"],
        "max_equivalence_diff": metadata.get("reduced_prefill_unmodified_next_logit_max_abs_diff"),
        "peak_cuda_memory_bytes": metadata.get("cuda_max_memory_allocated_bytes"),
        "runtime_seconds": metadata.get("prefill_runtime_seconds"),
    }


def main() -> None:
    args = parse_args()
    eight, sixty_four = select_examples(Path(args.manifest), Path(args.sampling_policy_json))
    results = []
    for record, label, bins in ((eight, "bins8", 8), (sixty_four, "bins64", 64)):
        artifact_path = Path(args.output_root) / label / f"{record['question_id']}.json"
        if not args.skip_run:
            artifact_path = run_example(args, record, label)
        results.append(validate_artifact(artifact_path, bins))
    print(json.dumps({"status": "ok", "results": results}, indent=2))


if __name__ == "__main__":
    main()
