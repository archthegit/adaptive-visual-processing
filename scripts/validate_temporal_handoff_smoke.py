#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CONDITIONS = ("dense_custom", "handoff_mean", "hard_evict", "random_handoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Qwen temporal-handoff smoke-test artifacts.")
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_temporal_handoff/smoke_qwen")
    parser.add_argument("--dense-equivalence-atol", type=float, default=1e-4)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _finite_scores(artifact: dict[str, Any]) -> None:
    scores = artifact.get("answer_choice_scores") or {}
    for key in ("choice_logits", "choice_log_probabilities", "normalized_choice_probabilities"):
        values = scores.get(key)
        if not values:
            raise AssertionError(f"{artifact['condition']} missing {key}.")
        if not all(math.isfinite(float(item)) for item in values):
            raise AssertionError(f"{artifact['condition']} has non-finite {key}.")
    if not math.isfinite(float(scores.get("correct_choice_log_probability"))):
        raise AssertionError(f"{artifact['condition']} has non-finite correct-choice log probability.")
    if not math.isfinite(float(scores.get("correct_vs_strongest_incorrect_margin"))):
        raise AssertionError(f"{artifact['condition']} has non-finite answer margin.")


def validate(output_dir: Path, dense_equivalence_atol: float) -> dict[str, Any]:
    artifacts = {condition: load_json(output_dir / f"{condition}.json") for condition in EXPECTED_CONDITIONS}
    equivalence = load_json(output_dir / "dense_equivalence_report.json")
    if float(equivalence["stock_vs_dense_custom_max_logit_difference"]) > dense_equivalence_atol:
        raise AssertionError("dense_custom did not match stock dense logits within tolerance.")
    dense = artifacts["dense_custom"]
    dense_layers = dense["temporal_handoff"]["instrumentation"]["layers"]
    original_len = int(dense["metadata"]["original_sequence_length"])
    if any(layer["sequence_length_in"] != original_len or layer["sequence_length_out"] != original_len for layer in dense_layers):
        raise AssertionError("dense_custom changed sequence length.")

    for condition, artifact in artifacts.items():
        if artifact["status"] != "complete":
            raise AssertionError(f"{condition} is not complete.")
        if artifact.get("generation_supported") is not False:
            raise AssertionError(f"{condition} must mark generation as unsupported.")
        _finite_scores(artifact)

    handoff = artifacts["handoff_mean"]["temporal_handoff"]
    hard = artifacts["hard_evict"]["temporal_handoff"]
    random = artifacts["random_handoff"]["temporal_handoff"]
    handoff_layers = handoff["instrumentation"]["layers"]
    handoff_boundary = next(layer for layer in handoff_layers if layer["compaction_applied_after_layer"])
    compacted_len = int(handoff_boundary["sequence_length_out"])
    if compacted_len >= original_len:
        raise AssertionError("handoff_mean did not physically shorten the sequence.")
    if int(handoff_boundary["memory_token_count_out"]) <= 0:
        raise AssertionError("handoff_mean did not create memory tokens.")
    hard_boundary = next(layer for layer in hard["instrumentation"]["layers"] if layer["compaction_applied_after_layer"])
    if int(hard_boundary["memory_token_count_out"]) != 0:
        raise AssertionError("hard_evict must not create memory tokens.")
    if handoff["retained_temporal_regions"] != hard["retained_temporal_regions"]:
        raise AssertionError("handoff_mean and hard_evict must retain the same adaptive regions.")
    if len(random["retained_temporal_regions"]) != len(handoff["retained_temporal_regions"]):
        raise AssertionError("random_handoff must use the same retained-region budget.")

    dense_flops = int(dense["temporal_handoff"]["instrumentation"]["total_estimated_attention_flops"])
    handoff_flops = int(handoff["instrumentation"]["total_estimated_attention_flops"])
    hard_flops = int(hard["instrumentation"]["total_estimated_attention_flops"])
    random_flops = int(random["instrumentation"]["total_estimated_attention_flops"])
    if not handoff_flops < dense_flops:
        raise AssertionError("handoff_mean did not reduce estimated decoder attention FLOPs.")
    if not hard_flops < dense_flops:
        raise AssertionError("hard_evict did not reduce estimated decoder attention FLOPs.")
    if not random_flops < dense_flops:
        raise AssertionError("random_handoff did not reduce estimated decoder attention FLOPs.")

    return {
        "status": "passed",
        "conditions": list(EXPECTED_CONDITIONS),
        "stock_vs_dense_custom_max_logit_difference": equivalence["stock_vs_dense_custom_max_logit_difference"],
        "original_sequence_length": original_len,
        "handoff_compacted_sequence_length": compacted_len,
        "dense_estimated_attention_flops": dense_flops,
        "handoff_estimated_attention_flops": handoff_flops,
    }


def main() -> None:
    args = parse_args()
    result = validate(Path(args.output_dir), args.dense_equivalence_atol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

