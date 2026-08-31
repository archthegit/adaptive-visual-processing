from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.io import write_jsonl


def repeated_frame_control_record(primary: dict[str, Any]) -> dict[str, Any]:
    record = dict(primary)
    record["condition"] = "repeated_frame"
    record["control_type"] = "positional_bias_repeated_frame"
    return record


def reversed_video_control_record(primary: dict[str, Any]) -> dict[str, Any]:
    record = dict(primary)
    record["condition"] = "reversed_video"
    record["control_type"] = "content_position_reversal"
    return record


def mismatched_query_control_record(primary: dict[str, Any], mismatch: dict[str, Any]) -> dict[str, Any]:
    if primary["source_video_id"] == mismatch["mismatched_source_video_id"]:
        raise ValueError(f"Mismatched query for {primary['question_id']} uses the same source video.")
    if primary["category"] != mismatch["category"]:
        raise ValueError(f"Mismatched query for {primary['question_id']} changes category.")
    if primary["duration_group"] != mismatch["duration_group"]:
        raise ValueError(f"Mismatched query for {primary['question_id']} changes duration group.")
    record = dict(primary)
    record["condition"] = "mismatched_query"
    record["control_type"] = "same_video_wrong_query"
    record["override_question_id"] = mismatch["mismatched_question_id"]
    record["override_question"] = mismatch["question"]
    record["override_choices"] = list(mismatch["choices"])
    record["override_correct_idx"] = int(mismatch["correct_idx"])
    record["override_source_video_id"] = mismatch["mismatched_source_video_id"]
    record["mismatch_token_length_difference"] = mismatch["token_length_difference"]
    return record


def build_v2_control_records(
    primary_manifest: list[dict[str, Any]],
    mismatched_queries: dict[str, Any],
    control: str,
) -> list[dict[str, Any]]:
    mismatch_map = mismatched_queries.get("mismatches", {})
    records = []
    for primary in primary_manifest:
        if control == "repeated_frame":
            records.append(repeated_frame_control_record(primary))
        elif control == "reversed_video":
            records.append(reversed_video_control_record(primary))
        elif control == "mismatched_query":
            mismatch = mismatch_map.get(primary["question_id"])
            if mismatch is None:
                raise ValueError(f"No mismatched query exists for {primary['question_id']}.")
            records.append(mismatched_query_control_record(primary, mismatch))
        else:
            raise ValueError(f"Unsupported control: {control}")
    return records


def write_v2_control_manifest(
    primary_manifest_path: str | Path,
    mismatched_queries_path: str | Path,
    output_jsonl: str | Path,
    control: str,
) -> list[dict[str, Any]]:
    primary = []
    with Path(primary_manifest_path).open("r") as handle:
        for line in handle:
            if line.strip():
                primary.append(json.loads(line))
    mismatches = json.loads(Path(mismatched_queries_path).read_text()) if Path(mismatched_queries_path).exists() else {}
    records = build_v2_control_records(primary, mismatches, control)
    write_jsonl(output_jsonl, records)
    return records
