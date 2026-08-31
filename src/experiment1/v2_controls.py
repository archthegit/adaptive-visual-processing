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


def same_video_different_query_record(primary: dict[str, Any], additional: dict[str, Any]) -> dict[str, Any]:
    if primary["source_video_id"] != additional["source_video_id"]:
        raise ValueError(f"Additional question for {primary['question_id']} uses a different source video.")
    record = dict(primary)
    record["condition"] = "same_video_different_query"
    record["control_type"] = "same_video_different_query"
    record["override_question_id"] = additional["question_id"]
    record["override_question"] = additional["question"]
    record["override_choices"] = list(additional["choices"])
    record["override_correct_idx"] = int(additional["correct_idx"])
    record["same_video_primary_question_id"] = additional.get("primary_question_id", primary["question_id"])
    return record


def build_v2_control_records(
    primary_manifest: list[dict[str, Any]],
    mismatched_queries: dict[str, Any],
    control: str,
    additional_questions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    mismatch_map = mismatched_queries.get("mismatches", {})
    additional_by_primary: dict[str, dict[str, Any]] = {}
    for item in additional_questions or []:
        primary_id = item.get("primary_question_id")
        if primary_id and primary_id not in additional_by_primary:
            additional_by_primary[str(primary_id)] = item
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
        elif control == "same_video_different_query":
            additional = additional_by_primary.get(primary["question_id"])
            if additional is None:
                continue
            records.append(same_video_different_query_record(primary, additional))
        else:
            raise ValueError(f"Unsupported control: {control}")
    if control == "same_video_different_query" and not records:
        raise ValueError("No same-video different-query records could be built from additional_questions.")
    return records


def write_v2_control_manifest(
    primary_manifest_path: str | Path,
    mismatched_queries_path: str | Path,
    output_jsonl: str | Path,
    control: str,
    additional_questions_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    primary = []
    with Path(primary_manifest_path).open("r") as handle:
        for line in handle:
            if line.strip():
                primary.append(json.loads(line))
    mismatches = json.loads(Path(mismatched_queries_path).read_text()) if Path(mismatched_queries_path).exists() else {}
    additional_questions = []
    if additional_questions_path is not None and Path(additional_questions_path).exists():
        with Path(additional_questions_path).open("r") as handle:
            additional_questions = [json.loads(line) for line in handle if line.strip()]
    records = build_v2_control_records(primary, mismatches, control, additional_questions=additional_questions)
    write_jsonl(output_jsonl, records)
    return records
