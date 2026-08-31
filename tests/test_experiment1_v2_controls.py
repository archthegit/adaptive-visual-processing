import pytest

from src.experiment1.v2_controls import (
    build_v2_control_records,
    mismatched_query_control_record,
    same_video_different_query_record,
)


def _primary():
    return {
        "question_id": "q1",
        "source_video_id": "v1",
        "category": "gaze",
        "duration_group": "short",
        "question": "original?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
    }


def _mismatch():
    return {
        "mismatched_question_id": "q2",
        "mismatched_source_video_id": "v2",
        "category": "gaze",
        "duration_group": "short",
        "question": "wrong question?",
        "choices": ["F", "G", "H", "I", "J"],
        "correct_idx": 3,
        "token_length_difference": 1,
    }


def test_repeated_and_reversed_control_records_set_condition():
    records = build_v2_control_records([_primary()], {"mismatches": {"q1": _mismatch()}}, "repeated_frame")
    assert records[0]["condition"] == "repeated_frame"
    assert records[0]["control_type"] == "positional_bias_repeated_frame"
    records = build_v2_control_records([_primary()], {"mismatches": {"q1": _mismatch()}}, "reversed_video")
    assert records[0]["condition"] == "reversed_video"
    assert records[0]["control_type"] == "content_position_reversal"


def test_mismatched_query_control_record_overrides_prompt_but_keeps_video():
    record = mismatched_query_control_record(_primary(), _mismatch())
    assert record["condition"] == "mismatched_query"
    assert record["source_video_id"] == "v1"
    assert record["override_source_video_id"] == "v2"
    assert record["override_question"] == "wrong question?"
    assert record["override_choices"] == ["F", "G", "H", "I", "J"]
    assert record["override_correct_idx"] == 3


def test_mismatched_query_rejects_same_video_category_or_duration_mismatch():
    mismatch = dict(_mismatch(), mismatched_source_video_id="v1")
    with pytest.raises(ValueError, match="same source video"):
        mismatched_query_control_record(_primary(), mismatch)
    mismatch = dict(_mismatch(), category="ingredient")
    with pytest.raises(ValueError, match="changes category"):
        mismatched_query_control_record(_primary(), mismatch)
    mismatch = dict(_mismatch(), duration_group="long")
    with pytest.raises(ValueError, match="changes duration"):
        mismatched_query_control_record(_primary(), mismatch)


def test_same_video_different_query_control_overrides_question_from_additional_record():
    additional = dict(
        _primary(),
        question_id="q3",
        primary_question_id="q1",
        question="another question?",
        choices=["V", "W", "X", "Y", "Z"],
        correct_idx=4,
    )
    record = same_video_different_query_record(_primary(), additional)
    assert record["condition"] == "same_video_different_query"
    assert record["source_video_id"] == "v1"
    assert record["override_question_id"] == "q3"
    assert record["override_question"] == "another question?"
    assert record["override_correct_idx"] == 4


def test_same_video_different_query_builder_skips_missing_additional_records():
    additional = [
        dict(
            _primary(),
            question_id="q3",
            primary_question_id="q1",
            question="another question?",
            choices=["V", "W", "X", "Y", "Z"],
            correct_idx=4,
        )
    ]
    records = build_v2_control_records([_primary()], {}, "same_video_different_query", additional_questions=additional)
    assert len(records) == 1
    assert records[0]["override_question_id"] == "q3"
    with pytest.raises(ValueError, match="No same-video"):
        build_v2_control_records([_primary()], {}, "same_video_different_query", additional_questions=[])
