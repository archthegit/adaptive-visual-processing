import json

import pytest

from src.experiment1.v2_interventions import (
    build_v2_intervention_records,
    contiguous_high_attention_cluster,
    fusion_condition_layer,
    position_matched_random_bins,
    select_temporal_bins,
    uniform_temporal_bins,
    write_v2_intervention_manifest,
)


def _artifact(path, scores):
    path.write_text(
        json.dumps(
            {
                "question_id": path.stem,
                "temporal_relevance": {
                    "normalized_temporal_bin_scores": [
                        [1.0 / len(scores) for _ in scores],
                        scores,
                    ]
                },
            }
        )
    )


def _primary():
    return [
        {
            "question_id": "q1",
            "source_video_id": "v1",
            "participant_id": "p1",
            "category": "gaze",
            "duration_group": "short",
            "split": "dev",
        }
    ]


def _primary_with_test():
    return _primary() + [
        {
            "question_id": "q2",
            "source_video_id": "v2",
            "participant_id": "p2",
            "category": "gaze",
            "duration_group": "short",
            "split": "test",
        }
    ]


def test_select_temporal_bins_top_bottom_random_and_contiguous():
    scores = [0.05, 0.5, 0.1, 0.25, 0.1]
    assert select_temporal_bins(scores, "top", 0.4) == [1, 3]
    assert select_temporal_bins(scores, "bottom", 0.4) == [0, 4]
    assert select_temporal_bins(scores, "random", 0.4, seed=1, question_id="q") == select_temporal_bins(
        scores, "random", 0.4, seed=1, question_id="q"
    )
    assert select_temporal_bins(scores, "uniform", 0.4) == [0, 4]
    assert contiguous_high_attention_cluster(scores, 2) == [1, 2]
    with pytest.raises(ValueError, match="Unsupported"):
        select_temporal_bins(scores, "middle", 0.2)


def test_uniform_and_position_matched_random_are_distinct_and_equal_budget():
    scores = [0.9, 0.01, 0.01, 0.01, 0.8, 0.01, 0.01, 0.7, 0.01, 0.01]
    uniform = uniform_temporal_bins(len(scores), 3)
    random_a = position_matched_random_bins(scores, fraction=0.3, seed=11, question_id="q1")
    random_b = position_matched_random_bins(scores, fraction=0.3, seed=11, question_id="q1")

    assert uniform == [0, 4, 9]
    assert random_a == random_b
    assert len(random_a) == len(uniform) == 3
    assert random_a != uniform


def test_build_intervention_records_from_baseline_artifacts(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _artifact(baseline / "q1.json", [0.05, 0.5, 0.1, 0.25, 0.1])
    records = build_v2_intervention_records(
        _primary(),
        baseline,
        condition="mask_top20",
        strategy="top",
        removal_fraction=0.2,
        ranking_layer=-1,
        seed=7,
    )
    assert records[0]["selected_temporal_bins"] == [1]
    assert records[0]["ranking_layer"] == -1
    assert records[0]["selection_source"] == "baseline"
    assert records[0]["pre_encoder_mask_temporal_bins"] == [1]


def test_mismatched_top_uses_mismatched_artifact_directory(tmp_path):
    baseline = tmp_path / "baseline"
    mismatched = tmp_path / "mismatched"
    baseline.mkdir()
    mismatched.mkdir()
    _artifact(baseline / "q1.json", [0.9, 0.05, 0.05])
    _artifact(mismatched / "q1.json", [0.05, 0.9, 0.05])
    records = build_v2_intervention_records(
        _primary(),
        baseline,
        condition="mask_mismatched_top20",
        strategy="mismatched_top",
        removal_fraction=0.34,
        mismatched_output_dir=mismatched,
    )
    assert records[0]["selected_temporal_bins"] == [0, 1]
    assert records[0]["selection_source"] == "mismatched_query"


def test_mismatched_top_requires_mismatched_output_dir(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _artifact(baseline / "q1.json", [0.9, 0.05, 0.05])
    with pytest.raises(ValueError, match="mismatched_output_dir"):
        build_v2_intervention_records(_primary(), baseline, "mask_mismatched_top20", "mismatched_top")


def test_fusion_depth_conditions_use_decoder_direct_access_field(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _artifact(baseline / "q1.json", [0.05, 0.5, 0.1, 0.25, 0.1])
    records = build_v2_intervention_records(
        _primary(),
        baseline,
        condition="fusion_block_top20_after_layer_8",
        strategy="top",
        removal_fraction=0.2,
    )
    assert records[0]["decoder_direct_access_mask_temporal_bins"] == [1]
    assert records[0]["decoder_direct_access_through_layer"] == 8
    assert "pre_encoder_mask_temporal_bins" not in records[0]


def test_fusion_condition_layer_parses_and_rejects_bad_names():
    assert fusion_condition_layer("mask_top20") is None
    assert fusion_condition_layer("fusion_block_top20_after_layer_24") == 24
    with pytest.raises(ValueError, match="after_layer"):
        fusion_condition_layer("fusion_block_top20")
    with pytest.raises(ValueError, match="non-integer"):
        fusion_condition_layer("fusion_block_top20_after_layer_late")


def test_write_intervention_manifest_requires_frozen_layer_and_test_split(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _artifact(baseline / "q1.json", [0.05, 0.5, 0.1])
    _artifact(baseline / "q2.json", [0.05, 0.5, 0.1])
    primary = tmp_path / "primary.jsonl"
    primary.write_text("\n".join(json.dumps(record) for record in _primary_with_test()) + "\n")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"selected_layer": 1}))

    with pytest.raises(ValueError, match="frozen-reference-layer"):
        write_v2_intervention_manifest(primary, baseline, tmp_path / "bad.jsonl", "mask_top20", "top")

    output = tmp_path / "interventions.jsonl"
    records = write_v2_intervention_manifest(
        primary,
        baseline,
        output,
        "mask_top20",
        "top",
        frozen_reference_layer_path=frozen,
    )

    assert [record["question_id"] for record in records] == ["q2"]
    assert records[0]["ranking_layer"] == 1

    with pytest.raises(ValueError, match="does not match frozen"):
        write_v2_intervention_manifest(
            primary,
            baseline,
            tmp_path / "bad_layer.jsonl",
            "mask_top20",
            "top",
            ranking_layer=-1,
            frozen_reference_layer_path=frozen,
        )
