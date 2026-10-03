from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.dataset import VQAExample, VideoSegment
from src.experiment1.adaptive_compaction import (
    AdaptiveSplitConfig,
    candidate_routes,
    create_source_video_disjoint_splits,
    extract_router_features,
    retention_count,
    stable_action_id,
    validate_compact_action_artifact,
)
from src.experiment1.temporal_handoff import (
    TemporalHandoffConfig,
    run_compacted_decoder_from_prefix_cache,
    run_custom_decoder_prefill,
    run_dense_decoder_with_prefix_cache,
)
from scripts.run_qwen_adaptive_compaction import config_mismatches, validate_resumed_dense_artifact


def _example(qid: str, qtype: str, video: str, participant: str = "P01") -> VQAExample:
    segment = VideoSegment(
        input_key="video_0",
        video_id=video,
        participant_id=participant,
        start_seconds=0.0,
        end_seconds=10.0,
        image_time_seconds=None,
        raw={"id": video, "start_time": "00:00:00.0", "end_time": "00:00:10.0"},
    )
    return VQAExample(
        question_id=qid,
        question_type=qtype,
        question="Question?",
        choices=("A", "B", "C", "D", "E"),
        correct_idx=0,
        inputs=(segment,),
        raw={},
    )


def _many_examples(per_category: int = 6) -> list[VQAExample]:
    qtypes = {
        "fine_grained": "fine_grained_action_recognition",
        "gaze": "gaze_interaction_anticipation",
        "ingredient": "ingredient_ingredient_adding_localization",
        "object_motion": "object_motion_object_movement_itinerary",
    }
    examples = []
    for category, qtype in qtypes.items():
        for idx in range(per_category):
            examples.append(_example(f"{category}_{idx}", qtype, f"{category}-video-{idx}", participant=f"P{idx % 3}"))
    return examples


def test_source_video_disjoint_splits_are_deterministic_and_disjoint():
    examples = _many_examples(per_category=6)
    config = AdaptiveSplitConfig(train_per_category=2, development_per_category=2, test_per_category=2, seed=123, max_questions_per_source_video=4)
    splits_a, summary_a = create_source_video_disjoint_splits(examples, config)
    splits_b, summary_b = create_source_video_disjoint_splits(examples, config)
    assert [[item.question_id for item in splits_a[name]] for name in ("train", "development", "test")] == [
        [item.question_id for item in splits_b[name]] for name in ("train", "development", "test")
    ]
    assert summary_a == summary_b
    assert all(value == 0 for value in summary_a["source_video_overlap_counts"].values())
    assert all(value == 0 for value in summary_a["question_id_overlap_counts"].values())


def test_source_video_disjoint_splits_fail_when_impossible():
    examples = [_example(f"q{i}", "fine_grained_action_recognition", "shared-video") for i in range(5)]
    config = AdaptiveSplitConfig(train_per_category=1, development_per_category=1, test_per_category=1, seed=1)
    with pytest.raises(ValueError, match="shortages"):
        create_source_video_disjoint_splits(examples, config)


def test_retention_count_and_candidate_routes_are_stable_and_unique():
    assert retention_count(4, 0.25) == 1
    assert retention_count(4, 0.50) == 2
    assert retention_count(4, 0.75) == 3
    exhaustive = candidate_routes(
        question_id="q",
        frame_count=8,
        compaction_layer=8,
        native_temporal_cell_count=4,
        retention_fraction=0.50,
        seed=7,
    )
    assert len(exhaustive) == 6
    assert {route.route_family for route in exhaustive} == {"exhaustive"}
    capped = candidate_routes(
        question_id="q",
        frame_count=32,
        compaction_layer=8,
        native_temporal_cell_count=16,
        retention_fraction=0.50,
        seed=7,
    )
    assert len(capped) <= 24
    assert len({route.action_id for route in capped}) == len(capped)
    assert capped == candidate_routes(
        question_id="q",
        frame_count=32,
        compaction_layer=8,
        native_temporal_cell_count=16,
        retention_fraction=0.50,
        seed=7,
    )
    assert stable_action_id(
        question_id="q",
        frame_count=8,
        compaction_layer=8,
        native_temporal_cell_count=4,
        retention_fraction=0.5,
        retained_cells=(1, 3),
        route_family="exhaustive",
    ).startswith("act_")


class _FakeSelfAttention:
    is_causal = True
    config = SimpleNamespace(_attn_implementation="sdpa")


class _FakeLayer:
    def __init__(self, scale: float):
        self.scale = scale
        self.self_attn = _FakeSelfAttention()

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **_kwargs):
        return hidden_states + self.scale

    forward = __call__


class _FakeLmHead:
    def __call__(self, hidden_states):
        return hidden_states


def _layout():
    cells = []
    for temporal in range(2):
        for offset in range(4):
            token_index = temporal * 4 + offset
            cells.append({"token_index": token_index, "temporal_index": temporal})
    return {"visual_cells": cells, "question_token_indices": (8, 9)}


def test_cached_prefix_matches_full_prefix_compaction():
    torch = pytest.importorskip("torch")
    layers = [_FakeLayer(0.1), _FakeLayer(0.2), _FakeLayer(0.3), _FakeLayer(0.4)]
    hidden = torch.arange(12 * 4, dtype=torch.float32).reshape(1, 12, 4)
    layout = _layout()
    config = TemporalHandoffConfig(condition="hard_evict", handoff_layer=1, retained_temporal_regions=(1,), memory_tokens_per_region=0)
    full = run_custom_decoder_prefill(
        layers=layers,
        hidden_states=hidden.clone(),
        position_ids=None,
        layout=layout,
        config=config,
        lm_head=_FakeLmHead(),
        norm=None,
        num_attention_heads=1,
        head_dim=4,
    )
    dense, caches = run_dense_decoder_with_prefix_cache(
        layers=layers,
        hidden_states=hidden.clone(),
        position_ids=None,
        layout=layout,
        cache_boundaries=(1,),
        lm_head=_FakeLmHead(),
        norm=None,
        num_attention_heads=1,
        head_dim=4,
    )
    cached = run_compacted_decoder_from_prefix_cache(
        cache_entry=caches[1],
        layers=layers,
        layout=layout,
        config=config,
        lm_head=_FakeLmHead(),
        norm=None,
        num_attention_heads=1,
        head_dim=4,
    )
    assert torch.allclose(full.logits, cached.logits)
    assert dense.final_hidden_states.shape[1] == 12
    assert cached.final_hidden_states.shape[1] < dense.final_hidden_states.shape[1]
    assert [item.layer for item in cached.instrumentation] == [0, 1, 2, 3]
    assert cached.instrumentation[1].sequence_length_in == 12
    assert cached.instrumentation[1].sequence_length_out == cached.instrumentation[2].sequence_length_in
    assert cached.instrumentation[1].compaction_applied_after_layer is True
    expected_flops = (
        sum(item.estimated_qk_flops + item.estimated_av_flops for item in dense.instrumentation[:2])
        + sum(item.estimated_qk_flops + item.estimated_av_flops for item in cached.instrumentation[2:])
    )
    assert cached.total_estimated_attention_flops == expected_flops


def test_router_features_have_expected_shape_and_finite_values():
    torch = pytest.importorskip("torch")
    hidden = torch.ones((1, 12, 6), dtype=torch.float32)
    features = extract_router_features(hidden, _layout(), layer=8, frame_count=8)
    assert features["cell_mean_residual"].shape == (2, 6)
    assert features["question_mean_residual"].shape == (6,)
    assert features["token_count_per_cell"] == [4, 4]
    assert all(value == 0.0 for value in features["cell_residual_dispersion"])


def test_immutable_config_rejection_and_compact_artifact_schema():
    saved = {"schema_version": "x", "frame_counts": [8], "git_commit": "a"}
    requested = {"schema_version": "x", "frame_counts": [16], "git_commit": "a"}
    assert "frame_counts" in config_mismatches(saved, requested)
    artifact = {
        "question_id": "q",
        "source_video_id": "v",
        "split": "train",
        "frame_count": 8,
        "compaction_layer": 8,
        "native_temporal_cell_count": 4,
        "retention_fraction": 0.5,
        "retained_cell_ids": [1, 3],
        "route_family": "exhaustive",
        "correct_choice_log_probability": -1.0,
        "delta_correct_choice_log_probability_from_dense": -0.1,
        "answer_margin": 0.2,
        "delta_answer_margin_from_dense": -0.1,
        "predicted_answer": "A",
        "correct": True,
        "prediction_changed": False,
        "original_sequence_length": 100,
        "compacted_sequence_length": 60,
        "active_sequence_length_by_layer": [],
        "estimated_attention_flops": 123,
        "paired_flop_reduction_from_dense": 0.4,
        "memory_token_count": 0,
        "status": "complete",
        "git_commit": "abc",
        "run_config": {"schema_version": "x"},
    }
    validate_compact_action_artifact(artifact)
    artifact["retained_cell_ids"] = [1, 1]
    with pytest.raises(ValueError, match="duplicates"):
        validate_compact_action_artifact(artifact)


def test_validate_resumed_dense_artifact_requires_finite_router_features(tmp_path):
    torch = pytest.importorskip("torch")
    run_config = {"schema_version": "x", "git_commit": "abc"}
    artifact_dir = tmp_path / "artifacts" / "q" / "frames_8"
    feature_dir = tmp_path / "features" / "q" / "frames_8"
    artifact_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    feature = {
        "layer": 8,
        "native_temporal_cell_ids": [0, 1],
        "cell_mean_residual": torch.ones((2, 4)),
        "question_mean_residual": torch.ones((4,)),
    }
    feature_path = feature_dir / "layer_8.pt"
    torch.save(feature, feature_path)
    dense = {
        "question_id": "q",
        "frame_count": 8,
        "condition": "dense_custom",
        "native_temporal_cell_count": 2,
        "correct_choice_log_probability": -1.0,
        "answer_margin": 0.1,
        "router_features": [{"layer": 8, "feature_file": str(feature_path)}],
        "status": "complete",
        "git_commit": "abc",
        "run_config": run_config,
    }
    dense_path = artifact_dir / "dense_custom.json"
    dense_path.write_text(__import__("json").dumps(dense))
    (artifact_dir / "dense_equivalence_report.json").write_text('{"passed": true}')

    loaded = validate_resumed_dense_artifact(dense_path, run_config=run_config, output_dir=tmp_path)
    assert loaded["question_id"] == "q"

    feature["cell_mean_residual"][0, 0] = float("nan")
    torch.save(feature, feature_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_resumed_dense_artifact(dense_path, run_config=run_config, output_dir=tmp_path)
