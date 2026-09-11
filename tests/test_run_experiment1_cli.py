import sys

from types import SimpleNamespace

import pytest

from scripts.run_experiment1 import (
    _load_realtime_sampling_policy,
    _sampling_plan_for_record,
    condition_for_record,
    completed_question_ids,
    decoder_direct_access_through_layer,
    example_for_record,
    frame_batches_for_example,
    frames_per_video_input,
    intervention_bins,
    parse_args,
    records_filename,
    shard_records,
    summary_filename,
)
from src.io import append_jsonl, write_json


def test_run_experiment1_exposes_max_new_tokens(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment1.py",
            "--manifest",
            "manifest.jsonl",
            "--max-new-tokens",
            "24",
            "--sampling-mode",
            "fixed_budget",
            "--attention-extraction",
            "reduced_sdpa",
            "--model-backend",
            "vila_llama3",
            "--model-checkpoint",
            "Efficient-Large-Model/Llama-3-VILA1.5-8B",
            "--frames-per-bin",
            "1",
            "--decoder-mask-temporal-bin",
            "2",
            "--decoder-direct-access-through-layer",
            "8",
            "--condition",
            "repeated_frame",
        ],
    )
    args = parse_args()
    assert args.max_new_tokens == 24
    assert args.sampling_mode == "fixed_budget"
    assert args.attention_extraction == "reduced_sdpa"
    assert args.model_backend == "vila_llama3"
    assert args.model_checkpoint == "Efficient-Large-Model/Llama-3-VILA1.5-8B"
    assert args.frames_per_bin == 1
    assert args.frame_budget_mode == "total"
    assert args.resume is False
    assert args.shard_index == 0
    assert args.num_shards == 1
    assert args.decoder_mask_temporal_bin == [2]
    assert args.decoder_direct_access_through_layer == 8
    assert args.pre_encoder_mask_temporal_bin is None
    assert args.pre_encoder_keep_temporal_bin is None
    assert args.condition == "repeated_frame"


def test_frames_per_input_splits_total_budget_deterministically():
    assert frames_per_video_input(8, 2, "total") == [4, 4]
    assert frames_per_video_input(5, 2, "total") == [3, 2]
    assert frames_per_video_input(4, 1, "total") == [4]
    assert frames_per_video_input(8, 0, "total") == []


def test_frames_per_input_supports_legacy_per_input_mode():
    assert frames_per_video_input(4, 2, "per-input") == [4, 4]


def test_frames_per_input_rejects_too_small_total_budget():
    with pytest.raises(ValueError, match="smaller than the 3 video inputs"):
        frames_per_video_input(2, 3, "total")


def test_shard_records_is_deterministic():
    records = [{"question_id": f"q{i}"} for i in range(7)]
    assert [record["question_id"] for record in shard_records(records, 1, 3)] == ["q1", "q4"]
    with pytest.raises(ValueError, match="shard-index"):
        shard_records(records, 3, 3)


def test_sharded_output_filenames_are_isolated():
    assert records_filename(0, 1) == "records.jsonl"
    assert summary_filename(0, 1) == "summary.json"
    assert records_filename(2, 8) == "records_shard-00002-of-00008.jsonl"
    assert summary_filename(2, 8) == "summary_shard-00002-of-00008.json"


def test_completed_question_ids_only_skips_existing_complete_artifacts(tmp_path):
    artifact = tmp_path / "q1.json"
    write_json(artifact, {"ok": True})
    records_path = tmp_path / "records.jsonl"
    append_jsonl(records_path, {"question_id": "q1", "status": "complete", "artifact": str(artifact)})
    append_jsonl(records_path, {"question_id": "q2", "status": "complete", "artifact": str(tmp_path / "missing.json")})
    append_jsonl(records_path, {"question_id": "q3", "status": "failed"})
    assert completed_question_ids(records_path) == {"q1"}


def test_frame_batches_for_example_samples_reference_images_as_one_frame(monkeypatch):
    calls = []

    class Segment:
        def __init__(self, is_image):
            self.is_image = is_image

        def path_under(self, mp4_dir):
            return f"{mp4_dir}/video.mp4"

    class FakeSampler:
        def __init__(self, num_frames):
            self.num_frames = num_frames

        def sample_video(self, path, segment):
            calls.append((self.num_frames, segment.is_image))
            from src.frame_sampling import FrameBatch

            return FrameBatch(
                frames=[],
                frame_indices=tuple(range(self.num_frames)),
                timestamps=tuple(float(idx) for idx in range(self.num_frames)),
                video_path=None,
                metadata={},
            )

    monkeypatch.setattr("src.frame_sampling.UniformFrameSampler", FakeSampler)
    example = SimpleNamespace(inputs=(Segment(False), Segment(True), Segment(False)))

    batches = frame_batches_for_example(example, "mp4s", 8, "total")

    assert calls == [(4, False), (1, True), (4, False)]
    assert [batch.metadata["input_modality"] for batch in batches] == ["video", "image", "video"]


def test_realtime_sampling_uses_development_durations_from_manifest(tmp_path):
    policy = tmp_path / "split_summary.json"
    policy.write_text(
        '{"realtime_sampling_policy":{"development_duration_p95_seconds":640,"delta_t_seconds":10,'
        '"frames_per_bin":2,"max_bins":64,"max_frames":128}}'
    )
    loaded = _load_realtime_sampling_policy(policy)
    assert loaded["delta_t_seconds"] == 10
    with pytest.raises(ValueError, match="sampling-policy-json"):
        _load_realtime_sampling_policy(None)


def test_realtime_sampling_frames_per_bin_override_uses_one_frame_per_bin(monkeypatch):
    from scripts import run_experiment1

    monkeypatch.setattr(run_experiment1, "_probe_video_for_sampling", lambda path: {"fps": 10.0, "num_frames": 1000})
    monkeypatch.setattr(run_experiment1, "_decord_video_length", lambda path: 1000)
    record = {
        "analyzed_start_seconds": 0.0,
        "analyzed_end_seconds": 8.0,
        "_realtime_sampling_policy": {
            "delta_t_seconds": 1.0,
            "frames_per_bin": 2,
            "max_bins": 64,
            "min_bins": 8,
        },
        "_sampling_policy_json": "split_summary.json",
    }
    plan, metadata = run_experiment1._sampling_plan_for_record(
        record,
        "video.mp4",
        "realtime",
        frames_per_bin_override=1,
    )
    assert len(plan) == 8
    assert all(len(item["source_frame_indices"]) == 1 for item in plan)
    assert metadata["policy"]["frames_per_bin"] == 1
    assert metadata["frames_per_bin_override"] == 1


def test_cross_model_8_sampling_uses_eight_center_frames(monkeypatch):
    from scripts import run_experiment1

    monkeypatch.setattr(run_experiment1, "_probe_video_for_sampling", lambda path: {"fps": 10.0, "num_frames": 1000})
    monkeypatch.setattr(run_experiment1, "_decord_video_length", lambda path: 1000)
    record = {"analyzed_start_seconds": 10.0, "analyzed_end_seconds": 18.0}

    plan, metadata = _sampling_plan_for_record(record, "video.mp4", "cross_model_8")

    assert len(plan) == 8
    assert [item["source_frame_indices"] for item in plan] == [[105], [115], [125], [135], [145], [155], [165], [175]]
    assert metadata["mode"] == "cross_model_8"
    assert metadata["policy"]["name"] == "cross_model_8_bins_1_center_frame"
    assert metadata["num_bins"] == 8


def test_v2_sampling_mode_requires_manifest_record():
    example = SimpleNamespace(inputs=(SimpleNamespace(is_image=False),))
    with pytest.raises(ValueError, match="manifest record"):
        frame_batches_for_example(example, "mp4s", 8, sampling_mode="fixed_budget")


def test_condition_for_record_prefers_cli_then_manifest_then_baseline():
    assert condition_for_record({"condition": "reversed_video"}, None) == "reversed_video"
    assert condition_for_record({"condition": "reversed_video"}, "baseline") == "baseline"
    assert condition_for_record({}, None) == "baseline"


def test_intervention_bins_supports_keep_and_rejects_mixed_modes():
    args = SimpleNamespace(
        decoder_mask_temporal_bin=None,
        pre_encoder_mask_temporal_bin=None,
        pre_encoder_keep_temporal_bin=None,
    )
    assert intervention_bins({"keep_temporal_bins": [1, 2]}, args) == ((), (), (1, 2))
    args = SimpleNamespace(
        decoder_mask_temporal_bin=[0],
        pre_encoder_mask_temporal_bin=None,
        pre_encoder_keep_temporal_bin=[1],
    )
    with pytest.raises(ValueError, match="separate interventions"):
        intervention_bins({}, args)


def test_decoder_direct_access_through_layer_prefers_cli_then_manifest():
    args = SimpleNamespace(decoder_direct_access_through_layer=None)
    assert decoder_direct_access_through_layer({"decoder_direct_access_through_layer": 12}, args) == 12
    args = SimpleNamespace(decoder_direct_access_through_layer=8)
    assert decoder_direct_access_through_layer({"decoder_direct_access_through_layer": 12}, args) == 8
    assert decoder_direct_access_through_layer({}, SimpleNamespace(decoder_direct_access_through_layer=None)) is None


def test_example_for_record_applies_mismatched_query_override():
    from src.dataset import parse_vqa_example

    example = parse_vqa_example(
        "q1",
        {
            "inputs": {"video 1": {"id": "P01-video", "start_time": "00:00:00.000", "end_time": "00:00:10.000"}},
            "question": "Original?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": 0,
        },
    )
    updated = example_for_record(
        example,
        {
            "override_question_id": "q2",
            "override_question": "Different?",
            "override_choices": ["V", "W", "X", "Y", "Z"],
            "override_correct_idx": 4,
        },
    )
    assert updated.question_id == "q1"
    assert updated.inputs == example.inputs
    assert updated.question == "Different?"
    assert updated.choices == ("V", "W", "X", "Y", "Z")
    assert updated.correct_idx == 4
    assert updated.raw["experiment1_v2_original_question_id"] == "q1"
    assert updated.raw["experiment1_v2_override_question_id"] == "q2"
