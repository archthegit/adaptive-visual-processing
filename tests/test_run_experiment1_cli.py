import sys

from types import SimpleNamespace

import pytest

from scripts.run_experiment1 import (
    completed_question_ids,
    frame_batches_for_example,
    frames_per_video_input,
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
            "--attention-extraction",
            "reduced_sdpa",
            "--remove-temporal-bin",
            "2",
        ],
    )
    args = parse_args()
    assert args.max_new_tokens == 24
    assert args.attention_extraction == "reduced_sdpa"
    assert args.frame_budget_mode == "total"
    assert args.resume is False
    assert args.shard_index == 0
    assert args.num_shards == 1
    assert args.remove_temporal_bin == [2]


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
