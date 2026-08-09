import sys

from types import SimpleNamespace

import pytest

from scripts.run_experiment1 import frame_batches_for_example, frames_per_video_input, parse_args


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
        ],
    )
    args = parse_args()
    assert args.max_new_tokens == 24
    assert args.attention_extraction == "reduced_sdpa"
    assert args.frame_budget_mode == "total"


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
