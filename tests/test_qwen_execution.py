import pytest

from src.experiment1.qwen_execution import (
    cuda_memory_metadata,
    effective_sample_fps,
    frame_indices_for_temporal_bins,
    mask_frame_batch_temporal_bins,
    next_token_topk_from_outputs,
    normalize_video_kwargs,
    represented_sampled_frames,
    validate_scalar_video_fps_compatibility,
)
from src.frame_sampling import FrameBatch


class _Batch:
    frame_indices = tuple(range(8))
    timestamps = tuple(float(i) for i in range(8))


def test_normalize_video_kwargs_collapses_single_fps_list():
    assert normalize_video_kwargs({"fps": [1.0], "do_sample_frames": False}) == {
        "fps": 1.0,
        "do_sample_frames": False,
    }


def test_normalize_video_kwargs_collapses_identical_multi_fps_list():
    assert normalize_video_kwargs({"fps": [1.0, 1.0]}) == {"fps": 1.0}


def test_normalize_video_kwargs_preserves_multi_fps_list():
    with pytest.raises(ValueError, match="Refusing to collapse different FPS"):
        normalize_video_kwargs({"fps": [1.0, 2.0]})


def test_effective_sample_fps_uses_sampled_timestamp_span():
    class Batch:
        timestamps = (10.0, 20.0, 30.0, 40.0)

    assert effective_sample_fps(Batch()) == 0.1


def test_validate_scalar_video_fps_rejects_mixed_multi_video_rates():
    class Batch:
        metadata = {"input_modality": "video"}

        def __init__(self, timestamps):
            self.timestamps = timestamps

    with pytest.raises(ValueError, match="different effective FPS"):
        validate_scalar_video_fps_compatibility([
            Batch((0.0, 1.0, 2.0)),
            Batch((0.0, 10.0, 20.0)),
        ])


def test_validate_scalar_video_fps_ignores_reference_images():
    class Batch:
        def __init__(self, timestamps, modality):
            self.timestamps = timestamps
            self.metadata = {"input_modality": modality}

    validate_scalar_video_fps_compatibility([
        Batch((0.0, 1.0, 2.0), "video"),
        Batch((10.0,), "image"),
    ])


def test_next_token_topk_from_outputs_extracts_compact_logits():
    torch = pytest.importorskip("torch")

    class Outputs:
        logits = torch.tensor([[[0.0, 2.0, 1.0]]])

    topk = next_token_topk_from_outputs(Outputs(), k=2)
    assert topk == [{"token_id": 1, "logit": 2.0}, {"token_id": 2, "logit": 1.0}]


def test_cuda_memory_metadata_is_empty_without_cuda():
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    assert cuda_memory_metadata(FakeTorch()) == {}


def test_represented_sampled_frames_labels_two_frames_per_temporal_bin():
    represented = represented_sampled_frames(_Batch(), temporal_index=2, grid_t=4)
    assert represented["sampled_frame_indices"] == [4, 5]
    assert represented["sampled_timestamps"] == [4.0, 5.0]
    assert "two sampled frames" in represented["note"]


def test_pre_encoder_temporal_mask_maps_bins_to_sampled_frames():
    import numpy as np

    assert frame_indices_for_temporal_bins(8, 4, (1, 3)) == [2, 3, 6, 7]
    batch = FrameBatch(
        frames=np.ones((8, 2, 2, 3), dtype=np.uint8),
        frame_indices=tuple(range(8)),
        timestamps=tuple(float(idx) for idx in range(8)),
        video_path=None,
        metadata={"input_modality": "video"},
    )

    masked = mask_frame_batch_temporal_bins(batch, 4, (1, 3))

    assert masked.metadata["pre_encoder_removed_temporal_bins"] == [1, 3]
    assert masked.metadata["pre_encoder_masked_sample_positions"] == [2, 3, 6, 7]
    assert masked.frames[[2, 3, 6, 7]].sum() == 0
    assert masked.frames[[0, 1, 4, 5]].sum() > 0
