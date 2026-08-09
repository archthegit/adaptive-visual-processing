from src.experiment1.qwen_execution import cuda_memory_metadata, normalize_video_kwargs, represented_sampled_frames


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
    assert normalize_video_kwargs({"fps": [1.0, 2.0]}) == {"fps": [1.0, 2.0]}


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
