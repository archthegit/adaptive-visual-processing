from src.experiment1.qwen_execution import normalize_video_kwargs


def test_normalize_video_kwargs_collapses_single_fps_list():
    assert normalize_video_kwargs({"fps": [1.0], "do_sample_frames": False}) == {
        "fps": 1.0,
        "do_sample_frames": False,
    }


def test_normalize_video_kwargs_collapses_identical_multi_fps_list():
    assert normalize_video_kwargs({"fps": [1.0, 1.0]}) == {"fps": 1.0}


def test_normalize_video_kwargs_preserves_multi_fps_list():
    assert normalize_video_kwargs({"fps": [1.0, 2.0]}) == {"fps": [1.0, 2.0]}
