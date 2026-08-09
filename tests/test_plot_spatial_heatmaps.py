import subprocess

from scripts.plot_spatial_heatmaps import available_temporal_bins
from scripts.plot_spatial_heatmaps import read_frame_ffmpeg
from scripts.plot_spatial_heatmaps import represented_frame_index


def test_represented_frame_index_uses_middle_sampled_frame():
    cells = [
        {
            "video_input_index": 0,
            "temporal_bin": 1,
            "sampled_frame_indices": [2, 3],
        }
    ]
    assert represented_frame_index(cells, input_index=0, temporal_bin=1) == 3


def test_available_temporal_bins_uses_selected_input_only():
    cells = [
        {"video_input_index": 0, "temporal_bin": 0},
        {"video_input_index": 0, "temporal_bin": 1},
        {"video_input_index": 1, "temporal_bin": 0},
    ]
    assert available_temporal_bins(cells, input_index=1) == [0]


def test_read_frame_ffmpeg_reads_synthetic_mp4(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x24:rate=5:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    frame = read_frame_ffmpeg(str(video_path), 2)
    assert frame.shape == (24, 32, 3)
