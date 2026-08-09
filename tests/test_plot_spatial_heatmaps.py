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

