from types import SimpleNamespace

import numpy as np

from src.experiment1.qwen_vision_attention import (
    VisionAttentionCapture,
    block_diagonal_attention,
    vision_module_to_layer,
)


def test_block_diagonal_attention_reconstructs_split_windows():
    chunks = [
        np.ones((2, 2, 2), dtype=np.float64),
        np.full((2, 3, 3), 2.0, dtype=np.float64),
    ]
    full = block_diagonal_attention(chunks)
    assert full.shape == (2, 5, 5)
    np.testing.assert_allclose(full[:, :2, :2], 1.0)
    np.testing.assert_allclose(full[:, 2:, 2:], 2.0)
    np.testing.assert_allclose(full[:, :2, 2:], 0.0)


def test_vision_module_to_layer_finds_nested_visual_attention_modules():
    attn0 = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    attn1 = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    visual = SimpleNamespace(blocks=[SimpleNamespace(attn=attn0), SimpleNamespace(attn=attn1)])

    class Model:
        def named_modules(self):
            return iter((("", self), ("backbone.visual", visual)))

    assert vision_module_to_layer(Model()) == {id(attn0): 0, id(attn1): 1}


def test_vision_attention_capture_pools_chunks_to_temporal_bins():
    attn = SimpleNamespace()
    capture = VisionAttentionCapture(
        module_to_layer={id(attn): 0},
        grid_thw=[2, 1, 2],
        spatial_merge_size=1,
        reverse_indices=None,
    )
    chunk = np.asarray(
        [
            [
                [0.4, 0.4, 0.1, 0.1],
                [0.4, 0.4, 0.1, 0.1],
                [0.4, 0.4, 0.1, 0.1],
                [0.4, 0.4, 0.1, 0.1],
            ]
        ],
        dtype=np.float64,
    )
    capture.chunks_by_layer[0] = [chunk]
    temporal = capture.temporal_attention_for_layer(0)
    assert temporal.shape == (1, 2)
    np.testing.assert_allclose(temporal[0], [0.8, 0.2])
    artifact = capture.to_json_dict(expected_layers=1)
    assert artifact["num_layers"] == 1
    assert artifact["num_heads"] == 1
    assert artifact["num_temporal_bins"] == 2
