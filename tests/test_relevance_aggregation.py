import numpy as np

from src.experiment1.relevance import compute_layerwise_relevance, normalize_distribution
from src.experiment1.token_layout import TokenLayout, VisualTokenCell


def _layout():
    cells = (
        VisualTokenCell(2, 0, "video", 0, 0, 0, 0, 2, 1, 2),
        VisualTokenCell(3, 1, "video", 0, 0, 0, 1, 2, 1, 2),
        VisualTokenCell(4, 2, "video", 0, 1, 0, 0, 2, 1, 2),
        VisualTokenCell(5, 3, "video", 0, 1, 0, 1, 2, 1, 2),
    )
    return TokenLayout(
        question_token_indices=(6, 7),
        prompt_token_indices=tuple(range(8)),
        visual_token_indices=(2, 3, 4, 5),
        visual_cells=cells,
        visual_grid_metadata={"video_grid_thw": [[2, 1, 4]], "spatial_merge_size": 2},
        query_scope="question",
    )


def test_normalize_distribution_handles_zero_rows():
    arr = normalize_distribution(np.array([[0.0, 0.0], [1.0, 3.0]]), axis=1)
    np.testing.assert_allclose(arr[0], [0.0, 0.0])
    np.testing.assert_allclose(arr[1], [0.25, 0.75])


def test_compute_layerwise_relevance_preserves_layer_dimension():
    attn0 = np.zeros((2, 8, 8))
    attn1 = np.zeros((2, 8, 8))
    attn0[:, [6, 7], 2:6] = [1.0, 1.0, 2.0, 2.0]
    attn1[:, [6, 7], 2:6] = [0.0, 0.0, 4.0, 4.0]
    relevance = compute_layerwise_relevance([attn0, attn1], _layout())
    assert relevance.raw_token_scores.shape == (2, 4)
    assert relevance.raw_frame_scores.shape == (2, 2)
    np.testing.assert_allclose(relevance.raw_frame_scores[0], [2.0, 4.0])
    np.testing.assert_allclose(relevance.normalized_frame_scores[1], [0.0, 1.0])
    assert len(relevance.concentration_by_layer) == 2
