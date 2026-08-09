import numpy as np

from src.experiment1.relevance import (
    _attention_array,
    build_layerwise_relevance_from_token_scores,
    compute_layerwise_relevance,
    normalize_distribution,
)
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


def test_attention_array_casts_tensor_like_bfloat_before_numpy():
    class FakeBFloatTensor:
        def __init__(self, cast_to_float=False):
            self.cast_to_float = cast_to_float

        def detach(self):
            return self

        def float(self):
            return FakeBFloatTensor(cast_to_float=True)

        def cpu(self):
            return self

        def numpy(self):
            if not self.cast_to_float:
                raise TypeError("Got unsupported ScalarType BFloat16")
            return np.zeros((1, 2, 3, 3), dtype=np.float32)

    arr = _attention_array(FakeBFloatTensor())
    assert arr.shape == (2, 3, 3)


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
    assert relevance.metadata["extraction_method"] == "returned_full_attention_reduced_after_forward"


def test_build_layerwise_relevance_from_reduced_token_scores():
    token_scores = np.array([[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 4.0, 4.0]])
    relevance = build_layerwise_relevance_from_token_scores(token_scores, _layout(), "unit_test_reduced_scores")
    assert relevance.raw_token_scores.shape == (2, 4)
    np.testing.assert_allclose(relevance.raw_frame_scores[0], [2.0, 4.0])
    assert relevance.metadata["extraction_method"] == "unit_test_reduced_scores"
