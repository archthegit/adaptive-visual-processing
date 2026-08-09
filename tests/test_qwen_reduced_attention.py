import numpy as np

from src.experiment1.qwen_reduced_attention import (
    ReducedAttentionCapture,
    VisualAccessIntervention,
    _set_attention_implementation,
)


class _Config:
    def __init__(self, implementation="eager"):
        self._attn_implementation = implementation


class _Module:
    def __init__(self, config, layer_idx=None):
        self.config = config
        self.layer_idx = layer_idx


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


def test_reduced_attention_capture_orders_layers():
    capture = ReducedAttentionCapture(question_token_indices=(2,), visual_token_indices=(0, 1))
    capture.reduced_by_layer[1] = np.array([0.2, 0.8])
    capture.reduced_by_layer[0] = np.array([0.7, 0.3])
    np.testing.assert_allclose(capture.ordered_token_scores(expected_layers=2), [[0.7, 0.3], [0.2, 0.8]])


def test_set_attention_implementation_only_changes_decoder_layers():
    text_config = _Config()
    vision_config = _Config()
    model = _Model([_Module(vision_config), _Module(text_config, layer_idx=0), _Module(text_config, layer_idx=1)])
    changed = _set_attention_implementation(model, "custom")
    assert text_config._attn_implementation == "custom"
    assert vision_config._attn_implementation == "eager"
    assert changed == [(text_config, "eager")]


def test_custom_attention_implementations_register_causal_masks():
    import pytest

    torch = pytest.importorskip("torch")
    from transformers.masking_utils import create_causal_mask

    from src.experiment1 import qwen_reduced_attention as reduced

    reduced.register_reduced_attention()

    class Config:
        is_causal = True

        def __init__(self, implementation):
            self._attn_implementation = implementation

    for implementation in [reduced.ATTENTION_IMPLEMENTATION, reduced.MASKED_EAGER_IMPLEMENTATION]:
        mask = create_causal_mask(
            config=Config(implementation),
            inputs_embeds=torch.zeros(1, 4, 8),
            attention_mask=None,
            past_key_values=None,
        )
        assert mask is not None
        assert mask.shape == (1, 1, 4, 4)
        assert mask[0, 0, 0, 1] < -1e20
        assert mask[0, 0, 3, 0] == 0


def test_reduced_attention_matches_full_attention_for_question_visual_rows():
    import pytest

    torch = pytest.importorskip("torch")

    from src.experiment1 import qwen_reduced_attention as reduced

    class Module:
        num_key_value_groups = 1
        training = False
        layer_idx = 0

    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    value = torch.eye(3).reshape(1, 1, 3, 3)
    capture = reduced.ReducedAttentionCapture(question_token_indices=(1, 2), visual_token_indices=(0, 2))
    old_capture = reduced._ACTIVE_CAPTURE
    reduced._ACTIVE_CAPTURE = capture
    try:
        output, weights = reduced.qwen_relevance_reduced_sdpa_forward(
            Module(), query, key, value, None, scaling=1.0
        )
    finally:
        reduced._ACTIVE_CAPTURE = old_capture

    assert weights is None
    assert output.shape == (1, 3, 1, 3)
    full = torch.softmax(torch.matmul(query, key.transpose(2, 3)), dim=-1)
    expected = full[:, :, [1, 2], :][:, :, :, [0, 2]].mean(dim=(0, 1, 2)).numpy()
    np.testing.assert_allclose(capture.ordered_token_scores(), [expected], rtol=1e-6, atol=1e-6)


def test_reduced_attention_respects_additive_causal_mask():
    import pytest

    torch = pytest.importorskip("torch")
    from transformers.masking_utils import create_causal_mask

    from src.experiment1 import qwen_reduced_attention as reduced

    class Config:
        is_causal = True
        _attn_implementation = reduced.ATTENTION_IMPLEMENTATION

    class Module:
        num_key_value_groups = 1
        training = False
        layer_idx = 0

    reduced.register_reduced_attention()
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]])
    value = torch.eye(3).reshape(1, 1, 3, 3)
    mask = create_causal_mask(
        config=Config(),
        inputs_embeds=torch.zeros(1, 3, 2),
        attention_mask=None,
        past_key_values=None,
    )
    capture = reduced.ReducedAttentionCapture(question_token_indices=(0, 1), visual_token_indices=(2,))
    old_capture = reduced._ACTIVE_CAPTURE
    reduced._ACTIVE_CAPTURE = capture
    try:
        reduced.qwen_relevance_reduced_sdpa_forward(Module(), query, key, value, mask, scaling=1.0)
    finally:
        reduced._ACTIVE_CAPTURE = old_capture

    masked_logits = torch.matmul(query, key.transpose(2, 3)) + mask
    expected = torch.softmax(masked_logits, dim=-1)[:, :, [0, 1], :][:, :, :, [2]].mean(dim=(0, 1, 2)).numpy()
    np.testing.assert_allclose(capture.ordered_token_scores(), [expected], rtol=1e-6, atol=1e-6)
    assert expected[0] == 0.0


def test_visual_access_intervention_changes_attention_after_cutoff():
    import pytest

    torch = pytest.importorskip("torch")

    from src.experiment1 import qwen_reduced_attention as reduced

    class Module:
        num_key_value_groups = 1
        training = False
        layer_idx = 1

    query = torch.tensor([[[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]]])
    key = torch.tensor([[[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]]])
    value = torch.tensor([[[[10.0, 0.0], [0.0, 1.0], [0.0, 2.0]]]])
    module = Module()
    baseline, _ = reduced.qwen_relevance_masked_eager_forward(module, query, key, value, None, scaling=1.0)

    old_intervention = reduced._ACTIVE_VISUAL_ACCESS
    reduced._ACTIVE_VISUAL_ACCESS = VisualAccessIntervention(
        through_layer=0,
        visual_token_indices=(0,),
        prompt_seq_len=3,
    )
    try:
        blocked, weights = reduced.qwen_relevance_masked_eager_forward(module, query, key, value, None, scaling=1.0)
    finally:
        reduced._ACTIVE_VISUAL_ACCESS = old_intervention

    assert not torch.allclose(baseline, blocked)
    assert torch.all(weights[:, :, 1:, 0] == 0)


def test_visual_access_intervention_keeps_attention_before_cutoff():
    import pytest

    torch = pytest.importorskip("torch")

    from src.experiment1 import qwen_reduced_attention as reduced

    class Module:
        num_key_value_groups = 1
        training = False
        layer_idx = 0

    query = torch.tensor([[[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]]])
    key = query.clone()
    value = torch.tensor([[[[10.0, 0.0], [0.0, 1.0], [0.0, 2.0]]]])
    module = Module()
    baseline, _ = reduced.qwen_relevance_masked_eager_forward(module, query, key, value, None, scaling=1.0)

    old_intervention = reduced._ACTIVE_VISUAL_ACCESS
    reduced._ACTIVE_VISUAL_ACCESS = VisualAccessIntervention(
        through_layer=0,
        visual_token_indices=(0,),
        prompt_seq_len=3,
    )
    try:
        unblocked, _ = reduced.qwen_relevance_masked_eager_forward(module, query, key, value, None, scaling=1.0)
    finally:
        reduced._ACTIVE_VISUAL_ACCESS = old_intervention

    assert torch.allclose(baseline, unblocked)
