from __future__ import annotations

from typing import Any

from src.experiment1.relevance import LayerwiseRelevance, compute_layerwise_relevance
from src.experiment1.token_layout import TokenLayout


class QwenDecoderSelfAttentionRelevanceExtractor:
    """Extract question-row to visual-column relevance from decoder self-attention."""

    def compute_relevance(self, question: str, frames: Any, model_outputs: Any) -> LayerwiseRelevance:
        if not hasattr(model_outputs, "attentions") or model_outputs.attentions is None:
            raise ValueError("model_outputs must expose decoder attentions.")
        if not hasattr(model_outputs, "token_layout"):
            raise ValueError("model_outputs must include a TokenLayout as token_layout.")
        layout: TokenLayout = model_outputs.token_layout
        return compute_layerwise_relevance(model_outputs.attentions, layout)
