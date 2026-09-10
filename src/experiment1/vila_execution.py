from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from src.frame_sampling import FrameBatch
from src.models.base import format_multiple_choice_prompt, parse_choice_response

from .answer_scoring import score_answer_choices_from_outputs
from .resolution import ResolutionConfig
from .temporal import (
    TemporalLayerStats,
    TemporalRelevance,
    bins_to_attention_mass,
    gini_coefficient,
    normalized_temporal_entropy,
    spearman_rank_correlation,
    temporal_rank_order,
    top1_mass,
    top_fraction_mass,
    topk_overlap,
)


VILA_DEFAULT_CHECKPOINT = "Efficient-Large-Model/Llama-3-VILA1.5-8B"
VILA_REPO_URL = "https://github.com/NVlabs/VILA.git"
VILA_PINNED_COMMIT = "0f1426e8da9181e6e6653e10bc15f62d515fa2f6"
VILA_DEFAULT_CONV_MODE = "llama_3"


@dataclass(frozen=True)
class PreparedVILAInputs:
    model_inputs: dict[str, Any]
    rendered_prompt: str
    input_ids: Sequence[int]
    question_token_indices: tuple[int, ...]
    visual_token_indices: tuple[int, ...]
    visual_token_frame_indices: tuple[int, ...]
    prepared_frame_indices: tuple[int, ...]
    truncation_occurred: bool = False
    image_feature_lengths: tuple[int, ...] = ()
    base_image_token_positions: tuple[int, ...] = ()
    base_to_expanded_token_positions: dict[int, int] = field(default_factory=dict)
    expanded_sequence_length: int | None = None
    context_limit: int | None = None


class VILALlama3Wrapper:
    """Experiment 1 adapter for the official NVLabs/VILA implementation."""

    def __init__(
        self,
        checkpoint: str = VILA_DEFAULT_CHECKPOINT,
        max_new_tokens: int = 16,
        conv_mode: str = VILA_DEFAULT_CONV_MODE,
    ):
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.conv_mode = conv_mode
        self.model_name: str | None = None
        self.context_len: int | None = None
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._llava_modules: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from llava import conversation as conversation_lib
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except ImportError as exc:
            raise RuntimeError(
                "VILA inference requires the official NVLabs/VILA code installed at pinned commit "
                f"{VILA_PINNED_COMMIT}. Run scripts/setup_vila_environment.sh."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; refusing to run VILA inference.")
        if self.conv_mode not in conversation_lib.conv_templates:
            raise RuntimeError(f"VILA conversation mode {self.conv_mode!r} is not available in pinned VILA.")
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.conv_mode].copy()
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.checkpoint)
        self._tokenizer, self._model, self._image_processor, self.context_len = load_pretrained_model(
            self.checkpoint,
            self.model_name,
            None,
        )
        _set_vila_attention_backend(self._model, "sdpa")
        self._model.eval()

    @property
    def tokenizer(self) -> Any:
        self._load()
        return self._tokenizer

    def prepare_inputs(self, example: Any, prompt: str, frame_batches: list[FrameBatch]) -> PreparedVILAInputs:
        self._load()
        return prepare_vila_inputs_from_decoded_frames(
            model=self._model,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            example=example,
            prompt=prompt,
            frame_batches=frame_batches,
            conv_mode=self.conv_mode,
        )

    def forward(self, prepared: PreparedVILAInputs, output_attentions: bool = True) -> Any:
        self._load()
        return self._model(**prepared.model_inputs, output_attentions=output_attentions, use_cache=False)

    def generate(self, prepared: PreparedVILAInputs) -> Any:
        self._load()
        generation_inputs = dict(prepared.model_inputs)
        generation_inputs.pop("labels", None)
        return self._model.generate(
            **generation_inputs,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )

    def decode_new_tokens(self, output_ids: Any, input_length: int) -> str:
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "batch_decode"):
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return ""


def coerce_prepared_vila_inputs(payload: Any) -> PreparedVILAInputs:
    if isinstance(payload, PreparedVILAInputs):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("VILA preparer must return PreparedVILAInputs or a mapping with equivalent fields.")
    required = {
        "model_inputs",
        "rendered_prompt",
        "input_ids",
        "question_token_indices",
        "visual_token_indices",
        "visual_token_frame_indices",
        "prepared_frame_indices",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"VILA prepared input payload is missing required fields: {missing}")
    return PreparedVILAInputs(
        model_inputs=dict(payload["model_inputs"]),
        rendered_prompt=str(payload["rendered_prompt"]),
        input_ids=tuple(int(item) for item in _as_sequence(payload["input_ids"])),
        question_token_indices=tuple(int(item) for item in payload["question_token_indices"]),
        visual_token_indices=tuple(int(item) for item in payload["visual_token_indices"]),
        visual_token_frame_indices=tuple(int(item) for item in payload["visual_token_frame_indices"]),
        prepared_frame_indices=tuple(int(item) for item in payload["prepared_frame_indices"]),
        truncation_occurred=bool(payload.get("truncation_occurred", False)),
        image_feature_lengths=tuple(int(item) for item in payload.get("image_feature_lengths", ())),
        base_image_token_positions=tuple(int(item) for item in payload.get("base_image_token_positions", ())),
        base_to_expanded_token_positions={
            int(key): int(value) for key, value in dict(payload.get("base_to_expanded_token_positions", {})).items()
        },
        expanded_sequence_length=(
            None if payload.get("expanded_sequence_length") is None else int(payload["expanded_sequence_length"])
        ),
        context_limit=None if payload.get("context_limit") is None else int(payload["context_limit"]),
    )


def decoded_rgb_frames_as_pil(frame_batches: Sequence[FrameBatch]) -> list[Any]:
    from PIL import Image

    frames = []
    for batch in frame_batches:
        for frame in batch.frames:
            frames.append(Image.fromarray(frame).convert("RGB"))
    return frames


def official_vila_conversation(prompt: str, num_frames: int) -> list[dict[str, str]]:
    try:
        from llava.constants import MEDIA_TOKENS
    except ImportError as exc:
        raise RuntimeError("Official VILA modules are not importable; run scripts/setup_vila_environment.sh.") from exc
    if num_frames <= 0:
        raise ValueError("VILA prompt construction requires at least one decoded frame.")
    media_prompt = "".join(MEDIA_TOKENS.get("image", "<image>") for _ in range(num_frames)) + "\n" + prompt
    return [{"from": "human", "value": media_prompt}]


def json_safe_conversation(conversation: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{item['from']}: {item['value']}" for item in conversation)


def _tensor_to_device(value: Any, device: Any, dtype: Any | None = None) -> Any:
    if hasattr(value, "to"):
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return value.to(**kwargs)
    return value


def _process_vila_images(images: Sequence[Any], image_processor: Any, model_config: Any) -> Any:
    try:
        from llava.mm_utils import process_images
    except ImportError as exc:
        raise RuntimeError("Official VILA process_images is unavailable.") from exc
    image_tensors = process_images(list(images), image_processor, model_config)
    return image_tensors


def _feature_lengths_from_native_image_encoder(model: Any, image_media: list[Any], image_media_config: dict[str, Any]) -> tuple[int, ...]:
    import torch

    with torch.inference_mode():
        if not hasattr(model, "encoders") or "image" not in model.encoders:
            raise RuntimeError("Loaded pinned VILA model does not expose model.encoders['image'].")
        features = model.encoders["image"](image_media, image_media_config)
    if isinstance(features, (list, tuple)):
        lengths = []
        for feature in features:
            if len(feature.shape) == 3 and int(feature.shape[0]) == 1:
                lengths.append(int(feature.shape[1]))
            else:
                lengths.append(int(feature.shape[0]))
        return tuple(lengths)
    if hasattr(features, "shape") and len(features.shape) == 3:
        return tuple(int(features.shape[1]) for _ in range(int(features.shape[0])))
    if hasattr(features, "shape") and len(features.shape) == 2 and len(image_media) == 1:
        return (int(features.shape[0]),)
    shape = tuple(features.shape) if hasattr(features, "shape") else type(features).__name__
    raise RuntimeError(f"Cannot derive per-frame VILA image feature lengths from shape {shape}.")


def expanded_positions_from_image_features(
    input_ids: Sequence[int],
    image_token_index: int,
    feature_lengths: Sequence[int],
) -> tuple[tuple[int, ...], dict[int, int], tuple[int, ...]]:
    feature_lengths = tuple(int(item) for item in feature_lengths)
    visual_positions: list[int] = []
    image_positions: list[int] = []
    base_to_expanded: dict[int, int] = {}
    cursor = 0
    feature_cursor = 0
    for base_pos, token_id in enumerate(input_ids):
        if int(token_id) == int(image_token_index):
            if feature_cursor >= len(feature_lengths):
                raise ValueError("Input contains more VILA image placeholders than encoded image features.")
            length = feature_lengths[feature_cursor]
            image_positions.append(base_pos)
            visual_positions.extend(range(cursor, cursor + length))
            cursor += length
            feature_cursor += 1
        else:
            base_to_expanded[base_pos] = cursor
            cursor += 1
    if feature_cursor != len(feature_lengths):
        raise ValueError("Encoded VILA image feature count does not match prompt image placeholders.")
    return tuple(visual_positions), base_to_expanded, tuple(image_positions)


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, ...]:
    if not subsequence:
        return tuple()
    n = len(subsequence)
    seq = list(sequence)
    sub = list(subsequence)
    for start in range(0, len(seq) - n + 1):
        if seq[start : start + n] == sub:
            return tuple(range(start, start + n))
    return tuple()


def derive_vila_question_rows(
    tokenizer: Any,
    base_input_ids: Sequence[int],
    base_to_expanded: dict[int, int],
    question_text: str,
) -> tuple[int, ...]:
    question_ids = _tokenizer_ids(tokenizer(question_text, add_special_tokens=False))
    base_question_positions = _find_subsequence([int(item) for item in base_input_ids], question_ids)
    if not base_question_positions:
        raise ValueError("Could not locate question tokens in official VILA prompt tokenization.")
    missing = [pos for pos in base_question_positions if pos not in base_to_expanded]
    if missing:
        raise ValueError(f"Question token positions overlap VILA image placeholders: {missing}")
    return tuple(int(base_to_expanded[pos]) for pos in base_question_positions)


def prepare_vila_inputs_from_decoded_frames(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    example: Any,
    prompt: str,
    frame_batches: Sequence[FrameBatch],
    conv_mode: str = VILA_DEFAULT_CONV_MODE,
) -> PreparedVILAInputs:
    import torch

    try:
        from llava.utils.tokenizer import tokenize_conversation
    except ImportError as exc:
        raise RuntimeError("Official VILA tokenize_conversation utility is unavailable.") from exc
    images = decoded_rgb_frames_as_pil(frame_batches)
    conversation = official_vila_conversation(prompt, len(images))
    input_ids = tokenize_conversation(conversation, tokenizer, add_generation_prompt=True).unsqueeze(0)
    base_ids = _as_sequence(input_ids[0])
    image_tensors = _process_vila_images(images, image_processor, model.config)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    dtype = getattr(model, "dtype", None)
    if isinstance(image_tensors, list):
        image_media = [_tensor_to_device(item, device, dtype=dtype) for item in image_tensors]
    else:
        image_tensors = _tensor_to_device(image_tensors, device, dtype=dtype)
        image_media = [image for image in image_tensors]
    media_config = defaultdict(dict)
    media = {"image": image_media}
    image_token_id = int(tokenizer.media_token_ids["image"])
    feature_lengths = _feature_lengths_from_native_image_encoder(model, image_media, media_config["image"])
    visual_positions, base_to_expanded, image_positions = expanded_positions_from_image_features(
        base_ids,
        image_token_id,
        feature_lengths,
    )
    question_rows = derive_vila_question_rows(tokenizer, base_ids, base_to_expanded, example.question)
    visual_frame_indices: list[int] = []
    for frame_position, length in enumerate(feature_lengths):
        visual_frame_indices.extend([frame_position] * int(length))
    prepared_frame_indices = _all_frame_indices(frame_batches)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.inference_mode():
        inputs_embeds, _, expanded_attention_mask = model._embed(
            input_ids.to(device=device, non_blocking=True),
            media,
            media_config,
            None,
            attention_mask.to(device=device, non_blocking=True),
        )
    actual_expanded_length = int(inputs_embeds.shape[1])
    reconstructed_expanded_length = len([idx for idx in base_ids if int(idx) != image_token_id]) + sum(feature_lengths)
    if actual_expanded_length != reconstructed_expanded_length:
        raise ValueError(
            "Pinned VILA expanded sequence length mismatch: "
            f"reconstructed={reconstructed_expanded_length}, actual={actual_expanded_length}."
        )
    context_limit = int(
        getattr(tokenizer, "model_max_length", 0)
        or getattr(getattr(model, "config", None), "max_position_embeddings", 0)
        or actual_expanded_length
    )
    truncation_occurred = actual_expanded_length > context_limit
    model_inputs = {
        "input_ids": input_ids.to(device=device, non_blocking=True),
        "media": media,
        "media_config": media_config,
        "attention_mask": attention_mask.to(device=device, non_blocking=True),
    }
    return PreparedVILAInputs(
        model_inputs=model_inputs,
        rendered_prompt=json_safe_conversation(conversation),
        input_ids=tuple(base_ids),
        question_token_indices=question_rows,
        visual_token_indices=tuple(int(item) for item in visual_positions),
        visual_token_frame_indices=tuple(visual_frame_indices),
        prepared_frame_indices=prepared_frame_indices,
        truncation_occurred=truncation_occurred,
        image_feature_lengths=tuple(int(item) for item in feature_lengths),
        base_image_token_positions=image_positions,
        base_to_expanded_token_positions=base_to_expanded,
        expanded_sequence_length=actual_expanded_length,
        context_limit=context_limit,
    )


def _as_sequence(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _tokenizer_ids(tokenizer_output: Any) -> list[int]:
    if isinstance(tokenizer_output, dict):
        return _as_sequence(tokenizer_output["input_ids"])
    if hasattr(tokenizer_output, "input_ids"):
        return _as_sequence(tokenizer_output.input_ids)
    return _as_sequence(tokenizer_output)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _set_vila_attention_backend(model: Any, implementation: str) -> list[tuple[Any, str]]:
    changed: list[tuple[Any, str]] = []
    seen: set[int] = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or not hasattr(config, "_attn_implementation") or id(config) in seen:
            continue
        seen.add(id(config))
        changed.append((config, config._attn_implementation))
        config._attn_implementation = implementation
    return changed


@dataclass
class VILAReducedAttentionCapture:
    question_token_indices: tuple[int, ...]
    visual_token_indices: tuple[int, ...]
    reduced_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    tensor_shapes_by_layer: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    sdpa_calls_by_layer: dict[int, int] = field(default_factory=dict)

    def record_shape(self, layer: int, stage: str, shape: Sequence[int]) -> None:
        self.tensor_shapes_by_layer.setdefault(int(layer), []).append(
            {"stage": stage, "shape": [int(item) for item in shape]}
        )

    def ordered_token_scores(self, expected_layers: int | None = None) -> np.ndarray:
        if not self.reduced_by_layer:
            raise RuntimeError("No VILA reduced attention scores were captured.")
        layer_ids = sorted(self.reduced_by_layer)
        if expected_layers is not None and layer_ids != list(range(expected_layers)):
            raise RuntimeError(f"Captured VILA layers {layer_ids}, expected {list(range(expected_layers))}.")
        return np.stack([self.reduced_by_layer[layer_idx] for layer_idx in layer_ids], axis=0)


_ACTIVE_VILA_CAPTURE: VILAReducedAttentionCapture | None = None
_ACTIVE_VILA_LAYER: int | None = None


def _vila_decoder_attention_modules(model: Any) -> list[tuple[int, Any]]:
    modules: list[tuple[int, Any]] = []
    for module in model.modules():
        if not (hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            layer_idx = len(modules)
        modules.append((int(layer_idx), module))
    if not modules:
        raise RuntimeError("Could not find VILA/Llama decoder attention modules.")
    modules.sort(key=lambda item: item[0])
    return modules


def _num_vila_decoder_layers(model: Any) -> int:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    return len(_vila_decoder_attention_modules(model))


def _slice_vila_mask(attn_mask: Any, question_rows: Any, q_len: int, key_len: int) -> Any:
    if attn_mask is None:
        return None
    if len(attn_mask.shape) == 4:
        if int(attn_mask.shape[2]) == q_len:
            local_rows = question_rows - (key_len - q_len)
            return attn_mask[:, :, local_rows, :]
        return attn_mask[:, :, question_rows, :]
    if len(attn_mask.shape) == 3:
        return attn_mask[:, question_rows, :]
    return attn_mask


def _causal_mask_for_question_rows(query: Any, key: Any, question_rows: Any, is_causal: bool) -> Any | None:
    if not is_causal:
        return None
    import torch

    key_len = int(key.shape[-2])
    mask = torch.zeros((1, 1, int(question_rows.numel()), key_len), dtype=query.dtype, device=query.device)
    blocked_value = torch.finfo(query.dtype).min
    key_positions = torch.arange(key_len, device=query.device).view(1, -1)
    blocked = key_positions > question_rows.view(-1, 1)
    mask[:, :, :, :] = torch.where(blocked.view(1, 1, blocked.shape[0], blocked.shape[1]), blocked_value, 0.0)
    return mask


def _repeat_kv_to_query_heads(key: Any, query_heads: int) -> Any:
    key_heads = int(key.shape[-3])
    if key_heads == query_heads:
        return key
    if query_heads % key_heads != 0:
        raise ValueError(f"Cannot align VILA Q/K heads: query_heads={query_heads}, key_heads={key_heads}.")
    repeat = query_heads // key_heads
    return key.repeat_interleave(repeat, dim=-3)


@contextmanager
def vila_reduced_sdpa_context(model: Any, prepared: PreparedVILAInputs) -> Iterator[VILAReducedAttentionCapture]:
    import torch
    import torch.nn.functional as F

    global _ACTIVE_VILA_CAPTURE, _ACTIVE_VILA_LAYER
    capture = VILAReducedAttentionCapture(
        question_token_indices=tuple(prepared.question_token_indices),
        visual_token_indices=tuple(prepared.visual_token_indices),
    )
    previous_capture = _ACTIVE_VILA_CAPTURE
    previous_layer = _ACTIVE_VILA_LAYER
    previous_configs = _set_vila_attention_backend(model, "sdpa")
    modules = _vila_decoder_attention_modules(model)
    original_forwards: list[tuple[Any, Any]] = []
    original_sdpa = F.scaled_dot_product_attention

    def make_forward(layer_idx: int, original_forward: Any) -> Any:
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            global _ACTIVE_VILA_LAYER
            previous = _ACTIVE_VILA_LAYER
            _ACTIVE_VILA_LAYER = int(layer_idx)
            try:
                return original_forward(*args, **kwargs)
            finally:
                _ACTIVE_VILA_LAYER = previous

        return wrapped_forward

    def reduced_sdpa(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        output = original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            *args,
            **kwargs,
        )
        active_capture = _ACTIVE_VILA_CAPTURE
        layer = _ACTIVE_VILA_LAYER
        if active_capture is not None and layer is not None:
            device = query.device
            q_len = int(query.shape[-2])
            key_len = int(key.shape[-2])
            question_rows = torch.as_tensor(active_capture.question_token_indices, device=device, dtype=torch.long)
            question_rows = question_rows[(question_rows >= key_len - q_len) & (question_rows < key_len)]
            if int(question_rows.numel()) > 0:
                local_rows = question_rows - (key_len - q_len)
                visual_cols = torch.as_tensor(active_capture.visual_token_indices, device=device, dtype=torch.long)
                visual_cols = visual_cols[visual_cols < key_len]
                query_rows = query.index_select(-2, local_rows)
                active_capture.record_shape(int(layer), "vila_question_rows", tuple(query_rows.shape))
                key_for_logits = _repeat_kv_to_query_heads(key, int(query.shape[-3]))
                logits = torch.matmul(query_rows, key_for_logits.transpose(-2, -1))
                logits = logits * (float(scale) if scale is not None else (float(query.shape[-1]) ** -0.5))
                reduced_mask = _slice_vila_mask(attn_mask, question_rows, q_len, key_len)
                causal_mask = _causal_mask_for_question_rows(query, key, question_rows, is_causal)
                for candidate in (reduced_mask, causal_mask):
                    if candidate is not None:
                        logits = logits + candidate
                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                qv = probs.index_select(-1, visual_cols)
                active_capture.record_shape(int(layer), "vila_question_by_visual_probs", tuple(qv.shape))
                active_capture.reduced_by_layer[int(layer)] = qv.mean(dim=(0, 1, 2)).detach().cpu().numpy()
                active_capture.sdpa_calls_by_layer[int(layer)] = active_capture.sdpa_calls_by_layer.get(int(layer), 0) + 1
        return output

    _ACTIVE_VILA_CAPTURE = capture
    try:
        for layer_idx, module in modules:
            original_forwards.append((module, module.forward))
            module.forward = make_forward(layer_idx, module.forward)
        F.scaled_dot_product_attention = reduced_sdpa
        yield capture
    finally:
        F.scaled_dot_product_attention = original_sdpa
        for module, original_forward in original_forwards:
            module.forward = original_forward
        _ACTIVE_VILA_CAPTURE = previous_capture
        _ACTIVE_VILA_LAYER = previous_layer
        for config, previous_implementation in previous_configs:
            config._attn_implementation = previous_implementation


def _all_frame_indices(frame_batches: Sequence[FrameBatch]) -> tuple[int, ...]:
    indices: list[int] = []
    for batch in frame_batches:
        presented = batch.metadata.get("presented_source_frame_indices")
        if presented is not None:
            indices.extend(int(index) for index in presented)
        else:
            indices.extend(int(index) for index in batch.frame_indices)
    return tuple(indices)


def sample_position_to_analysis_bin(batch: FrameBatch) -> dict[int, int]:
    return {
        int(item["sample_position"]): int(item["analysis_bin"])
        for item in (batch.metadata.get("frame_bin_mapping", []) or [])
    }


def validate_prepared_vila_mapping(prepared: PreparedVILAInputs, frame_batches: Sequence[FrameBatch]) -> None:
    expected_frames = _all_frame_indices(frame_batches)
    if prepared.prepared_frame_indices != expected_frames:
        raise ValueError(
            "VILA frame order mismatch: prepared frames do not match decoded Experiment 1 frames. "
            f"expected={expected_frames}, prepared={prepared.prepared_frame_indices}"
        )
    if prepared.truncation_occurred:
        raise ValueError("VILA reported truncation; refusing to run Experiment 1 temporal replication.")
    if len(prepared.visual_token_indices) != len(prepared.visual_token_frame_indices):
        raise ValueError("VILA visual token/frame mapping length mismatch.")
    allowed_positions = set(range(len(expected_frames)))
    bad = [idx for idx in prepared.visual_token_frame_indices if idx not in allowed_positions]
    if bad:
        raise ValueError(f"VILA visual tokens reference nonexistent frame positions: {bad[:5]}")
    if not prepared.question_token_indices:
        raise ValueError("VILA prepared inputs did not identify question-token rows.")
    if not prepared.visual_token_indices:
        raise ValueError("VILA prepared inputs did not identify visual-token columns.")


def visual_token_analysis_bins(prepared: PreparedVILAInputs, batch: FrameBatch) -> tuple[int, ...]:
    mapping = sample_position_to_analysis_bin(batch)
    if not mapping:
        raise ValueError("FrameBatch is missing frame_bin_mapping required for VILA temporal aggregation.")
    bins: list[int] = []
    for frame_position in prepared.visual_token_frame_indices:
        if int(frame_position) not in mapping:
            raise ValueError(f"Visual token references sample position {frame_position}, absent from frame_bin_mapping.")
        bins.append(mapping[int(frame_position)])
    return tuple(bins)


def extract_temporal_scores_from_vila_attentions(
    attentions: Sequence[Any],
    question_token_indices: Sequence[int],
    visual_token_indices: Sequence[int],
    visual_token_bins: Sequence[int],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(visual_token_indices) != len(visual_token_bins):
        raise ValueError("visual_token_indices and visual_token_bins must have the same length.")
    raw = np.zeros((len(attentions), num_bins), dtype=np.float64)
    absolute = np.zeros((len(attentions),), dtype=np.float64)
    q_idx = np.asarray(tuple(int(item) for item in question_token_indices), dtype=np.int64)
    v_idx = np.asarray(tuple(int(item) for item in visual_token_indices), dtype=np.int64)
    bin_idx = np.asarray(tuple(int(item) for item in visual_token_bins), dtype=np.int64)
    for layer_index, attention in enumerate(attentions):
        array = _to_numpy(attention)
        if array.ndim != 4:
            raise ValueError(f"Expected VILA attention shape [batch, heads, q, k], got {array.shape}.")
        if array.shape[0] != 1:
            raise ValueError("Experiment 1 VILA extraction expects batch size 1.")
        selected = array[0][:, q_idx, :][:, :, v_idx]
        token_mass = selected.mean(axis=(0, 1))
        absolute[layer_index] = float(token_mass.sum())
        for visual_position, analysis_bin in enumerate(bin_idx):
            raw[layer_index, int(analysis_bin)] += float(token_mass[visual_position])
    return raw, absolute


def temporal_relevance_from_raw_scores(
    raw_temporal: np.ndarray,
    absolute_mass: np.ndarray,
    temporal_bins: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    topk: int = 3,
) -> TemporalRelevance:
    raw = np.asarray(raw_temporal, dtype=np.float64)
    normalized = np.zeros_like(raw)
    for layer_index, scores in enumerate(raw):
        total = float(scores.sum())
        normalized[layer_index] = scores / total if total > 0.0 else scores
    final_order = temporal_rank_order(normalized[-1]) if len(normalized) else tuple()
    metrics: list[TemporalLayerStats] = []
    for layer_index, layer_scores in enumerate(normalized):
        order = temporal_rank_order(layer_scores)
        count80, fraction80 = bins_to_attention_mass(layer_scores, 0.8)
        overlap, overlap_fraction = topk_overlap(order, final_order, topk)
        metrics.append(
            TemporalLayerStats(
                layer=layer_index,
                normalized_temporal_entropy=normalized_temporal_entropy(layer_scores),
                top1_temporal_bin_mass=top1_mass(layer_scores),
                top20_temporal_bin_mass=top_fraction_mass(layer_scores, 0.2),
                temporal_gini=gini_coefficient(layer_scores),
                first_bin_mass=float(layer_scores[0]) if len(layer_scores) else 0.0,
                last_bin_mass=float(layer_scores[-1]) if len(layer_scores) else 0.0,
                bins_to_80pct_mass=count80,
                fraction_bins_to_80pct_mass=fraction80,
                temporal_bin_rank_order=order,
                spearman_with_final_layer_ordering=spearman_rank_correlation(order, final_order)
                if final_order
                else 0.0,
                topk_overlap_with_final_layer=overlap,
                topk_overlap_fraction_with_final_layer=overlap_fraction,
            )
        )
    return TemporalRelevance(
        raw_temporal_bin_scores=raw,
        normalized_temporal_bin_scores=normalized,
        absolute_question_to_visual_attention_mass=np.asarray(absolute_mass, dtype=np.float64),
        temporal_bins=tuple(dict(item) for item in temporal_bins),
        layer_metrics=tuple(metrics),
        metadata=dict(metadata),
    )


def vila_temporal_bin_metadata(batch: FrameBatch, visual_token_bins: Sequence[int]) -> tuple[dict[str, Any], ...]:
    mapping = batch.metadata.get("frame_bin_mapping", []) or []
    num_bins = max(int(item["analysis_bin"]) for item in mapping) + 1
    token_counts = {index: 0 for index in range(num_bins)}
    for analysis_bin in visual_token_bins:
        token_counts[int(analysis_bin)] += 1
    metadata: list[dict[str, Any]] = []
    for bin_index in range(num_bins):
        frames = [item for item in mapping if int(item["analysis_bin"]) == bin_index]
        metadata.append(
            {
                "input_index": 0,
                "temporal_bin": bin_index,
                "analysis_bin": bin_index,
                "num_visual_tokens": int(token_counts[bin_index]),
                "sampled_frame_indices": [int(item["source_frame_index"]) for item in frames],
                "sampled_timestamps": [float(item["timestamp_seconds"]) for item in frames],
                "note": "VILA visual tokens are aggregated by verified frame-position-to-analysis-bin mapping.",
            }
        )
    return tuple(metadata)


def vila_visual_token_cell_metadata(prepared: PreparedVILAInputs, visual_token_bins: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "token_index": int(token_index),
            "visual_index": visual_index,
            "sample_position": int(frame_position),
            "analysis_bin": int(analysis_bin),
            "modality": "video",
        }
        for visual_index, (token_index, frame_position, analysis_bin) in enumerate(
            zip(prepared.visual_token_indices, prepared.visual_token_frame_indices, visual_token_bins)
        )
    ]


def _profile_memory() -> dict[str, Any]:
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = None
    cuda: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            cuda = {
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    except Exception:
        pass
    return {"cpu_rss_bytes": rss, **cuda}


def run_vila_relevance_example(
    model: Any,
    example: Any,
    frame_batches: list[FrameBatch],
    resolution: ResolutionConfig,
    query_scope: str = "question",
    attention_extraction: str = "reduced_sdpa",
    vision_access_through_layer: str | None = None,
    decoder_direct_access_mask_temporal_bins: tuple[int, ...] = (),
    decoder_direct_access_through_layer: int | None = None,
    pre_encoder_remove_temporal_bins: tuple[int, ...] = (),
    pre_encoder_keep_temporal_bins: tuple[int, ...] = (),
    condition: str | None = None,
    profiler: Any | None = None,
) -> dict[str, Any]:
    if query_scope != "question":
        raise ValueError("VILA cross-model replication currently supports query_scope='question' only.")
    if decoder_direct_access_mask_temporal_bins or pre_encoder_remove_temporal_bins or pre_encoder_keep_temporal_bins:
        raise ValueError("VILA replication currently supports descriptive controls, not causal interventions.")
    if attention_extraction not in {"reduced_sdpa", "full"}:
        raise ValueError("attention_extraction must be 'full' or 'reduced_sdpa'.")
    if vision_access_through_layer not in {None, "none"}:
        raise ValueError("VILA replication does not implement Qwen decoder-access cutoff conditions.")

    from .qwen_execution import apply_frame_control

    stage = profiler.stage if profiler is not None else None
    null_stage = _NullStage()
    stage_fn = (lambda name: stage(name)) if stage is not None else (lambda name: null_stage)

    started = time.time()
    controlled_batches = apply_frame_control(frame_batches, condition)
    prompt = format_multiple_choice_prompt(example)
    with stage_fn("vila_prepare_inputs"):
        prepared = model.prepare_inputs(example, prompt, controlled_batches)
        prepared = coerce_prepared_vila_inputs(prepared)
        validate_prepared_vila_mapping(prepared, controlled_batches)
    visual_bins = visual_token_analysis_bins(prepared, controlled_batches[0])
    num_bins = max(visual_bins) + 1

    with stage_fn("vila_decoder_prefill_attention_extraction"):
        target_model = getattr(model, "_model", model)
        if getattr(model, "allow_full_attention_test_fallback", False):
            outputs = model.forward(prepared, output_attentions=True)
            reduced_prefill_next_logits = (
                _to_numpy(outputs.logits[0, -1]) if getattr(outputs, "logits", None) is not None else None
            )
            attentions = getattr(outputs, "attentions", None)
            if attentions is None:
                raise ValueError("VILA test fallback requested full attentions but none were returned.")
            raw_temporal, absolute_mass = extract_temporal_scores_from_vila_attentions(
                attentions,
                prepared.question_token_indices,
                prepared.visual_token_indices,
                visual_bins,
                num_bins,
            )
        else:
            expected_layers = _num_vila_decoder_layers(target_model)
            with vila_reduced_sdpa_context(target_model, prepared) as capture:
                outputs = model.forward(prepared, output_attentions=False)
            reduced_prefill_next_logits = (
                _to_numpy(outputs.logits[0, -1]) if getattr(outputs, "logits", None) is not None else None
            )
            missing_layers = [
                layer for layer in range(expected_layers) if capture.sdpa_calls_by_layer.get(layer, 0) != 1
            ]
            if missing_layers:
                raise RuntimeError(
                    "VILA reduced SDPA capture did not observe every decoder layer exactly once: "
                    f"calls={capture.sdpa_calls_by_layer}, expected_layers={expected_layers}."
                )
            raw_token_scores = capture.ordered_token_scores(expected_layers=expected_layers)
            raw_temporal = np.zeros((raw_token_scores.shape[0], num_bins), dtype=np.float64)
            for visual_index, analysis_bin in enumerate(visual_bins):
                raw_temporal[:, int(analysis_bin)] += raw_token_scores[:, visual_index]
            absolute_mass = raw_token_scores.sum(axis=1)
            if profiler is not None:
                profiler.add_tensor_shapes("vila_decoder_question_visual_reduction", capture.tensor_shapes_by_layer)
            del outputs
    temporal_relevance = temporal_relevance_from_raw_scores(
        raw_temporal,
        absolute_mass,
        vila_temporal_bin_metadata(controlled_batches[0], visual_bins),
        {
            "num_layers": int(raw_temporal.shape[0]),
            "num_temporal_bins": int(raw_temporal.shape[1]),
            "num_visual_tokens": len(prepared.visual_token_indices),
            "num_question_tokens": len(prepared.question_token_indices),
            "query_scope": query_scope,
            "extraction_method": "vila_llama3_question_visual_rows",
            "input_index": 0,
            "topk": 3,
        },
    )
    with stage_fn("vila_answer_scoring"):
        scoring_outputs = model.forward(prepared, output_attentions=False)
        scoring_next_logits = (
            _to_numpy(scoring_outputs.logits[0, -1]) if getattr(scoring_outputs, "logits", None) is not None else None
        )
        if reduced_prefill_next_logits is not None and scoring_next_logits is not None:
            prefill_equivalence = float(np.max(np.abs(reduced_prefill_next_logits - scoring_next_logits)))
        else:
            prefill_equivalence = None
        answer_choice_scores = score_answer_choices_from_outputs(
            scoring_outputs,
            model.tokenizer,
            example.correct_idx,
            len(example.choices),
        )
    with stage_fn("vila_generation"):
        output_ids = model.generate(prepared)
        raw_response = model.decode_new_tokens(output_ids, len(prepared.input_ids))
    predicted_idx = parse_choice_response(raw_response, len(example.choices))

    return {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "question": example.question,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "correct_answer": example.choices[example.correct_idx],
        "video_clip": [
            {
                "input_key": segment.input_key,
                "video_id": segment.video_id,
                "participant_id": segment.participant_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "image_time_seconds": segment.image_time_seconds,
            }
            for segment in example.inputs
        ],
        "raw_response": raw_response,
        "predicted_idx": predicted_idx,
        "correct": predicted_idx == example.correct_idx,
        "answer_choice_scores": answer_choice_scores,
        "intervention_answer_choice_scores": {},
        "sampled_frame_indices": [batch.frame_indices for batch in controlled_batches],
        "sampled_timestamps": [batch.timestamps for batch in controlled_batches],
        "frame_bin_mappings": [batch.metadata.get("frame_bin_mapping", []) for batch in controlled_batches],
        "presented_to_original_frame_bin_mappings": [
            batch.metadata.get("presented_to_original_frame_bin_mapping", []) for batch in controlled_batches
        ],
        "sampling_metadata": [batch.metadata.get("sampling", {}) for batch in controlled_batches],
        "token_layout": {
            "question_token_indices": list(prepared.question_token_indices),
            "visual_token_indices": list(prepared.visual_token_indices),
            "visual_grid_metadata": {
                "backend": "vila_llama3",
                "visual_token_frame_indices": list(prepared.visual_token_frame_indices),
                "image_feature_lengths": list(prepared.image_feature_lengths),
                "base_image_token_positions": list(prepared.base_image_token_positions),
                "expanded_sequence_length": prepared.expanded_sequence_length,
                "context_limit": prepared.context_limit,
            },
            "num_visual_tokens": len(prepared.visual_token_indices),
            "query_scope": query_scope,
            "visual_token_cells": vila_visual_token_cell_metadata(prepared, visual_bins),
        },
        "temporal_relevance": temporal_relevance.to_json_dict(),
        "encoder_temporal": {"available": False, "reason": "VILA replication measures decoder prefill only."},
        "encoder_attention_temporal": {"available": False, "reason": "VILA replication measures decoder prefill only."},
        "metadata": {
            "model_backend": "vila_llama3",
            "model_id": getattr(model, "checkpoint", VILA_DEFAULT_CHECKPOINT),
            "checkpoint": getattr(model, "checkpoint", VILA_DEFAULT_CHECKPOINT),
            "actual_num_frames": sum(len(batch.frame_indices) for batch in controlled_batches),
            "actual_num_visual_tokens": len(prepared.visual_token_indices),
            "num_decoder_layers": int(raw_temporal.shape[0]),
            "question_token_indices": list(prepared.question_token_indices),
            "visual_token_indices": list(prepared.visual_token_indices),
            "truncation_occurred": bool(prepared.truncation_occurred),
            "expanded_sequence_length": prepared.expanded_sequence_length,
            "context_limit": prepared.context_limit,
            "image_feature_lengths": list(prepared.image_feature_lengths),
            "condition": condition or "baseline",
            "resolution": resolution.to_metadata(),
            "prefill_runtime_seconds": time.time() - started,
            "source_video_paths": [str(batch.video_path) if batch.video_path else None for batch in controlled_batches],
            "attention_extraction": attention_extraction,
            "answer_choice_score_source": "separate_unmodified_vila_prefill_forward",
            "reduced_prefill_unmodified_next_logit_max_abs_diff": prefill_equivalence,
            **_profile_memory(),
        },
    }


class _NullStage:
    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False
