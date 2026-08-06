from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .base import ModelPrediction, format_multiple_choice_prompt, parse_choice_response
from ..dataset import VQAExample
from ..frame_sampling import FrameBatch


@dataclass
class QwenConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 4


class Qwen25VLWrapper:
    model_name = "qwen2.5-vl-7b-instruct"

    def __init__(self, config: QwenConfig | None = None):
        self.config = config or QwenConfig()
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen inference requires torch and transformers with Qwen2.5-VL support."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; refusing to run 7B Qwen inference.")

        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
        )
        self._model.eval()

    def predict(self, example: VQAExample, frame_batches: list[FrameBatch]) -> ModelPrediction:
        self._load()
        assert self._model is not None
        assert self._processor is not None

        prompt = format_multiple_choice_prompt(example)
        start = time.time()

        content: list[dict[str, Any]] = []
        for batch in frame_batches:
            for frame in batch.frames:
                content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[frame for batch in frame_batches for frame in batch.frames],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        raw_response = self._processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        predicted_idx = parse_choice_response(raw_response, len(example.choices))

        return ModelPrediction(
            raw_response=raw_response,
            predicted_idx=predicted_idx,
            correct_idx=example.correct_idx,
            correct=predicted_idx == example.correct_idx,
            frame_indices=tuple(batch.frame_indices for batch in frame_batches),
            timestamps=tuple(batch.timestamps for batch in frame_batches),
            metadata={
                "model_id": self.config.model_id,
                "runtime_seconds": time.time() - start,
                "num_visual_inputs": len(frame_batches),
                "num_frames": sum(len(batch.frame_indices) for batch in frame_batches),
            },
        )
