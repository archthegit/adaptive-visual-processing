from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .resolution import ResolutionConfig


QWEN_DEFAULT_CHECKPOINT = "Qwen/Qwen2.5-VL-7B-Instruct"
VILA_LLAMA3_DEFAULT_CHECKPOINT = "Efficient-Large-Model/Llama-3-VILA1.5-8B"


@dataclass
class Experiment1ModelBackend:
    backend: str
    checkpoint: str
    model: Any

    def run_example(
        self,
        example: Any,
        frame_batches: list[Any],
        resolution: ResolutionConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.backend == "qwen":
            from .qwen_execution import run_qwen_relevance_example

            return run_qwen_relevance_example(self.model, example, frame_batches, resolution, **kwargs)
        if self.backend == "vila_llama3":
            from .vila_execution import run_vila_relevance_example

            return run_vila_relevance_example(self.model, example, frame_batches, resolution, **kwargs)
        raise ValueError(f"Unsupported Experiment 1 model backend: {self.backend!r}")


def create_model_backend(
    backend: str,
    checkpoint: str | None = None,
    max_new_tokens: int = 16,
) -> Experiment1ModelBackend:
    if backend == "qwen":
        from src.models.qwen import Qwen25VLWrapper, QwenConfig

        resolved = checkpoint or QWEN_DEFAULT_CHECKPOINT
        return Experiment1ModelBackend(
            backend="qwen",
            checkpoint=resolved,
            model=Qwen25VLWrapper(QwenConfig(model_id=resolved, max_new_tokens=max_new_tokens)),
        )
    if backend == "vila_llama3":
        from .vila_execution import VILALlama3Wrapper

        resolved = checkpoint or VILA_LLAMA3_DEFAULT_CHECKPOINT
        return Experiment1ModelBackend(
            backend="vila_llama3",
            checkpoint=resolved,
            model=VILALlama3Wrapper(checkpoint=resolved, max_new_tokens=max_new_tokens),
        )
    raise ValueError("model backend must be one of: qwen, vila_llama3")
