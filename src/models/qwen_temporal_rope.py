from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass
from typing import Any


SUPPORTED_TRANSFORMERS_VERSION = "5.14.1"
BUGGY_PATTERN = "tokens_per_second * int(next(second_per_grid_ts))"
PATCHED_PATTERN = "int(tokens_per_second * float(next(second_per_grid_ts)))"


@dataclass(frozen=True)
class TemporalRopePatchInfo:
    transformers_version: str
    temporal_rope_patch_active: bool
    patched_class: str
    buggy_pattern: str
    patched_pattern: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "transformers_version": self.transformers_version,
            "temporal_rope_patch_active": self.temporal_rope_patch_active,
            "temporal_rope_patched_class": self.patched_class,
            "temporal_rope_buggy_pattern": self.buggy_pattern,
            "temporal_rope_patched_pattern": self.patched_pattern,
        }


def temporal_position_interval(tokens_per_second: int | float, second_per_grid_ts: int | float) -> int:
    return int(float(tokens_per_second) * float(second_per_grid_ts))


def install_qwen_temporal_rope_patch(
    model_cls: type[Any] | None = None,
    transformers_module: Any | None = None,
) -> TemporalRopePatchInfo:
    if transformers_module is None:
        import transformers as transformers_module

    version = str(getattr(transformers_module, "__version__", "unknown"))
    if version != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Refusing to patch Qwen temporal mRoPE because the installed transformers version is "
            f"{version!r}, expected {SUPPORTED_TRANSFORMERS_VERSION!r}."
        )

    if model_cls is None:
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as model_cls
        except Exception as exc:
            raise RuntimeError("Could not import Qwen2_5_VLModel for temporal mRoPE compatibility patch.") from exc

    if getattr(model_cls, "_adaptive_temporal_rope_patch_active", False):
        return TemporalRopePatchInfo(
            transformers_version=version,
            temporal_rope_patch_active=True,
            patched_class=f"{model_cls.__module__}.{model_cls.__name__}",
            buggy_pattern=BUGGY_PATTERN,
            patched_pattern=PATCHED_PATTERN,
        )

    original = getattr(model_cls, "get_rope_index", None)
    if original is None:
        raise RuntimeError("Qwen temporal mRoPE patch failed: Qwen2_5_VLModel.get_rope_index is missing.")
    source = inspect.getsource(original)
    if PATCHED_PATTERN in source:
        raise RuntimeError(
            "Qwen temporal mRoPE patch expected the upstream buggy source, but the corrected pattern is already "
            "present. Review the local compatibility patch before continuing."
        )
    if BUGGY_PATTERN not in source:
        raise RuntimeError(
            "Qwen temporal mRoPE patch expected source pattern "
            f"{BUGGY_PATTERN!r}, but it was not found. Upstream implementation changed; inspect before running."
        )

    patched_source = textwrap.dedent(source).replace(BUGGY_PATTERN, PATCHED_PATTERN)
    namespace = dict(original.__globals__)
    exec(patched_source, namespace)
    patched = namespace.get(original.__name__)
    if patched is None:
        raise RuntimeError("Qwen temporal mRoPE patch failed: patched get_rope_index was not created.")

    setattr(model_cls, "get_rope_index", patched)
    setattr(model_cls, "_adaptive_temporal_rope_patch_active", True)
    setattr(model_cls, "_adaptive_temporal_rope_patch_info", {
        "buggy_pattern": BUGGY_PATTERN,
        "patched_pattern": PATCHED_PATTERN,
    })
    return TemporalRopePatchInfo(
        transformers_version=version,
        temporal_rope_patch_active=True,
        patched_class=f"{model_cls.__module__}.{model_cls.__name__}",
        buggy_pattern=BUGGY_PATTERN,
        patched_pattern=PATCHED_PATTERN,
    )


def temporal_rope_patch_metadata(transformers_module: Any | None = None, model_cls: type[Any] | None = None) -> dict[str, Any]:
    if transformers_module is None:
        try:
            import transformers as transformers_module
        except Exception:
            transformers_module = None
    version = str(getattr(transformers_module, "__version__", "unknown"))
    active = bool(getattr(model_cls, "_adaptive_temporal_rope_patch_active", False)) if model_cls is not None else False
    return {
        "transformers_version": version,
        "temporal_rope_patch_active": active,
    }
