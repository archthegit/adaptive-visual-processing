from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ResolutionConfig:
    name: str
    min_pixels: int
    max_pixels: int
    description: str

    def to_processor_kwargs(self) -> dict[str, int]:
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


RESOLUTION_CONFIGS = {
    "low": ResolutionConfig("low", min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28, description="Small visual-token budget for debugging."),
    "medium": ResolutionConfig("medium", min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28, description="Moderate visual-token budget."),
    "high": ResolutionConfig("high", min_pixels=512 * 28 * 28, max_pixels=1280 * 28 * 28, description="Higher visual-token budget for fine detail."),
}


def get_resolution_config(name: str) -> ResolutionConfig:
    try:
        return RESOLUTION_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown resolution config {name!r}. Expected one of {sorted(RESOLUTION_CONFIGS)}.") from exc
