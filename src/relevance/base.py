from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SpatialRegion:
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RelevanceOutput:
    frame_scores: np.ndarray
    spatial_maps: np.ndarray | None
    selected_regions: tuple[SpatialRegion, ...]
    metadata: dict[str, Any]


class QueryRelevanceExtractor(Protocol):
    def compute_relevance(self, question: str, frames: np.ndarray, model_outputs: Any) -> RelevanceOutput:
        ...


class NotImplementedRelevanceExtractor:
    def compute_relevance(self, question: str, frames: np.ndarray, model_outputs: Any) -> RelevanceOutput:
        raise NotImplementedError(
            "Query-relevance extraction is intentionally left unimplemented until the experiment definition is finalized."
        )


class RegionPropagator(Protocol):
    def propagate(self, regions: list[SpatialRegion], frames: np.ndarray) -> list[SpatialRegion]:
        ...
