from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterator


def peak_cpu_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def cuda_memory_snapshot(torch_module: Any | None = None) -> dict[str, int]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return {}
    try:
        if not torch_module.cuda.is_available():
            return {}
        return {
            "cuda_memory_allocated_bytes": int(torch_module.cuda.memory_allocated()),
            "cuda_memory_reserved_bytes": int(torch_module.cuda.memory_reserved()),
            "cuda_max_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            "cuda_max_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
        }
    except Exception:
        return {}


@dataclass
class StageProfiler:
    enabled: bool = False
    log_progress: bool = False
    stages: list[dict[str, Any]] = field(default_factory=list)
    tensor_shapes: list[dict[str, Any]] = field(default_factory=list)

    def log(self, message: str) -> None:
        if self.log_progress:
            print(message, file=sys.stderr, flush=True)

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self.log(f"[experiment1] start {name}")
        started = time.time()
        start_rss = peak_cpu_rss_bytes()
        try:
            yield
        finally:
            cuda = cuda_memory_snapshot()
            record = {
                "stage": name,
                "elapsed_seconds": time.time() - started,
                "start_peak_cpu_rss_bytes": start_rss,
                "end_peak_cpu_rss_bytes": peak_cpu_rss_bytes(),
                **cuda,
                **metadata,
            }
            self.stages.append(record)
            self.log(
                "[experiment1] end "
                f"{name} elapsed={record['elapsed_seconds']:.3f}s "
                f"peak_rss={record['end_peak_cpu_rss_bytes']}"
            )

    def add_tensor_shapes(self, stage: str, shapes: Any) -> None:
        if self.enabled and shapes:
            self.tensor_shapes.append({"stage": stage, "shapes": shapes})

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stages": self.stages,
            "tensor_shapes": self.tensor_shapes,
            "peak_cpu_rss_bytes": peak_cpu_rss_bytes() if self.enabled else None,
            **cuda_memory_snapshot(),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2))
