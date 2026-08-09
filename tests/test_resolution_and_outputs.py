import json

import numpy as np

from src.experiment1.outputs import write_experiment_artifact
from src.experiment1.resolution import get_resolution_config


def test_resolution_config_exposes_explicit_pixel_budgets():
    cfg = get_resolution_config("low")
    assert cfg.min_pixels > 0
    assert cfg.max_pixels > cfg.min_pixels
    assert cfg.to_processor_kwargs()["max_pixels"] == cfg.max_pixels


def test_experiment_artifact_serializes_numpy_arrays(tmp_path):
    path = tmp_path / "artifact.json"
    write_experiment_artifact(path, {"scores": np.array([1.0, 2.0])})
    data = json.loads(path.read_text())
    assert data["scores"] == [1.0, 2.0]
