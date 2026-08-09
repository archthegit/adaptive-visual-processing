#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.plots import plot_spatial_heatmap_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spatial query-relevance overlays from an Experiment 1 artifact.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--temporal-bin", type=int, action="append", default=None)
    return parser.parse_args()


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("Install decord to extract frames for spatial overlays.") from exc
    reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frame_index = max(0, min(int(frame_index), len(reader) - 1))
    return reader[frame_index].asnumpy()


def represented_frame_index(cells: list[dict[str, Any]], input_index: int, temporal_bin: int) -> int:
    matches = [
        cell
        for cell in cells
        if cell["video_input_index"] == input_index and cell["temporal_bin"] == temporal_bin
    ]
    if not matches:
        raise ValueError(f"No visual-token cells for input {input_index}, temporal bin {temporal_bin}.")
    frame_indices = matches[0].get("sampled_frame_indices", [])
    if not frame_indices:
        raise ValueError(f"No sampled frame mapping for input {input_index}, temporal bin {temporal_bin}.")
    return int(frame_indices[len(frame_indices) // 2])


def main() -> None:
    args = parse_args()
    artifact = json.loads(Path(args.artifact).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial = np.asarray(artifact["relevance"]["normalized_spatial_scores_by_input"], dtype=np.float64)
    layer = args.layer if args.layer >= 0 else spatial.shape[0] + args.layer
    if layer < 0 or layer >= spatial.shape[0]:
        raise ValueError(f"Layer {args.layer} is outside available range 0-{spatial.shape[0] - 1}.")
    input_index = args.input_index
    temporal_bins = args.temporal_bin or list(range(spatial.shape[2]))
    video_path = artifact["metadata"]["source_video_paths"][input_index]
    cells = artifact["token_layout"]["visual_token_cells"]

    written = []
    for temporal_bin in temporal_bins:
        frame_index = represented_frame_index(cells, input_index, temporal_bin)
        frame = read_frame(video_path, frame_index)
        heatmap = spatial[layer, input_index, temporal_bin]
        path = output_dir / f"{artifact['question_id']}_input{input_index}_t{temporal_bin}_layer{layer}.png"
        plot_spatial_heatmap_overlay(
            frame,
            heatmap,
            path,
            title=f"{artifact['question_id']} input {input_index} temporal bin {temporal_bin} layer {layer}",
        )
        written.append(str(path))
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()

