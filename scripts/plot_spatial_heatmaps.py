#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
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
    except ImportError:
        return read_frame_ffmpeg(video_path, frame_index)
    else:
        reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frame_index = max(0, min(int(frame_index), len(reader) - 1))
        return reader[frame_index].asnumpy()


def probe_video(video_path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    if fps <= 0:
        raise RuntimeError(f"Could not determine a positive FPS for {video_path}.")
    if stream.get("nb_frames") and stream["nb_frames"] != "N/A":
        num_frames = int(stream["nb_frames"])
    else:
        num_frames = max(1, int(round(float(stream["duration"]) * fps)))
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps, "num_frames": num_frames}


def read_frame_ffmpeg(video_path: str, frame_index: int) -> np.ndarray:
    info = probe_video(video_path)
    frame_index = max(0, min(int(frame_index), info["num_frames"] - 1))
    timestamp = frame_index / info["fps"]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    frame = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = info["height"] * info["width"] * 3
    if frame.size != expected:
        raise RuntimeError(f"ffmpeg returned {frame.size} bytes, expected {expected}.")
    return frame.reshape((info["height"], info["width"], 3))


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
