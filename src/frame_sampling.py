from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import VideoSegment


@dataclass(frozen=True)
class FrameBatch:
    frames: np.ndarray
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    video_path: Path | None
    metadata: dict[str, Any]


class UniformFrameSampler:
    def __init__(self, num_frames: int = 8, resize: int | None = None):
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        self.num_frames = num_frames
        self.resize = resize

    def sample_indices(self, start_frame: int, end_frame_exclusive: int) -> list[int]:
        if end_frame_exclusive <= start_frame:
            end_frame_exclusive = start_frame + 1
        stop = end_frame_exclusive - 1
        if self.num_frames == 1:
            return [start_frame]
        step = (stop - start_frame) / float(self.num_frames - 1)
        return [int(start_frame + step * i) for i in range(self.num_frames)]

    def sample_array(
        self,
        frames: np.ndarray,
        fps: float,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> FrameBatch:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install numpy to sample from in-memory frame arrays.") from exc

        if frames.ndim < 1:
            raise ValueError("frames must have a frame dimension.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        start_frame = int(round((start_seconds or 0.0) * fps))
        end_frame = int(round(end_seconds * fps)) if end_seconds is not None else len(frames)
        start_frame = max(0, min(start_frame, len(frames) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(frames)))
        indices = self.sample_indices(start_frame, end_frame)
        indices_array = np.asarray(indices, dtype=np.int64)
        timestamps = tuple((indices_array / fps).astype(float).tolist())
        return FrameBatch(
            frames=frames[indices_array],
            frame_indices=tuple(int(i) for i in indices),
            timestamps=timestamps,
            video_path=None,
            metadata={"backend": "array", "fps": fps, "source_num_frames": int(len(frames))},
        )

    def sample_video(self, video_path: str | Path, segment: VideoSegment) -> FrameBatch:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)

        try:
            import decord
        except ImportError:
            return self._sample_video_ffmpeg(path, segment)

        kwargs: dict[str, int] = {}
        if self.resize is not None:
            kwargs = {"width": self.resize, "height": self.resize}
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), **kwargs)
        fps = float(reader.get_avg_fps())
        start_frame = int(round((segment.start_seconds or 0.0) * fps))
        end_frame = int(round(segment.end_seconds * fps)) if segment.end_seconds is not None else len(reader)
        start_frame = max(0, min(start_frame, len(reader) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(reader)))
        indices = self.sample_indices(start_frame, end_frame)
        frames = reader.get_batch(indices).asnumpy()
        timestamps = tuple(float(index / fps) for index in indices)
        return FrameBatch(
            frames=frames,
            frame_indices=tuple(int(i) for i in indices),
            timestamps=timestamps,
            video_path=path,
            metadata={
                "backend": "decord",
                "fps": fps,
                "source_num_frames": int(len(reader)),
                "resize": self.resize,
            },
        )

    def _sample_video_ffmpeg(self, path: Path, segment: VideoSegment) -> FrameBatch:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install numpy to use the ffmpeg video sampling fallback.") from exc

        info = self._probe_video(path)
        fps = info["fps"]
        start_frame = int(round((segment.start_seconds or 0.0) * fps))
        end_frame = int(round(segment.end_seconds * fps)) if segment.end_seconds is not None else info["num_frames"]
        start_frame = max(0, min(start_frame, info["num_frames"] - 1))
        end_frame = max(start_frame + 1, min(end_frame, info["num_frames"]))
        indices = self.sample_indices(start_frame, end_frame)

        frames = []
        width = self.resize or info["width"]
        height = self.resize or info["height"]
        for index in indices:
            timestamp = index / fps
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
            ]
            if self.resize is not None:
                cmd.extend(["-vf", f"scale={self.resize}:{self.resize}"])
            cmd.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
            frame = np.frombuffer(result.stdout, dtype=np.uint8)
            expected = height * width * 3
            if frame.size != expected:
                raise RuntimeError(f"ffmpeg returned {frame.size} bytes, expected {expected}.")
            frames.append(frame.reshape((height, width, 3)))

        return FrameBatch(
            frames=np.stack(frames, axis=0),
            frame_indices=tuple(int(i) for i in indices),
            timestamps=tuple(float(index / fps) for index in indices),
            video_path=path,
            metadata={
                "backend": "ffmpeg",
                "fps": fps,
                "source_num_frames": info["num_frames"],
                "resize": self.resize,
            },
        )

    def _probe_video(self, path: Path) -> dict[str, Any]:
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
            str(path),
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        numerator, denominator = stream["avg_frame_rate"].split("/")
        fps = float(numerator) / float(denominator)
        if fps <= 0:
            raise RuntimeError(f"Could not determine a positive FPS for {path}.")
        if stream.get("nb_frames") and stream["nb_frames"] != "N/A":
            num_frames = int(stream["nb_frames"])
        else:
            num_frames = max(1, int(round(float(stream["duration"]) * fps)))
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "num_frames": num_frames,
        }
