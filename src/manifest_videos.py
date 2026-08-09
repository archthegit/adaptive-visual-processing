from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_HD_EPIC_BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/Videos"


@dataclass(frozen=True)
class ManifestVideo:
    video_id: str
    participant_id: str
    url: str
    output_path: Path

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["output_path"] = str(self.output_path)
        return data


def video_ids_from_manifest(path: str | Path) -> list[str]:
    video_ids: set[str] = set()
    with Path(path).open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for clip in record.get("video_clip", []):
                video_id = clip.get("video_id")
                if video_id:
                    video_ids.add(str(video_id))
    return sorted(video_ids)


def manifest_videos(
    manifest_path: str | Path,
    output_dir: str | Path,
    base_url: str = DEFAULT_HD_EPIC_BASE_URL,
) -> list[ManifestVideo]:
    videos = []
    for video_id in video_ids_from_manifest(manifest_path):
        participant_id = video_id.split("-")[0]
        videos.append(
            ManifestVideo(
                video_id=video_id,
                participant_id=participant_id,
                url=f"{base_url.rstrip('/')}/{participant_id}/{video_id}.mp4",
                output_path=Path(output_dir) / participant_id / f"{video_id}.mp4",
            )
        )
    return videos


def remote_size_bytes(url: str, timeout: float = 20.0) -> int | None:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size is not None else None
    except (HTTPError, URLError, TimeoutError):
        return None


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def write_download_plan(path: str | Path, videos: Iterable[ManifestVideo], sizes: dict[str, int | None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for video in videos:
        size = sizes.get(video.video_id)
        records.append(video.to_json() | {"size_bytes": size, "size": format_bytes(size)})
    path.write_text(json.dumps(records, indent=2))
