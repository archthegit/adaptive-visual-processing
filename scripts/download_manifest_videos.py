#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manifest_videos import (
    DEFAULT_HD_EPIC_BASE_URL,
    format_bytes,
    manifest_videos,
    remote_size_bytes,
    write_download_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only HD-EPIC MP4s referenced by a JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="data/hd_epic_mp4")
    parser.add_argument("--base-url", default=DEFAULT_HD_EPIC_BASE_URL)
    parser.add_argument("--plan-json", default="outputs/manifest_video_download_plan.json")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--video-id", action="append", default=None, help="Download/check only this video id. Can be repeated.")
    parser.add_argument("--skip-size-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually download files. Prevents accidental large dataset pulls.",
    )
    return parser.parse_args()


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with urlopen(url, timeout=60) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    videos = manifest_videos(args.manifest, args.output_dir, args.base_url)
    if args.video_id:
        requested = set(args.video_id)
        videos = [video for video in videos if video.video_id in requested]
        missing_requested = requested - {video.video_id for video in videos}
        if missing_requested:
            raise ValueError(f"Requested video ids are not present in manifest: {sorted(missing_requested)}")
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    sizes = {}
    if not args.skip_size_check:
        for video in videos:
            sizes[video.video_id] = remote_size_bytes(video.url)
    write_download_plan(args.plan_json, videos, sizes)

    known_total = sum(size for size in sizes.values() if size is not None)
    unknown_count = sum(1 for size in sizes.values() if size is None)
    existing = [video for video in videos if video.output_path.exists()]
    missing = [video for video in videos if not video.output_path.exists()]

    print(f"Manifest videos: {len(videos)}")
    print(f"Already present: {len(existing)}")
    print(f"Missing: {len(missing)}")
    print(f"Known total size: {format_bytes(known_total)}")
    print(f"Unknown size count: {unknown_count}")
    print(f"Plan JSON: {args.plan_json}")

    if args.dry_run or not args.yes:
        print("Dry-run only. Re-run with --yes to download missing videos.")
        return

    for index, video in enumerate(missing, start=1):
        size = format_bytes(sizes.get(video.video_id))
        print(f"[{index}/{len(missing)}] downloading {video.video_id} ({size})")
        download_file(video.url, video.output_path)


if __name__ == "__main__":
    main()
