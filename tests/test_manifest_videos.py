import json

from src.manifest_videos import manifest_videos, video_ids_from_manifest


def test_manifest_video_extraction_is_unique_and_sorted(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"video_clip": [{"video_id": "P02-b"}, {"video_id": "P01-a"}]}),
                json.dumps({"video_clip": [{"video_id": "P01-a"}]}),
            ]
        )
    )
    assert video_ids_from_manifest(manifest) == ["P01-a", "P02-b"]
    videos = manifest_videos(manifest, tmp_path / "mp4", "https://example.test/Videos")
    assert videos[0].url == "https://example.test/Videos/P01/P01-a.mp4"
    assert videos[0].output_path.name == "P01-a.mp4"
