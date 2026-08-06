import unittest
import subprocess
from pathlib import Path

from src.frame_sampling import UniformFrameSampler
from src.dataset import parse_video_segment


class FrameSamplingTests(unittest.TestCase):
    def test_uniform_indices_include_segment_bounds_with_repeats_for_short_segments(self):
        sampler = UniformFrameSampler(num_frames=4)
        self.assertEqual(sampler.sample_indices(2, 6), [2, 3, 4, 5])
        self.assertEqual(sampler.sample_indices(5, 6), [5, 5, 5, 5])

    def test_sample_array_returns_frames_indices_and_timestamps(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        frames = np.arange(10 * 2).reshape(10, 2)
        batch = UniformFrameSampler(num_frames=3).sample_array(
            frames, fps=2.0, start_seconds=1.0, end_seconds=4.0
        )
        self.assertEqual(batch.frame_indices, (2, 4, 7))
        self.assertEqual(batch.timestamps, (1.0, 2.0, 3.5))
        self.assertEqual(batch.frames.tolist(), frames[[2, 4, 7]].tolist())

    def test_sample_video_reads_exact_uniform_frames_from_synthetic_mp4(self):
        try:
            import numpy as np
        except ImportError:
            self.fail("numpy must be installed for MP4 frame sampling tests")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "synthetic.mp4"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=10:duration=1",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ]
            subprocess.run(cmd, check=True)
            segment = parse_video_segment(
                "video 1",
                {
                    "id": "P00-synthetic",
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:01.000",
                },
            )
            batch = UniformFrameSampler(num_frames=5).sample_video(video_path, segment)

        self.assertEqual(batch.frames.shape, (5, 48, 64, 3))
        self.assertEqual(batch.frame_indices, (0, 2, 4, 6, 9))
        self.assertEqual(len(batch.timestamps), 5)
        self.assertTrue(all(a <= b for a, b in zip(batch.frame_indices, batch.frame_indices[1:])))
        self.assertTrue(all(a <= b for a, b in zip(batch.timestamps, batch.timestamps[1:])))
        np.testing.assert_allclose(batch.timestamps, (0.0, 0.2, 0.4, 0.6, 0.9))


if __name__ == "__main__":
    unittest.main()
