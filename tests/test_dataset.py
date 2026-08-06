from pathlib import Path
import unittest

from src.dataset import HDEpicVQADataset, seconds_from_time_str


class DatasetTests(unittest.TestCase):
    def test_seconds_from_time_str(self):
        self.assertEqual(seconds_from_time_str("01:02:03.500"), 3723.5)

    def test_dataset_parses_official_shape(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            questions_dir = Path(tmp) / "vqa"
            questions_dir.mkdir()
            (questions_dir / "fine_grained_action_recognition.json").write_text(
                """
{
  "fine_grained_action_recognition_0": {
    "inputs": {
      "video 1": {
        "id": "P03-20240216-205923",
        "start_time": "00:01:15.539",
        "end_time": "00:01:16.310"
      }
    },
    "question": "Which action is happening?",
    "choices": ["A0", "A1", "A2", "A3", "A4"],
    "correct_idx": 1,
    "others": {"actions_num": "1"}
  }
}
""".strip()
            )
            dataset = HDEpicVQADataset(questions_dir)
            example = dataset[0]
            self.assertEqual(example.question_id, "fine_grained_action_recognition_0")
            self.assertEqual(example.question_type, "fine_grained_action_recognition")
            self.assertEqual(example.choices, ("A0", "A1", "A2", "A3", "A4"))
            self.assertEqual(example.correct_idx, 1)
            self.assertEqual(example.inputs[0].video_id, "P03-20240216-205923")
            self.assertEqual(example.inputs[0].participant_id, "P03")
            self.assertEqual(example.inputs[0].start_seconds, 75.539)
            self.assertEqual(
                example.inputs[0].path_under("/data/mp4"),
                Path("/data/mp4/P03/P03-20240216-205923.mp4"),
            )


if __name__ == "__main__":
    unittest.main()
