import unittest

from src.dataset import parse_vqa_example
from src.models.base import format_multiple_choice_prompt, parse_choice_response


class AnswerParsingTests(unittest.TestCase):
    def test_parse_choice_response_matches_official_first_capital_letter(self):
        self.assertEqual(parse_choice_response("C", 5), 2)
        self.assertEqual(parse_choice_response("answer is B.", 5), 1)
        self.assertEqual(parse_choice_response("The answer is B.", 5), -1)
        self.assertEqual(parse_choice_response("Z", 5), -1)
        self.assertEqual(parse_choice_response("no capital choice", 5), -1)

    def test_format_prompt_parses_official_time_and_bbox_tags(self):
        example = parse_vqa_example(
            "fixture_0",
            {
                "inputs": {"video 1": {"id": "P01-1"}},
                "question": "Where is <BBOX 0 704 1408 1408> at <TIME 00:03:1.8 video 1>?",
                "choices": ["0", "1", "2", "3", "4"],
                "correct_idx": 0,
            },
        )
        prompt = format_multiple_choice_prompt(example)
        self.assertIn("(0, 500, 1000, 1000)", prompt)
        self.assertIn("03:01", prompt)


if __name__ == "__main__":
    unittest.main()
