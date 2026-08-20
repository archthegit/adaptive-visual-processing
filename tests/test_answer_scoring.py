from collections import UserDict

import numpy as np
import pytest

from src.experiment1.answer_scoring import score_answer_choice_logits, validate_choice_token_mapping


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        mapping = {"A": [10], "B": [11], "C": [12], "D": [13], "E": [14]}
        return {"input_ids": mapping.get(text, [99, 100])}


def test_answer_choice_scoring_uses_validated_single_token_letters():
    logits = np.zeros(20)
    logits[10:15] = [0.0, 1.0, 3.0, 2.0, -1.0]
    scores = score_answer_choice_logits(logits, FakeTokenizer(), correct_idx=2).to_json_dict()
    assert [item["token_id"] for item in scores["choice_token_mapping"]] == [10, 11, 12, 13, 14]
    assert scores["normalized_choice_probabilities"][2] == max(scores["normalized_choice_probabilities"])
    assert scores["correct_vs_strongest_incorrect_margin"] == 1.0


def test_choice_mapping_rejects_multi_token_letters():
    class BadTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}

    with pytest.raises(ValueError, match="does not map to exactly one"):
        validate_choice_token_mapping(BadTokenizer(), ("A",))


def test_choice_mapping_accepts_mapping_like_tokenizer_outputs():
    class MappingTokenizer:
        def __call__(self, text, add_special_tokens=False):
            mapping = {"A": [10], "B": [11], "C": [12], "D": [13], "E": [14]}
            return UserDict({"input_ids": mapping.get(text, [99, 100])})

    mapping = validate_choice_token_mapping(MappingTokenizer(), ("A", "B"))

    assert [item.token_id for item in mapping] == [10, 11]
