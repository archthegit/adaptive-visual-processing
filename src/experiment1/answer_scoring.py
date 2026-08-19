from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np


CHOICE_LETTERS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class ChoiceTokenMapping:
    letter: str
    token_text: str
    token_id: int


@dataclass(frozen=True)
class AnswerChoiceScores:
    choice_token_mapping: tuple[ChoiceTokenMapping, ...]
    choice_logits: tuple[float, ...]
    choice_log_probabilities: tuple[float, ...]
    normalized_choice_probabilities: tuple[float, ...]
    correct_choice_log_probability: float
    correct_vs_strongest_incorrect_margin: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "choice_token_mapping": [asdict(item) for item in self.choice_token_mapping],
            "choice_logits": list(self.choice_logits),
            "choice_log_probabilities": list(self.choice_log_probabilities),
            "normalized_choice_probabilities": list(self.normalized_choice_probabilities),
            "correct_choice_log_probability": self.correct_choice_log_probability,
            "correct_vs_strongest_incorrect_margin": self.correct_vs_strongest_incorrect_margin,
        }


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def validate_choice_token_mapping(
    tokenizer: Any,
    letters: Sequence[str] = CHOICE_LETTERS,
    token_variants: Sequence[str] = ("{letter}", " {letter}"),
) -> tuple[ChoiceTokenMapping, ...]:
    mappings: list[ChoiceTokenMapping] = []
    seen_ids: set[int] = set()
    for letter in letters:
        valid: list[ChoiceTokenMapping] = []
        errors: list[str] = []
        for pattern in token_variants:
            token_text = pattern.format(letter=letter)
            ids = _token_ids(tokenizer, token_text)
            if len(ids) == 1:
                valid.append(ChoiceTokenMapping(letter=letter, token_text=token_text, token_id=ids[0]))
            else:
                errors.append(f"{token_text!r}->{ids}")
        if not valid:
            raise ValueError(f"Choice {letter!r} does not map to exactly one tokenizer token. Tried: {errors}")
        chosen = valid[0]
        if chosen.token_id in seen_ids:
            raise ValueError(f"Choice tokenizer mapping is not one-to-one; token_id={chosen.token_id} repeats.")
        seen_ids.add(chosen.token_id)
        mappings.append(chosen)
    return tuple(mappings)


def _as_numpy_logits(next_token_logits: Any) -> np.ndarray:
    if hasattr(next_token_logits, "detach"):
        next_token_logits = next_token_logits.detach()
        if hasattr(next_token_logits, "float"):
            next_token_logits = next_token_logits.float()
        next_token_logits = next_token_logits.cpu().numpy()
    logits = np.asarray(next_token_logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError(f"Expected one-dimensional next-token logits, got shape {logits.shape}.")
    return logits


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


def score_answer_choice_logits(
    next_token_logits: Any,
    tokenizer: Any,
    correct_idx: int,
    num_choices: int = 5,
) -> AnswerChoiceScores:
    if not 0 <= correct_idx < num_choices:
        raise ValueError(f"correct_idx={correct_idx} is outside num_choices={num_choices}.")
    letters = CHOICE_LETTERS[:num_choices]
    mapping = validate_choice_token_mapping(tokenizer, letters)
    logits = _as_numpy_logits(next_token_logits)
    token_ids = [item.token_id for item in mapping]
    if max(token_ids) >= logits.shape[0]:
        raise ValueError("Choice token id exceeds the logits vocabulary size.")
    choice_logits = logits[token_ids]
    full_log_probs = _log_softmax(logits)
    choice_log_probs = full_log_probs[token_ids]
    choice_probs = np.exp(choice_logits - np.max(choice_logits))
    choice_probs = choice_probs / choice_probs.sum()
    incorrect = [idx for idx in range(num_choices) if idx != correct_idx]
    margin = float(choice_logits[correct_idx] - max(choice_logits[idx] for idx in incorrect))
    return AnswerChoiceScores(
        choice_token_mapping=mapping,
        choice_logits=tuple(float(item) for item in choice_logits),
        choice_log_probabilities=tuple(float(item) for item in choice_log_probs),
        normalized_choice_probabilities=tuple(float(item) for item in choice_probs),
        correct_choice_log_probability=float(choice_log_probs[correct_idx]),
        correct_vs_strongest_incorrect_margin=margin,
    )


def score_answer_choices_from_outputs(
    outputs: Any,
    tokenizer: Any,
    correct_idx: int,
    num_choices: int = 5,
) -> dict[str, Any]:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        return {}
    next_logits = logits[0, -1]
    return score_answer_choice_logits(next_logits, tokenizer, correct_idx, num_choices).to_json_dict()
