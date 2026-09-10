from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.experiment1.resolution import get_resolution_config
from src.experiment1.vila_execution import (
    PreparedVILAInputs,
    extract_temporal_scores_from_vila_attentions,
    run_vila_relevance_example,
    validate_prepared_vila_mapping,
    visual_token_analysis_bins,
)
from src.frame_sampling import FrameBatch


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        letter_ids = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        return [letter_ids[text.strip()]]

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["A"]


class FakeVILAModel:
    checkpoint = "fake-vila"

    def __init__(self, layers=3, visual_token_frame_indices=(0, 0, 1, 2, 2), truncation=False):
        self.layers = layers
        self.visual_token_frame_indices = tuple(visual_token_frame_indices)
        self.truncation = truncation
        self.tokenizer = FakeTokenizer()

    def prepare_inputs(self, example, prompt, frame_batches):
        return PreparedVILAInputs(
            model_inputs={"input_ids": np.asarray([[10, 11, 12, 13, 14, 15, 16, 17]])},
            rendered_prompt=prompt,
            input_ids=(10, 11, 12, 13, 14, 15, 16, 17),
            question_token_indices=(6, 7),
            visual_token_indices=(1, 2, 3, 4, 5),
            visual_token_frame_indices=self.visual_token_frame_indices,
            prepared_frame_indices=tuple(int(index) for batch in frame_batches for index in batch.metadata.get(
                "presented_source_frame_indices", batch.frame_indices
            )),
            truncation_occurred=self.truncation,
        )

    def forward(self, prepared, output_attentions=True):
        if prepared is None:
            prepared = SimpleNamespace(input_ids=(10, 11, 12, 13, 14, 15, 16, 17))
        vocab = 8
        logits = np.zeros((1, len(prepared.input_ids), vocab), dtype=np.float64)
        logits[0, -1, 0:5] = [5.0, 4.0, 3.0, 2.0, 1.0]
        if not output_attentions:
            return SimpleNamespace(logits=logits)
        attentions = []
        for layer in range(self.layers):
            array = np.zeros((1, 2, 8, 8), dtype=np.float64)
            # Non-question row has large visual mass; extraction must ignore it.
            for head in range(2):
                array[0, head, 0, [1, 2, 3, 4, 5]] = 100.0
                array[0, head, 6, [1, 2, 3, 4, 5]] = [0.1 + layer, 0.2, 0.3, 0.4, 0.5]
                array[0, head, 7, [1, 2, 3, 4, 5]] = [0.2 + layer, 0.1, 0.4, 0.3, 0.6]
            attentions.append(array)
        return SimpleNamespace(attentions=attentions, logits=logits)

    def generate(self, prepared):
        return np.asarray([[10, 11, 12, 13, 14, 15, 16, 17, 0]])

    def decode_new_tokens(self, output_ids, input_length):
        return "A"


def fake_example():
    segment = SimpleNamespace(
        input_key="video 1",
        video_id="video-a",
        participant_id="P01",
        start_seconds=0.0,
        end_seconds=3.0,
        image_time_seconds=None,
    )
    return SimpleNamespace(
        question_id="q1",
        question_type="gaze_interaction_anticipation",
        question="What happens?",
        choices=("A", "B", "C", "D", "E"),
        correct_idx=0,
        inputs=(segment,),
    )


def frame_batch():
    frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    mapping = [
        {
            "sample_position": 0,
            "analysis_bin": 0,
            "source_frame_index": 10,
            "timestamp_seconds": 1.0,
            "bin_start_seconds": 0.0,
            "bin_end_seconds": 1.0,
            "original_temporal_position": 0,
            "presented_temporal_position": 0,
        },
        {
            "sample_position": 1,
            "analysis_bin": 1,
            "source_frame_index": 20,
            "timestamp_seconds": 2.0,
            "bin_start_seconds": 1.0,
            "bin_end_seconds": 2.0,
            "original_temporal_position": 1,
            "presented_temporal_position": 1,
        },
        {
            "sample_position": 2,
            "analysis_bin": 2,
            "source_frame_index": 30,
            "timestamp_seconds": 3.0,
            "bin_start_seconds": 2.0,
            "bin_end_seconds": 3.0,
            "original_temporal_position": 2,
            "presented_temporal_position": 2,
        },
    ]
    return FrameBatch(
        frames=frames,
        frame_indices=(10, 20, 30),
        timestamps=(1.0, 2.0, 3.0),
        video_path=None,
        metadata={
            "frame_bin_mapping": mapping,
            "sampling": {"mode": "realtime", "num_bins": 3, "policy": {"frames_per_bin": 1}},
        },
    )


def test_visual_token_to_bin_mapping_supports_variable_tokens_per_frame():
    batch = frame_batch()
    prepared = FakeVILAModel(visual_token_frame_indices=(0, 0, 1, 2, 2)).prepare_inputs(
        fake_example(), "prompt", [batch]
    )
    assert visual_token_analysis_bins(prepared, batch) == (0, 0, 1, 2, 2)


def test_question_rows_only_and_absolute_mass_is_not_normalized():
    attentions = FakeVILAModel(layers=1).forward(None, output_attentions=True).attentions
    raw, absolute = extract_temporal_scores_from_vila_attentions(
        attentions,
        question_token_indices=(6, 7),
        visual_token_indices=(1, 2, 3, 4, 5),
        visual_token_bins=(0, 0, 1, 2, 2),
        num_bins=3,
    )
    np.testing.assert_allclose(raw[0], [0.3, 0.35, 0.9])
    assert absolute[0] == pytest.approx(1.55)


def test_run_vila_relevance_artifact_has_normalized_distributions_and_variable_layer_count():
    artifact = run_vila_relevance_example(
        FakeVILAModel(layers=4),
        fake_example(),
        [frame_batch()],
        get_resolution_config("low"),
        condition="baseline",
    )
    scores = np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"])
    assert scores.shape == (4, 3)
    np.testing.assert_allclose(scores.sum(axis=1), np.ones(4))
    assert artifact["metadata"]["num_decoder_layers"] == 4
    assert artifact["metadata"]["actual_num_frames"] == 3
    assert artifact["metadata"]["actual_num_visual_tokens"] == 5
    assert artifact["answer_choice_scores"]["correct_choice_log_probability"] < 0.0
    assert artifact["correct"] is True


def test_repeated_frame_preserves_frame_mapping_and_positions():
    artifact = run_vila_relevance_example(
        FakeVILAModel(),
        fake_example(),
        [frame_batch()],
        get_resolution_config("low"),
        condition="repeated_frame",
    )
    assert artifact["sampled_frame_indices"] == [(10, 20, 30)]
    assert artifact["frame_bin_mappings"][0][0]["analysis_bin"] == 0
    assert artifact["metadata"]["condition"] == "repeated_frame"


def test_reversed_video_requires_presented_frame_order_and_records_reverse_mapping():
    artifact = run_vila_relevance_example(
        FakeVILAModel(),
        fake_example(),
        [frame_batch()],
        get_resolution_config("low"),
        condition="reversed_video",
    )
    assert artifact["presented_to_original_frame_bin_mappings"][0][0]["original_source_frame_index"] == 30
    assert artifact["metadata"]["condition"] == "reversed_video"


def test_truncation_detection_fails_loudly():
    with pytest.raises(ValueError, match="truncation"):
        run_vila_relevance_example(
            FakeVILAModel(truncation=True),
            fake_example(),
            [frame_batch()],
            get_resolution_config("low"),
        )


def test_prepared_mapping_detects_reordered_frames():
    batch = frame_batch()
    prepared = PreparedVILAInputs(
        model_inputs={},
        rendered_prompt="prompt",
        input_ids=(1, 2, 3),
        question_token_indices=(2,),
        visual_token_indices=(0, 1),
        visual_token_frame_indices=(0, 1),
        prepared_frame_indices=(20, 10, 30),
        truncation_occurred=False,
    )
    with pytest.raises(ValueError, match="frame order mismatch"):
        validate_prepared_vila_mapping(prepared, [batch])
