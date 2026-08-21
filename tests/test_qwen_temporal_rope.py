from types import SimpleNamespace

import pytest

from src.models.qwen import Qwen25VLWrapper
from src.models.qwen_temporal_rope import install_qwen_temporal_rope_patch, temporal_position_interval


class SyntheticQwenModel:
    def get_rope_index(self, second_per_grid_ts):
        tokens_per_second = 25
        time_interval = tokens_per_second * int(next(second_per_grid_ts))
        return time_interval


class PatchedAlreadyQwenModel:
    def get_rope_index(self, second_per_grid_ts):
        tokens_per_second = 25
        time_interval = int(tokens_per_second * float(next(second_per_grid_ts)))
        return time_interval


class UnsupportedSourceQwenModel:
    def get_rope_index(self, second_per_grid_ts):
        tokens_per_second = 25
        time_interval = round(tokens_per_second * float(next(second_per_grid_ts)))
        return time_interval


def test_temporal_position_interval_keeps_subsecond_grid_duration():
    assert temporal_position_interval(25, 0.0833) == 2
    assert temporal_position_interval(25, 0.643) == 16


def test_temporal_rope_patch_changes_subsecond_intervals():
    assert SyntheticQwenModel().get_rope_index(iter([0.643])) == 0

    install_qwen_temporal_rope_patch(
        model_cls=SyntheticQwenModel,
        transformers_module=SimpleNamespace(__version__="5.14.1"),
    )

    assert SyntheticQwenModel().get_rope_index(iter([0.0833])) == 2
    assert SyntheticQwenModel().get_rope_index(iter([0.643])) == 16


def test_temporal_rope_patch_rejects_unsupported_version():
    with pytest.raises(RuntimeError, match="expected '5.14.1'"):
        install_qwen_temporal_rope_patch(
            model_cls=SyntheticQwenModel,
            transformers_module=SimpleNamespace(__version__="5.15.0"),
        )


def test_temporal_rope_patch_rejects_changed_upstream_source():
    with pytest.raises(RuntimeError, match="corrected pattern is already present"):
        install_qwen_temporal_rope_patch(
            model_cls=PatchedAlreadyQwenModel,
            transformers_module=SimpleNamespace(__version__="5.14.1"),
        )
    with pytest.raises(RuntimeError, match="Upstream implementation changed"):
        install_qwen_temporal_rope_patch(
            model_cls=UnsupportedSourceQwenModel,
            transformers_module=SimpleNamespace(__version__="5.14.1"),
        )


def test_qwen_wrapper_installs_temporal_rope_patch_before_inference(monkeypatch):
    events = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            events.append("processor")
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            events.append("model_load")
            assert events[0] == "patch"
            return cls()

        def eval(self):
            events.append("eval")

    def fake_patch():
        events.append("patch")
        return SimpleNamespace(
            to_metadata=lambda: {
                "transformers_version": "5.14.1",
                "temporal_rope_patch_active": True,
            }
        )

    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        FakeTorch,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoProcessor=FakeProcessor, Qwen2_5_VLForConditionalGeneration=FakeModel),
    )
    monkeypatch.setattr("src.models.qwen.install_qwen_temporal_rope_patch", fake_patch)

    wrapper = Qwen25VLWrapper()
    wrapper._load()

    assert events == ["patch", "processor", "model_load", "eval"]
    assert wrapper._temporal_rope_patch_info is not None
