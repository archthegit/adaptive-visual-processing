from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import pytest


PINNED_COMMIT = "0f1426e8da9181e6e6653e10bc15f62d515fa2f6"


def pinned_vila_source() -> Path:
    path = Path(os.environ.get("VILA_SOURCE_DIR", "/tmp/VILA-pinned"))
    if not path.exists():
        pytest.skip("Pinned NVLabs/VILA source is not available locally.")
    return path


def test_pinned_vila_api_signatures(monkeypatch):
    source = pinned_vila_source()
    monkeypatch.syspath_prepend(str(source))
    constants = importlib.import_module("llava.constants")
    assert not hasattr(constants, "IMAGE_TOKEN_INDEX")
    assert constants.MEDIA_TOKENS["image"] == "<image>"

    try:
        mm_utils = importlib.import_module("llava.mm_utils")
    except ImportError as exc:
        pytest.skip(f"Pinned VILA runtime dependencies are not installed: {exc}")
    tokenizer_image_token_sig = inspect.signature(mm_utils.tokenizer_image_token)
    assert list(tokenizer_image_token_sig.parameters) == ["prompt", "tokenizer", "return_tensors"]

    tokenizer_utils = importlib.import_module("llava.utils.tokenizer")
    tokenize_conversation_sig = inspect.signature(tokenizer_utils.tokenize_conversation)
    assert "messages" in tokenize_conversation_sig.parameters
    assert "tokenizer" in tokenize_conversation_sig.parameters
    assert "add_generation_prompt" in tokenize_conversation_sig.parameters

    from src.experiment1.vila_execution import VILA_PINNED_COMMIT

    assert VILA_PINNED_COMMIT == PINNED_COMMIT
