# tests/test_agent.py

import os
import sys
import importlib
import pytest
from pathlib import Path

import torch
from peft import PeftModel
import transformers

# We want a small model to keep downloads faster in slow tests
SMALL_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


@pytest.mark.slow
def test_generate_solution_runs_without_error(tmp_path):
    # This will actually download a small model and tokenizer
    from agent import LoraHuggingFaceAgent

    agent = LoraHuggingFaceAgent(
        model_name=SMALL_MODEL,
        generation_config={"max_length": 20},
        device="cpu",
    )
    prompt = "print('hello world')"
    out = agent.generate_solution(prompt)
    assert isinstance(out, str)
    # Should at least contain part of the prompt or not crash
    assert "hello" in out.lower()


@pytest.mark.slow
def test_initialize_with_adapter_path(monkeypatch, tmp_path):
    # Create a dummy adapter folder
    adapter_dir = tmp_path / "mock_adapter"
    adapter_dir.mkdir()

    # Monkey-patch PeftModel.from_pretrained so it won't actually try to load anything
    called = {}

    def fake_from_pretrained(model, path, is_trainable=False):
        called["path"] = path
        return model

    monkeypatch.setattr(PeftModel, "from_pretrained", fake_from_pretrained)

    # Now initialize agent with adapter_path
    from agent import LoraHuggingFaceAgent

    agent = LoraHuggingFaceAgent(
        model_name=SMALL_MODEL,
        adapter_path=str(adapter_dir),
        device="cpu",
    )
    # ensure our fake loader was called with exactly that path
    assert called.get("path") == str(adapter_dir)


def test_get_base_model_strips_peft_wrapper():
    from agent import LoraHuggingFaceAgent

    # create an agent (will download tokenizer/model, mark as slow if needed)
    agent = LoraHuggingFaceAgent(
        model_name=SMALL_MODEL,
        device="cpu",
    )

    # replace its model with a dummy having get_base_model()
    class DummyPEFT:
        def get_base_model(self):
            return "base-model"

    agent.model = DummyPEFT()
    agent._get_base_model()
    assert agent.model == "base-model"
