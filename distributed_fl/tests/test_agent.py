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


@pytest.mark.slow
def test_load_latest_adapter_uses_env_and_central_latest(monkeypatch):
    # Point PATH_TO_ADAPTERS at our tmp structure
    adapters_root = Path("/tmp/fake_adapters")
    central_latest = adapters_root / "central" / "latest"
    central_latest.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PATH_TO_ADAPTERS", str(adapters_root))
    # force agent to re-import so it picks up new PATH_TO_ADAPTERS
    if "agent" in sys.modules:
        del sys.modules["agent"]
    import agent  # noqa: F401

    importlib.reload(agent)

    # stub out PeftModel.from_pretrained again
    called = {}

    def fake_from_pretrained(model, path, is_trainable=False):
        called["path"] = path
        return model

    monkeypatch.setattr(agent.PeftModel, "from_pretrained", fake_from_pretrained)

    # initialize and then call load_latest_adapter
    from agent import LoraHuggingFaceAgent

    agent = LoraHuggingFaceAgent(
        model_name=SMALL_MODEL,
        device="cpu",
    )
    agent.load_latest_adapter()
    expected = str(central_latest)
    assert called.get("path") == expected


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
