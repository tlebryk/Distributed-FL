import pytest
import torch
from inference_server import app, shared_state


# --- Dummy components for testing ---
class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self._last_prompt = None

    def __call__(self, prompt, return_tensors=None):
        # Store prompt for decode
        self._last_prompt = prompt
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, token_ids, skip_special_tokens=False):
        # Return prompt + suffix
        return self._last_prompt + " world"


class DummyModel:
    def generate(self, input_ids, **kwargs):
        # Return dummy token sequence
        return torch.tensor([[1, 2, 3]])


class DummyAgent:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel()


# --- Pytest fixtures ---
@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    # Ensure is_training is False by default and agent is fresh per test
    monkeypatch.setattr(shared_state, "is_training", False)
    yield


@pytest.fixture
def client():
    # Flask test client
    with app.test_client() as client:
        yield client


# --- Tests ---


def test_when_training_then_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(shared_state, "is_training", True)
    response = client.post("/generate", json={"prompt": "test"})
    assert response.status_code == 503
    assert (
        response.get_json()["error"]
        == "Model is currently training. Please try again later."
    )


def test_model_not_initialized_yet(client, monkeypatch):
    # Agent missing model/tokenizer
    dummy = DummyAgent()
    dummy.model = None
    dummy.tokenizer = None
    monkeypatch.setattr(shared_state, "agent", dummy)
    response = client.post("/generate", json={"prompt": "test"})
    assert response.status_code == 503
    assert response.get_json()["error"] == "Model not initialized yet"


def test_missing_prompt_field(client, monkeypatch):
    # Valid agent
    monkeypatch.setattr(shared_state, "agent", DummyAgent())
    # Empty JSON
    response = client.post("/generate", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing 'prompt' field in request"
    # No JSON body
    response = client.post("/generate")
    assert response.status_code == 415


def test_generate_success_returns_generated_text(client, monkeypatch):
    dummy = DummyAgent()
    monkeypatch.setattr(shared_state, "agent", dummy)
    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 200
    data = response.get_json()
    # DummyTokenizer.decode returns 'hello world', so generated_only => 'world'
    assert data["generated_text"] == "world"


def test_generation_exception_returns_server_error(client, monkeypatch):
    # Agent whose generate throws
    class ErrorModel:
        def generate(self, *args, **kwargs):
            raise RuntimeError("boom")

    dummy = DummyAgent()
    dummy.model = ErrorModel()
    monkeypatch.setattr(shared_state, "agent", dummy)
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 500
    assert "Generation error: boom" in response.get_json()["error"]
