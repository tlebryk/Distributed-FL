import pytest
import torch
from inference_server import app, shared_state


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.last_prompt = None

    def __call__(self, prompt, return_tensors):
        self.last_prompt = prompt
        # Return a dummy tensor for input_ids
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, skip_special_tokens=True):
        # Simulate decoding by prepending the stored prompt
        return self.last_prompt + " generated_text"


class DummyModel:
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature,
        top_p,
        num_return_sequences,
        pad_token_id,
        do_sample,
    ):
        # Return a dummy output tensor
        return torch.tensor([[0, 1, 2]])


class ErrorModel:
    def generate(self, *args, **kwargs):
        # Simulate an error during generation
        raise RuntimeError("test error")


@pytest.fixture(autouse=True)
def reset_state():
    """
    Reset shared_state before each test.
    """
    shared_state.model = None
    shared_state.tokenizer = None
    shared_state.is_training = False
    yield


@pytest.fixture
def client():
    """
    Provide a Flask test client for sending requests.
    """
    return app.test_client()


def test_generate_when_training(client):
    # Model is in training mode
    shared_state.is_training = True

    response = client.post("/generate", json={"prompt": "test"})
    assert response.status_code == 503
    assert (
        response.get_json()["error"]
        == "Model is currently training. Please try again later."
    )


def test_generate_model_not_initialized(client):
    # Model and tokenizer not set
    response = client.post("/generate", json={"prompt": "test"})
    assert response.status_code == 503
    assert response.get_json()["error"] == "Model not initialized yet"


def test_generate_missing_prompt_field(client):
    # Model and tokenizer initialized but missing prompt
    shared_state.model = DummyModel()
    shared_state.tokenizer = DummyTokenizer()

    response = client.post("/generate", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing 'prompt' field in request"


def test_generate_success(client):
    # Successful generation
    shared_state.model = DummyModel()
    shared_state.tokenizer = DummyTokenizer()

    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["generated_text"] == "generated_text"


def test_generate_exception(client):
    # Simulate an exception in model.generate
    shared_state.model = ErrorModel()
    shared_state.tokenizer = DummyTokenizer()

    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 500
    err = response.get_json()["error"]
    assert "Generation error" in err
