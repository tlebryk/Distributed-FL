# test_client.py
import io
import time
import zlib
import pytest
import grpc
from unittest import mock
import torch
from transformers import AutoModelForCausalLM
from client import FederatedClient, get_adapter_update
import model_update_pb2 as model_update_pb2
import model_update_pb2_grpc as model_update_pb2_grpc


@pytest.fixture
def mock_stub():
    """Fixture to mock the FederatedLearningServiceStub."""
    stub = mock.Mock(spec=model_update_pb2_grpc.FederatedLearningServiceStub)
    return stub


@pytest.fixture
def mock_channel():
    """Fixture to mock the gRPC channel."""
    channel = mock.Mock(spec=grpc.Channel)
    return channel


@pytest.fixture
def client(mock_stub, mock_channel):
    """Fixture to create a FederatedClient with mocked stub and channel."""
    with mock.patch("grpc.insecure_channel", return_value=mock_channel):
        with mock.patch(
            "distributed_fl.model_update_pb2_grpc.FederatedLearningServiceStub",
            return_value=mock_stub,
        ):
            client = FederatedClient(client_id="client_1")
            client.channel = mock_channel
            client.stub = mock_stub
            client.initialize_model()
            return client


def test_initialize_model(client):
    """Test that the model is initialized correctly."""
    assert client.model is not None
    assert isinstance(client.model, AutoModelForCausalLM)


def test_connect_to_server_success(client, mock_stub):
    """Test successful connection to the server without updates."""
    response = model_update_pb2.ConnectResponse(
        latest_version=1, update_available=False, model_state=b""
    )
    mock_stub.ConnectClient.return_value = response

    success = client.connect_to_server()
    assert success
    assert client.current_version == 1


def test_connect_to_server_with_update(client, mock_stub):
    """Test connection to the server with a model update."""
    # Create a dummy adapter state
    adapter_state = {"key": torch.tensor([1, 2, 3])}
    buffer = io.BytesIO()
    torch.save(adapter_state, buffer)
    compressed_state = zlib.compress(buffer.getvalue())

    response = model_update_pb2.ConnectResponse(
        latest_version=2, update_available=True, model_state=compressed_state
    )
    mock_stub.ConnectClient.return_value = response

    with mock.patch.object(client.model, "load_state_dict") as mock_load_state_dict:
        success = client.connect_to_server()
        assert success
        assert client.current_version == 2
        mock_load_state_dict.assert_called_with(adapter_state, strict=False)


def test_get_latest_model(client, mock_stub):
    """Test fetching the latest model from the server."""
    adapter_state = {"key": torch.tensor([4, 5, 6])}
    buffer = io.BytesIO()
    torch.save(adapter_state, buffer)
    compressed_state = zlib.compress(buffer.getvalue())

    response = model_update_pb2.AggregatedModel(version=3, model_state=compressed_state)
    mock_stub.GetAggregatedModel.return_value = response

    with mock.patch.object(client.model, "load_state_dict") as mock_load_state_dict:
        success = client.get_latest_model()
        assert success
        assert client.current_version == 3
        mock_load_state_dict.assert_called_with(adapter_state, strict=False)


def test_train_and_submit_success(client, mock_stub):
    """Test successful training and submission of an update."""
    ack = model_update_pb2.SubmitAck(success=True, message="Update received")
    mock_stub.SubmitUpdate.return_value = ack

    with mock.patch("time.sleep", return_value=None):
        success = client.train_and_submit()
        assert success


def test_train_and_submit_version_mismatch(client, mock_stub):
    """Test handling of version mismatch during submission."""
    ack = model_update_pb2.SubmitAck(
        success=False, message="Update rejected due to outdated model"
    )
    mock_stub.SubmitUpdate.return_value = ack

    with mock.patch.object(client, "get_latest_model") as mock_get_latest_model:
        with mock.patch("time.sleep", return_value=None):
            success = client.train_and_submit()
            assert not success
            mock_get_latest_model.assert_called()


def test_run_training_loop_interrupt(client):
    """Test graceful shutdown of the training loop."""
    with mock.patch.object(client, "train_and_submit", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit):
            client.run_training_loop()


def test_shutdown(client):
    """Test client shutdown procedure."""
    client.shutdown()
    assert not client.running
    client.channel.close.assert_called_once()
