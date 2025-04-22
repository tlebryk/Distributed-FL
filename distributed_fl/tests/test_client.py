# test_client.py
import io
import time
import zlib
import pytest
import torch
import grpc
import unittest.mock as mock
from transformers import GPT2LMHeadModel
from client import serialize_state_dict, get_model_update, run
import model_update_pb2
import model_update_pb2_grpc


@pytest.fixture
def gpt2_model():
    """Fixture to provide a GPT2 model for testing."""
    return GPT2LMHeadModel.from_pretrained("gpt2")


@pytest.fixture
def mock_grpc_channel():
    """Fixture to mock the gRPC channel."""
    with mock.patch("grpc.insecure_channel") as mock_channel:
        yield mock_channel


@pytest.fixture
def mock_stub():
    """Fixture to mock the FederatedLearningServiceStub."""
    stub = mock.Mock(spec=model_update_pb2_grpc.FederatedLearningServiceStub)
    return stub


class TestSerializeStateDict:
    """Tests for the state dict serialization functionality."""

    def test_serialize_state_dict(self, gpt2_model):
        """Test that a model state dict can be serialized."""
        state_dict = gpt2_model.state_dict()
        serialized = serialize_state_dict(state_dict)

        # Verify the output is bytes
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        # Verify the serialized data can be deserialized back
        buffer = io.BytesIO(serialized)
        deserialized_state_dict = torch.load(buffer)

        # Check keys match
        assert set(deserialized_state_dict.keys()) == set(state_dict.keys())

        # Check some tensor values match
        for key in state_dict:
            assert torch.allclose(deserialized_state_dict[key], state_dict[key])


class TestGetModelUpdate:
    """Tests for the model update generation functionality."""

    def test_get_model_update(self, gpt2_model):
        """Test that a model update can be generated with noise applied."""
        # Get the original state dict for comparison
        original_state_dict = gpt2_model.state_dict()

        # Get the compressed update
        compressed_update = get_model_update(gpt2_model)

        # Verify the output is bytes and is compressed
        assert isinstance(compressed_update, bytes)

        # Decompress and deserialize
        decompressed = zlib.decompress(compressed_update)
        buffer = io.BytesIO(decompressed)
        updated_state_dict = torch.load(buffer)

        # Check keys match
        assert set(updated_state_dict.keys()) == set(original_state_dict.keys())

        # Verify that tensors are not identical (noise was added)
        for key in original_state_dict:
            assert not torch.allclose(
                updated_state_dict[key], original_state_dict[key], atol=1e-6
            )

            # But verify they're close (since we only add small noise)
            assert torch.allclose(
                updated_state_dict[key], original_state_dict[key], atol=0.01
            )


class TestClientRun:
    """Tests for the main client run function."""

    def test_run_successfully_sends_update(
        self, mock_grpc_channel, mock_stub, gpt2_model
    ):
        """Test that the client can successfully send an update to the server."""
        # Set up mocks
        mock_grpc_channel.return_value = mock.Mock()
        mock_grpc_channel.return_value.__enter__ = mock.Mock(
            return_value=mock_grpc_channel.return_value
        )
        mock_grpc_channel.return_value.__exit__ = mock.Mock(return_value=None)

        # Set up the stub mock to return an acknowledgment
        ack = model_update_pb2.Acknowledgement(success=True, message="Update received")
        mock_stub.SubmitUpdate.return_value = ack

        # Set up the mock for GetAggregatedModel
        model_state = serialize_state_dict(gpt2_model.state_dict())
        compressed_state = zlib.compress(model_state)
        aggregated_model = model_update_pb2.AggregatedModel(
            model_state=compressed_state, version=2
        )
        mock_stub.GetAggregatedModel.return_value = aggregated_model

        # Patch the stub creation
        with mock.patch(
            "model_update_pb2_grpc.FederatedLearningServiceStub", return_value=mock_stub
        ):
            # Patch time.sleep to avoid waiting
            with mock.patch("time.sleep"):
                # Patch the model loading to use our fixture
                with mock.patch(
                    "transformers.GPT2LMHeadModel.from_pretrained",
                    return_value=gpt2_model,
                ):
                    # Run the client
                    run()

        # Verify the stub was called with appropriate arguments
        mock_stub.SubmitUpdate.assert_called_once()
        update_msg = mock_stub.SubmitUpdate.call_args[0][0]
        assert update_msg.client_id == "client_1"
        assert isinstance(update_msg.update, bytes)
        assert update_msg.version == 1
        assert isinstance(update_msg.timestamp, int)

        # Verify GetAggregatedModel was called
        mock_stub.GetAggregatedModel.assert_called_once()
        request = mock_stub.GetAggregatedModel.call_args[0][0]
        assert request.client_id == "client_1"
        assert request.current_version == 1

    def test_run_handles_missing_global_model(
        self, mock_grpc_channel, mock_stub, gpt2_model
    ):
        """Test that the client handles the case when the global model is not yet updated."""
        # Set up mocks
        mock_grpc_channel.return_value = mock.Mock()
        mock_grpc_channel.return_value.__enter__ = mock.Mock(
            return_value=mock_grpc_channel.return_value
        )
        mock_grpc_channel.return_value.__exit__ = mock.Mock(return_value=None)

        # Set up the stub mock to return an acknowledgment
        ack = model_update_pb2.Acknowledgement(success=True, message="Update received")
        mock_stub.SubmitUpdate.return_value = ack

        # Set up the mock for GetAggregatedModel with empty model_state
        aggregated_model = model_update_pb2.AggregatedModel(
            model_state=b"", version=1  # Empty bytes
        )
        mock_stub.GetAggregatedModel.return_value = aggregated_model

        # Patch the stub creation
        with mock.patch(
            "model_update_pb2_grpc.FederatedLearningServiceStub", return_value=mock_stub
        ):
            # Patch time.sleep to avoid waiting
            with mock.patch("time.sleep"):
                # Patch the model loading to use our fixture
                with mock.patch(
                    "transformers.GPT2LMHeadModel.from_pretrained",
                    return_value=gpt2_model,
                ):
                    # Capture stdout to verify the output
                    with mock.patch("builtins.print") as mock_print:
                        # Run the client
                        run()

                        # Verify the appropriate message was printed
                        mock_print.assert_any_call("Global model not updated yet.")

    def test_run_handles_grpc_error(self, mock_grpc_channel, mock_stub, gpt2_model):
        """Test that the client handles gRPC errors gracefully."""
        # Set up mocks
        mock_grpc_channel.return_value = mock.Mock()
        mock_grpc_channel.return_value.__enter__ = mock.Mock(
            return_value=mock_grpc_channel.return_value
        )
        mock_grpc_channel.return_value.__exit__ = mock.Mock(return_value=None)

        # Make SubmitUpdate raise a gRPC error
        mock_stub.SubmitUpdate.side_effect = grpc.RpcError("Connection failed")

        # Patch the stub creation
        with mock.patch(
            "model_update_pb2_grpc.FederatedLearningServiceStub", return_value=mock_stub
        ):
            # Patch time.sleep to avoid waiting
            with mock.patch("time.sleep"):
                # Patch the model loading to use our fixture
                with mock.patch(
                    "transformers.GPT2LMHeadModel.from_pretrained",
                    return_value=gpt2_model,
                ):
                    # Test that an exception is raised
                    with pytest.raises(grpc.RpcError):
                        run()

                    # Verify that SubmitUpdate was called
                    mock_stub.SubmitUpdate.assert_called_once()
