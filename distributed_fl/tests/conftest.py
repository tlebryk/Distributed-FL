# conftest.py
import pytest
from unittest.mock import MagicMock
import sys


# Mock classes to mimic the generated protobuf files for testing
class MockModelUpdate:
    def __init__(self, client_id="", update=b"", version=0, timestamp=0):
        self.client_id = client_id
        self.update = update
        self.version = version
        self.timestamp = timestamp


class MockAggregatedModel:
    def __init__(self, model_state=b"", version=0):
        self.model_state = model_state
        self.version = version


class MockAcknowledgement:
    def __init__(self, success=False, message=""):
        self.success = success
        self.message = message


class MockClientRequest:
    def __init__(self, client_id="", current_version=0):
        self.client_id = client_id
        self.current_version = current_version


class MockClientConnection:
    def __init__(self, client_id="", current_version=0, ready_for_training=False):
        self.client_id = client_id
        self.current_version = current_version
        self.ready_for_training = ready_for_training


class MockModelVersionInfo:
    def __init__(self, update_available=False, latest_version=0, model_state=b""):
        self.update_available = update_available
        self.latest_version = latest_version
        self.model_state = model_state


class MockUpdateNotification:
    def __init__(self, new_version=0, update_type=""):
        self.new_version = new_version
        self.update_type = update_type


# Create a pytest fixture to mock the model_update_pb2 module
@pytest.fixture(autouse=True)
def mock_model_update_pb2(monkeypatch):
    mock_module = MagicMock()

    # Assign our mock classes to the mock module
    mock_module.ModelUpdate = MockModelUpdate
    mock_module.AggregatedModel = MockAggregatedModel
    mock_module.Acknowledgement = MockAcknowledgement
    mock_module.ClientRequest = MockClientRequest
    mock_module.ClientConnection = MockClientConnection
    mock_module.ModelVersionInfo = MockModelVersionInfo
    mock_module.UpdateNotification = MockUpdateNotification

    # Patch the module
    sys.modules["model_update_pb2"] = mock_module

    return mock_module
