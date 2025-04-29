# # test_server.py
# import io
# import threading
# import time
# import zlib
# from unittest.mock import MagicMock, patch

# import grpc
# import pytest
# import torch

# import model_update_pb2
# from server import FederatedLearningServiceServicer, cleanup_disconnected_clients


# class TestFederatedLearningServiceServicer:
#     @pytest.fixture
#     def service(self):
#         return FederatedLearningServiceServicer()

#     @pytest.fixture
#     def mock_context(self):
#         context = MagicMock()
#         context.is_active.return_value = True
#         return context

#     @pytest.fixture
#     def sample_adapter_state(self):
#         """Create a simple mock adapter state for testing"""
#         return {
#             "lora_A.weight": torch.ones(10, 10),
#             "lora_B.weight": torch.ones(10, 10) * 2,
#         }

#     def create_model_update(self, client_id, adapter_state, version=1):
#         """Helper to create a model update request"""
#         buffer = io.BytesIO()
#         torch.save(adapter_state, buffer)
#         serialized = buffer.getvalue()
#         compressed = zlib.compress(serialized)

#         return model_update_pb2.ModelUpdate(
#             client_id=client_id,
#             update=compressed,
#             version=version,
#             timestamp=int(time.time()),
#         )

#     def test_submit_update(self, service, mock_context, sample_adapter_state):
#         # Test submitting a single update
#         request = self.create_model_update("client_1", sample_adapter_state)
#         response = service.SubmitUpdate(request, mock_context)

#         assert response.success
#         assert "received" in response.message.lower()
#         assert len(service.updates) == 1
#         assert (
#             service.global_adapter_state is None
#         )  # We need 2+ updates to trigger aggregation

#         # Submit a second update to trigger aggregation
#         request2 = self.create_model_update("client_2", sample_adapter_state)
#         response2 = service.SubmitUpdate(request2, mock_context)

#         assert response2.success
#         assert service.global_adapter_state is not None
#         assert (
#             service.version == 2
#         )  # Initial version was 1, incremented after aggregation
#         assert (
#             len(service.updates) == 0
#         )  # Updates list should be reset after aggregation

#     def test_submit_update_with_outdated_version(
#         self, service, mock_context, sample_adapter_state
#     ):
#         # First, create an initial aggregated state
#         service.global_adapter_state = sample_adapter_state
#         service.version = 5

#         # Now submit an update with outdated version
#         request = self.create_model_update("client_1", sample_adapter_state, version=3)
#         response = service.SubmitUpdate(request, mock_context)

#         assert not response.success
#         assert "outdated" in response.message.lower()

#     def test_get_aggregated_model(self, service, mock_context, sample_adapter_state):
#         # Set up an aggregated model
#         service.global_adapter_state = sample_adapter_state
#         service.version = 3

#         # Test getting the model
#         request = model_update_pb2.ClientRequest(
#             client_id="client_1", current_version=1
#         )
#         response = service.GetAggregatedModel(request, mock_context)

#         assert response.version == 3
#         assert len(response.model_state) > 0

#         # Test that we can deserialize the response
#         decompressed = zlib.decompress(response.model_state)
#         buffer = io.BytesIO(decompressed)
#         loaded_state = torch.load(buffer)

#         # Check that the loaded state matches the original
#         assert set(loaded_state.keys()) == set(sample_adapter_state.keys())
#         for key in sample_adapter_state:
#             assert torch.all(loaded_state[key] == sample_adapter_state[key])

#     def test_get_aggregated_model_when_none_available(self, service, mock_context):
#         # Test getting model when none exists yet
#         request = model_update_pb2.ClientRequest(
#             client_id="client_1", current_version=1
#         )
#         response = service.GetAggregatedModel(request, mock_context)

#         assert response.version == 1  # Default version
#         assert response.model_state == b""  # Empty bytes when no model

#     def test_connect_client(self, service, mock_context, sample_adapter_state):
#         # Test connecting a client with no update available
#         request = model_update_pb2.ClientConnection(
#             client_id="client_1", current_version=1, ready_for_training=True
#         )
#         response = service.ConnectClient(request, mock_context)

#         assert not response.update_available
#         assert response.latest_version == 1
#         assert "client_1" in service.connected_clients

#         # Test connecting a client when update is available
#         service.global_adapter_state = sample_adapter_state
#         service.version = 2

#         request = model_update_pb2.ClientConnection(
#             client_id="client_2", current_version=1, ready_for_training=True
#         )
#         response = service.ConnectClient(request, mock_context)

#         assert response.update_available
#         assert response.latest_version == 2
#         assert len(response.model_state) > 0
#         assert "client_2" in service.connected_clients

#     def test_subscribe_to_updates(self, service, mock_context):
#         # Test subscription setup
#         request = model_update_pb2.ClientRequest(
#             client_id="client_1", current_version=1
#         )

#         # Mock the generator behavior
#         def mock_updates_generator():
#             # Return a single notification
#             notification = model_update_pb2.UpdateNotification(
#                 new_version=1, update_type="FULL"
#             )
#             yield notification
#             # Then simulate connection close by raising StopIteration
#             raise StopIteration()

#         # Replace the generator function with our mock
#         with patch.object(
#             service, "SubscribeToUpdates", return_value=mock_updates_generator()
#         ):
#             generator = service.SubscribeToUpdates(request, mock_context)
#             notifications = list(generator)

#             assert len(notifications) == 1
#             assert notifications[0].new_version == 1
#             assert notifications[0].update_type == "FULL"

#     def test_notify_clients_of_update(self, service):
#         # Setup clients with notification queues
#         service.notification_queues = {"client_1": MagicMock(), "client_2": MagicMock()}
#         service.version = 3

#         # Test notification
#         service._notify_clients_of_update()

#         # Check that notifications were queued for both clients
#         for client_id, queue in service.notification_queues.items():
#             queue.put_nowait.assert_called_once()
#             # Get the notification argument
#             args, _ = queue.put_nowait.call_args
#             notification = args[0]
#             assert notification.new_version == 3
#             assert notification.update_type == "FULL"

#     def test_cleanup_disconnected_clients(self, service):
#         # Setup clients with different last_seen times
#         current_time = time.time()
#         service.connected_clients = {
#             "active_client": {"last_seen": current_time, "version": 1},
#             "inactive_client": {
#                 "last_seen": current_time - 400,
#                 "version": 1,
#             },  # 400s ago (> 300s timeout)
#         }
#         service.notification_queues = {
#             "active_client": MagicMock(),
#             "inactive_client": MagicMock(),
#         }

#         # Run cleanup
#         cleanup_disconnected_clients(service)

#         # Check that inactive client was removed
#         assert "active_client" in service.connected_clients
#         assert "inactive_client" not in service.connected_clients
#         assert "active_client" in service.notification_queues
#         assert "inactive_client" not in service.notification_queues

#     def test_aggregation_logic(self, service, mock_context, sample_adapter_state):
#         """Test that model aggregation is performed correctly"""
#         # Create two slightly different adapter states
#         adapter_state1 = {
#             "lora_A.weight": torch.ones(10, 10),
#             "lora_B.weight": torch.ones(10, 10) * 2,
#         }

#         adapter_state2 = {
#             "lora_A.weight": torch.ones(10, 10) * 3,
#             "lora_B.weight": torch.ones(10, 10) * 4,
#         }

#         # Submit both updates
#         request1 = self.create_model_update("client_1", adapter_state1)
#         service.SubmitUpdate(request1, mock_context)

#         request2 = self.create_model_update("client_2", adapter_state2)
#         service.SubmitUpdate(request2, mock_context)

#         # Check that aggregation happened
#         assert service.global_adapter_state is not None

#         # Check that aggregation is the average of the two updates
#         for key in adapter_state1:
#             expected_avg = (adapter_state1[key] + adapter_state2[key]) / 2
#             assert torch.all(
#                 torch.isclose(service.global_adapter_state[key], expected_avg)
#             )


# # Test StreamInterceptor for gRPC
# class TestStreamContext:
#     @pytest.fixture
#     def stream_context(self):
#         context = MagicMock()
#         # Setup the context to simulate active for some time
#         context.is_active.side_effect = [True] * 5 + [False]
#         return context

#     def test_subscribe_to_updates_stream_termination(self, stream_context):
#         service = FederatedLearningServiceServicer()
#         client_id = "test_client"
#         service.notification_queues[client_id] = MagicMock()
#         service.notification_queues[client_id].get.side_effect = Exception(
#             "Test interruption"
#         )

#         request = model_update_pb2.ClientRequest(client_id=client_id, current_version=1)

#         # Test that the subscription stream handles termination gracefully
#         notifications = list(service.SubscribeToUpdates(request, stream_context))

#         # We should get 0 notifications because of the exception
#         assert len(notifications) == 0

#         # The client should still be in the connected clients list
#         assert client_id in service.connected_clients
