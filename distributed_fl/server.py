# server.py
import io
import queue
import threading
import time
import zlib
from concurrent import futures

import grpc
import model_update_pb2
import model_update_pb2_grpc
import torch


class FederatedLearningServiceServicer(
    model_update_pb2_grpc.FederatedLearningServiceServicer
):
    def __init__(self):
        # Store received adapter updates
        self.updates = []
        self.global_adapter_state = None  # Aggregated adapter weights
        self.version = 1

        # New attributes for hybrid approach
        self.connected_clients = {}  # {client_id: {version, last_seen}}
        self.update_lock = threading.Lock()  # For thread safety

        # Map of client_id to notification queues for update streaming
        self.notification_queues = {}  # {client_id: Queue()}

    def SubmitUpdate(self, request, context):
        client_id = request.client_id
        client_version = request.version

        # Check if client is submitting an update with outdated version
        if self.global_adapter_state is not None and client_version < self.version:
            return model_update_pb2.Acknowledgement(
                success=False,
                message=f"Client has outdated model (client: {client_version}, latest: {self.version})",
            )

        try:
            # Decompress and deserialize the adapter update
            decompressed = zlib.decompress(request.update)
            buffer = io.BytesIO(decompressed)
            adapter_update = torch.load(buffer)
            print(
                f"Received adapter update from {request.client_id} (version: {request.version})."
            )

            # Update the client's tracked version
            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = client_version
                self.connected_clients[client_id]["last_seen"] = time.time()

            self.updates.append(adapter_update)

            # Aggregate once two or more updates are received (for testing)
            with self.update_lock:
                if len(self.updates) >= 2:
                    aggregated_state = {}
                    # Assume all updates have matching keys
                    for key in self.updates[0].keys():
                        aggregated_state[key] = sum(
                            update[key] for update in self.updates
                        ) / len(self.updates)
                    self.global_adapter_state = aggregated_state
                    self.version += 1
                    print("Aggregated global adapter state updated.")
                    self.updates = []  # Reset for the next round

                    # Notify all subscribed clients of the new model
                    self._notify_clients_of_update()

            return model_update_pb2.Acknowledgement(
                success=True, message="Adapter update received."
            )
        except Exception as e:
            print("Error in SubmitUpdate:", e)
            return model_update_pb2.Acknowledgement(success=False, message=str(e))

    def GetAggregatedModel(self, request, context):
        client_id = request.client_id

        # Update client's last seen timestamp
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

        if self.global_adapter_state is None:
            print("No aggregated adapter state available yet.")
            return model_update_pb2.AggregatedModel(
                model_state=b"", version=self.version
            )

        try:
            # Serialize and compress the aggregated adapter state
            buffer = io.BytesIO()
            torch.save(self.global_adapter_state, buffer)
            serialized = buffer.getvalue()
            compressed = zlib.compress(serialized)
            print(
                f"Sending aggregated adapter state (version: {self.version}) to {request.client_id}."
            )

            # Update client's tracked version
            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = self.version

            return model_update_pb2.AggregatedModel(
                model_state=compressed, version=self.version
            )
        except Exception as e:
            print("Error in GetAggregatedModel:", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_update_pb2.AggregatedModel()

    # New method for client connection
    def ConnectClient(self, request, context):
        client_id = request.client_id
        client_version = request.current_version

        print(f"Client {client_id} connected with version {client_version}")

        # Track client connection
        self.connected_clients[client_id] = {
            "version": client_version,
            "last_seen": time.time(),
            "ready_for_training": request.ready_for_training,
        }

        # Check if client needs an update
        update_available = (
            self.global_adapter_state is not None and client_version < self.version
        )

        response = model_update_pb2.ModelVersionInfo(
            update_available=update_available, latest_version=self.version
        )

        # Automatically send model if update is available
        if update_available:
            try:
                # Serialize and compress the aggregated adapter state
                buffer = io.BytesIO()
                torch.save(self.global_adapter_state, buffer)
                serialized = buffer.getvalue()
                compressed = zlib.compress(serialized)
                response.model_state = compressed
                print(f"Sent latest model version {self.version} to client {client_id}")
            except Exception as e:
                print(f"Error preparing model for client {client_id}: {e}")

        return response

    # New method for update notifications
    def SubscribeToUpdates(self, request, context):
        client_id = request.client_id
        print(f"Client {client_id} subscribed to update notifications")

        # Create a queue for this client if it doesn't exist
        if client_id not in self.notification_queues:
            self.notification_queues[client_id] = queue.Queue()

        # Update client's last seen timestamp
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()
        else:
            self.connected_clients[client_id] = {
                "version": request.current_version,
                "last_seen": time.time(),
            }

        # Send initial notification if needed
        if self.global_adapter_state is not None:
            yield model_update_pb2.UpdateNotification(
                new_version=self.version, update_type="FULL"
            )

        # Process notifications from the queue
        try:
            while context.is_active():
                try:
                    # Non-blocking queue check with timeout
                    notification = self.notification_queues[client_id].get(
                        block=True, timeout=30
                    )
                    yield notification
                    self.notification_queues[client_id].task_done()
                except queue.Empty:
                    # Timeout occurred, update last_seen and continue
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]["last_seen"] = time.time()
                    # Send a ping to check if connection is still active
                    continue
        except Exception as e:
            print(f"Error in subscription stream for client {client_id}: {e}")
        finally:
            # Clean up when client disconnects
            print(f"Client {client_id} unsubscribed from updates")

    # Helper method to notify clients
    def _notify_clients_of_update(self):
        notification = model_update_pb2.UpdateNotification(
            new_version=self.version, update_type="FULL"
        )

        # Put the notification in each client's queue
        for client_id in list(self.notification_queues.keys()):
            try:
                # Use put_nowait to avoid blocking if a queue is full
                self.notification_queues[client_id].put_nowait(notification)
                print(
                    f"Queued notification for client {client_id} about new model version {self.version}"
                )
            except queue.Full:
                print(f"Notification queue full for client {client_id}")
            except Exception as e:
                print(f"Error notifying client {client_id}: {e}")


def cleanup_disconnected_clients(servicer):
    """Remove clients that haven't been seen for more than 5 minutes"""
    current_time = time.time()
    timeout = 300  # 5 minutes

    clients_to_remove = []
    for client_id, info in servicer.connected_clients.items():
        if current_time - info.get("last_seen", 0) > timeout:
            clients_to_remove.append(client_id)

    for client_id in clients_to_remove:
        print(f"Removing inactive client: {client_id}")
        if client_id in servicer.connected_clients:
            del servicer.connected_clients[client_id]
            # Also clean up any notification queues
            if client_id in servicer.notification_queues:
                del servicer.notification_queues[client_id]
            print(f"Removed inactive client: {client_id}")


def serve():
    servicer = FederatedLearningServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_update_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051.")

    # Start a background thread for periodic client cleanup
    def cleanup_thread():
        while True:
            try:
                cleanup_disconnected_clients(servicer)
                time.sleep(60)  # Run cleanup every minute
            except Exception as e:
                print(f"Error in cleanup thread: {e}")

    cleanup_task = threading.Thread(target=cleanup_thread, daemon=True)
    cleanup_task.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        print("Server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
