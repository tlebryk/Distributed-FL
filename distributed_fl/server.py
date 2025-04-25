# server.py
import io
import queue
import threading
import os
import tempfile
import time
import zlib
from concurrent import futures

import grpc
import model_update_pb2
import model_update_pb2_grpc
import torch
from eval_script import evaluate
from benchmark import HumanEvalBenchmark
from agent import LoraHuggingFaceAgent
from safetensors.torch import load_file
from logger import get_logger

logger = get_logger(__name__)

PATH_TO_ADAPTERS = "./distributed_fl/adapters"


class FederatedLearningServiceServicer(
    model_update_pb2_grpc.FederatedLearningServiceServicer
):
    def __init__(self, mode="test"):
        # Store received adapter updates
        self.updates = []
        self.global_adapter_state = None  # Aggregated adapter weights
        self.version = 1

        # New attributes for hybrid approach
        self.connected_clients = {}  # {client_id: {version, last_seen}}
        self.update_lock = threading.Lock()  # For thread safety

        # Map of client_id to notification queues for update streaming
        self.notification_queues = {}  # {client_id: Queue()}
        self.benchmark = HumanEvalBenchmark()
        self.benchmark.load_dataset()

        if mode == "test":
            self.benchmark.dataset = self.benchmark.dataset.select(range(3))

    @staticmethod
    def _load_safetensors_from_bytes(raw: bytes):
        # save the buffer to a file then read from teh file
        # def tensors_from_bytes_tmp(raw: bytes):
        # TODO: make this persistent, save as my client version of this.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name  # close + flush
        try:
            return load_file(tmp_path)  # dict[str, torch.Tensor]
        finally:
            os.remove(tmp_path)

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
            client_adapter_path = os.path.join(
                PATH_TO_ADAPTERS, "clients", client_id, client_version
            )
            adapter_update = self._load_safetensors_from_bytes(request.update)

            logger.info(
                f"Received adapter update from {request.client_id} (version: {request.version})."
            )

            # Update the client's tracked version
            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = client_version
                self.connected_clients[client_id]["last_seen"] = time.time()

            self.updates.append(adapter_update)

            # Aggregate once two or more updates are received (for testing)
            with self.update_lock:
                if len(self.updates) >= 1:  # 2
                    aggregated_state = {}
                    # Assume all updates have matching keys
                    for key in self.updates[0].keys():
                        aggregated_state[key] = sum(
                            update[key] for update in self.updates
                        ) / len(self.updates)
                    self.global_adapter_state = aggregated_state
                    self.version += 1
                    logger.info("Aggregated global adapter state updated.")
                    self.updates = []  # Reset for the next round

                    # Notify all subscribed clients of the new model
                    self._notify_clients_of_update()

            return model_update_pb2.Acknowledgement(
                success=True, message="Adapter update received."
            )
        except Exception as e:
            # log traceback
            import traceback

            traceback.print_exc()
            logger.info("Error in SubmitUpdate:", e)
            return model_update_pb2.Acknowledgement(success=False, message=str(e))

    def GetAggregatedModel(self, request, context):
        client_id = request.client_id

        # Update client's last seen timestamp
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

        if self.global_adapter_state is None:
            logger.info("No aggregated adapter state available yet.")
            return model_update_pb2.AggregatedModel(
                model_state=b"", version=self.version
            )

        try:
            # Serialize and compress the aggregated adapter state
            buffer = io.BytesIO()
            # TODO: retry logic
            # implement eval loop and send if good update

            code_agent = LoraHuggingFaceAgent(
                model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                adapter_path="./distributed_fl/adapters/central/latest",
            )
            # TODO: integrate latest adapter...
            result = evaluate(
                code_agent, self.benchmark, results_csv="experiments.csv", mode="prod"
            )
            if result:
                torch.save(self.global_adapter_state, buffer)
                serialized = buffer.getvalue()
                compressed = zlib.compress(serialized)
                logger.info(
                    f"Sending aggregated adapter state (version: {self.version}) to {request.client_id}."
                )

                # Update client's tracked version
                if client_id in self.connected_clients:
                    self.connected_clients[client_id]["version"] = self.version
                return model_update_pb2.AggregatedModel(
                    model_state=compressed, version=self.version
                )
            else:
                context.set_details("Bad update")
                context.set_code(grpc.StatusCode.INTERNAL)
                model_update_pb2.AggregatedModel()
        except Exception as e:
            logger.info("Error in GetAggregatedModel:", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_update_pb2.AggregatedModel()

    # New method for client connection
    def ConnectClient(self, request, context):
        client_id = request.client_id
        client_version = request.current_version

        logger.info(f"Client {client_id} connected with version {client_version}")

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
                logger.info(
                    f"Sent latest model version {self.version} to client {client_id}"
                )
            except Exception as e:
                logger.info(f"Error preparing model for client {client_id}: {e}")

        return response

    # New method for update notifications
    def SubscribeToUpdates(self, request, context):
        client_id = request.client_id
        logger.info(f"Client {client_id} subscribed to update notifications")

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
            logger.info(f"Error in subscription stream for client {client_id}: {e}")
        finally:
            # Clean up when client disconnects
            logger.info(f"Client {client_id} unsubscribed from updates")

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
                logger.info(
                    f"Queued notification for client {client_id} about new model version {self.version}"
                )
            except queue.Full:
                logger.info(f"Notification queue full for client {client_id}")
            except Exception as e:
                logger.info(f"Error notifying client {client_id}: {e}")


def cleanup_disconnected_clients(servicer):
    """Remove clients that haven't been seen for more than 5 minutes"""
    current_time = time.time()
    timeout = 300  # 5 minutes

    clients_to_remove = []
    for client_id, info in servicer.connected_clients.items():
        if current_time - info.get("last_seen", 0) > timeout:
            clients_to_remove.append(client_id)

    for client_id in clients_to_remove:
        logger.info(f"Removing inactive client: {client_id}")
        if client_id in servicer.connected_clients:
            del servicer.connected_clients[client_id]
            # Also clean up any notification queues
            if client_id in servicer.notification_queues:
                del servicer.notification_queues[client_id]
            logger.info(f"Removed inactive client: {client_id}")


def serve():
    servicer = FederatedLearningServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_update_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("Server started on port 50051.")

    # Start a background thread for periodic client cleanup
    def cleanup_thread():
        while True:
            try:
                cleanup_disconnected_clients(servicer)
                time.sleep(60)  # Run cleanup every minute
            except Exception as e:
                logger.info(f"Error in cleanup thread: {e}")

    cleanup_task = threading.Thread(target=cleanup_thread, daemon=True)
    cleanup_task.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
