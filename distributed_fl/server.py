# server.py
import io
import queue
import threading
import os
import tempfile
import time
from typing import NamedTuple, Dict, Any
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
from kazoo.client import KazooClient
from utils import find_latest_adapter_version, load_safetensors_from_bytes
from safetensors.torch import save_file
import shutil


logger = get_logger(__name__)

PATH_TO_ADAPTERS = "./distributed_fl/adapters"


# @dataclass
# class ClientUpdate:
#     client_id: str
#     version: int
#     weight: float
#     adapter_state: Dict[str, Any]  # your tensor dict


class DecodedModelUpdate(NamedTuple):
    client_id: str
    update: Dict[str, Any]
    version: int
    timestamp: int
    pylint_score: float


class FederatedLearningServiceServicer(
    model_update_pb2_grpc.FederatedLearningServiceServicer
):
    def __init__(self, mode="test", zk_hosts="127.0.0.1:2181"):
        # Store received adapter updates
        self.update_requests = []
        self.global_adapter_state = None  # Aggregated adapter weights
        self.version = 0

        # New attributes for hybrid approach
        self.connected_clients = {}  # {client_id: {version, last_seen}}
        self.update_lock = threading.Lock()  # For thread safety

        # Map of client_id to notification queues for update streaming
        self.notification_queues = {}  # {client_id: Queue()}
        self.benchmark = HumanEvalBenchmark()

        if mode == "test":
            self.benchmark.dataset = self.benchmark.dataset.select(range(2))
        try:
            self.zk = KazooClient(hosts=zk_hosts)
            self.zk.start()
        except:
            self.zk = None

    def update_client_weight(self, client_id, weight):

        # 3. Ensure the parent path exists
        self.zk.ensure_path("/myapp/clients")

        # 4. Create or update a znode for client_id → weight
        # client_id = "client123"
        # weight = 0.42

        path = f"/myapp/clients/{client_id}"
        data = str(weight).encode("utf-8")

        if self.zk.exists(path):
            self.zk.set(path, data)
        else:
            self.zk.create(path, data, makepath=True)

        print(f"Set {path} = {weight}")

    def get_client_weight(self, client_id):

        path = f"/myapp/clients/{client_id}"
        weight = 0.5
        if self.zk.exists(path):
            raw, stat = self.zk.get(path)
            weight = float(raw.decode("utf-8"))
            print(f"Weight for {client_id}: {weight}")
        else:
            print(f"No entry for {client_id}")
            self.update_client_weight(client_id, weight)

        return weight

    def weighted_average(self, use_pylint=True):
        aggregated_state = {}
        for key in self.update_requests[0].update.keys():
            aggregated_state[key] = 0
            total_weight = 0
            for update_requests in self.update_requests:
                # allow some zk tolerance
                if self.zk is not None:
                    weight = self.get_client_weight(update_requests.client_id)
                else:
                    weight = 1
                if use_pylint:
                    logger.info(f"Weight for {update_requests.client_id}: {weight}")
                    logger.info(
                        f"pylint_score for {update_requests.client_id}: {update_requests.pylint_score}"
                    )
                    weight *= (update_requests.pylint_score + 0.01) / 10
                aggregated_state[key] += update_requests.update[key] * weight
                total_weight += weight

            aggregated_state[key] /= total_weight
        return aggregated_state

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
            decoded_dict = load_safetensors_from_bytes(request.update)
            decoded_msg = DecodedModelUpdate(
                client_id=request.client_id,
                update=decoded_dict,
                version=request.version,
                timestamp=request.timestamp,
                pylint_score=request.pylint_score,
            )

            logger.info(
                f"Received adapter update from {request.client_id} (version: {request.version})."
            )

            # Update the client's tracked version
            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = client_version
                self.connected_clients[client_id]["last_seen"] = time.time()

            # Add to update requests with the lock, but return immediately
            with self.update_lock:
                self.update_requests.append(decoded_msg)
                should_aggregate = len(self.update_requests) >= 2

            # Immediately return acknowledgment to client
            ack = model_update_pb2.Acknowledgement(
                success=True, message="Adapter update received."
            )

            # Kick off aggregation in a separate thread if we have enough updates
            if should_aggregate:
                threading.Thread(
                    target=self._perform_aggregation_and_evaluation, daemon=True
                ).start()

            return ack
        except Exception as e:
            # log traceback
            import traceback

            traceback.print_exc()
            logger.info("Error in SubmitUpdate:", e)
            return model_update_pb2.Acknowledgement(success=False, message=str(e))

    def _perform_aggregation_and_evaluation(self):
        """Background thread to perform aggregation and evaluation."""
        try:
            # TODO: this lock too long... probably could split into two locks?
            with self.update_lock:
                # Check if another thread already processed these updates
                if len(self.update_requests) < 2:
                    return

                # Perform the aggregation
                aggregated_state = self.weighted_average()
                self.update_requests = []  # Reset for the next round
                # Assume all updates have matching keys
                self.global_adapter_state = aggregated_state

                logger.info("Aggregated global adapter state updated.")
                round_path = os.path.join(
                    PATH_TO_ADAPTERS,
                    "server",
                    "rounds",
                )
                latest_round = find_latest_adapter_version(round_path)
                updated_round = latest_round + 1
                output_dir = os.path.join(round_path, f"v{updated_round}")
                os.makedirs(output_dir, exist_ok=True)
                save_file(
                    self.global_adapter_state,
                    os.path.join(output_dir, "adapter_model.safetensors"),
                )
                shutil.copy2(
                    os.path.join(PATH_TO_ADAPTERS, "adapter_config.json"),
                    os.path.join(output_dir, "adapter_config.json"),
                )

                code_agent = LoraHuggingFaceAgent(
                    model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    adapter_path=output_dir,
                )
                result = evaluate(
                    code_agent,
                    self.benchmark,
                    results_csv="experiments.csv",
                    mode="prod",
                )
                # TODO: retry logic
                # implement eval loop and send if good update
                if result:
                    main_path = os.path.join(PATH_TO_ADAPTERS, "server", "central")
                    # get latest version
                    latest_version = find_latest_adapter_version(main_path)
                    updated_version = latest_version + 1
                    output_dir = os.path.join(main_path, f"v{updated_version}")
                    os.makedirs(output_dir, exist_ok=True)
                    save_file(
                        self.global_adapter_state,
                        os.path.join(output_dir, "adapter_model.safetensors"),
                    )
                    self.version = updated_version
                    # Notify all subscribed clients of the new model
                    self._notify_clients_of_update()
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.info(f"Error in background aggregation: {e}")

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
            latest_model_dir = os.path.join(
                PATH_TO_ADAPTERS, "server", "central", f"v{self.version}"
            )
            with open(
                os.path.join(latest_model_dir, "adapter_model.safetensors"), "rb"
            ) as f:
                bytes_ = f.read()

            # Update client's tracked version
            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = self.version
            return model_update_pb2.AggregatedModel(
                model_state=bytes_, version=self.version
            )
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
            latest_model_dir = os.path.join(
                PATH_TO_ADAPTERS, "server", "central", f"v{self.version}"
            )
            with open(
                os.path.join(latest_model_dir, "adapter_model.safetensors"), "rb"
            ) as f:
                bytes_ = f.read()
            response.model_state = bytes_

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
        server.zk.stop()
        server.zk.close()


if __name__ == "__main__":
    serve()
