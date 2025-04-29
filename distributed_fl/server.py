# server.py
import io
import queue
import threading
import os
import tempfile
import time
import socket  # ← NEW
from typing import NamedTuple, Dict, Any
import zlib
import json
from concurrent import futures
import traceback

import grpc
import torch
from eval_script import evaluate
from benchmark import HumanEvalBenchmark
from agent import LoraHuggingFaceAgent
from safetensors.torch import load_file
from logger import get_logger
from kazoo.client import KazooClient
from kazoo.retry import KazooRetry
from kazoo.exceptions import NoNodeError  # ← NEW
from peft import PeftModel, PeftConfig

import model_update_pb2
import model_update_pb2_grpc
from safetensors.torch import save_file

from utils import load_safetensors_from_bytes

logger = get_logger(__name__)

PATH_TO_ADAPTERS = os.environ.get("PATH_TO_ADAPTERS", "./distributed_fl/adapters")

# ---------------------------------------------------------------------------
# Leader / replica constants  (NEW)
# ---------------------------------------------------------------------------
LEADER_ELECTION_ROOT = "/fl/servers"  # parent path for election znodes
META_VERSION_ZNODE = "/fl/meta/version"  # current aggregated version
SYNC_POLL_SEC = 3  # replica polling cadence (seconds)
# ---------------------------------------------------------------------------


class DecodedModelUpdate(NamedTuple):
    client_id: str
    update: Dict[str, Any]
    version: int
    timestamp: int


class FederatedLearningServiceServicer(
    model_update_pb2_grpc.FederatedLearningServiceServicer
):
    def __init__(self, mode="test", zk_hosts="127.0.0.1:2181"):
        # Store received adapter updates
        self.update_requests = []
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

        self.host = socket.gethostname()  # ← NEW
        self.is_leader = False  # ← NEW

        # Connect to ZooKeeper (existing logic)
        try:
            print(f"trying to connect to {zk_hosts}")
            retry = KazooRetry(
                max_tries=1,
                delay=1.0,
                backoff=2,
                max_delay=1,
            )
            self.zk = KazooClient(
                hosts=zk_hosts,
                command_retry=retry,
                connection_retry=retry,
            )
            self.zk.start()
            logger.info(f"Connected to Zookeeper at {zk_hosts}")
        except Exception:
            self.zk = None

        # -------------------------------------------------------------------
        # Leader / replica initialisation (NEW)
        # -------------------------------------------------------------------
        if self.zk is not None:
            self._elect_leader()
            self._load_version()
            if not self.is_leader:
                threading.Thread(target=self._replica_sync_loop, daemon=True).start()
        # -------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Leader / replica helpers  (NEW)
    # -----------------------------------------------------------------------
    def _elect_leader(self):
        """Create an ephemeral-sequential znode and determine our role."""
        self.zk.ensure_path(LEADER_ELECTION_ROOT)
        self.my_znode = self.zk.create(
            f"{LEADER_ELECTION_ROOT}/n_",
            value=self.host.encode(),
            ephemeral=True,
            sequence=True,
        )
        self._refresh_role()  # sets self.is_leader and installs watch

    def _refresh_role(self, *_):
        """Callback when watched znode disappears → recalc leader."""
        children = sorted(self.zk.get_children(LEADER_ELECTION_ROOT))
        my_node = self.my_znode.split("/")[-1]
        self.is_leader = children and my_node == children[0]

        if not self.is_leader and children:
            idx = children.index(my_node)
            watch_target = f"{LEADER_ELECTION_ROOT}/{children[idx - 1]}"
            self.zk.exists(watch_target, watch=self._refresh_role)

    def _get_leader_host(self) -> str:
        """Return hostname stored in leader’s znode."""
        try:
            children = sorted(self.zk.get_children(LEADER_ELECTION_ROOT))
            leader_node = f"{LEADER_ELECTION_ROOT}/{children[0]}"
            data, _ = self.zk.get(leader_node)
            return data.decode()
        except Exception:
            return ""

    def _persist_version(self):
        """Write current version to ZooKeeper so replicas can catch up."""
        try:
            self.zk.ensure_path(META_VERSION_ZNODE)
            self.zk.set(META_VERSION_ZNODE, str(self.version).encode())
        except Exception as e:
            logger.warning(f"Persist version failed: {e}")

    def _load_version(self):
        """Load persisted version at startup (if any)."""
        try:
            data, _ = self.zk.get(META_VERSION_ZNODE)
            self.version = int(data.decode())
        except NoNodeError:
            # First boot: nothing persisted yet.
            pass

    def _replica_sync_loop(self):
        """Background loop for replicas to pull latest state."""
        while True:
            try:
                data, _ = self.zk.get(META_VERSION_ZNODE)
                latest = int(data.decode())
                if latest > self.version:
                    self.version = latest
                    adapter_dir = os.path.join(
                        PATH_TO_ADAPTERS, "rounds", f"v{self.version}"
                    )
                    path = os.path.join(adapter_dir, "adapter_model.safetensors")
                    if os.path.exists(path):
                        with open(path, "rb") as fh:
                            self.global_adapter_state = load_safetensors_from_bytes(
                                fh.read()
                            )
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.warning(f"Replica sync error: {e}")
            time.sleep(SYNC_POLL_SEC)

    # -----------------------------------------------------------------------

    def update_client_weight(self, client_id, weight):

        # 3. Ensure the parent path exists
        if self.zk is None:
            return

        self.zk.ensure_path("/myapp/clients")

        # 4. Create or update a znode for client_id → weight
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
        if self.zk is not None and self.zk.exists(path):
            raw, stat = self.zk.get(path)
            weight = float(raw.decode("utf-8"))
            logger.debug(f"Weight for {client_id}: {weight}")
        else:
            logger.debug(f"No entry for {client_id}")

        return weight

    # -----------------------------------------------------------------------
    # RPCs
    # -----------------------------------------------------------------------
    def SubmitUpdate(self, request, context):
        # ---------------- leader-only gate ----------------
        if self.zk is not None and not self.is_leader:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Not leader")
            context.set_trailing_metadata((("leader-host", self._get_leader_host()),))
            return model_update_pb2.Acknowledgement()
        # --------------------------------------------------

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
                PATH_TO_ADAPTERS, "clients", client_id, f"v{client_version}"
            )
            decoded_dict = load_safetensors_from_bytes(request.update)
            decoded_msg = DecodedModelUpdate(
                client_id=request.client_id,
                update=decoded_dict,
                version=request.version,
                timestamp=request.timestamp,
            )

            logger.info(
                f"Received adapter update from {request.client_id} (version: {request.version})."
            )

            # Update the client's tracked version
            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = client_version
                self.connected_clients[client_id]["last_seen"] = time.time()

            self.update_requests.append(decoded_msg)

            # Aggregate once two or more updates are received (for testing)
            with self.update_lock:
                if len(self.update_requests) >= 1:  # 2
                    self.version += 1

                    aggregated_state = {}
                    # Assume all updates have matching keys
                    for key in self.update_requests[0].update.keys():
                        aggregated_state[key] = 0
                        total_weight = 0
                        for update_requests in self.update_requests:
                            # allow some zk tolerance
                            if self.zk is not None:
                                weight = self.get_client_weight(
                                    update_requests.client_id
                                )
                            else:
                                weight = 1
                            aggregated_state[key] += (
                                update_requests.update[key] * weight
                            )
                            total_weight += weight

                        aggregated_state[key] /= total_weight

                    self.global_adapter_state = aggregated_state
                    adapter_folder = os.path.join(
                        PATH_TO_ADAPTERS, "rounds", f"v{self.version}"
                    )
                    os.makedirs(adapter_folder, exist_ok=True)
                    # TODO: Figure out adapter_config.json. For now let's cheat
                    model_path = os.path.join(
                        PATH_TO_ADAPTERS, "central", "latest"
                    )  # distributed_fl/adapters/central/latest
                    config = PeftConfig.from_pretrained(model_path)
                    # save adapter.config_json
                    config.save_pretrained(adapter_folder)

                    save_file(
                        aggregated_state,
                        os.path.join(adapter_folder, "adapter_model.safetensors"),
                    )

                    logger.info("Aggregated global adapter state updated.")
                    self.update_requests = []  # Reset for the next round

                    # NEW: persist version for replicas
                    if self.zk is not None:
                        self._persist_version()

                    # Notify all subscribed clients of the new model
                    self._notify_clients_of_update()

            return model_update_pb2.Acknowledgement(
                success=True, message="Adapter update received."
            )
        except Exception as e:
            traceback.print_exc()
            logger.info("Error in SubmitUpdate: %s", e)
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
            adapter_file_path = os.path.join(
                PATH_TO_ADAPTERS, "rounds", f"v{self.version}"
            )
            code_agent = LoraHuggingFaceAgent(
                model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                adapter_path=adapter_file_path,
            )
            result = evaluate(
                code_agent, self.benchmark, results_csv="experiments.csv", mode="prod"
            )
            if result:
                path = os.path.join(adapter_file_path, "adapter_model.safetensors")
                with open(path, "rb") as f:
                    bytes_ = f.read()
                # Update client's tracked version
                if client_id in self.connected_clients:
                    self.connected_clients[client_id]["version"] = self.version
                return model_update_pb2.AggregatedModel(
                    model_state=bytes_, version=self.version
                )
            else:
                context.set_details("Bad update")
                context.set_code(grpc.StatusCode.INTERNAL)
                model_update_pb2.AggregatedModel()
        except Exception as e:
            traceback.print_exc()
            logger.info("Error in GetAggregatedModel:: %s", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_update_pb2.AggregatedModel()

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
                    notification = self.notification_queues[client_id].get(
                        block=True, timeout=30
                    )
                    yield notification
                    self.notification_queues[client_id].task_done()
                except queue.Empty:
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]["last_seen"] = time.time()
                    continue
        except Exception as e:
            logger.info(f"Error in subscription stream for client {client_id}: {e}")
        finally:
            logger.info(f"Client {client_id} unsubscribed from updates")

    # Helper method to notify clients
    def _notify_clients_of_update(self):
        notification = model_update_pb2.UpdateNotification(
            new_version=self.version, update_type="FULL"
        )

        for client_id in list(self.notification_queues.keys()):
            try:
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
            if client_id in servicer.notification_queues:
                del servicer.notification_queues[client_id]
            logger.info(f"Removed inactive client: {client_id}")


def serve():
    zk_hosts = os.getenv("ZK_HOSTS", "127.0.0.1:2181")

    servicer = FederatedLearningServiceServicer(zk_hosts=zk_hosts)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_update_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("Server started on port 50051.")

    # Background thread for periodic client cleanup
    def cleanup_thread():
        while True:
            try:
                cleanup_disconnected_clients(servicer)
                time.sleep(60)
            except Exception as e:
                logger.info(f"Error in cleanup thread: {e}")

    cleanup_task = threading.Thread(target=cleanup_thread, daemon=True)
    cleanup_task.start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)
        if servicer.zk is not None:
            servicer.zk.stop()
            servicer.zk.close()


if __name__ == "__main__":
    serve()
