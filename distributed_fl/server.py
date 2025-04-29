# server.py
import io
import os
import queue
import socket
import tempfile
import threading
import time
import traceback
import zlib
from concurrent import futures
from typing import Any, Dict, NamedTuple

import grpc
import torch
from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError
from kazoo.retry import KazooRetry
from kazoo.recipe.watchers import DataWatch, ChildrenWatch
from safetensors.torch import load_file, save_file
from peft import PeftConfig

import model_update_pb2
import model_update_pb2_grpc
from agent import LoraHuggingFaceAgent
from benchmark import HumanEvalBenchmark
from eval_script import evaluate
from logger import get_logger
from utils import load_safetensors_from_bytes, find_latest_adapter_version

logger = get_logger(__name__)

PATH_TO_ADAPTERS = os.environ.get("PATH_TO_ADAPTERS", "./distributed_fl/adapters")

# ---- Zookeeper paths ---------------------------------------------------------
ZK_ELECTION_PATH = "/fl/servers"
ZK_VERSION_PATH = "/fl/meta/version"
ZK_LEADER_INFO_PATH = "/fl/meta/leader"
# -----------------------------------------------------------------------------


class DecodedModelUpdate(NamedTuple):
    client_id: str
    update: Dict[str, Any]
    version: int
    timestamp: int


# -----------------------------------------------------------------------------
# Federated learning service
# -----------------------------------------------------------------------------
class FederatedLearningServiceServicer(
    model_update_pb2_grpc.FederatedLearningServiceServicer
):
    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------
    def __init__(
        self,
        mode: str = "test",
        zk_hosts: str = "127.0.0.1:2181",
        host: str | None = None,
    ):
        # -------------------- in-memory state --------------------------------
        self.update_requests: list[DecodedModelUpdate] = []
        self.global_adapter_state: Dict[str, Any] | None = None
        self.version: int = 1

        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.notification_queues: Dict[
            str, "queue.Queue[model_update_pb2.UpdateNotification]"
        ] = {}  # noqa: E501

        self.update_lock = threading.Lock()
        self.is_leader = False
        self.host = host or socket.gethostname()

        # -------------------- benchmarking -----------------------------------
        self.benchmark = HumanEvalBenchmark()
        self.benchmark.load_dataset()
        if mode == "test":
            self.benchmark.dataset = self.benchmark.dataset.select(range(3))

        # -------------------- Zookeeper connection & election ----------------
        self._init_zookeeper(zk_hosts)

        # -------------------- replica sync watch -----------------------------
        self._install_version_watch()

    # -------------------------------------------------------------------------
    # Zookeeper helpers
    # -------------------------------------------------------------------------
    def _init_zookeeper(self, zk_hosts: str) -> None:
        retry = KazooRetry(max_tries=1, delay=1.0, backoff=2, max_delay=1)
        self.zk = KazooClient(
            hosts=zk_hosts,
            connection_retry=retry,
            command_retry=retry,
        )
        self.zk.start()
        logger.info(f"Connected to Zookeeper at {zk_hosts}")

        # Ensure base znodes exist
        for path in (ZK_ELECTION_PATH, "/fl/meta", "/fl/clients"):
            self.zk.ensure_path(path)

        # Start leader election in a background thread
        threading.Thread(target=self._participate_in_election, daemon=True).start()

    # -------------------------------------------------------------------------
    # Leader election
    # -------------------------------------------------------------------------
    def _participate_in_election(self) -> None:
        """
        Blocks until this process becomes leader, then keeps the role while the
        ephemeral node exists. When we lose the session the thread exits and
        another server takes over.
        """
        election = self.zk.Election(ZK_ELECTION_PATH, self.host)

        def _on_elected() -> None:
            self.is_leader = True
            logger.info("🏆  Became leader (%s)", self.host)
            # Advertise leader host for replicas/clients
            self.zk.set(ZK_LEADER_INFO_PATH, self.host.encode("utf-8"), version=-1)
            # Persist version if not present, or load existing
            if self.zk.exists(ZK_VERSION_PATH):
                self._load_version()
            else:
                self._persist_version()

        election.run(_on_elected)

    def _persist_version(self) -> None:
        data = str(self.version).encode()
        if self.zk.exists(ZK_VERSION_PATH):
            self.zk.set(ZK_VERSION_PATH, data)
        else:
            self.zk.create(ZK_VERSION_PATH, data, makepath=True)

    def _load_version(self) -> None:
        data, _ = self.zk.get(ZK_VERSION_PATH)
        self.version = int(data.decode())
        logger.info("Loaded version %s from ZooKeeper", self.version)
        self._load_adapter_from_disk(self.version)

    def _get_current_leader_host(self) -> str | None:
        if not self.zk.exists(ZK_LEADER_INFO_PATH):
            return None
        data, _ = self.zk.get(ZK_LEADER_INFO_PATH)
        return data.decode() if data else None

    # -------------------------------------------------------------------------
    # Replica sync
    # -------------------------------------------------------------------------
    def _install_version_watch(self) -> None:
        @DataWatch(self.zk, ZK_VERSION_PATH)
        def _watch(data, _stat, _event) -> None:  # noqa: N803 (kazoo naming)
            # This fires both on initial set and on every change
            if data is None:
                return
            new_version = int(data.decode())
            if new_version > self.version:
                logger.info("Replica detected new version %s → loading", new_version)
                self.version = new_version
                self._load_adapter_from_disk(new_version)

    def _load_adapter_from_disk(self, version: int) -> None:
        try:
            adapter_path = os.path.join(
                PATH_TO_ADAPTERS, "rounds", f"v{version}", "adapter_model.safetensors"
            )
            if not os.path.exists(adapter_path):
                logger.warning("Adapter v%s not found on disk yet", version)
                return
            self.global_adapter_state = load_file(adapter_path)
            logger.info("Loaded aggregated adapter v%s into memory", version)
        except Exception as e:
            logger.error("Failed to load adapter v%s: %s", version, e, exc_info=True)

    # -------------------------------------------------------------------------
    # RPC utilities
    # -------------------------------------------------------------------------
    def _redirect_to_leader_ack(
        self, prefix: str = ""
    ) -> model_update_pb2.Acknowledgement:
        host = self._get_current_leader_host()
        msg = f"{prefix} REDIRECT:{host}" if host else f"{prefix} NO_LEADER_AVAILABLE"
        return model_update_pb2.Acknowledgement(success=False, message=msg)

    # -------------------------------------------------------------------------
    # RPC implementations
    # -------------------------------------------------------------------------
    def SubmitUpdate(self, request, context):
        # Only the leader mutates global state
        if not self.is_leader:
            return self._redirect_to_leader_ack("Not leader:")

        client_id = request.client_id
        client_version = request.version

        if self.global_adapter_state is not None and client_version < self.version:
            return model_update_pb2.Acknowledgement(
                success=False,
                message=f"Client outdated (client v{client_version}, latest v{self.version})",
            )

        try:
            decoded_dict = load_safetensors_from_bytes(request.update)
            decoded_msg = DecodedModelUpdate(
                client_id=request.client_id,
                update=decoded_dict,
                version=request.version,
                timestamp=request.timestamp,
            )
            logger.info("Received update from %s (v%s)", client_id, client_version)

            if client_id in self.connected_clients:
                self.connected_clients[client_id]["version"] = client_version
                self.connected_clients[client_id]["last_seen"] = time.time()

            self.update_requests.append(decoded_msg)

            # ------------------------------------------------ aggregation block
            with self.update_lock:
                if len(self.update_requests) >= 1:  # ⇐ use 2+ in prod
                    self.version += 1
                    aggregated_state: Dict[str, Any] = {}
                    total_weights: Dict[str, float] = {}

                    for update in self.update_requests:
                        weight = self._get_client_weight(update.client_id)
                        for k, v in update.update.items():
                            aggregated_state.setdefault(k, 0.0)
                            total_weights.setdefault(k, 0.0)
                            aggregated_state[k] += v * weight
                            total_weights[k] += weight

                    for k in aggregated_state:
                        aggregated_state[k] /= total_weights[k]

                    self.global_adapter_state = aggregated_state
                    self._store_new_adapter()

                    logger.info("Aggregated adapter v%s ready", self.version)
                    self.update_requests.clear()

                    # Persist and notify
                    self._persist_version()
                    self._notify_clients_of_update()

            return model_update_pb2.Acknowledgement(success=True, message="ok")
        except Exception as e:
            logger.error("SubmitUpdate failed: %s", e, exc_info=True)
            return model_update_pb2.Acknowledgement(success=False, message=str(e))

    def _store_new_adapter(self) -> None:
        adapter_folder = os.path.join(PATH_TO_ADAPTERS, "rounds", f"v{self.version}")
        os.makedirs(adapter_folder, exist_ok=True)
        latest_version = find_latest_adapter_version(
            os.path.join(PATH_TO_ADAPTERS, "central")
        )
        model_path = os.path.join(PATH_TO_ADAPTERS, "central", f"v{latest_version}")
        config = PeftConfig.from_pretrained(model_path)
        config.save_pretrained(adapter_folder)
        save_file(
            self.global_adapter_state,
            os.path.join(adapter_folder, "adapter_model.safetensors"),
        )

    def GetAggregatedModel(self, request, context):
        client_id = request.client_id
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

        if self.global_adapter_state is None:
            return model_update_pb2.AggregatedModel(
                model_state=b"", version=self.version
            )

        try:
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
                with open(
                    os.path.join(adapter_file_path, "adapter_model.safetensors"), "rb"
                ) as f:
                    bytes_ = f.read()
                if client_id in self.connected_clients:
                    self.connected_clients[client_id]["version"] = self.version
                return model_update_pb2.AggregatedModel(
                    model_state=bytes_, version=self.version
                )

            context.set_details("Bad update")
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_update_pb2.AggregatedModel()
        except Exception as e:
            logger.error("GetAggregatedModel failed: %s", e, exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_update_pb2.AggregatedModel()

    def ConnectClient(self, request, context):
        client_id = request.client_id
        client_version = request.current_version
        self.connected_clients[client_id] = {
            "version": client_version,
            "last_seen": time.time(),
            "ready_for_training": request.ready_for_training,
        }

        update_available = (
            self.global_adapter_state is not None and client_version < self.version
        )
        response = model_update_pb2.ModelVersionInfo(
            update_available=update_available, latest_version=self.version
        )

        if update_available:
            try:
                adapter_path = os.path.join(
                    PATH_TO_ADAPTERS,
                    "rounds",
                    f"v{self.version}",
                    "adapter_model.safetensors",
                )
                with open(adapter_path, "rb") as f:
                    response.model_state = f.read()
            except Exception as e:
                logger.error("Prepare model for client %s failed: %s", client_id, e)

        return response

    def SubscribeToUpdates(self, request, context):
        client_id = request.client_id
        self.notification_queues.setdefault(client_id, queue.Queue())
        self.connected_clients.setdefault(
            client_id,
            {
                "version": request.current_version,
                "last_seen": time.time(),
            },
        )
        if self.global_adapter_state is not None:
            yield model_update_pb2.UpdateNotification(
                new_version=self.version, update_type="FULL"
            )

        try:
            while context.is_active():
                try:
                    note = self.notification_queues[client_id].get(
                        block=True, timeout=30
                    )
                    yield note
                except queue.Empty:
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]["last_seen"] = time.time()
                    continue
        finally:
            logger.info("Client %s unsubscribed", client_id)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _notify_clients_of_update(self) -> None:
        note = model_update_pb2.UpdateNotification(
            new_version=self.version, update_type="FULL"
        )
        for cid, q in self.notification_queues.items():
            try:
                q.put_nowait(note)
            except queue.Full:
                logger.warning("Notification queue full for %s", cid)

    def _get_client_weight(self, client_id: str) -> float:
        zk_path = f"/fl/clients/{client_id}"
        if self.zk.exists(zk_path):
            data, _ = self.zk.get(zk_path)
            return float(data.decode())
        return 1.0  # default

    # -------------------------------------------------------------------------
    # Cleanup inactive clients (runs in background thread from serve())
    # -------------------------------------------------------------------------
    def cleanup_disconnected_clients(self) -> None:
        now = time.time()
        timeout = 300
        stale = [
            cid
            for cid, info in self.connected_clients.items()
            if now - info.get("last_seen", 0) > timeout
        ]
        for cid in stale:
            logger.info("Removing inactive client %s", cid)
            self.connected_clients.pop(cid, None)
            self.notification_queues.pop(cid, None)


# -----------------------------------------------------------------------------
# gRPC server bootstrap
# -----------------------------------------------------------------------------
def serve() -> None:
    zk_hosts = os.getenv("ZK_HOSTS", "127.0.0.1:2181")
    host = socket.gethostname()

    servicer = FederatedLearningServiceServicer(zk_hosts=zk_hosts, host=host)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_update_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("Server %s started on port 50051", host)

    def cleanup_loop() -> None:
        while True:
            try:
                servicer.cleanup_disconnected_clients()
            except Exception as e:
                logger.error("Cleanup thread error: %s", e, exc_info=True)
            time.sleep(60)

    threading.Thread(target=cleanup_loop, daemon=True).start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        server.stop(0)
        servicer.zk.stop()
        servicer.zk.close()


if __name__ == "__main__":
    serve()
