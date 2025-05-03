import os
import time
import threading
import grpc
import shutil
from typing import Optional
import model_update_pb2
import model_update_pb2_grpc
from safetensors.torch import save_file, load_file
from utils import load_safetensors_from_bytes, find_latest_adapter_version
from logger import get_logger

logger = get_logger(__name__)

PATH_TO_ADAPTERS = "./distributed_fl/adapters"


class ReplicaManager:
    """Manages a replica server that syncs with a leader server."""

    def __init__(
        self,
        replica_id: str,
        leader_address: str,
        adapters_path: str = PATH_TO_ADAPTERS,
        heartbeat_interval: int = 5,
        max_missed_heartbeats: int = 3,
    ):
        self.replica_id = replica_id
        self.leader_address = leader_address
        self.path_to_adapters = adapters_path
        self.current_version = 0
        self.running = True
        self.lock = threading.Lock()

        # Heartbeat configuration
        self.heartbeat_interval = heartbeat_interval  # seconds
        self.max_missed_heartbeats = max_missed_heartbeats
        self.missed_heartbeats = 0
        self.leader_is_alive = True

        # Create gRPC channel to leader
        self.channel = grpc.insecure_channel(leader_address)
        self.stub = model_update_pb2_grpc.FederatedLearningServiceStub(self.channel)

        # Make sure directories exist
        self.replica_model_path = os.path.join(
            self.path_to_adapters, "replica", "models"
        )
        os.makedirs(self.replica_model_path, exist_ok=True)

        # Check if we have any existing models
        self.current_version = find_latest_adapter_version(self.replica_model_path)
        logger.info(f"Initialized replica with model version {self.current_version}")

    def connect_to_leader(self) -> bool:
        """Connect to the leader server and check for model updates."""
        try:
            connect_request = model_update_pb2.ClientConnection(
                client_id=f"replica_{self.replica_id}",
                current_version=self.current_version,
                ready_for_training=False,  # Replicas don't perform training
            )

            version_info = self.stub.ConnectClient(connect_request)
            logger.info(
                f"Connected to leader. Latest version: {version_info.latest_version}"
            )

            if version_info.update_available and version_info.model_state:
                self._update_local_model(
                    version_info.model_state, version_info.latest_version
                )

            return True
        except Exception as e:
            logger.error(f"Error connecting to leader: {e}")
            return False

    def subscribe_to_updates(self) -> Optional[threading.Thread]:
        """Listen for model update notifications from the leader."""

        def update_listener():
            try:
                subscription_request = model_update_pb2.ClientRequest(
                    client_id=f"replica_{self.replica_id}",
                    current_version=self.current_version,
                )

                for notification in self.stub.SubscribeToUpdates(subscription_request):
                    if not self.running:
                        break

                    logger.info(
                        f"Update notification: New version {notification.new_version} available"
                    )

                    with self.lock:
                        if notification.new_version > self.current_version:
                            self._get_latest_model()
            except Exception as e:
                if self.running:
                    logger.error(f"Update subscription error: {e}")
                    logger.info("Attempting to reconnect to leader in 10 seconds...")
                    time.sleep(10)
                    if self.running:
                        self.subscribe_to_updates()

        listener_thread = threading.Thread(target=update_listener, daemon=True)
        listener_thread.start()
        return listener_thread

    def _get_latest_model(self) -> bool:
        """Fetch the latest model from the leader."""
        try:
            client_request = model_update_pb2.ClientRequest(
                client_id=f"replica_{self.replica_id}",
                current_version=self.current_version,
            )

            aggregated = self.stub.GetAggregatedModel(client_request)

            if aggregated.model_state:
                logger.info(
                    f"Received model state of length {len(aggregated.model_state)}"
                )
                return self._update_local_model(
                    aggregated.model_state, aggregated.version
                )
            else:
                logger.info("No model state received or no newer model available")
                return False
        except Exception as e:
            logger.error(f"Error getting latest model: {e}")
            return False

    def _update_local_model(self, model_state: bytes, version: int) -> bool:
        """Update local model with received model state."""
        try:
            # Decode model state
            decoded_dict = load_safetensors_from_bytes(model_state)

            # Create version directory
            version_dir = os.path.join(self.replica_model_path, f"v{version}")
            os.makedirs(version_dir, exist_ok=True)

            # Save model state
            save_file(
                decoded_dict, os.path.join(version_dir, "adapter_model.safetensors")
            )

            # Copy adapter config
            shutil.copy2(
                os.path.join(self.path_to_adapters, "adapter_config.json"),
                os.path.join(version_dir, "adapter_config.json"),
            )

            # Update current version
            self.current_version = version
            logger.info(f"Updated local model to version {self.current_version}")

            return True
        except Exception as e:
            logger.error(f"Error updating local model: {e}")
            return False

    def check_leader_heartbeat(self) -> bool:
        """Send a heartbeat to the leader and check response."""
        try:
            heartbeat_request = model_update_pb2.HeartbeatRequest(
                sender_id=f"replica_{self.replica_id}", timestamp=int(time.time())
            )

            response = self.stub.SendHeartbeat(heartbeat_request, timeout=2)

            # If we get here, the leader is alive
            if self.missed_heartbeats > 0:
                logger.info(
                    f"Leader is now responsive after {self.missed_heartbeats} missed heartbeats"
                )
            self.missed_heartbeats = 0
            self.leader_is_alive = True
            # TODO: turn to debug
            logger.info("Heartbeat to leader successful")
            # Check if there's a newer version available
            if response.current_version > self.current_version:
                logger.info(
                    f"Heartbeat revealed newer model version: {response.current_version}"
                )
                self._get_latest_model()

            return True
        except Exception as e:
            # Increment missed heartbeats counter
            self.missed_heartbeats += 1
            logger.warning(
                f"Missed heartbeat from leader ({self.missed_heartbeats}/{self.max_missed_heartbeats}): {e}"
            )

            # Check if we've reached the threshold for declaring leader dead
            if self.missed_heartbeats >= self.max_missed_heartbeats:
                self.leader_is_alive = False
                logger.error(
                    f"Leader considered down after {self.missed_heartbeats} missed heartbeats"
                )

                # Here you would trigger leadership takeover if implemented
                # self._assume_leadership()

            return False

    # Add this method to the ReplicaManager class

    def _assume_leadership(self):
        """
        Simple leadership takeover implementation.
        """
        logger.info(
            "LEADERSHIP TAKEOVER: Replica assuming leader role due to leader failure"
        )

        # Stop heartbeat checking and update subscription
        self.running = False

        # Close existing channel to the failed leader
        if self.channel:
            self.channel.close()

        # Import necessary modules for server functionality
        from concurrent import futures
        import server

        # Find the latest model we have
        latest_model_path = os.path.join(
            self.replica_model_path, f"v{self.current_version}"
        )

        # Copy our latest model to the server central directory
        server_central_path = os.path.join(
            self.path_to_adapters, "server", "central", f"v{self.current_version}"
        )
        os.makedirs(os.path.dirname(server_central_path), exist_ok=True)

        if os.path.exists(latest_model_path):
            if not os.path.exists(server_central_path):
                os.makedirs(server_central_path, exist_ok=True)

                # Copy model files
                shutil.copy2(
                    os.path.join(latest_model_path, "adapter_model.safetensors"),
                    os.path.join(server_central_path, "adapter_model.safetensors"),
                )
                shutil.copy2(
                    os.path.join(latest_model_path, "adapter_config.json"),
                    os.path.join(server_central_path, "adapter_config.json"),
                )
                logger.info(
                    f"Copied model files from {latest_model_path} to {server_central_path}"
                )

        # Initialize server components
        servicer = server.FederatedLearningServiceServicer()
        servicer.version = self.current_version  # Set the version to our latest

        # If we have a model, load it
        if os.path.exists(
            os.path.join(server_central_path, "adapter_model.safetensors")
        ):
            servicer.global_adapter_state = load_file(
                os.path.join(server_central_path, "adapter_model.safetensors")
            )
            logger.info(f"Loaded model state from {server_central_path}")

        # Start the gRPC server
        port = int(
            self.leader_address.split(":")[-1]
        )  # Use the same port as the leader
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        model_update_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
            servicer, grpc_server
        )
        grpc_server.add_insecure_port(f"[::]:{port}")
        grpc_server.start()

        logger.info(
            f"LEADERSHIP TAKEOVER COMPLETE: Now serving as leader on port {port}"
        )

        # Start a background thread for periodic client cleanup
        def cleanup_thread():
            while True:
                try:
                    server.cleanup_disconnected_clients(servicer)
                    time.sleep(60)  # Run cleanup every minute
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")

        cleanup_task = threading.Thread(target=cleanup_thread, daemon=True)
        cleanup_task.start()

        try:
            # Keep main thread alive
            while True:
                time.sleep(86400)  # Sleep for a day
        except KeyboardInterrupt:
            logger.info("Leader server shutting down...")
            grpc_server.stop(0)

    def start_heartbeat_monitoring(self) -> threading.Thread:
        """Start a thread to monitor leader heartbeats."""

        def heartbeat_thread():
            logger.info(
                f"Starting heartbeat monitoring of leader at {self.leader_address}"
            )

            while self.running:
                self.check_leader_heartbeat()
                time.sleep(self.heartbeat_interval)

        monitor_thread = threading.Thread(target=heartbeat_thread, daemon=True)
        monitor_thread.start()
        return monitor_thread

    def start(self):
        """Start the replica manager."""
        if not self.connect_to_leader():
            logger.error("Failed to connect to leader. Retrying in 10 seconds...")
            time.sleep(10)
            self.start()
            return

        # Subscribe to model updates
        update_thread = self.subscribe_to_updates()

        # Start heartbeat monitoring
        heartbeat_thread = self.start_heartbeat_monitoring()

        try:
            consecutive_failures = 0
            max_consecutive_failures = 3  # Number of consecutive checks before takeover
            check_interval = 10  # Seconds between leadership checks

            while self.running:
                time.sleep(check_interval)

                # Check if leader is down
                if not self.leader_is_alive:
                    consecutive_failures += 1
                    logger.warning(
                        f"Leader is down! ({consecutive_failures}/{max_consecutive_failures}) Waiting for recovery..."
                    )

                    # Try reconnecting
                    if self.connect_to_leader():
                        logger.info("Successfully reconnected to leader")
                        consecutive_failures = 0
                        self.leader_is_alive = True
                        continue

                    # Check if we should take over leadership
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"Leader failed to recover after {consecutive_failures} attempts. Taking over leadership."
                        )
                        self._assume_leadership()
                        return  # Exit this method as we've now become the leader
                else:
                    consecutive_failures = 0

        except KeyboardInterrupt:
            logger.info("Replica manager shutting down...")
        finally:
            self.shutdown()
            if update_thread:
                update_thread.join(timeout=2)
            if heartbeat_thread:
                heartbeat_thread.join(timeout=2)

    def shutdown(self):
        """Clean shutdown of the replica manager."""
        logger.info("Shutting down replica manager...")
        self.running = False
        if self.channel:
            self.channel.close()
