import os
import time
import threading
import grpc
import shutil
import socket
from typing import Optional, List, Dict
import model_update_pb2
import model_update_pb2_grpc
from safetensors.torch import save_file, load_file
from utils import load_safetensors_from_bytes, find_latest_adapter_version
from logger import get_logger
from zk_manager import ZKManager

logger = get_logger(__name__)

PATH_TO_ADAPTERS = "./distributed_fl/adapters"


def normalize_address(address):
    """
    Normalize server addresses to prevent false leader changes
    when the same server is referenced with different names.

    Converts hostnames like 'host:port' to either 'localhost:port'
    or 'ip:port' to ensure consistent addressing.
    """
    if not address:
        return address

    parts = address.split(":")
    if len(parts) != 2:
        return address  # Not a valid host:port format

    hostname, port = parts

    # Check if this is a local hostname
    try:
        import socket

        local_hostname = socket.gethostname()

        # If this matches the local machine, use localhost
        if hostname == local_hostname:
            return f"localhost:{port}"

        # Try to resolve the hostname
        try:
            ip = socket.gethostbyname(hostname)

            # Check if it's a loopback address (127.x.x.x)
            if ip.startswith("127."):
                return f"localhost:{port}"

            # Check if it's one of this machine's IP addresses
            for addr_info in socket.getaddrinfo(local_hostname, None):
                if addr_info[4][0] == ip:
                    return f"localhost:{port}"

            # Otherwise return the IP form for consistency
            return f"{ip}:{port}"
        except socket.gaierror:
            # Can't resolve hostname, return as is
            return address
    except:
        # If any error occurs, return the original address
        return address


class ReplicaManager:
    """Manages a replica server that syncs with a leader server."""

    def __init__(
        self,
        replica_id: str,
        leader_address: str,
        server_port: int,
        adapters_path: str = PATH_TO_ADAPTERS,
        heartbeat_interval: int = 5,
        max_missed_heartbeats: int = 3,
        zk_hosts: str = "127.0.0.1:2181",
    ):
        self.replica_id = replica_id
        self.leader_address = leader_address
        self.server_port = server_port  # Store port for election
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

        # ZooKeeper for simple coordination
        self.zk_manager = ZKManager(hosts=zk_hosts)

        # Register this replica with ZooKeeper
        self._register_replica()

        # Make sure directories exist
        self.replica_model_path = os.path.join(
            self.path_to_adapters, "server", "central"
        )
        os.makedirs(self.replica_model_path, exist_ok=True)

        # Check if we have any existing models
        self.current_version = find_latest_adapter_version(self.replica_model_path)
        logger.info(f"Initialized replica with model version {self.current_version}")

    def _register_replica(self):
        """Register this replica in ZooKeeper for election purposes."""
        if self.zk_manager.zk:
            # Create a replicas directory if it doesn't exist
            self.zk_manager.zk.ensure_path("/myapp/replicas")

            # Register this replica with its port number
            replica_path = f"/myapp/replicas/{self.server_port}"
            if not self.zk_manager.zk.exists(replica_path):
                self.zk_manager.zk.create(
                    replica_path,
                    str(self.replica_id).encode("utf-8"),
                    ephemeral=True,  # This node disappears if the replica crashes
                )
                logger.info(f"Registered replica in ZooKeeper: {replica_path}")
            else:
                self.zk_manager.zk.set(
                    replica_path, str(self.replica_id).encode("utf-8")
                )
                logger.info(f"Updated replica in ZooKeeper: {replica_path}")

    def _register_as_leader(self):
        """Register this server as the current leader in ZooKeeper."""
        if self.zk_manager.zk:
            # Create leader path if needed
            self.zk_manager.zk.ensure_path("/myapp/leader")

            # Always use localhost:port format for consistency
            leader_data = f"localhost:{self.server_port}".encode("utf-8")

            # Set leader info
            if self.zk_manager.zk.exists("/myapp/leader/current"):
                self.zk_manager.zk.set("/myapp/leader/current", leader_data)
            else:
                self.zk_manager.zk.create("/myapp/leader/current", leader_data)

            logger.info(
                f"Registered as leader with address localhost:{self.server_port}"
            )
            return True

    def _discover_leader(self):
        """Check if leader has changed and update connection if needed."""
        if not self.zk_manager.zk:
            logger.warning("No ZooKeeper connection for leader discovery")
            return False

        try:
            # Check if leader info exists
            if self.zk_manager.zk.exists("/myapp/leader/current"):
                # Get current leader address from ZooKeeper
                leader_data, _ = self.zk_manager.zk.get("/myapp/leader/current")
                new_leader_address = leader_data.decode("utf-8")

                # Normalize both addresses to prevent false detection of changes
                normalized_current = normalize_address(self.leader_address)
                normalized_new = normalize_address(new_leader_address)

                # Check if leader address has meaningfully changed
                if normalized_new != normalized_current:
                    logger.info(
                        f"Discovered potential new leader at {new_leader_address} (old: {self.leader_address})"
                    )

                    # Test the new connection BEFORE closing the old one
                    try:
                        test_channel = grpc.insecure_channel(new_leader_address)
                        test_stub = model_update_pb2_grpc.FederatedLearningServiceStub(
                            test_channel
                        )

                        # Try a simple heartbeat to verify the connection works
                        heartbeat_request = model_update_pb2.HeartbeatRequest(
                            sender_id=f"replica_{self.replica_id}",
                            timestamp=int(time.time()),
                        )

                        # Use a short timeout to avoid hanging
                        response = test_stub.SendHeartbeat(heartbeat_request, timeout=3)

                        if response.alive:
                            logger.info(
                                f"Successfully verified new leader connection to {new_leader_address}"
                            )

                            # Now that we've verified the new connection works, close the old one
                            if self.channel:
                                self.channel.close()

                            # Update leader address
                            self.leader_address = new_leader_address

                            # Create new connection
                            self.channel = test_channel  # Reuse the test channel we already verified
                            self.stub = test_stub

                            logger.info(
                                f"Updated connection to new leader at {self.leader_address}"
                            )
                            return True
                        else:
                            logger.warning(
                                f"New leader at {new_leader_address} responded but reports not alive"
                            )
                            test_channel.close()
                    except Exception as e:
                        logger.warning(
                            f"Failed to connect to potential new leader at {new_leader_address}: {e}"
                        )
                        return False
            else:
                # Addresses are equivalent after normalization
                return False
        except Exception as e:
            logger.error(f"Error discovering leader: {e}")

        return False

    def _elect_leader(self) -> bool:
        """
        Simple deterministic leader election.
        Returns True if this replica should become the leader.
        """
        if not self.zk_manager.zk:
            logger.warning("No ZooKeeper connection, assuming leadership by default")
            return True

        try:
            # Get all registered replicas
            replicas = self.zk_manager.zk.get_children("/myapp/replicas")
            if not replicas:
                logger.warning("No replicas found in ZooKeeper")
                return True

            # Convert to integers (port numbers)
            replica_ports = [int(port) for port in replicas]

            # Sort ports to find the lowest
            replica_ports.sort()

            # Check if this replica has the lowest port
            if replica_ports[0] == self.server_port:
                logger.info(
                    f"This replica (port {self.server_port}) has the lowest port number and will become leader"
                )
                return True
            else:
                logger.info(
                    f"Replica with port {replica_ports[0]} should become the leader (our port: {self.server_port})"
                )
                return False
        except Exception as e:
            logger.error(f"Error in leader election: {e}")
            # In case of error, default to taking over
            return True

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

            return False

    def _assume_leadership(self):
        """
        Assume leadership after being elected.
        """
        logger.info(
            "LEADERSHIP TAKEOVER: Replica assuming leader role due to leader failure"
        )

        # Shutdown replica operations
        self._shutdown_replica_operations()

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
        port = self.server_port
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        model_update_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
            servicer, grpc_server
        )
        grpc_server.add_insecure_port(f"[::]:{port}")
        grpc_server.start()

        # Register as leader in ZooKeeper for other replicas to discover
        self._register_as_leader()

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

    def _shutdown_replica_operations(self):
        """
        Shut down replica-specific operations before becoming a leader.
        """
        logger.info("Shutting down replica operations before becoming leader")

        # Mark as not running to stop threads
        self.running = False

        # Close connection to former leader
        if self.channel:
            self.channel.close()
            self.channel = None

        # Sleep briefly to allow threads to terminate
        time.sleep(2)

        logger.info("Replica operations shutdown complete")

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

            # Try to discover current leader in case the original one failed
            if self._discover_leader():
                logger.info(f"Found new leader at {self.leader_address}, connecting")
                if not self.connect_to_leader():
                    logger.error("Failed to connect to new leader. Retrying...")
                    time.sleep(10)
                    self.start()
                    return
            else:
                self.start()
                return

        # Subscribe to model updates
        update_thread = self.subscribe_to_updates()

        # Start heartbeat monitoring
        heartbeat_thread = self.start_heartbeat_monitoring()

        try:
            consecutive_failures = 0
            max_consecutive_failures = 3  # Number of consecutive checks before takeover
            check_interval = 3  # Seconds between leadership checks

            while self.running:
                time.sleep(check_interval)

                # Periodically check for leader changes
                if consecutive_failures == 0:  # Only check when things seem normal
                    self._discover_leader()

                # Check if leader is down
                if not self.leader_is_alive:
                    consecutive_failures += 1
                    logger.warning(
                        f"Leader is down! ({consecutive_failures}/{max_consecutive_failures}) Waiting for recovery..."
                    )

                    # Check for new leader first before trying to reconnect
                    if self._discover_leader():
                        logger.info(
                            f"Discovered new leader at {self.leader_address}, attempting to connect"
                        )
                        if self.connect_to_leader():
                            logger.info("Successfully connected to new leader")
                            consecutive_failures = 0
                            self.leader_is_alive = True
                            continue

                    # Try reconnecting to current leader
                    if self.connect_to_leader():
                        logger.info("Successfully reconnected to leader")
                        consecutive_failures = 0
                        self.leader_is_alive = True
                        continue

                    # Check if we should initiate election
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"Leader failed to recover after {consecutive_failures} attempts. Initiating election."
                        )

                        # Simple deterministic election based on port number
                        if self._elect_leader():
                            logger.info(
                                "This replica won the election and will become the new leader"
                            )
                            self._assume_leadership()
                            return  # Exit this method as we've now become the leader
                        else:
                            logger.info(
                                "Another replica was elected as leader, continuing as replica"
                            )
                            # Reset failure counter to give the new leader time to start
                            consecutive_failures = 0
                            # Wait a bit longer for the new leader to start up
                            time.sleep(15)
                            # Try to discover and connect to the new leader
                            if self._discover_leader():
                                logger.info(
                                    f"Discovered new leader at {self.leader_address}, attempting to connect"
                                )
                                if self.connect_to_leader():
                                    logger.info("Successfully connected to new leader")
                                    self.leader_is_alive = True
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
