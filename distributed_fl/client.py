# client.py
# %%
import argparse
import datetime
import io
import json
import os
import re
import shutil
import threading
import time
from statistics import mean

import grpc
import model_update_pb2
import model_update_pb2_grpc
import torch
from agent import LoraHuggingFaceAgent
from logger import get_logger
from python_extractor import create_huggingface_dataset
from safetensors.torch import save_file
from training import ModelArguments, train_model
from utils import find_latest_adapter_version, load_safetensors_from_bytes

logger = get_logger(__name__)

torch.set_num_threads(4)

PATH_TO_ADAPTERS = os.environ.get("PATH_TO_ADAPTERS", "./distributed_fl/adapters")

os.makedirs(PATH_TO_ADAPTERS, exist_ok=True)

# TODO: switch to streaming
CHANNEL_OPTS = [
    ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 100 MiB
    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
]


def serialize_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


class FederatedClient:
    def __init__(
        self, client_id, server_address="localhost:50051", fallback_addresses=None
    ):
        self.client_id = client_id
        self.server_address = server_address
        # Default fallback addresses if none provided
        self.fallback_addresses = fallback_addresses or []
        self.current_version = 0
        self.channel = None
        self.stub = None
        self.initialize_connection()
        self.initialize_agent()
        self.running = True
        self.lock = threading.Lock()

    def initialize_connection(self):
        """Initialize or reinitialize the gRPC connection to the server."""
        if self.channel:
            self.channel.close()

        logger.info(f"Connecting to server at {self.server_address}")
        self.channel = grpc.insecure_channel(self.server_address, options=CHANNEL_OPTS)
        self.stub = model_update_pb2_grpc.FederatedLearningServiceStub(self.channel)

    def update_server_address(self, new_address):
        """Update the server address and reconnect."""
        if new_address and new_address != self.server_address:
            logger.info(
                f"Updating server address from {self.server_address} to {new_address}"
            )
            self.server_address = new_address
            self.initialize_connection()
            return True
        return False

    def try_fallback_servers(self):
        """Try connecting to fallback servers when primary connection fails."""
        if not self.fallback_addresses:
            logger.warning("No fallback addresses configured")
            return False

        # Store original address to restore if all fallbacks fail
        original_address = self.server_address

        for address in self.fallback_addresses:
            if address == original_address:
                continue  # Skip the current address

            logger.info(f"Trying fallback server at {address}")
            self.update_server_address(address)

            try:
                # Try a simple heartbeat check to see if this server is alive
                request = model_update_pb2.HeartbeatRequest(
                    sender_id=self.client_id, timestamp=int(time.time())
                )
                response = self.stub.SendHeartbeat(request, timeout=3)

                if response.alive:
                    logger.info(
                        f"Successfully connected to fallback server at {address}"
                    )
                    return True
            except Exception as e:
                logger.warning(f"Fallback server at {address} unavailable: {e}")

        # If we get here, all fallbacks failed
        logger.error("All fallback servers unavailable")

        # Restore original address
        self.update_server_address(original_address)
        return False

    def check_connection(self, max_retries=2):
        """Check current connection and try fallbacks if needed."""
        for attempt in range(max_retries):
            try:
                # Try a simple heartbeat to check connection
                request = model_update_pb2.HeartbeatRequest(
                    sender_id=self.client_id, timestamp=int(time.time())
                )
                response = self.stub.SendHeartbeat(request, timeout=3)
                if response.alive:
                    return True
            except Exception as e:
                logger.warning(f"Connection check failed (attempt {attempt+1}): {e}")

                # Try fallback servers
                if self.try_fallback_servers():
                    return True

                # Small delay before retry
                time.sleep(1)

        return False

    def initialize_agent(
        self, model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct", adapter_path=None
    ):
        if adapter_path is not None:
            latest_version = find_latest_adapter_version(
                os.path.join(PATH_TO_ADAPTERS, "client", "central")
            )
            adapter_path = os.path.join(
                PATH_TO_ADAPTERS, "client", "central", f"v{latest_version}"
            )
        self.agent = LoraHuggingFaceAgent(
            model_name=model_id, adapter_path=adapter_path
        )

    def connect_to_server(self, max_retries=3):
        """Establish connection with the server and update model if necessary."""
        retries = 0

        while retries < max_retries:
            try:
                # Check if current connection is working
                if retries > 0 and not self.check_connection():
                    logger.warning(
                        "Server connection failed, trying fallback servers..."
                    )
                    if not self.try_fallback_servers():
                        retries += 1
                        time.sleep(2)
                        continue

                # Now proceed with regular connection
                connect_request = model_update_pb2.ClientConnection(
                    client_id=self.client_id,
                    current_version=self.current_version,
                    ready_for_training=True,
                )

                version_info = self.stub.ConnectClient(connect_request)
                logger.info(
                    f"Connected to server at {self.server_address}. Latest version: {version_info.latest_version}"
                )

                if version_info.update_available and version_info.model_state:
                    with self.lock:
                        self.current_version = version_info.latest_version
                        logger.info(
                            f"Received newer model (version {self.current_version})"
                        )
                        decoded_dict = load_safetensors_from_bytes(
                            version_info.model_state
                        )
                        # save the dict to a safetensors file
                        path_dir = os.path.join(
                            PATH_TO_ADAPTERS,
                            "client",
                            "central",
                            f"v{version_info.latest_version}",
                        )
                        os.makedirs(path_dir, exist_ok=True)
                        save_file(
                            decoded_dict,
                            os.path.join(
                                path_dir,
                                "adapter_model.safetensors",
                            ),
                        )
                        shutil.copy2(
                            os.path.join(PATH_TO_ADAPTERS, "adapter_config.json"),
                            os.path.join(
                                path_dir,
                                "adapter_config.json",
                            ),
                        )
                        self.agent.model.load_state_dict(decoded_dict, strict=False)
                        logger.info("Updated local model with latest adapter state")
                return True
            except Exception as e:
                logger.warning(f"Connection attempt {retries+1} failed: {e}")
                retries += 1

                # Try fallback servers if primary connection failed
                if retries <= 1:  # Only try fallbacks on first retry
                    logger.info("Trying fallback servers...")
                    if self.try_fallback_servers():
                        continue

                # Wait before retrying
                time.sleep(2)

        logger.error(f"Failed to connect to server after {max_retries} attempts")
        return False

    def subscribe_to_updates(self):
        """Listen for model update notifications from the server."""

        def update_listener():
            while self.running:
                try:
                    # Check if connection is alive
                    if not self.check_connection():
                        logger.warning(
                            "Lost connection to server. Trying to reconnect..."
                        )
                        if not self.connect_to_server():
                            logger.error(
                                "Failed to reconnect to server. Retrying in 10 seconds..."
                            )
                            time.sleep(10)
                            continue

                    subscription_request = model_update_pb2.ClientRequest(
                        client_id=self.client_id, current_version=self.current_version
                    )
                    for notification in self.stub.SubscribeToUpdates(
                        subscription_request
                    ):
                        if not self.running:
                            break
                        logger.info(
                            f"Update notification: New version {notification.new_version} available"
                        )
                        with self.lock:
                            if notification.new_version > self.current_version:
                                self.get_latest_model()

                    # If we exit the loop normally, there was likely a disconnection
                    if self.running:
                        logger.warning(
                            "Update subscription ended unexpectedly. Reconnecting..."
                        )
                        time.sleep(5)

                except Exception as e:
                    if self.running:
                        logger.warning(f"Update subscription error: {e}")
                        logger.info("Attempting to reconnect in 10 seconds...")
                        time.sleep(10)

        listener_thread = threading.Thread(target=update_listener, daemon=True)
        listener_thread.start()
        return listener_thread

    def get_latest_model(self, max_retries=3):
        """Request the latest aggregated model from the server."""
        retries = 0

        while retries < max_retries:
            try:
                # Check if connection is alive
                if retries > 0 and not self.check_connection():
                    logger.warning("Connection to server lost. Trying to reconnect...")
                    if not self.connect_to_server():
                        retries += 1
                        time.sleep(2)
                        continue

                client_request = model_update_pb2.ClientRequest(
                    client_id=self.client_id, current_version=self.current_version
                )
                aggregated = self.stub.GetAggregatedModel(client_request)

                if aggregated.model_state:
                    logger.info(
                        f"Received model state of length {len(aggregated.model_state)}"
                    )
                    logger.info("Successfully loaded adapter state")
                    adapter_state = load_safetensors_from_bytes(aggregated.model_state)
                    self.agent.model.load_state_dict(adapter_state, strict=False)
                    logger.info("Successfully loaded adapter state into model")
                    version_dir = os.path.join(
                        PATH_TO_ADAPTERS, "client", "central", f"v{aggregated.version}"
                    )
                    os.makedirs(version_dir, exist_ok=True)
                    logger.info(f"Saving adapter state to {version_dir}")
                    self.save_adapter_to_disk(version_dir, aggregated.version)
                    shutil.copy2(
                        os.path.join(PATH_TO_ADAPTERS, "adapter_config.json"),
                        os.path.join(version_dir, "adapter_config.json"),
                    )
                    self.current_version = aggregated.version
                    logger.info(f"Updated model to version {self.current_version}")
                    return True
                else:
                    logger.info("No model state received or no newer model available")
                    return False

            except Exception as e:
                logger.warning(f"Error getting latest model (attempt {retries+1}): {e}")
                retries += 1
                time.sleep(2)

        logger.error(f"Failed to get latest model after {max_retries} attempts")
        return False

    def train_and_submit(self, mode="debug", code_path="./data", max_retries=3):
        """Simulate training and submit local update to the server."""
        retries = 0

        while retries < max_retries:
            try:
                # Check if connection is alive
                if retries > 0 and not self.check_connection():
                    logger.warning("Connection to server lost. Trying to reconnect...")
                    if not self.connect_to_server():
                        retries += 1
                        time.sleep(2)
                        continue

                logger.info("Training local model...")
                with self.lock:
                    update_payload, pylint_score = self.get_adapter_update(
                        self.agent, code_path
                    )
                    if mode == "debug":
                        payload_size_bytes = len(update_payload)
                        payload_size_kb = payload_size_bytes / 1024
                        payload_size_mb = payload_size_kb / 1024
                        logger.info(
                            f"Update payload size: {payload_size_bytes:,} bytes ({payload_size_kb:.2f} KB, {payload_size_mb:.4f} MB)"
                        )

                    current_ver = self.current_version
                    update_message = model_update_pb2.ModelUpdate(
                        client_id=self.client_id,
                        update=update_payload,
                        version=current_ver,
                        timestamp=int(time.time()),
                        pylint_score=pylint_score,
                    )
                    logger.info(
                        f"Submitting update to server (version: {current_ver})..."
                    )
                    ack = self.stub.SubmitUpdate(update_message)
                    logger.info(f"SubmitUpdate result: {ack.message}")

                    if not ack.success and "outdated model" in ack.message:
                        logger.info("Server rejected update due to outdated model.")
                        self.get_latest_model()
                        return False
                    return ack.success

            except Exception as e:
                logger.warning(
                    f"Error in training and submitting update (attempt {retries+1}): {e}"
                )
                retries += 1
                time.sleep(2)

        logger.error(f"Failed to train and submit update after {max_retries} attempts")
        return False

    def get_adapter_update(self, agent, code_path="./data"):
        """
        Extract adapter-specific parameters, simulate a local training update by adding
        small Gaussian noise, and then compress the serialized adapter update.
        """
        state = agent.model.state_dict()
        adapter_state = {k: v for k, v in state.items() if "lora_" in k}
        if not adapter_state:
            logger.info(
                "No adapter parameters found! Check your PEFT adapter configuration."
            )
        # train agent.model here
        # get training data
        # TODO: figure out file paths
        train_dataset = create_huggingface_dataset(code_path)
        print(f"{len(train_dataset)=}")
        # train agent.model
        personal_adapters = os.path.join(PATH_TO_ADAPTERS, "client", "personal")
        personal_latest_version = self._get_latest_version(personal_adapters)
        output_dir = os.path.join(personal_adapters, f"v{personal_latest_version + 1}")
        model_args = ModelArguments(output_dir=output_dir)

        agent.model, metrics = train_model(
            agent.model,
            agent.tokenizer,
            train_dataset,
            model_args=model_args,
        )
        with open(os.path.join(output_dir, "adapter_model.safetensors"), "rb") as f:
            bytes_ = f.read()
        # TODO: decouple pylint and bytes and get cleaner average?
        pylint_score = mean(train_dataset["pylint_score"])
        logger.info(f"Average pylint score: {pylint_score}")

        return bytes_, pylint_score

    def run_training_loop(self, interval=10, code_path="./data"):
        """Main training loop with periodic update submissions."""
        try:
            while self.running:
                success = self.train_and_submit(code_path=code_path)
                if not success:
                    logger.warning("Training and update submission failed. Will retry.")
                time.sleep(interval)
        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            logger.error(f"Training loop error: {e}")
            if self.running:
                logger.info("Restarting training loop...")
                time.sleep(5)
                self.run_training_loop(interval, code_path)

    def shutdown(self):
        """Cleanly shut down the client."""
        logger.info("Shutting down client...")
        self.running = False
        if self.channel:
            self.channel.close()

    @staticmethod
    def get_version(path: str) -> int:
        """
        Extracts the last version number from the given path. If no version is found,
        returns 0. Version segments are of the form 'v<number>'. The last matching
        segment in the path is used.
        """
        version = 0
        for segment in path.split("/"):
            match = re.fullmatch(r"v(\d+)", segment)
            if match:
                version = int(match.group(1))
        return version

    def _get_latest_version(self, dir_path: str) -> int:
        """
        Scans the contents of the given directory and returns the highest version
        number found among its entries, based on 'v<number>' segments.
        If no versions are found or the directory is invalid, returns 0.
        """
        max_version = 0
        try:
            for entry in os.listdir(dir_path):
                # Only consider the name of the entry for version extraction
                version = self.get_version(entry)
                if version > max_version:
                    max_version = version
        except (OSError, FileNotFoundError):
            return 0
        return max_version

    def _get_round_from_path(self, path):
        """Extract round number from adapter path."""
        try:
            dirname = os.path.basename(path)
            parts = dirname.split("_")
            if len(parts) >= 3 and parts[0] == "adapter" and parts[2].startswith("r"):
                return int(parts[2][1:])  # Return the round part without 'r' (42)
        except:
            pass
        return 0  # Default if parsing fails

    def save_adapter_to_disk(self, version_dir, version, create_symlink=True):
        """Save adapter state to disk using the versioning system"""

        self.agent.model.save_pretrained(version_dir)

        # Create metadata file
        metadata = {
            "version": version,
            "timestamp": datetime.datetime.now().isoformat(),
            "client_id": self.client_id,
            "server_address": self.server_address,
        }

        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved adapter version {version} to {version_dir}")
        return version_dir


def run(
    client_id="client_1",
    server_address="localhost:50051",
    fallback_addresses=None,
    interval=3,
    code_path="./data",
):
    """
    Create and run the FederatedClient. The function accepts keyword arguments
    for customization.

    Args:
        client_id: Unique identifier for this client
        server_address: Primary server address in format host:port
        fallback_addresses: List of fallback server addresses to try if primary fails
        interval: Interval (in seconds) between training submissions
        code_path: Path to the code directory
    """
    # Default fallback servers if none provided (will be empty list if None)
    if fallback_addresses is None:
        # Generate fallback addresses based on common replica ports
        base_parts = server_address.split(":")
        if len(base_parts) == 2:
            host = base_parts[0]
            port = int(base_parts[1])
            # Create fallbacks with incrementing port numbers
            fallback_addresses = [f"{host}:{port+i}" for i in range(1, 4)]
            logger.info(
                f"Using auto-generated fallback addresses: {fallback_addresses}"
            )

    # Create the client with the initial server address and fallbacks
    client = FederatedClient(
        client_id=client_id,
        server_address=server_address,
        fallback_addresses=fallback_addresses,
    )

    # Try to connect to the server
    max_retries = 5
    for attempt in range(max_retries):
        if client.connect_to_server():
            logger.info(f"Successfully connected to server at {client.server_address}")
            break
        else:
            if attempt < max_retries - 1:
                logger.warning(f"Connection attempt {attempt+1} failed. Retrying...")
                time.sleep(5)
            else:
                logger.error(
                    f"Failed to connect after {max_retries} attempts. Exiting."
                )
                return

    # Start the update subscription
    update_thread = client.subscribe_to_updates()

    try:
        # Run the main training loop
        client.run_training_loop(interval=interval, code_path=code_path)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # Clean shutdown
        client.shutdown()
        update_thread.join(timeout=2)


def parse_args():
    """
    Parse command-line arguments and return them.
    """
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--client_id",
        "-c",
        type=str,
        default="client_1",
        help="Unique identifier for this client",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:50051",
        help="Primary server address in format host:port",
    )
    parser.add_argument(
        "--fallback_addresses",
        type=str,
        nargs="*",  # Accept multiple values or none
        help="List of fallback server addresses to try if primary fails (e.g., localhost:50052 localhost:50053)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Interval (in seconds) between training submissions",
    )
    parser.add_argument(
        "--code_path",
        type=str,
        default="./data",
        help="Path to the code directory",
    )
    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()
    run(
        client_id=args.client_id,
        server_address=args.server_address,
        fallback_addresses=args.fallback_addresses,
        interval=args.interval,
        code_path=args.code_path,
    )
