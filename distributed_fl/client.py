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
import traceback
import zlib

import grpc
import model_update_pb2
import model_update_pb2_grpc
import torch
from logger import get_logger
from peft import PeftModel
from python_extractor import create_huggingface_dataset
from training import train_model, ModelArguments
from agent import LoraHuggingFaceAgent
from utils import load_safetensors_from_bytes, find_latest_adapter_version
from safetensors.torch import save_file

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
    def __init__(self, client_id, server_address="localhost:50051"):
        self.client_id = client_id
        self.server_address = server_address
        self.current_version = 0
        self.channel = grpc.insecure_channel(server_address)
        self.stub = model_update_pb2_grpc.FederatedLearningServiceStub(self.channel)
        self.initialize_agent()
        self.running = True
        self.lock = threading.Lock()

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

    def connect_to_server(self):
        """Establish connection with the server and update model if necessary."""
        try:
            connect_request = model_update_pb2.ClientConnection(
                client_id=self.client_id,
                current_version=self.current_version,
                ready_for_training=True,
            )

            version_info = self.stub.ConnectClient(connect_request)
            logger.info(
                f"Connected to server. Latest version: {version_info.latest_version}"
            )

            if version_info.update_available and version_info.model_state:
                with self.lock:
                    self.current_version = version_info.latest_version
                    logger.info(
                        f"Received newer model (version {self.current_version})"
                    )
                    decoded_dict = load_safetensors_from_bytes(version_info.model_state)
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
            traceback.print_exc()
            logger.info(f"Error connecting to server: {e}")
            return False

    def subscribe_to_updates(self):
        """Listen for model update notifications from the server."""

        def update_listener():
            try:
                subscription_request = model_update_pb2.ClientRequest(
                    client_id=self.client_id, current_version=self.current_version
                )
                for notification in self.stub.SubscribeToUpdates(subscription_request):
                    if not self.running:
                        break
                    logger.info(
                        f"Update notification: New version {notification.new_version} available"
                    )
                    with self.lock:
                        if notification.new_version > self.current_version:
                            self.get_latest_model()
            except Exception as e:
                if self.running:
                    logger.info(f"Update subscription error: {e}")
                    logger.info("Attempting to reconnect in 10 seconds...")
                    time.sleep(10)
                    if self.running:
                        self.subscribe_to_updates()

        listener_thread = threading.Thread(target=update_listener, daemon=True)
        listener_thread.start()
        return listener_thread

    def get_latest_model(self):
        """Request the latest aggregated model from the server."""
        try:
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
            logger.info(f"Error getting latest model: {e}")
            return False

    def train_and_submit(self, mode="debug"):
        """Simulate training and submit local update to the server."""
        try:
            logger.info("Training local model...")
            with self.lock:
                update_payload = self.get_adapter_update(self.agent)
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
                )
                logger.info(f"Submitting update to server (version: {current_ver})...")
                ack = self.stub.SubmitUpdate(update_message)
                logger.info(f"SubmitUpdate result: {ack.message}")

                if not ack.success and "outdated model" in ack.message:
                    logger.info("Server rejected update due to outdated model.")
                    self.get_latest_model()
                    return False
                return ack.success
        except Exception as e:
            import traceback

            logger.info(f"Error in training and submitting update: {e}")
            traceback.print_exc()
            return False

    def get_adapter_update(self, agent):
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
        train_dataset = create_huggingface_dataset(
            "/home/tlebryk/262_distributed_systems/Distributed-FL/data"
        )
        print(f"{len(train_dataset)=}")
        # train agent.model
        personal_adapters = os.path.join(PATH_TO_ADAPTERS, "client", "personal")
        personal_latest_version = self._get_latest_version(personal_adapters)
        output_dir = os.path.join(personal_adapters, str(personal_latest_version + 1))
        model_args = ModelArguments(output_dir=output_dir)

        agent.model, metrics = train_model(
            agent.model,
            agent.tokenizer,
            train_dataset,
            model_args=model_args,
        )
        with open(os.path.join(output_dir, "adapter_model.safetensors"), "rb") as f:
            bytes_ = f.read()
        return bytes_

    def run_training_loop(self, interval=10):
        """Main training loop with periodic update submissions."""
        try:
            while self.running:
                self.train_and_submit()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            logger.info(f"Training loop error: {e}")
            if self.running:
                logger.info("Restarting training loop...")
                time.sleep(5)
                self.run_training_loop(interval)

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

        # Create empty adapter.json file (or save actual config if available)
        # with open(os.path.join(version_dir, "adapter.json"), "w") as f:
        #     if hasattr(self.agent.model, "peft_config") and self.agent.model.peft_config:
        #         # If we have actual config, save it
        #         json.dump(self.agent.model.peft_config, f, indent=2)
        #     else:
        #         # Otherwise create an empty JSON object
        #         json.dump({}, f)

        # Update symlink to point to latest version
        if create_symlink:

            latest_link = os.path.join(PATH_TO_ADAPTERS, "client", "central", "latest")
            if os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                else:
                    shutil.rmtree(latest_link)
            os.symlink(f"v{version}", latest_link, target_is_directory=True)

        logger.info(f"Saved adapter version {version} to {version_dir}")
        return version_dir


def run(client_id="client_1", server_address="localhost:50051", interval=10):
    """
    Create and run the FederatedClient. The function accepts keyword arguments
    for customization.
    """
    client = FederatedClient(client_id=client_id, server_address=server_address)
    if not client.connect_to_server():
        logger.info("Failed to connect to server. Exiting.")
        return

    update_thread = client.subscribe_to_updates()
    try:
        client.run_training_loop(interval=interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        client.shutdown()
        update_thread.join(timeout=2)


def parse_args():
    """
    Parse command-line arguments and return them.
    """
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--client_id",
        type=str,
        default="client_1",
        help="Unique identifier for this client",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:50051",
        help="Server address in format host:port",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval (in seconds) between training submissions",
    )
    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
