# client.py
# %%
import argparse
import datetime
import io
import json
import logging
import os
import shutil
import threading
import time
import zlib

import grpc
import model_update_pb2
import model_update_pb2_grpc
import torch
from logger import get_logger
from peft import LoraConfig, get_peft_model, PeftModel
from python_extractor import create_huggingface_dataset
from training import LoraArguments, train_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = get_logger(__name__)

torch.set_num_threads(4)

PATH_TO_ADAPTERS = "./adapters"

os.makedirs(PATH_TO_ADAPTERS, exist_ok=True)


def serialize_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def get_adapter_update(model, tokenizer, lora_config):
    """
    Extract adapter-specific parameters, simulate a local training update by adding
    small Gaussian noise, and then compress the serialized adapter update.
    """
    state = model.state_dict()
    adapter_state = {k: v for k, v in state.items() if "lora_" in k}
    if not adapter_state:
        logger.info(
            "No adapter parameters found! Check your PEFT adapter configuration."
        )
    # train model here
    # get training data
    # TODO: figure out file paths
    train_dataset = create_huggingface_dataset(
        "/home/tlebryk/262_distributed_systems/Distributed-FL/distributed_fl/tests"
    )
    print(f"{len(train_dataset)=}")
    # print(f"{train_dataset[0]=}")
    # train model
    model, metrics = train_model(
        model,
        tokenizer,
        train_dataset,
        # model_args=model_args,
        # training_args=training_args,
        # data_args=data_args,
        lora_args=lora_config,
    )
    updated_adapter_state = {}
    for key, tensor in adapter_state.items():
        # noise = torch.randn_like(tensor) * 0.001
        updated_adapter_state[key] = tensor  # + noise
    payload = serialize_state_dict(updated_adapter_state)
    compressed_payload = zlib.compress(payload)
    return compressed_payload


class FederatedClient:
    def __init__(self, client_id, server_address="localhost:50051"):
        self.client_id = client_id
        self.server_address = server_address
        self.current_version = 1
        self.channel = grpc.insecure_channel(server_address)
        self.stub = model_update_pb2_grpc.FederatedLearningServiceStub(self.channel)
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.running = True
        self.lock = threading.Lock()

    def find_latest_adapter_version(self):
        """Find the latest adapter version on disk"""

        # Check if the latest symlink exists and is valid
        latest_link = os.path.join(PATH_TO_ADAPTERS, "latest")
        if os.path.islink(latest_link) and os.path.exists(
            os.path.realpath(latest_link)
        ):
            # Read the metadata file to get the version
            try:
                with open(os.path.join(latest_link, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                    return metadata.get("version", 0)
            except Exception as e:
                logger.info(f"Error reading latest adapter metadata: {e}")

        # If no valid symlink, scan all version directories
        try:
            version_dirs = [
                d
                for d in os.listdir(PATH_TO_ADAPTERS)
                if d.startswith("v")
                and os.path.isdir(os.path.join(PATH_TO_ADAPTERS, d))
            ]

            if not version_dirs:
                logger.info("No adapter versions found")
                return 0

            # Extract version numbers and find the max
            versions = [int(d[1:]) for d in version_dirs if d[1:].isdigit()]
            if versions:
                return max(versions)
            return 0
        except Exception as e:
            logger.info(f"Error finding latest adapter version: {e}")
            return 0

    def initialize_model(self, model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Load Qwen/Qwen2.5-Coder-0.5B-Instruct model and configure the PEFT adapter."""
        logger.info(
            "Loading Qwen/Qwen2.5-Coder-0.5B-Instruct model and configuring PEFT adapter..."
        )
        # list folders in PATH_TO_ADAPTERS
        adapter_folders = os.listdir(PATH_TO_ADAPTERS)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # device_maps="auto",
        )
        self.model = PeftModel.from_pretrained(
            self.model,  # Get the original base model without adapters
            os.path.join(PATH_TO_ADAPTERS, "latest"),
            is_trainable=False,  # Set as needed
        )
        self.lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            # lora_dropout=0.1,
            # target_modules=[
            #     "q_proj",
            #     "v_proj",
            #     "k_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
            # ],
        )

        self.load_latest_adapter()
        logger.info("Model with PEFT adapter loaded.")

    def initialize_tokenizer(self, model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Load Qwen/Qwen2.5-Coder-0.5B-Instruct tokenizer."""
        logger.info("Loading Qwen/Qwen2.5-Coder-0.5B-Instruct tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # if no pad token, add it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded.")

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
                    decompressed = zlib.decompress(version_info.model_state)
                    buffer = io.BytesIO(decompressed)
                    adapter_state = torch.load(buffer)
                    self.model.load_state_dict(adapter_state, strict=False)
                    logger.info("Updated local model with latest adapter state")
            return True
        except Exception as e:
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
                with self.lock:
                    decompressed = zlib.decompress(aggregated.model_state)
                    buffer = io.BytesIO(decompressed)
                    adapter_state = torch.load(buffer)
                    self.model.load_state_dict(adapter_state, strict=False)
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
            time.sleep(3)  # Simulate training time
            with self.lock:
                update_payload = get_adapter_update(
                    self.model, self.tokenizer, self.lora_config
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

    def _extract_version_from_path(self, path):
        """Extract version number from adapter path."""
        # Example path: ./adapters/adapter_1.2.3_r42_20250423
        try:
            dirname = os.path.basename(path)
            parts = dirname.split("_")
            if len(parts) >= 2 and parts[0] == "adapter":
                return parts[1]  # Return the version part (1.2.3)
        except:
            pass
        return "0.0.0"  # Default if parsing fails

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

    def save_adapter_to_disk(self, version, create_symlink=True):
        """Save adapter state to disk using the versioning system"""
        version_dir = os.path.join(PATH_TO_ADAPTERS, f"v{version}")

        # Create version directory if it doesn't exist
        os.makedirs(version_dir, exist_ok=True)

        # Save adapter files
        # adapter_path = os.path.join(version_dir, "adapter.safetensors")
        # buffer = io.BytesIO()
        # torch.save(adapter_state, buffer)
        # with open(adapter_path, "wb") as f:
        #     f.write(buffer.getvalue())
        self.model.save_pretrained(version_dir)

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
        #     if hasattr(self.model, "peft_config") and self.model.peft_config:
        #         # If we have actual config, save it
        #         json.dump(self.model.peft_config, f, indent=2)
        #     else:
        #         # Otherwise create an empty JSON object
        #         json.dump({}, f)

        # Update symlink to point to latest version
        if create_symlink:
            latest_link = os.path.join(PATH_TO_ADAPTERS, "latest")
            if os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                else:
                    shutil.rmtree(latest_link)
            os.symlink(f"v{version}", latest_link, target_is_directory=True)

        logger.info(f"Saved adapter version {version} to {version_dir}")
        return version_dir

    def load_latest_adapter(self):
        """Load the latest adapter from disk"""
        # Find the latest version
        # latest_version = self.find_latest_adapter_version()
        self.model = PeftModel.from_pretrained(
            self.model.get_base_model(),  # Get the original base model without adapters
            os.path.join(PATH_TO_ADAPTERS, f"latest"),
            is_trainable=False,  # Set as needed
        )


def run(client_id="client_1", server_address="localhost:50051", interval=10):
    """
    Create and run the FederatedClient. The function accepts keyword arguments
    for customization.
    """
    client = FederatedClient(client_id=client_id, server_address=server_address)
    client.initialize_model()
    client.initialize_tokenizer()
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
