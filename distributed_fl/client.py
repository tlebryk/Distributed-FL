import io
import threading
import time
import zlib

import grpc
import model_update_pb2
import model_update_pb2_grpc
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel


def serialize_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def get_adapter_update(model):
    """
    Extract the adapter-specific parameters from the model by filtering for keys
    that contain "lora_". Simulate a local update by adding small Gaussian noise to
    each adapter parameter. Then serialize and compress the adapter update.
    """
    # Get the complete state dict and filter adapter parameters
    state = model.state_dict()
    adapter_state = {k: v for k, v in state.items() if "lora_" in k}
    if not adapter_state:
        print("No adapter parameters found! Check your PEFT adapter configuration.")
    updated_adapter_state = {}
    for key, tensor in adapter_state.items():
        # Add slight noise to simulate a local training update
        noise = torch.randn_like(tensor) * 0.001
        updated_adapter_state[key] = tensor + noise
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
        self.running = True
        self.lock = threading.Lock()

    def initialize_model(self):
        """Load GPT-2 model and configure PEFT adapter"""
        print("Loading GPT-2 model and configuring PEFT adapter...")
        # Load GPT-2 model
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Configure a LoRA adapter using PEFT for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, lora_config)
        print("Model with PEFT adapter loaded.")

    def connect_to_server(self):
        """Establish initial connection with server and check for model updates"""
        try:
            connect_request = model_update_pb2.ClientConnection(
                client_id=self.client_id,
                current_version=self.current_version,
                ready_for_training=True,
            )

            version_info = self.stub.ConnectClient(connect_request)
            print(f"Connected to server. Latest version: {version_info.latest_version}")

            # Update model if server sent a newer version
            if version_info.update_available and version_info.model_state:
                with self.lock:
                    self.current_version = version_info.latest_version
                    print(f"Received newer model (version {self.current_version})")

                    # Update local model with received adapter state
                    decompressed = zlib.decompress(version_info.model_state)
                    buffer = io.BytesIO(decompressed)
                    adapter_state = torch.load(buffer)

                    # Load adapter weights into model
                    self.model.load_state_dict(adapter_state, strict=False)
                    print("Updated local model with latest adapter state")

            return True
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return False

    def subscribe_to_updates(self):
        """Listen for model update notifications from server"""

        def update_listener():
            try:
                subscription_request = model_update_pb2.ClientRequest(
                    client_id=self.client_id, current_version=self.current_version
                )

                for notification in self.stub.SubscribeToUpdates(subscription_request):
                    if not self.running:
                        break

                    print(
                        f"Update notification: New version {notification.new_version} available"
                    )

                    # Only update if the notified version is newer
                    with self.lock:
                        if notification.new_version > self.current_version:
                            self.get_latest_model()
            except Exception as e:
                if self.running:  # Only log if we're not shutting down
                    print(f"Update subscription error: {e}")
                    print("Attempting to reconnect in 10 seconds...")
                    time.sleep(10)
                    if self.running:
                        self.subscribe_to_updates()  # Try again

        # Start update listener in a background thread
        listener_thread = threading.Thread(target=update_listener, daemon=True)
        listener_thread.start()
        return listener_thread

    def get_latest_model(self):
        """Request the latest aggregated model from the server"""
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

                    # Update model
                    self.model.load_state_dict(adapter_state, strict=False)
                    self.current_version = aggregated.version
                    print(f"Updated model to version {self.current_version}")
                return True
            else:
                print("No model state received or no newer model available")
                return False
        except Exception as e:
            print(f"Error getting latest model: {e}")
            return False

    def train_and_submit(self):
        """Simulate local training and submit updates to the server"""
        try:
            print("Training local model...")
            time.sleep(3)  # Simulate training time

            with self.lock:
                # Generate update
                update_payload = get_adapter_update(self.model)
                current_ver = (
                    self.current_version
                )  # Capture current version within lock

            # Submit update to server
            update_message = model_update_pb2.ModelUpdate(
                client_id=self.client_id,
                update=update_payload,
                version=current_ver,
                timestamp=int(time.time()),
            )

            print(f"Submitting update to server (version: {current_ver})...")
            ack = self.stub.SubmitUpdate(update_message)
            print(f"SubmitUpdate result: {ack.message}")

            # If server rejected due to version mismatch, get latest model
            if not ack.success and "outdated model" in ack.message:
                print("Server rejected update due to outdated model.")
                self.get_latest_model()
                return False

            return ack.success
        except Exception as e:
            print(f"Error in training and submitting update: {e}")
            return False

    def run_training_loop(self, interval=10):
        """Main training loop with periodic submissions"""
        try:
            while self.running:
                success = self.train_and_submit()
                # Wait between training rounds
                time.sleep(interval)
        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            print(f"Training loop error: {e}")
            if self.running:
                print("Restarting training loop...")
                time.sleep(5)
                self.run_training_loop(interval)

    def shutdown(self):
        """Cleanly shut down the client"""
        print("Shutting down client...")
        self.running = False
        if self.channel:
            self.channel.close()


def run():
    # Create and run client
    client = FederatedClient(client_id="client_1")
    client.initialize_model()

    # Connect to server and get initial model if needed
    if not client.connect_to_server():
        print("Failed to connect to server. Exiting.")
        return

    # Subscribe to model updates
    update_thread = client.subscribe_to_updates()

    # Run the main training loop
    try:
        client.run_training_loop()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client.shutdown()
        update_thread.join(timeout=2)


if __name__ == "__main__":
    run()
