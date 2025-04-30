# agent.py
import os
from abc import ABC, abstractmethod

import torch
from logger import get_logger
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import find_latest_adapter_version


logger = get_logger(__name__)

PATH_TO_ADAPTERS = os.environ.get("PATH_TO_ADAPTERS", "./distributed_fl/adapters")

os.makedirs(PATH_TO_ADAPTERS, exist_ok=True)


class CodeGenerationAgent(ABC):
    @abstractmethod
    def generate_solution(self, prompt: str) -> str:
        """
        Generate a solution based on the provided prompt.
        Must be implemented by subclasses.
        """
        pass


class LoraHuggingFaceAgent(CodeGenerationAgent):
    """
    A CodeGenerationAgent that uses a Hugging Face model for code generation.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: dict = None,
        device: str = "cpu",
        model_load_kwargs: dict = {},
        adapter_path: str = None,
    ):
        """
        Initialize the agent with a model name and optional generation configuration.

        Args:
            model_name (str): Name or path of the Hugging Face model.
            generation_config (dict, optional): Generation configuration parameters (e.g., max_length, temperature).
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device

        # Use the provided generation config or rely on the model's defaults
        self.generation_config = generation_config
        self.adapter_path = adapter_path
        self.initialize_tokenizer(model_name)
        self.initialize_model(model_name, adapter_path)

    def generate_solution(self, prompt: str) -> str:
        """
        Generate a solution based on the prompt using the Hugging Face model.

        Args:
            prompt (str): The input prompt for the coding task.

        Returns:
            str: Generated code as a string.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # If generation_config is provided, use it, otherwise rely on model defaults
        if self.generation_config:
            output_ids = self.model.generate(input_ids, **self.generation_config)
        else:
            output_ids = self.model.generate(input_ids)

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    def initialize_model(
        self,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        adapter_path=None,
        adapter_config=None,
    ):
        """Load Qwen/Qwen2.5-Coder-0.5B-Instruct model and configure the PEFT adapter."""
        logger.info(
            "Loading Qwen/Qwen2.5-Coder-0.5B-Instruct model and configuring PEFT adapter..."
        )
        print(f"Loading {model_id} model...")
        # list folders in PATH_TO_ADAPTERS

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # torch_dtype=torch.float16,
            # device_maps="auto",
        )
        if adapter_config is None:
            adapter_config = LoraConfig()
        print(f"{adapter_path=}")
        # TODO: set a default adapter config...
        if adapter_path is None:
            print("Using adapter config from code")
            self.model = get_peft_model(self.model, adapter_config)
        else:
            print("Loading adapter from path")
            self.model = PeftModel.from_pretrained(
                self.model,  # Get the original base model without adapters
                adapter_path,
                is_trainable=False,  # Set as needed
            )
        logger.info("Model with PEFT adapter loaded.")

    def initialize_tokenizer(self, model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Load Qwen/Qwen2.5-Coder-0.5B-Instruct tokenizer."""
        logger.info("Loading Qwen/Qwen2.5-Coder-0.5B-Instruct tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # if no pad token, add it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded.")

    def load_latest_adapter(self, adapter_path=None):
        """Load the latest adapter from disk"""
        # Find the latest version
        if adapter_path is None:
            adapter_path = os.path.join(
                PATH_TO_ADAPTERS,
                "client",
                "central",
            )
        latest_version = find_latest_adapter_version(adapter_path=adapter_path)
        full_adapter_path = os.path.join(adapter_path, f"v{latest_version}")

        self.model = PeftModel.from_pretrained(
            self.model.get_base_model(),  # Get the original base model without adapters
            os.path.join(full_adapter_path),
            is_trainable=False,  # Set as needed
        )

    def _get_base_model(self):
        self.model = self.model.get_base_model()
