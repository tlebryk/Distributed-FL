# training.py
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import os
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from dataclasses import dataclass, asdict
from python_extractor import extract_from_file

from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


@dataclass
class LoraArguments:
    """Arguments for LoRA fine-tuning configuration."""

    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"  # Options: "none", "all", "lora_only"
    target_modules: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None
    dtype: Optional[str] = None  # "float16", "float32", "bfloat16"
    # SKIP QUANTIZATION FOR CPU INDEFINITELY
    # quantization_config: Optional[Dict[str, Any]] = None
    # use_8bit_quantization: bool = False
    # use_4bit_quantization: bool = False


@dataclass
class DataArguments:
    """Arguments for data processing configuration."""

    prompt_column: str = "prompt"
    completion_column: str = "completion"
    separator: str = " "
    max_length: int = 1024
    preprocessing_fn: Optional[Callable] = None


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    output_dir: str = "./model_output"
    logging_dir: str = "./logs"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class CustomTrainingArguments:
    """Custom wrapper for HuggingFace TrainingArguments with simplified parameters."""

    num_train_epochs: float = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 2
    save_steps: int = 1000
    eval_steps: int = 1000
    fp16: bool = True
    bf16: bool = False
    save_total_limit: int = 3
    report_to: Union[List[str], str] = "none"
    # remove_unused_columns: bool = False
    # Add any other training arguments you want to expose

    def to_transformers_args(
        self, output_dir: str, has_val_dataset: bool
    ) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        args_dict = asdict(self)

        # Calculate warmup steps from ratio
        # We'll need to set this dynamically in the train function after seeing dataset size

        args_dict.update(
            {
                "output_dir": output_dir,
                "overwrite_output_dir": True,
                # "evaluation_strategy": "steps" if has_val_dataset else "no",
                "save_strategy": "steps",
                # "load_best_model_at_end": has_val_dataset,
                "metric_for_best_model": "eval_loss" if has_val_dataset else None,
                "greater_is_better": False,
            }
        )

        return TrainingArguments(**args_dict)


def get_training_data():
    """Naive implementation: get all python files"""
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                extract_from_file(os.path.join(root, file))


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Union[Dataset, Dict[str, List]],
    val_dataset: Optional[Union[Dataset, Dict[str, List]]] = None,
    model_args: Optional[ModelArguments] = None,
    training_args: Optional[CustomTrainingArguments] = None,
    data_args: Optional[DataArguments] = None,
) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Train a causal language model on a dataset of prompt-completion pairs.
    Supports standard fine-tuning and parameter-efficient fine-tuning with LoRA.

    Args:
        model: A Hugging Face causal language model
        tokenizer: Tokenizer corresponding to the model
        train_dataset: Dataset containing prompt-completion pairs
        val_dataset: Optional validation dataset
        model_args: Configuration for model output and logging
        training_args: Configuration for training hyperparameters
        data_args: Configuration for data processing
        lora_args: Configuration for LoRA adapters

    Returns:
        The trained model and training metrics
    """
    # Set default configurations if not provided
    model_args = model_args or ModelArguments()
    training_args = training_args or CustomTrainingArguments()
    data_args = data_args or DataArguments()

    # Create output directory if it doesn't exist
    os.makedirs(model_args.output_dir, exist_ok=True)
    os.makedirs(model_args.logging_dir, exist_ok=True)

    # Ensure datasets are in the correct format
    if isinstance(train_dataset, dict):
        train_dataset = Dataset.from_dict(train_dataset)

    if val_dataset is not None and isinstance(val_dataset, dict):
        val_dataset = Dataset.from_dict(val_dataset)

    def formatting_prompts_func(example):
        output_texts = []
        text = (
            f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
        )
        output_texts.append(text)
        return text

    response_template = " ### Answer:"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer, padding_free=True
    )
    transformers_training_args = SFTConfig(
        output_dir="./tmp",
        gradient_checkpointing=True,
        num_train_epochs=1,
        learning_rate=5e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=False,
        fp16=False,
        save_total_limit=3,
        per_device_train_batch_size=1,
        packing=True,
        gradient_accumulation_steps=8,
        logging_steps=1,  # Log after every batch; adjust as needed.
        disable_tqdm=False,  # Ensure the tqdm progress bar is enabled.
    )
    model.train()
    if isinstance(model, PeftModel):
        for adapter in model.active_adapters:
            # Enable training for all active adapters
            model.peft_config[adapter].inference_mode = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        adapter_name = model.active_adapter
        for name, param in model.named_parameters():

            if (
                adapter_name in name or "lora_" in name
            ):  # Adjust this condition as necessary for your adapter type
                param.requires_grad = True
                # print(f"  - Enabling grad for: {name}") # Uncomment for detailed logging
        model.print_trainable_parameters()
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=transformers_training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"{name} is frozen.")

    training_results = trainer.train()

    # Train the model
    # training_results = trainer.train(
    #     resume_from_checkpoint=model_args.resume_from_checkpoint
    # )

    # TODO: figure out when and where to save model
    trainer.save_model(model_args.output_dir)
    tokenizer.save_pretrained(model_args.output_dir)

    # Also save training arguments
    with open(os.path.join(model_args.output_dir, "training_args.txt"), "w") as f:
        f.write(str(transformers_training_args))

    # Return the trained model and results
    return model, training_results


def load_and_quantize_model(
    model_name_or_path: str,
    use_8bit: bool = False,
    use_4bit: bool = False,
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model with optional quantization.

    Args:
        model_name_or_path: Name or path of the pre-trained model
        use_8bit: Whether to load in 8-bit mode
        use_4bit: Whether to load in 4-bit mode
        device_map: Device mapping strategy
        quantization_config: Additional quantization parameters

    Returns:
        Loaded model and tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Set up quantization config if needed
    bnb_config = None
    if use_4bit or use_8bit or quantization_config:
        if quantization_config is None:
            quantization_config = {}

        # Base quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            load_in_8bit=use_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            **quantization_config,
        )

    # Load model with quantization if specified
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# Example of usage
def run_example():
    """Example of how to use the training function with LoRA."""
    # Load model with quantization
    model, tokenizer = load_and_quantize_model(
        model_name_or_path="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        # use_8bit=True,  # Use 8-bit quantization
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=[
        #     "q_proj",
        #     "v_proj",
        #     "k_proj",
        #     "o_proj",
        #     "gate_proj",
        #     "up_proj",
        #     "down_proj",
        # ],
    )
    model = get_peft_model(model, lora_config)
    # Create or load your dataset
    train_data = {
        "instruction": [
            "What is the capital of France?",
            "Tell me about machine learning",
        ],
        "output": [
            " The capital of France is Paris.",
            " Machine learning is a field of AI...",
        ],
    }
    train_dataset = Dataset.from_dict(train_data)

    # Optional validation dataset
    val_data = {
        "instruction": ["What is the largest planet?"],
        "output": [" Jupiter is the largest planet in our solar system."],
    }
    val_dataset = Dataset.from_dict(val_data)

    # Configure LoRA
    lora_args = LoraArguments(
        enabled=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        # Target specific modules - will use defaults if not specified
    )

    # Configure training
    training_args = CustomTrainingArguments(
        num_train_epochs=1, per_device_train_batch_size=4, learning_rate=2e-5, fp16=True
    )

    # Configure model output
    model_args = ModelArguments(output_dir="./fine-tuned-lora-model")

    # Train the model
    model, training_results = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_args=model_args,
        training_args=training_args,
        lora_args=lora_args,
    )

    # Test the model with a sample prompt
    sample_prompt = "What is the capital of Germany?"
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    run_example()
