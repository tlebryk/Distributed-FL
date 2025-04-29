import os
import pytest
import torch
from pathlib import Path
from transformers import TrainingArguments

import training


# --- Dataclass defaults tests ---


def test_lora_arguments_defaults():
    args = training.LoraArguments()
    assert args.enabled is False
    assert args.r == 8
    assert args.lora_alpha == 16
    assert abs(args.lora_dropout - 0.05) < 1e-6
    assert args.bias == "none"
    assert args.target_modules is None
    assert args.task_type == "CAUSAL_LM"


def test_data_arguments_defaults():
    da = training.DataArguments()
    assert da.prompt_column == "prompt"
    assert da.completion_column == "completion"
    assert da.separator == " "
    assert da.max_length == 1024
    assert da.preprocessing_fn is None


def test_model_arguments_defaults(tmp_path, monkeypatch):
    # Default dirs
    ma = training.ModelArguments()
    assert ma.output_dir == "./model_output"
    assert ma.logging_dir == "./logs"
    # test that train_model will create these directories
    out = tmp_path / "mout"
    log = tmp_path / "mlog"
    ma2 = training.ModelArguments(output_dir=str(out), logging_dir=str(log))

    # monkeypatch trainer so it doesn't actually train
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self):
            return {}

        def save_model(self, *_):
            pass

    monkeypatch.setattr(training, "SFTTrainer", DummyTrainer)
    # also patch collator and config
    monkeypatch.setattr(
        training, "DataCollatorForCompletionOnlyLM", lambda *a, **k: None
    )
    monkeypatch.setattr(training, "SFTConfig", lambda *a, **k: None)

    # dummy model & tokenizer
    class DummyModel:
        def train(self):
            pass

        def named_parameters(self):
            return []

    class DummyTok:
        def encode(self, *a, **k):
            return [0, 1, 2]

        def save_pretrained(self, out_dir):
            pass

        pad_token = None
        eos_token = "<eos>"

    # run
    _ = training.train_model(
        model=DummyModel(),
        tokenizer=DummyTok(),
        train_dataset={"instruction": ["x"], "output": ["y"]},
        val_dataset=None,
        model_args=ma2,
        training_args=training.CustomTrainingArguments(),
        data_args=training.DataArguments(),
    )
    # dirs should now exist
    assert out.exists() and out.is_dir()
    assert log.exists() and log.is_dir()


# --- to_transformers_args tests ---


@pytest.mark.parametrize(
    "has_val, expected_best, expected_metric",
    [
        (True, True, "eval_loss"),
        (False, False, None),
    ],
)
def test_to_transformers_args(has_val, expected_best, expected_metric):
    cta = training.CustomTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        fp16=False,
    )
    ta: TrainingArguments = cta.to_transformers_args(
        output_dir="out_dir", has_val_dataset=has_val
    )
    assert isinstance(ta, TrainingArguments)
    assert ta.output_dir == "out_dir"
    assert ta.overwrite_output_dir is True
    assert ta.save_strategy == "steps"
    # assert ta.load_best_model_at_end == expected_best
    # metric_for_best_model is set only if has_val_dataset
    assert ta.metric_for_best_model == expected_metric
    # learning_rate, epochs, batch size transferred
    assert abs(ta.learning_rate - 1e-4) < 1e-9
    assert ta.num_train_epochs == 3
    assert ta.per_device_train_batch_size == 4


# --- get_training_data tests ---


def test_get_training_data_invokes_extractor(tmp_path, monkeypatch):
    # Create a small directory tree
    root = tmp_path / "proj"
    sub = root / "src"
    sub.mkdir(parents=True)
    f_py = sub / "mod.py"
    f_py.write_text("def a(): pass")
    f_txt = sub / "readme.md"
    f_txt.write_text("hello")

    called = []

    def fake_extract(path):
        called.append(path)

    monkeypatch.chdir(root)
    monkeypatch.setattr(training, "extract_from_file", fake_extract)

    training.get_training_data()
    assert len(called) == 1
    assert called[0].endswith(os.path.join("src", "mod.py"))


# --- load_and_quantize_model tests ---


class DummyAModel:
    pass


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<EOS>"

    def save_pretrained(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    import transformers

    # Patch the from_pretrained methods
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: DummyAModel()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: DummyTokenizer()),
    )
    # Patch BitsAndBytesConfig so it just captures inputs
    monkeypatch.setattr(
        transformers, "BitsAndBytesConfig", lambda **kwargs: {"bnb": kwargs}
    )
    yield


@pytest.mark.parametrize(
    "use_8bit,use_4bit,extra_cfg",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (False, False, {"foo": "bar"}),
    ],
)
def test_load_and_quantize_model_variants(use_8bit, use_4bit, extra_cfg):
    # Should not error and should return our dummy instances
    model, tok = training.load_and_quantize_model(
        model_name_or_path="dummy",
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        device_map="cpu",
        quantization_config=extra_cfg,
    )
    assert isinstance(model, DummyAModel)
    assert isinstance(tok, DummyTokenizer)
    # pad_token should have been set to eos_token if it was None
    assert tok.pad_token == "<EOS>"
