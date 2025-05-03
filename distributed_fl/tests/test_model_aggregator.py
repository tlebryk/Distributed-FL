import os
import shutil
import csv
import pytest
from model_aggregator import ModelAggregator
import model_aggregator


# Dummy update request for weighted_average tests
class DummyUpdateRequest:
    def __init__(self, client_id, weight, pylint_score, update):
        self.client_id = client_id
        self.weight = weight
        self.pylint_score = pylint_score
        self.update = update


# --- Tests for ModelAggregator.weighted_average ---
def test_weighted_average_empty():
    agg = ModelAggregator(None, None)
    assert agg.weighted_average([]) is None


def test_weighted_average_single():
    agg = ModelAggregator(None, None)
    update = {"a": 2.0, "b": 4.0}
    req = DummyUpdateRequest("c1", 1.0, 0.01, update)
    result = agg.weighted_average([req])
    assert pytest.approx(result["a"], rel=1e-6) == 2.0
    assert pytest.approx(result["b"], rel=1e-6) == 4.0


def test_weighted_average_multiple():
    agg = ModelAggregator(None, None)
    u1 = DummyUpdateRequest("c1", 1.0, 0.02, {"x": 1.0})
    u2 = DummyUpdateRequest("c2", 2.0, 0.01, {"x": 3.0})
    # weights: u1:1*0.01/10=0.001; u2:2*0.01/10=0.002
    result = agg.weighted_average([u1, u2])
    expected = (1.0 * 0.001 + 3.0 * 0.002) / (0.001 + 0.002)
    assert pytest.approx(result["x"], rel=1e-6) == expected


# --- Tests for leave_one_out_batches ---
def test_leave_one_out_batches():
    agg = ModelAggregator(None, None)
    updates = [1, 2, 3]
    batches = list(agg.leave_one_out_batches(updates))
    assert batches == [[2, 3], [1, 3], [1, 2]]


# --- Tests for find_latest_adapter_version ---
def test_find_latest_adapter_version_empty(tmp_path):
    path = tmp_path / "empty_dir"
    # directory does not exist
    assert ModelAggregator.find_latest_adapter_version(str(path)) == 0
    # now create directory but no v* dirs
    os.makedirs(path, exist_ok=True)
    assert ModelAggregator.find_latest_adapter_version(str(path)) == 0


def test_find_latest_adapter_version_with_versions(tmp_path):
    path = tmp_path / "versions"
    os.makedirs(path)
    for v in ["v1", "v2", "v10", "vX"]:
        (path / v).mkdir()
    # 'vX' should be ignored
    assert ModelAggregator.find_latest_adapter_version(str(path)) == 10


def test_find_latest_adapter_version_on_file(tmp_path):
    # create a file instead of a directory
    file_path = tmp_path / "afile"
    file_path.write_text("content")
    # should catch exception and return 0
    assert ModelAggregator.find_latest_adapter_version(str(file_path)) == 0


# --- Tests for save_run_result ---
def test_save_run_result(tmp_path):
    csv_path = tmp_path / "results.csv"
    info1 = {"a": "1", "b": "2"}
    ModelAggregator.save_run_result(info1, str(csv_path))
    # file should be created with header and one row
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == ["a", "b"]
        assert rows == [info1]
    # append second record
    info2 = {"a": "3", "b": "4"}
    ModelAggregator.save_run_result(info2, str(csv_path))
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert rows == [info1, info2]


# --- Tests for save_aggregated_model ---
def test_save_aggregated_model(tmp_path, monkeypatch):
    adapters_path = tmp_path / "adapters"
    # create adapter_config.json for copying
    (adapters_path / "adapter_config.json").parent.mkdir(parents=True, exist_ok=True)
    (adapters_path / "adapter_config.json").write_text("config")
    agg = ModelAggregator(None, None, adapters_path=str(adapters_path))
    called = {}

    def fake_save(state, path):
        called["state"] = state
        called["path"] = path
        # simulate file creation
        open(path, "w").write("data")

    monkeypatch.setattr(model_aggregator, "save_file", fake_save)
    aggregated_state = {"w": 1.0}
    output = agg.save_aggregated_model(aggregated_state, version=3)
    expected_dir = os.path.join(str(adapters_path), "server", "central", "v3")
    assert output == expected_dir
    # save_file should have been called with correct args
    assert called["state"] == aggregated_state
    assert called["path"] == os.path.join(expected_dir, "adapter_model.safetensors")
    # config file should be copied
    assert os.path.isfile(os.path.join(expected_dir, "adapter_config.json"))


# --- Tests for perform_eval ---
class FakeZK:
    def __init__(self, past_accuracy):
        self.past = str(past_accuracy)
        self.updated = None

    def get_past_accuracy(self):
        return self.past

    def update_past_accuracy(self, new):
        self.updated = new


class FakeAgent:
    def __init__(self, model_name, adapter_path):
        self.model_name = model_name
        self.adapter_path = adapter_path


def test_perform_eval_none():
    agg = ModelAggregator(None, None)
    result, info = agg.perform_eval(None)
    assert result is False
    assert info is None


def test_perform_eval_better(tmp_path, monkeypatch):
    adapters_path = tmp_path / "adapters"
    # prepare server rounds directory and config
    (adapters_path / "server" / "rounds").mkdir(parents=True)
    (adapters_path / "adapter_config.json").parent.mkdir(parents=True, exist_ok=True)
    (adapters_path / "adapter_config.json").write_text("config")
    zk = FakeZK(40.0)
    agg = ModelAggregator(None, zk, adapters_path=str(adapters_path))
    # patch latest version
    monkeypatch.setattr(
        ModelAggregator, "find_latest_adapter_version", staticmethod(lambda path: 5)
    )
    # patch save_file
    monkeypatch.setattr(
        model_aggregator, "save_file", lambda state, path: open(path, "w").write("")
    )
    # patch agent creation and evaluation
    monkeypatch.setattr(model_aggregator, "LoraHuggingFaceAgent", FakeAgent)
    monkeypatch.setattr(
        model_aggregator,
        "evaluate",
        lambda agent, bm, results_csv, mode: {"accuracy": "50.0", "run_id": "id"},
    )
    # run evaluation
    aggregated_state = {"x": 1.0}
    result, info = agg.perform_eval(aggregated_state)
    assert result is True
    assert info["accuracy"] == "50.0"
    assert zk.updated == 50.0


def test_perform_eval_worse(tmp_path, monkeypatch):
    adapters_path = tmp_path / "adapters"
    (adapters_path / "server" / "rounds").mkdir(parents=True)
    (adapters_path / "adapter_config.json").parent.mkdir(parents=True, exist_ok=True)
    (adapters_path / "adapter_config.json").write_text("config")
    zk = FakeZK(60.0)
    agg = ModelAggregator(None, zk, adapters_path=str(adapters_path))
    monkeypatch.setattr(
        ModelAggregator, "find_latest_adapter_version", staticmethod(lambda path: 1)
    )
    monkeypatch.setattr(
        model_aggregator, "save_file", lambda state, path: open(path, "w").write("")
    )
    monkeypatch.setattr(model_aggregator, "LoraHuggingFaceAgent", FakeAgent)
    monkeypatch.setattr(
        model_aggregator,
        "evaluate",
        lambda agent, bm, results_csv, mode: {"accuracy": "50.0"},
    )
    result, info = agg.perform_eval({"x": 1.0})
    assert result is False
    assert info["accuracy"] == "50.0"
    assert zk.updated is None
