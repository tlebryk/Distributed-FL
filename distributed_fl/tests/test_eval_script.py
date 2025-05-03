import os
import csv
import json
from datetime import datetime as _real_datetime
import pytest

import eval_script

# --- Tests for persistence helpers ---


def test_load_previous_results_no_file(tmp_path):
    # Should return empty list when file does not exist
    path = tmp_path / "nonexistent.csv"
    results = eval_script.load_previous_results(str(path))
    assert results == []


def test_load_previous_results_with_file(tmp_path):
    # Create a CSV file with headers and rows
    path = tmp_path / "results.csv"
    header = ["run_id", "accuracy", "eval_rows"]
    rows = [
        {"run_id": "r1", "accuracy": "50.0", "eval_rows": "10"},
        {"run_id": "r2", "accuracy": "75.0", "eval_rows": "20"},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    # Load and verify
    loaded = eval_script.load_previous_results(str(path))
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert all(isinstance(r, dict) for r in loaded)
    assert loaded == rows


def test_get_best_success_empty():
    assert eval_script.get_best_success([]) == 0.0


def test_get_best_success_non_empty():
    runs = [
        {"accuracy": "33.33"},
        {"accuracy": "66.67"},
        {"accuracy": "50.00"},
    ]
    best = eval_script.get_best_success(runs)
    assert isinstance(best, float)
    assert best == pytest.approx(66.67)


def test_compute_accuracy_empty():
    assert eval_script.compute_accuracy([]) == 0.0


def test_compute_accuracy_various():
    results = [
        {"success": True},
        {"success": False},
        {"success": True},
        {"success": False},
    ]
    # 2 successes out of 4 => 50%
    acc = eval_script.compute_accuracy(results)
    assert isinstance(acc, float)
    assert acc == pytest.approx(50.0)
    # All successes
    all_success = [{"success": True} for _ in range(5)]
    assert eval_script.compute_accuracy(all_success) == pytest.approx(100.0)
    # No successes
    none_success = [{"success": False} for _ in range(3)]
    assert eval_script.compute_accuracy(none_success) == pytest.approx(0.0)


# --- Tests for evaluate function ---


class DummyAgent:
    def __init__(self):
        # model and tokenizer are not used by DummyBenchmark
        self.model = None
        self.tokenizer = None


class DummyBenchmark:
    def __init__(self, results, dataset_size):
        # results: list of dicts to return from run
        self._results = results
        # dataset attribute used for eval_rows
        self.dataset = [None] * dataset_size

    def run(self, model, tokenizer):
        # Return predetermined results
        return self._results


def test_evaluate_creates_csv_and_returns_run_info(tmp_path, monkeypatch):
    # Prepare dummy results: 3 total, 2 successes
    dummy_results = [
        {"success": True},
        {"success": False},
        {"success": True},
    ]
    dataset_size = 3
    agent = DummyAgent()
    benchmark = DummyBenchmark(dummy_results, dataset_size)

    # Freeze datetime.now() and datetime.utcnow()
    fixed_now = _real_datetime(2020, 1, 2, 3, 4, 5)
    fixed_utcnow = _real_datetime(2019, 12, 31, 23, 59, 59)

    class FakeDateTime:
        @staticmethod
        def now():
            return fixed_now

        @staticmethod
        def utcnow():
            return fixed_utcnow

    # Monkeypatch datetime in eval_script module
    monkeypatch.setattr(eval_script, "datetime", FakeDateTime)
    # Change working dir to temp
    monkeypatch.chdir(tmp_path)

    # Run evaluation
    run_info = eval_script.evaluate(agent, benchmark)

    # Check return structure
    assert isinstance(run_info, dict)
    assert run_info["run_id"] == fixed_utcnow.isoformat()
    assert run_info["timestamp"] == fixed_now.isoformat()
    # Accuracy: 2/3 => 66.666... => formatted 66.67
    assert run_info["accuracy"] == "66.67"
    # eval_rows should match dataset_size
    assert run_info["eval_rows"] == str(dataset_size)
    # hyperparameters is a JSON string of empty dict
    assert run_info["hyperparameters"] == json.dumps({})

    # Check that a CSV file was created in the cwd
    # Expect filename: results_<timestamp>.csv
    timestamp_str = fixed_now.isoformat().replace(":", "-")
    expected_filename = f"results_{timestamp_str}.csv"
    assert (tmp_path / expected_filename).exists()
    # Optionally, verify CSV content matches dummy_results keys
    # Read CSV and compare number of rows
    import pandas as pd

    df = pd.read_csv(tmp_path / expected_filename)
    assert len(df) == len(dummy_results)
    # Ensure 'success' column exists
    assert "success" in df.columns
