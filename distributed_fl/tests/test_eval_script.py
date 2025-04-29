# tests/test_eval_script.py

import pytest
from datetime import datetime

import eval_script as es


def test_load_previous_results_file_not_exist(tmp_path):
    path = tmp_path / "no_such.csv"
    assert es.load_previous_results(str(path)) == []


def test_save_and_load_previous_results(tmp_path):
    path = tmp_path / "runs.csv"
    run1 = {"a": "1", "b": "x"}
    es.save_run_result(run1, str(path))

    loaded = es.load_previous_results(str(path))
    assert loaded == [run1]

    run2 = {"a": "2", "b": "y"}
    es.save_run_result(run2, str(path))
    loaded2 = es.load_previous_results(str(path))
    assert loaded2 == [run1, run2]


def test_compute_percent_success_and_get_best_success():
    # empty inputs
    assert es.compute_percent_success([]) == 0.0
    assert es.get_best_success([]) == 0.0

    # non-empty
    results = [{"success": True}, {"success": False}, {"success": True}]
    expected_pct = 2 / 3 * 100
    assert es.compute_percent_success(results) == pytest.approx(expected_pct)

    runs = [
        {"percent_success": "50.00"},
        {"percent_success": "75.5"},
        {"percent_success": "25.0"},
    ]
    assert es.get_best_success(runs) == pytest.approx(75.5)


# ————————————————————————————————————————————————————————————————
# Now tests for `evaluate()`


class FakeBenchmark:
    """Mimics a human‐eval benchmark that returns only 'generated_text' & 'success'."""

    def __init__(self, successes):
        self.dataset = [None] * len(successes)
        self._successes = successes

    def run(self, model, tokenizer):
        # produce exactly the two columns evaluate() expects
        return [{"generated_text": "dummy", "success": s} for s in self._successes]


class FakeAgent:
    """Just needs .model and .tokenizer attributes."""

    def __init__(self):
        self.model = object()
        self.tokenizer = object()


@pytest.fixture(autouse=True)
def patch_dataframe_and_datetime(monkeypatch):
    # 1) Patch pandas.DataFrame.to_csv so we don't create timestamped files
    import pandas as pd

    monkeypatch.setattr(es.pd.DataFrame, "to_csv", lambda self, path: None)

    # 2) Patch eval_script.datetime so that utcnow()/now() always return a fixed datetime
    fixed = datetime(2025, 4, 28, 12, 0, 0)

    class DummyDateTime:
        @classmethod
        def utcnow(cls):
            return fixed

        @classmethod
        def now(cls):
            return fixed

    monkeypatch.setattr(es, "datetime", DummyDateTime)


def test_evaluate_no_prior_runs(tmp_path):
    # no experiments.csv on disk → best_pct = 0.0
    results_csv = tmp_path / "experiments.csv"
    assert not results_csv.exists()

    benchmark = FakeBenchmark([True, True, False])
    agent = FakeAgent()

    ok = es.evaluate(agent, benchmark, results_csv=str(results_csv), mode="test")
    # current success = 2/3*100 > 0 → returns True
    assert ok is True

    # confirm that exactly one row was written
    runs = es.load_previous_results(str(results_csv))
    assert len(runs) == 1

    row = runs[0]
    # 2/3*100 = 66.67
    assert row["percent_success"] == f"{(2/3*100):.2f}"
    assert row["eval_rows"] == str(len(benchmark.dataset))


def test_evaluate_underperforms_prior(tmp_path):
    # pre‐write a prior run at 80%
    results_csv = tmp_path / "experiments.csv"
    prior = {
        "run_id": "old",
        "timestamp": "2025-04-28T12:00:00",
        "percent_success": "80.00",
        "eval_rows": "10",
        "hyperparameters": "{}",
    }
    es.save_run_result(prior, str(results_csv))

    # now current will be 1/3*100 = 33.33 < 80.00 → returns False
    benchmark = FakeBenchmark([False, False, True])
    agent = FakeAgent()

    ok = es.evaluate(agent, benchmark, results_csv=str(results_csv), mode="prod")
    assert ok is False

    runs = es.load_previous_results(str(results_csv))
    # we should now have 2 rows: old + new
    assert len(runs) == 2
    assert runs[0]["percent_success"] == "80.00"
    # new row:
    assert runs[1]["percent_success"] == f"{(1/3*100):.2f}"
    assert runs[1]["eval_rows"] == str(len(benchmark.dataset))
