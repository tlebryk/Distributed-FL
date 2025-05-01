# eval_script.py
# %%
import argparse
import csv
import json
import logging
from datetime import datetime
import os
import pandas as pd
from agent import LoraHuggingFaceAgent
from benchmark import HumanEvalBenchmark


# --- Persistence helpers ---


def load_previous_results(path):
    """
    Read past runs from CSV. Returns a list of dicts or empty list if none.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        return []


def get_best_success(runs):
    """
    Extract best accuracy from past runs.
    """
    if not runs:
        return 0.0
    return max(float(r["accuracy"]) for r in runs)


def compute_accuracy(results):
    """
    Compute percent of successful examples.
    """
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    return successes / total * 100 if total > 0 else 0.0


# --- Main evaluation with persistence ---


def evaluate(
    code_agent, benchmark, results_csv: str = "experiments.csv", mode: str = "test"
) -> bool:
    # mode is test or prod
    """
    Evaluate the current model using the HumanEval benchmark.

    Args:
        mode: Which mode to run in. test or prod

    Side effects:
        - Writes a line to the "experiments.csv" file.
        - Prints logging messages to the console.
    """
    logging.basicConfig(level=logging.INFO)

    # 1. Load previous experiments

    # 2. Setup model & dataset

    # %%
    results = benchmark.run(code_agent.model, code_agent.tokenizer)
    df = pd.DataFrame(results)
    # save df with current timestamp
    os.makedirs("results", exist_ok=True)
    # This is fine locally for now.
    df.to_csv(f"results_{datetime.now().isoformat().replace(':', '-')}.csv")
    # print(df[["generated_text", "success"]])

    current_pct = compute_accuracy(results)
    logging.info(f"Current run accuracy: {current_pct:.2f}%")

    # 5. Save this run
    run_info = {
        "run_id": datetime.utcnow().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "accuracy": f"{current_pct:.2f}",
        "eval_rows": str(len(benchmark.dataset)),
        "hyperparameters": json.dumps({}),  # fill in if needed
    }
    return run_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, default="experiments.csv")
    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()
    human_eval = HumanEvalBenchmark()
    code_agent = LoraHuggingFaceAgent(
        model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        adapter_path="/home/tlebryk/262_distributed_systems/Distributed-FL/distributed_fl/adapters/personal/1",
    )
    human_eval.dataset = human_eval.dataset.select(range(2))
    evaluate(code_agent, human_eval, results_csv=args.results_csv, mode=args.mode)
