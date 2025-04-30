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


def save_run_result(run_info, path):
    """
    Append a new run_info dict to the CSV (creates file if needed).
    """
    file_exists = False
    try:
        with open(path) as _:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_info.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_info)


def compute_percent_success(results):
    """
    Compute percent of successful examples.
    """
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    return successes / total * 100 if total > 0 else 0.0


def get_best_success(runs):
    """
    Extract best percent_success from past runs.
    """
    if not runs:
        return 0.0
    return max(float(r["percent_success"]) for r in runs)


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
    past_runs = load_previous_results(results_csv)
    best_pct = get_best_success(past_runs)
    logging.info(f"Best previous percent_success: {best_pct:.2f}%")

    # 2. Setup model & dataset

    # %%
    results = benchmark.run(code_agent.model, code_agent.tokenizer)
    df = pd.DataFrame(results)
    df.success.value_counts()
    # save df with current timestamp
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/results_{datetime.utcnow().isoformat()}.csv")
    print(df[["generated_text", "success"]])

    current_pct = compute_percent_success(results)
    logging.info(f"Current run percent_success: {current_pct:.2f}%")

    # 5. Save this run
    run_info = {
        "run_id": datetime.utcnow().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "percent_success": f"{current_pct:.2f}",
        "eval_rows": str(len(benchmark.dataset)),
        "hyperparameters": json.dumps({}),  # fill in if needed
    }
    save_run_result(run_info, results_csv)
    logging.info("Run result saved to experiments.csv")

    # 6. Compare to best and report
    if current_pct < best_pct:
        logging.warning("Current run underperforms best run — retraining advised.")
        return False
    else:
        # logging.info("Current run matches or exceeds best run.")
        return True


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
