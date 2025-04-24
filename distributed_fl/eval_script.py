# %%
from benchmark import HumanEvalBenchmark
from agent import LoraHuggingFaceAgent
import logging
import pandas as pd

# logging.basicConfig(level=logging.DEBUG)

human_eval = HumanEvalBenchmark()
code_agent = LoraHuggingFaceAgent(
    model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct", adapter_path="./adapters/latest"
)

human_eval.load_dataset()

model = code_agent.model
tokenizer = code_agent.tokenizer
# get first three rows of dataset
human_eval.dataset = human_eval.dataset.select(range(3))
# %%
results = human_eval.run(code_agent.model, code_agent.tokenizer)
df = pd.DataFrame(results)
df.success.value_counts()

# %%
import logging
import pandas as pd
import csv
import json
from datetime import datetime
from benchmark import HumanEvalBenchmark
from client import FederatedClient

# Constants
RESULTS_CSV = "experiments.csv"

# --- Persistence helpers ---


def load_previous_results(path=RESULTS_CSV):
    """
    Read past runs from CSV. Returns a list of dicts or empty list if none.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        return []


def save_run_result(run_info, path=RESULTS_CSV):
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. Load previous experiments
    past_runs = load_previous_results()
    best_pct = get_best_success(past_runs)
    logging.info(f"Best previous percent_success: {best_pct:.2f}%")

    # 2. Setup model & dataset
    human_eval = HumanEvalBenchmark()
    client = FederatedClient(client_id="client_1", server_address="localhost:50051")
    human_eval.load_dataset()
    client.initialize_model()
    client.initialize_tokenizer()

    # 3. Restrict to first N examples if desired
    human_eval.dataset = human_eval.dataset.select(range(5))

    # 4. Run benchmark
    results = human_eval.run_example_loop(client.model, client.tokenizer)
    current_pct = compute_percent_success(results)
    logging.info(f"Current run percent_success: {current_pct:.2f}%")

    # 5. Save this run
    run_info = {
        "run_id": datetime.utcnow().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "percent_success": f"{current_pct:.2f}",
        "hyperparameters": json.dumps({}),  # fill in if needed
    }
    save_run_result(run_info)
    logging.info("Run result saved to experiments.csv")

    # 6. Compare to best and report
    if current_pct < best_pct:
        logging.warning("Current run underperforms best run — retraining advised.")
    else:
        logging.info("Current run matches or exceeds best run.")
