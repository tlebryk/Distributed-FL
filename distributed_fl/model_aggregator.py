# model_aggregator.py
import os
import shutil
import torch
from safetensors.torch import save_file
from agent import LoraHuggingFaceAgent
from eval_script import evaluate
from logger import get_logger

logger = get_logger(__name__)


class ModelAggregator:
    def __init__(
        self, benchmark, zk_manager, adapters_path="./distributed_fl/adapters"
    ):
        self.benchmark = benchmark
        self.zk_manager = zk_manager
        self.path_to_adapters = adapters_path

    def weighted_average(self, update_requests, use_pylint=True):
        """Aggregate model updates with weighted averaging."""
        if not update_requests:
            return None

        # Use the pre-computed weights in the parameter aggregation
        aggregated_state = {}
        for key in update_requests[0].update.keys():
            aggregated_state[key] = 0
            total_weight = 0

            for update_request in update_requests:
                client_id = update_request.client_id
                weight = (
                    update_request.weight * min(update_request.pylint_score, 0.01) / 10
                )
                aggregated_state[key] += update_request.update[key] * weight
                total_weight += weight

            aggregated_state[key] /= total_weight

        return aggregated_state

    def perform_eval(self, aggregated_state, current_version=0):
        """Evaluate the aggregated model state against benchmarks."""
        if aggregated_state is None:
            return False, None

        logger.info("Evaluating aggregated model state.")

        # Save the model to a temp directory for evaluation
        round_path = os.path.join(
            self.path_to_adapters,
            "server",
            "rounds",
        )
        latest_round = self.find_latest_adapter_version(round_path)
        updated_round = latest_round + 1
        output_dir = os.path.join(round_path, f"v{updated_round}")
        os.makedirs(output_dir, exist_ok=True)

        save_file(
            aggregated_state,
            os.path.join(output_dir, "adapter_model.safetensors"),
        )
        shutil.copy2(
            os.path.join(self.path_to_adapters, "adapter_config.json"),
            os.path.join(output_dir, "adapter_config.json"),
        )

        # Create agent with the new adapter
        code_agent = LoraHuggingFaceAgent(
            model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            adapter_path=output_dir,
        )

        # Run evaluation
        run_info = evaluate(
            code_agent,
            self.benchmark,
            results_csv="experiments.csv",
            mode="prod",
        )

        # Get accuracy metrics
        current_pct = float(run_info["accuracy"])
        best_pct = float(self.zk_manager.get_past_accuracy())

        # Compare to best performance
        if current_pct < best_pct:
            logger.info(
                "Current run underperforms best run: {:.2f}% < {:.2f}%".format(
                    current_pct, best_pct
                )
            )
            return False, run_info
        else:
            logger.info(
                "Current run meets or exceeds best run: {:.2f}% >= {:.2f}%".format(
                    current_pct, best_pct
                )
            )
            self.zk_manager.update_past_accuracy(current_pct)
            return True, run_info

    def save_aggregated_model(self, aggregated_state, version):
        """Save the aggregated model to the central repository."""
        main_path = os.path.join(self.path_to_adapters, "server", "central")
        output_dir = os.path.join(main_path, f"v{version}")
        os.makedirs(output_dir, exist_ok=True)

        save_file(
            aggregated_state,
            os.path.join(output_dir, "adapter_model.safetensors"),
        )

        # Also copy the config file
        shutil.copy2(
            os.path.join(self.path_to_adapters, "adapter_config.json"),
            os.path.join(output_dir, "adapter_config.json"),
        )

        return output_dir

    def leave_one_out_batches(self, updates):
        """
        Generator that yields batches with one update left out each time.
        For N updates you'll get N batches:
        [1,2,3]     → drop 0
        [0,2,3]     → drop 1
        ...
        """
        for i in range(len(updates)):
            yield updates[:i] + updates[i + 1 :]

    @staticmethod
    def find_latest_adapter_version(path):
        """Find the latest version number in the provided directory path."""
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                return 0

            versions = []
            for dirname in os.listdir(path):
                if dirname.startswith("v") and dirname[1:].isdigit():
                    versions.append(int(dirname[1:]))

            return max(versions) if versions else 0
        except Exception as e:
            logger.error(f"Error finding latest adapter version: {e}")
            return 0

    @staticmethod
    def save_run_result(run_info, path):
        """
        Append a new run_info dict to the CSV (creates file if needed).
        """
        import csv

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
