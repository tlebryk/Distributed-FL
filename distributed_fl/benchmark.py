# benchmark.py
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datasets import load_dataset
from typing import List, Dict, Any
from tqdm import tqdm


class Benchmark(ABC):
    """
    Abstract base class for benchmarks.
    Subclasses must implement methods for loading the dataset and evaluating individual examples.
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.dataset = None

    @abstractmethod
    def load_dataset(self):
        """
        Load the benchmark dataset. This method should set self.dataset.
        """
        pass

    @abstractmethod
    def evaluate_example(self, example: dict, agent, **kwargs) -> dict:
        """
        Evaluate a single example using the provided agent.
        Returns a dictionary with details such as the prompt, generated code,
        whether the test passed, and any output or error messages.
        """
        pass

    def run_tests(self, generated_code: str, test_code: str) -> (int, str, str):
        """
        Combine the generated solution with the test code, write to a temporary file,
        and execute the code in an isolated subprocess.
        Returns a tuple: (returncode, stdout, stderr).
        """
        code_to_run = generated_code + "\n" + test_code
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code_to_run)
            tmp_filepath = tmp.name

        try:
            result = subprocess.run(
                ["python", tmp_filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                text=True,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return None, "", "Timeout"
        finally:
            os.remove(tmp_filepath)

    def run(self, agent) -> list:
        """
        Run the benchmark on all examples in the dataset using the provided agent.
        Returns a list of result dictionaries.
        """
        self.load_dataset()
        results = []
        for example in self.dataset:
            result = self.evaluate_example(example, agent)
            results.append(result)
        return results


class HumanEvalBenchmark(Benchmark):
    """
    Concrete benchmark for the HumanEval dataset.
    """

    def load_dataset(self):
        """
        Load the HumanEval dataset from Hugging Face.
        """
        self.dataset = load_dataset("openai/openai_humaneval", split="test")

    def evaluate_example(self, example: dict, agent) -> dict:
        """
        For each HumanEval example, use the agent to generate code from the prompt,
        run the generated code against the provided test code, and return the result.
        """
        prompt = example["prompt"]
        test_code = example["test"]
        candidate_code = agent.generate_solution(prompt)
        returncode, stdout, stderr = self.run_tests(candidate_code, test_code)
        success = returncode == 0
        return {
            "prompt": prompt,
            "generated_code": candidate_code,
            "success": success,
            "output": stdout if success else stderr,
        }

    # TODO: just use the extraction function and execute after the fact.
    @staticmethod
    def extract_check_function(code_string: str):
        """
        Extract the 'check' function from a string containing Python code.

        Args:
            code_string: A string containing Python code with a 'check' function.

        Returns:
            The 'check' function that can be called with a candidate function.

        Raises:
            ValueError: If no 'check' function is found in the code.
        """
        # Create a local scope
        local_scope = {}

        # Execute the code in the local scope
        exec(code_string, {}, local_scope)

        # Return the check function
        if "check" in local_scope:
            return local_scope["check"]
        else:
            raise ValueError("No check function found in the provided string")

    @staticmethod
    def extract_function(code_string: str, function_name: str):
        """
        Extract a function definition from a string of Python code.

        Args:
            code_string: A string containing Python code.
            function_name: The name of the function to extract.

        Returns:
            The function definition as a string, or None if not found.
        """
        local_scope = {}

        lines = code_string.split("\n")
        function_lines = []
        in_function = False
        function_indent = None

        for line in lines:
            # Check if this line starts the function definition
            if not in_function and line.strip().startswith(f"def {function_name}("):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                function_lines.append(line)
                continue

            # If we're in the function, check if this line is part of it
            if in_function:
                # If we hit a line that starts with "# Test cases", we've reached the end
                if line.strip().startswith("# Test"):
                    break

                # If line is blank, add it and continue
                if not line.strip():
                    function_lines.append(line)
                    continue

                # Check indentation
                current_indent = len(line) - len(line.lstrip())

                # If this line is not indented more than the function declaration,
                # and it's not a comment, we've exited the function
                if current_indent <= function_indent and not line.strip().startswith(
                    "#"
                ):
                    break

                # This line is part of the function
                function_lines.append(line)

        if function_lines:
            # Join the lines to form the complete function
            txt = "\n".join(function_lines)
            return txt
        else:
            return None

    @staticmethod
    def extract_and_execute_imports(code_string: str, function_name: str):
        """
        Extract and execute all code that appears above a given function in a string of Python code.

        Args:
            code_string: A string containing Python code.
            function_name: The name of the function to find.

        Returns:
            A tuple containing (extracted_imports, local_namespace) where:
            - extracted_imports is the string of code found above the function
            - local_namespace is a dictionary of the executed imports/variables
        """
        lines = code_string.split("\n")
        import_lines = []

        # Collect all lines until we find the function definition
        for line in lines:
            if line.strip().startswith(f"def {function_name}("):
                break
            import_lines.append(line)

        # Join the lines to form the complete imports section
        imports_code = "\n".join(import_lines)

        # Execute the imports in a controlled namespace
        local_namespace = {}
        exec(imports_code, {}, local_namespace)

        return imports_code, local_namespace

    @staticmethod
    def execute_check_with_function(
        imports_code, function_code, check_fn_code, function_name, local_namespace=None
    ):
        """
        Execute a check function with a generated function.

        Args:
            imports_code: Code containing imports and other definitions
            function_code: String containing the function definition to be checked
            check_fn_code: String containing the check function definition
            function_name: Name of the function being checked
            local_namespace: Optional namespace to use for execution (will create one if None)

        Returns:
            Result of running the check function on the generated function
        """
        # Create a namespace if one isn't provided
        if local_namespace is None:
            local_namespace = {}
            # Execute imports first
            logging.debug(f"Executing imports:\n{imports_code}")
            exec(imports_code, {}, local_namespace)

        # Execute the function code to create the function object
        logging.debug(f"Executing generated function:\n{function_code}")
        exec(function_code, {}, local_namespace)

        # Execute the check function code to create the check function
        logging.debug(f"Executing check function:\n{check_fn_code}")
        exec(check_fn_code, {}, local_namespace)

        # Get both function objects from the namespace
        function_obj = local_namespace[function_name]
        check_obj = local_namespace["check"]

        # Run the check function on the function object
        logging.debug(f"Running check function on {function_name}")
        result = check_obj(function_obj)
        logging.debug(f"Result: {result}")

        return result

    # TODO: convert to evaluate_example
    def run_example(self, example, model, tokenizer):
        prompt = example["prompt"]
        task_id = example["task_id"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        # %%
        output_ids = model.generate(input_ids, max_new_tokens=300)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # %%
        function_name = example["entry_point"]

        imports_code, local_namespace = self.extract_and_execute_imports(
            example["prompt"], function_name
        )

        # Extract the function code
        function_code = self.extract_function(generated_text, function_name)
        check_fn_code = self.extract_function(example["test"], "check")
        try:
            self.execute_check_with_function(
                imports_code,
                function_code,
                check_fn_code,
                function_name,
                local_namespace=local_namespace,
            )
            success = True
            msg = "Success"
        except Exception as e:
            success = False
            msg = "Error: " + str(e)
        result = {
            "task_id": task_id,
            "prompt": prompt,
            "test_code": check_fn_code,
            "function_name": function_name,
            "generated_text": generated_text,
            "extracted_function": function_code,
            "imports_code": imports_code,
            "check_function": check_fn_code,
            "success": success,
            "error_message": msg,
        }
        return result

    def run(self, model, tokenizer) -> List[Dict[str, Any]]:
        """
        Run the benchmark on all examples in the dataset using the provided model.

        Args:
            model: The model to generate solutions
            tokenizer: The tokenizer for the model
            output_dir: Optional directory to save results

        Returns:
            A list of result dictionaries.
        """
        results = []

        # Process all examples with a progress bar
        for example in tqdm(self.dataset, desc="Evaluating HumanEval"):
            result = self.run_example(example, model, tokenizer)
            results.append(result)

        return results


# %%
