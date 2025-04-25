import ast
import sys
from pathlib import Path
from pathlib import Path
import glob
from datasets import Dataset
import sys


def extract_function_parts(code_str, function_name=None):
    """
    Extract function signatures (definition + docstring) and bodies from Python code.

    Args:
        code_str (str): Python code as a string
        function_name (str, optional): Name of a specific function to extract. If None, extract all functions.

    Returns:
        dict: A dictionary mapping function names to tuples of (signature, body)
    """

    # Dictionary to store results
    results = {}

    try:
        # Parse the code
        tree = ast.parse(code_str)

        # Find function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                # Skip if we're looking for a specific function and this isn't it
                if function_name and node.name != function_name:
                    continue

                # Get function name
                func_name = node.name

                # Get function source
                func_start = node.lineno - 1  # Convert to 0-indexed
                func_end = node.end_lineno
                func_source = code_str.splitlines()[func_start:func_end]

                # Check for docstring
                docstring = ast.get_docstring(node)

                if docstring:
                    # The function has a docstring
                    # Find the line where the docstring ends
                    docstring_node = node.body[0]
                    docstring_end = docstring_node.end_lineno - node.lineno

                    # Signature includes function def and docstring
                    signature = "\n".join(func_source[: docstring_end + 1])

                    # Body is everything after the docstring
                    body = "\n".join(func_source[docstring_end + 1 :])
                else:
                    # No docstring
                    signature = func_source[0]
                    body = "\n".join(func_source[1:])

                results[func_name] = (signature, body)

    except SyntaxError as e:
        return {"error": f"Syntax error in code: {str(e)}"}

    return results


def extract_from_file(file_path, function_name=None):
    """Extract function parts from a file"""
    with open(file_path, "r") as f:
        code = f.read()
    return extract_function_parts(code, function_name)


def create_huggingface_dataset(directory_path, pattern="*.py", function_name=None):
    """
    Create a Huggingface Dataset from Python files in a directory.

    Args:
        directory_path (str): Path to directory containing Python files
        pattern (str): Glob pattern to match files (default: "*.py")
        function_name (str, optional): Name of a specific function to extract

    Returns:
        datasets.Dataset: Huggingface Dataset with 'prompt' and 'completion' columns
    """
    prompts = []
    completions = []
    function_names = []
    file_paths = []

    # Get all matching files in the directory
    path = Path(directory_path)
    all_files = list(path.glob(pattern))

    for file_path in all_files:
        try:
            # Extract functions from the file
            functions_dict = extract_from_file(file_path, function_name)

            # Skip files with errors
            if "error" in functions_dict:
                print(f"Error in {file_path}: {functions_dict['error']}")
                continue

            # Add each function to our dataset
            for func_name, (signature, body) in functions_dict.items():
                prompts.append(signature)
                completions.append(body)
                function_names.append(func_name)
                file_paths.append(str(file_path))

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Create the dataset
    data = {
        "instruction": prompts,
        "output": completions,
        "function_name": function_names,
        "file_path": file_paths,
    }
    # TODO: change this
    return Dataset.from_dict(data).select(range(3))


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_functions.py <python_file> [function_name]")
        sys.exit(1)

    file_path = sys.argv[1]
    specific_function = sys.argv[2] if len(sys.argv) > 2 else None

    functions = extract_from_file(file_path, specific_function)

    for name, (signature, body) in functions.items():
        print(f"Function: {name}")
        print("\nSignature:")
        print(signature)
        print("\nBody:")
        print(body)
        print("-" * 50)
