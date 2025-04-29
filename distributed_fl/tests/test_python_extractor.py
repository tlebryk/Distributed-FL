import pytest
import os
from datasets import Dataset

import python_extractor


# --- extract_function_parts tests ---

SAMPLE_CODE = """
def foo(x, y):
    \"\"\"Add x and y.\"\"\"
    return x + y

def bar(z):
    return z * 2
"""

ASYNC_CODE = """
async def spam(a):
    \"\"\"Async function.\"\"\"
    return a - 1
"""

SYNTAX_ERROR_CODE = "def oops(x):\n    return x+\n"


def test_extract_all_functions_with_and_without_docstrings():
    parts = python_extractor.extract_function_parts(SAMPLE_CODE)
    # Should find both foo and bar
    assert set(parts.keys()) == {"foo", "bar"}

    sig_foo, body_foo = parts["foo"]
    # Signature should include the def line and the docstring
    assert "def foo(x, y):" in sig_foo
    assert '"""Add x and y."""' in sig_foo
    # Body should be the return line only
    assert body_foo.strip() == "return x + y"

    sig_bar, body_bar = parts["bar"]
    # No docstring: signature is just the def line
    assert sig_bar.strip() == "def bar(z):"
    # Body is the return line
    assert body_bar.strip() == "return z * 2"


def test_extract_async_function():
    parts = python_extractor.extract_function_parts(ASYNC_CODE)
    assert "spam" in parts
    sig, body = parts["spam"]
    assert sig.startswith("async def spam(a):")
    assert '"""Async function."""' in sig
    assert body.strip() == "return a - 1"


def test_extract_specific_function():
    parts = python_extractor.extract_function_parts(SAMPLE_CODE, function_name="bar")
    assert set(parts.keys()) == {"bar"}
    assert "foo" not in parts


def test_syntax_error_returns_error_key():
    parts = python_extractor.extract_function_parts(SYNTAX_ERROR_CODE)
    assert "error" in parts
    assert "Syntax error in code" in parts["error"]


# --- extract_from_file tests ---


def test_extract_from_file(tmp_path):
    code = "def hey():\n    return 'ho'\n"
    file = tmp_path / "sample.py"
    file.write_text(code)

    parts = python_extractor.extract_from_file(str(file))
    assert "hey" in parts
    sig, body = parts["hey"]
    assert sig.strip() == "def hey():"
    assert body.strip() == "return 'ho'"


# --- create_huggingface_dataset tests ---


def test_create_huggingface_dataset_basic(tmp_path, capsys):
    # 1) Good file
    good = tmp_path / "good.py"
    good.write_text("def a():\n    return 1\n")
    # 2) Syntax-error file
    bad = tmp_path / "bad.py"
    bad.write_text("def b(:\n    pass\n")
    # 3) Non-py file (should be ignored)
    other = tmp_path / "ignore.txt"
    other.write_text("nothing here")

    ds = python_extractor.create_huggingface_dataset(str(tmp_path))

    # It should be a Dataset with one row (the one valid function)
    assert isinstance(ds, Dataset)
    assert ds.num_rows == 1

    # Columns must match the dict keys in create_huggingface_dataset
    assert set(ds.column_names) == {
        "instruction",
        "output",
        "function_name",
        "file_path",
    }

    # Check contents
    assert ds["function_name"] == ["a"]
    assert ds["instruction"][0].strip() == "def a():"
    assert ds["output"][0].strip() == "return 1"
    # file_path should point to our good.py
    assert os.path.basename(ds["file_path"][0]) == "good.py"

    # The bad file should have printed an error message
    captured = capsys.readouterr()
    assert "Error in" in captured.out
    assert "Syntax error" in captured.out
