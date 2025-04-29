# tests/test_benchmark.py
import os
import time
import pytest
from benchmark import Benchmark, HumanEvalBenchmark


# A minimal concrete Benchmark for testing run() and run_tests()
class DummyBenchmark(Benchmark):
    def load_dataset(self):
        # pretend we “load” three examples
        self.dataset = ["a", "b", "c"]

    def evaluate_example(self, example, agent):
        # return something simple so we can assert on it
        return {"example": example, "agent": agent}


def test_run_invokes_load_and_evaluate():
    dummy = DummyBenchmark(timeout=1)
    fake_agent = object()
    results = dummy.run(fake_agent)
    # we should get one result per item in our fake dataset
    assert results == [
        {"example": "a", "agent": fake_agent},
        {"example": "b", "agent": fake_agent},
        {"example": "c", "agent": fake_agent},
    ]


def test_run_tests_success(tmp_path):
    bench = DummyBenchmark(timeout=1)
    # generated code prints nothing; test_code just passes
    generated = "print('hello')"
    test_code = "assert True"
    returncode, stdout, stderr = bench.run_tests(generated, test_code)
    assert returncode == 0
    assert stdout == "hello\n"
    assert stderr == ""


def test_run_tests_failure(tmp_path):
    bench = DummyBenchmark(timeout=1)
    # test_code will fail the assertion
    generated = ""
    test_code = "assert False"
    returncode, stdout, stderr = bench.run_tests(generated, test_code)
    # non-zero exit for failed assertion
    assert returncode != 0
    assert "AssertionError" in stderr


def test_run_tests_timeout(tmp_path):
    # set a very small timeout so sleep(0.2) always times out
    bench = DummyBenchmark(timeout=0.05)
    generated = ""
    test_code = "import time\ntime.sleep(0.2)"
    returncode, stdout, stderr = bench.run_tests(generated, test_code)
    assert returncode is None
    assert stdout == ""
    assert stderr == "Timeout"


def test_extract_check_function_success():
    code = """
def helper(): pass

def check(fn):
    return fn(2) == 4
"""
    check_fn = HumanEvalBenchmark.extract_check_function(code)
    # check_fn should be callable and work as expected
    assert callable(check_fn)

    def double(x):
        return x * 2

    assert check_fn(double) is True


def test_extract_check_function_raises():
    code = "def not_check(): pass"
    with pytest.raises(ValueError):
        HumanEvalBenchmark.extract_check_function(code)


def test_extract_function_success():
    code = """
# Some header
def multiply(a, b):
    result = a * b
    return result

# Test cases follow
print(multiply(2,3))
"""
    fn_txt = HumanEvalBenchmark.extract_function(code, "multiply")
    assert fn_txt is not None
    # It should start with the def line and include the return
    lines = fn_txt.splitlines()
    assert lines[0].strip() == "def multiply(a, b):"
    assert any("return result" in l for l in lines)
    # It should not include the print(...) test line
    assert not any("print(" in l for l in lines)


def test_extract_function_none():
    code = "print('nothing here')"
    assert HumanEvalBenchmark.extract_function(code, "foo") is None


def test_extract_and_execute_imports():
    code = """
import math
x = 10
def foo(): pass
"""
    imports_code, ns = HumanEvalBenchmark.extract_and_execute_imports(code, "foo")
    # imports_code should include both lines
    assert "import math" in imports_code
    assert "x = 10" in imports_code
    # namespace should have math module and x variable
    assert ns["x"] == 10
    assert ns["math"].sqrt(9) == 3.0


def test_execute_check_with_function_success_and_failure():
    # case: success
    imports_code = ""
    fn_code = "def incr(x): return x + 1"
    check_code = "def check(fn): return fn(3) == 4"
    ok = HumanEvalBenchmark.execute_check_with_function(
        imports_code, fn_code, check_code, "incr"
    )
    assert ok is True

    # case: failure
    check_code2 = "def check(fn): return fn(3) == 5"
    ok2 = HumanEvalBenchmark.execute_check_with_function(
        imports_code, fn_code, check_code2, "incr"
    )
    assert ok2 is False
