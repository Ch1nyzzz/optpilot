"""LiveCodeBench benchmark loader and code-execution scorer.

Loads competitive programming problems from livecodebench/code_generation_lite
on HuggingFace.  Problems come from LeetCode (functional), AtCoder and
Codeforces (stdin).  Significantly harder than HumanEval.

Scoring works by executing the agent's generated code against private test
cases decoded from the dataset.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import re
import subprocess
import sys
import tempfile
import textwrap
import zlib
from pathlib import Path

from optpilot.data.benchmarks import BenchmarkExample


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_livecodebench_examples(
    limit: int = 200,
    version: str = "release_v5",
    difficulty: str | None = None,
) -> list[BenchmarkExample]:
    """Load LiveCodeBench code-generation problems as BenchmarkExamples.

    Args:
        limit: Maximum number of problems to load.
        version: Dataset version tag (release_v1 .. release_v6).
        difficulty: Optional filter — "easy", "medium", or "hard".
    """
    from datasets import load_dataset

    ds = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=version,
        split="test",
        trust_remote_code=True,
    )

    examples: list[BenchmarkExample] = []
    for item in ds:
        if len(examples) >= limit:
            break

        if difficulty and item.get("difficulty", "").lower() != difficulty.lower():
            continue

        # Decode private test cases (base64 -> zlib -> pickle -> JSON string)
        private_raw = item.get("private_test_cases", "")
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(private_raw)))
            if isinstance(decoded, str):
                decoded = json.loads(decoded)
            private_tests = decoded if isinstance(decoded, list) else []
        except Exception:
            private_tests = []

        # Decode public test cases (JSON string)
        public_raw = item.get("public_test_cases", "[]")
        try:
            public_tests = json.loads(public_raw) if isinstance(public_raw, str) else public_raw
            if not isinstance(public_tests, list):
                public_tests = []
        except (json.JSONDecodeError, TypeError):
            public_tests = []

        # Determine test type from first available test case
        all_tests = public_tests + private_tests
        test_type = "stdin"
        if all_tests:
            test_type = all_tests[0].get("testtype", "stdin")

        starter_code = item.get("starter_code", "") or ""
        question_content = item.get("question_content", "")
        question_title = item.get("question_title", "")
        platform = item.get("platform", "unknown")

        # Build the task prompt for the MAS
        if test_type == "functional" and starter_code:
            task_prompt = (
                f"Solve the following programming problem. "
                f"Complete the given starter code. Return ONLY the complete "
                f"Python solution (including the class/function definition). "
                f"Do NOT include test code or examples.\n\n"
                f"## Problem: {question_title}\n\n"
                f"{question_content}\n\n"
                f"## Starter Code\n```python\n{starter_code}\n```"
            )
        else:
            task_prompt = (
                f"Solve the following programming problem. "
                f"Write a Python solution that reads from stdin and prints to stdout. "
                f"Return ONLY the complete Python solution. "
                f"Do NOT include test code or examples.\n\n"
                f"## Problem: {question_title}\n\n"
                f"{question_content}"
            )

        examples.append(BenchmarkExample(
            benchmark_name="LiveCodeBench",
            task_id=item.get("question_id", f"lcb_{len(examples)}"),
            prompt=task_prompt,
            gold_answers=("",),  # no single gold answer — scored via test execution
            answer_type="code",
            metadata={
                "question_title": question_title,
                "platform": platform,
                "difficulty": item.get("difficulty", "unknown"),
                "test_type": test_type,
                "starter_code": starter_code,
                "public_test_cases": public_tests,
                "private_test_cases": private_tests,
            },
        ))
    return examples


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_from_response(response: str) -> str | None:
    """Extract Python code from the agent's response.

    Tries fenced code blocks first, then falls back to the raw response.
    """
    if not response:
        return None

    # Strategy 1: fenced code blocks
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip()

    # Strategy 2: if it looks like code, use directly
    if any(kw in response for kw in ("def ", "class ", "import ", "input()", "sys.stdin")):
        return response.strip()

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _build_stdin_test_script(code: str, test_input: str, expected_output: str) -> str:
    """Build a script that feeds stdin to the solution and checks stdout."""
    return textwrap.dedent(f"""\
        import sys, io
        sys.stdin = io.StringIO({test_input!r})
        _captured = io.StringIO()
        sys.stdout = _captured

        {textwrap.indent(code, '        ').strip()}

        sys.stdout = sys.__stdout__
        actual = _captured.getvalue().strip()
        expected = {expected_output!r}.strip()
        if actual != expected:
            sys.exit(1)
    """)


def _build_functional_test_script(
    code: str, starter_code: str, test_input: str, expected_output: str,
) -> str:
    """Build a script that calls the function and checks the return value."""
    return textwrap.dedent(f"""\
        import json, sys

        {code}

        _input = json.loads({test_input!r})
        _expected = json.loads({expected_output!r})

        # Find the Solution class method or standalone function
        if 'Solution' in dir():
            _sol = Solution()
            # Get the first public method
            _methods = [m for m in dir(_sol) if not m.startswith('_')]
            if _methods:
                _result = getattr(_sol, _methods[0])(*_input)
            else:
                sys.exit(1)
        else:
            sys.exit(1)

        if _result != _expected:
            sys.exit(1)
    """)


def score_livecodebench(
    prediction: str,
    example: BenchmarkExample,
    timeout: int = 30,
) -> float:
    """Score a LiveCodeBench prediction by executing against test cases.

    Returns 1.0 if ALL test cases pass, 0.0 otherwise.
    """
    code = extract_code_from_response(prediction)
    if code is None:
        return 0.0

    test_type = example.metadata.get("test_type", "stdin")
    starter_code = example.metadata.get("starter_code", "")
    # Use private tests if available, else public
    test_cases = example.metadata.get("private_test_cases") or []
    if not test_cases:
        test_cases = example.metadata.get("public_test_cases") or []
    if not test_cases:
        return 0.0

    for tc in test_cases:
        test_input = tc.get("input", "")
        expected_output = tc.get("output", "")

        if test_type == "functional" and starter_code:
            script = _build_functional_test_script(
                code, starter_code, test_input, expected_output,
            )
        else:
            script = _build_stdin_test_script(code, test_input, expected_output)

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, dir="/tmp",
        ) as f:
            f.write(script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return 1.0
