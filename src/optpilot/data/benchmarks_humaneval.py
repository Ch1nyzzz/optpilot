"""HumanEval benchmark loader and code-execution scorer.

Loads problems from openai/openai_humaneval on HuggingFace.
Scoring works by concatenating the agent's generated code with the
problem's test harness and executing via subprocess.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from optpilot.data.benchmarks import BenchmarkExample


def load_humaneval_examples(limit: int = 164) -> list[BenchmarkExample]:
    """Load HumanEval problems as BenchmarkExamples."""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    examples: list[BenchmarkExample] = []
    for item in ds:
        if len(examples) >= limit:
            break

        # The prompt for the MAS: ask agents to complete the function
        task_prompt = (
            "Complete the following Python function. Return ONLY the complete "
            "function implementation (including the signature and docstring). "
            "Do NOT include test code or examples.\n\n"
            f"{item['prompt']}"
        )

        examples.append(BenchmarkExample(
            benchmark_name="HumanEval",
            task_id=item["task_id"],
            prompt=task_prompt,
            gold_answers=(item["canonical_solution"],),
            answer_type="code",
            metadata={
                "entry_point": item["entry_point"],
                "test": item["test"],
                "function_signature": item["prompt"],
            },
        ))
    return examples


def extract_code_from_response(response: str, entry_point: str) -> str | None:
    """Extract a Python function from the agent's response.

    Tries multiple strategies:
    1. Fenced code block (```python ... ```)
    2. Find the function definition by entry_point name
    3. Use the entire response as code
    """
    if not response:
        return None

    # Strategy 1: Extract from fenced code blocks
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if fenced:
        # Use the longest fenced block that contains the entry point
        for block in sorted(fenced, key=len, reverse=True):
            if entry_point in block:
                return block.strip()
        # Fallback: longest block
        return max(fenced, key=len).strip()

    # Strategy 2: Find the function definition
    pattern = rf"((?:from\s+\S+\s+import\s+\S+\n|import\s+\S+\n)*\s*def\s+{re.escape(entry_point)}\s*\(.*)"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: If response looks like code, use it directly
    if "def " in response:
        return response.strip()

    return None


def score_humaneval(
    prediction: str,
    example: BenchmarkExample,
    timeout: int = 15,
) -> float:
    """Score a HumanEval prediction by executing test cases.

    Returns 1.0 if all tests pass, 0.0 otherwise.
    """
    entry_point = example.metadata["entry_point"]
    test_code = example.metadata["test"]
    function_signature = example.metadata["function_signature"]

    code = extract_code_from_response(prediction, entry_point)
    if code is None:
        return 0.0

    # If the extracted code doesn't include the function signature (just the
    # body), prepend the signature from the problem.
    if f"def {entry_point}" not in code:
        code = function_signature + code

    # Build the full test script: function + test harness + check() call
    test_script = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as f:
        f.write(test_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        Path(tmp_path).unlink(missing_ok=True)
