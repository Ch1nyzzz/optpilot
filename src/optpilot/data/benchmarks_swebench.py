"""SWE-bench Lite benchmark loader for HyperAgent topology.

Loads bug-fixing tasks from princeton-nlp/SWE-bench_Lite.
Each task: problem_statement → agent must produce a patch.
Scoring: check if key elements of the gold patch are present.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from optpilot.data.benchmarks import BenchmarkExample


@dataclass
class SWEBenchTask:
    """A single SWE-bench task with repo setup info."""
    instance_id: str
    repo: str
    problem_statement: str
    gold_patch: str
    test_patch: str
    base_commit: str
    version: str


def load_swebench_examples(limit: int = 50) -> list[BenchmarkExample]:
    """Load SWE-bench Lite examples as BenchmarkExamples."""
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    examples = []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        prompt = (
            f"Fix the following bug in the {item['repo']} repository.\n\n"
            f"## Problem Description\n{item['problem_statement']}\n\n"
            f"Use the available tools to navigate the code, understand the bug, "
            f"and apply a fix. When done, describe the changes you made."
        )
        examples.append(BenchmarkExample(
            benchmark_name="SWE-bench-Lite",
            task_id=item["instance_id"],
            prompt=prompt,
            gold_answers=(item["patch"],),
            answer_type="patch",
            metadata={
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "version": item["version"],
                "test_patch": item.get("test_patch", ""),
                "hints_text": item.get("hints_text", ""),
            },
        ))
    return examples


def score_swebench(
    prediction: str,
    gold_patch: str,
    workdir: str = "",
) -> float:
    """Score a SWE-bench prediction.

    If workdir is provided, runs `git diff` to compare actual file changes
    against the gold patch.  Otherwise falls back to text matching.
    """
    if not gold_patch:
        return 0.0

    # Extract key added lines from gold patch (the "new" code)
    gold_added = set()
    gold_removed = set()
    for line in gold_patch.splitlines():
        stripped = line.strip()
        if stripped.startswith("+") and not stripped.startswith("+++"):
            gold_added.add(stripped[1:].strip())
        elif stripped.startswith("-") and not stripped.startswith("---"):
            gold_removed.add(stripped[1:].strip())

    if not gold_added and not gold_removed:
        return 0.0

    # Try git diff if workdir available
    if workdir:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "diff"], cwd=workdir,
                capture_output=True, text=True, timeout=10,
            )
            actual_diff = result.stdout
            if actual_diff:
                # Check overlap between actual changes and gold
                actual_added = set()
                for line in actual_diff.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("+") and not stripped.startswith("+++"):
                        actual_added.add(stripped[1:].strip())

                if gold_added:
                    overlap = len(gold_added & actual_added)
                    return min(1.0, overlap / len(gold_added))
        except Exception:
            pass

    # Fallback: check if key changes appear in prediction text
    if not prediction:
        return 0.0
    pred_lower = prediction.lower()
    all_changes = gold_added | gold_removed
    matches = sum(1 for change in all_changes if change.lower() in pred_lower)
    return min(1.0, matches / max(1, len(all_changes)))
