"""AppWorld benchmark loader.

Loads real AppWorld tasks (728 total across train/dev/test splits).
Each task involves multi-step API interactions across 9 simulated apps
(Amazon, Spotify, Venmo, etc.) via a Supervisor-delegate pattern.

Requires: pip install appworld && appworld install && appworld download data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optpilot.data.benchmarks import BenchmarkExample

_APPWORLD_DATA = Path("data")  # AppWorld downloads to ./data/


def _find_appworld_data() -> Path:
    """Locate AppWorld data directory."""
    candidates = [
        Path("data"),
        Path.home() / ".appworld" / "data",
        Path(__file__).resolve().parents[4] / "data",  # project root
    ]
    for p in candidates:
        if (p / "tasks").exists() and (p / "datasets").exists():
            return p
    raise FileNotFoundError(
        "AppWorld data not found. Run: pip install appworld && appworld install && appworld download data"
    )


def load_appworld_examples(
    limit: int = 100,
    splits: tuple[str, ...] = ("train", "dev"),
    require_ground_truth: bool = True,
) -> list[BenchmarkExample]:
    """Load AppWorld tasks as BenchmarkExamples.

    Args:
        limit: Max total examples to load.
        splits: Which splits to load from (in order).
    """
    data_dir = _find_appworld_data()
    tasks_dir = data_dir / "tasks"
    datasets_dir = data_dir / "datasets"

    # Collect task IDs from requested splits
    task_ids: list[str] = []
    for split in splits:
        split_file = datasets_dir / f"{split}.txt"
        if split_file.exists():
            ids = split_file.read_text().strip().splitlines()
            task_ids.extend(ids)

    examples: list[BenchmarkExample] = []
    for tid in task_ids[:limit]:
        spec_path = tasks_dir / tid / "specs.json"
        if not spec_path.exists():
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        instruction = spec.get("instruction", "")
        supervisor = spec.get("supervisor", {})
        sup_name = f"{supervisor.get('first_name', '')} {supervisor.get('last_name', '')}".strip()

        prompt = (
            f"You are acting as {sup_name}. "
            f"Complete the following task using the available API services.\n\n"
            f"Task: {instruction}\n\n"
            f"Use the api_call tool to interact with services (amazon, spotify, venmo, "
            f"gmail, phone, todoist, file_system, admin, supervisor). "
            f"When done, use verify_result to submit your answer."
        )

        # Check for ground truth (answer.json contains a JSON string/value)
        answer_path = tasks_dir / tid / "ground_truth" / "answer.json"
        gold_answer = ""
        if answer_path.exists():
            try:
                raw = json.loads(answer_path.read_text(encoding="utf-8"))
                gold_answer = str(raw) if raw is not None else ""
            except (json.JSONDecodeError, OSError):
                pass

        if require_ground_truth and not gold_answer:
            continue

        examples.append(BenchmarkExample(
            benchmark_name="AppWorld",
            task_id=tid,
            prompt=prompt,
            gold_answers=(gold_answer,),
            answer_type="exact",
            metadata={
                "instruction": instruction,
                "supervisor": supervisor,
                "datetime": spec.get("datetime", ""),
                "task_dir": str(tasks_dir / tid),
            },
        ))

    return examples


def score_appworld(prediction: str, gold_answer: str) -> float:
    """Score AppWorld task: exact match on the final answer."""
    if not prediction or not gold_answer:
        return 0.0

    pred = prediction.strip().lower()
    gold = gold_answer.strip().lower()
    if pred == gold or gold in pred:
        return 1.0
    return 0.0
