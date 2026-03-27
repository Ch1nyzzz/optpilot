"""Official benchmark loaders and ground-truth scorers for online runs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from datasets import load_dataset
from sympy import simplify, sympify

from optpilot.dag.core import MASDAG
from optpilot.dag.executor import ExecutionTrace

MMLU_DATASET = "cais/mmlu"
AIME2025_DATASET = "opencompass/AIME2025"
OLYMPIADBENCH_DATASET = "lscpku/OlympiadBench-official"
OLYMPIADBENCH_DEFAULT_CONFIG = "maths_en_no_proof"
AIME2025_CONFIGS = ("AIME2025-I", "AIME2025-II")
MMLU_DEFAULT_SUBJECTS = (
    "abstract_algebra",
    "astronomy",
    "business_ethics",
    "college_chemistry",
    "econometrics",
    "formal_logic",
    "high_school_mathematics",
    "international_law",
    "professional_law",
    "virology",
)
_MMLU_ANSWER_LETTERS = ("A", "B", "C", "D")
_BOXED_TOKEN = r"\boxed{"


@dataclass(frozen=True)
class BenchmarkExample:
    """A single official benchmark example with its gold answer."""

    benchmark_name: str
    task_id: str
    prompt: str
    gold_answers: tuple[str, ...]
    answer_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OfficialBenchmarkSuite:
    """Bundle benchmark tasks with per-task benchmark lookup and scoring."""

    def __init__(self, examples: Sequence[BenchmarkExample]):
        self.examples = list(examples)
        self._by_prompt = {example.prompt: example for example in self.examples}
        if len(self._by_prompt) != len(self.examples):
            raise ValueError("Benchmark prompts must be unique.")

    def tasks(self) -> list[str]:
        """Return prompts in run order."""
        return [example.prompt for example in self.examples]

    def benchmark_name_for_task(self, task_prompt: str) -> str:
        """Resolve the benchmark name for a task prompt."""
        example = self._by_prompt.get(task_prompt)
        return example.benchmark_name if example is not None else "Unknown"

    def benchmark_counts(self) -> dict[str, int]:
        """Return the task count per benchmark."""
        return dict(Counter(example.benchmark_name for example in self.examples))

    def score_task(
        self,
        task_prompt: str,
        _dag: MASDAG,
        exec_trace: ExecutionTrace,
    ) -> float | None:
        """Score a task run against the benchmark ground truth."""
        example = self._by_prompt.get(task_prompt)
        if example is None:
            return None

        prediction = extract_prediction(example.benchmark_name, exec_trace)
        if example.benchmark_name == "MMLU":
            return 1.0 if prediction in example.gold_answers else 0.0

        return 1.0 if matches_math_answer(example, prediction) else 0.0


def load_online_benchmark_suite(
    total_tasks: int,
    *,
    mmlu_subjects: Sequence[str] = MMLU_DEFAULT_SUBJECTS,
    olympiad_config: str = OLYMPIADBENCH_DEFAULT_CONFIG,
) -> OfficialBenchmarkSuite:
    """Load a deterministic MMLU/AIME/OlympiadBench mix for the online pipeline.

    Tries equal 1/3 split; when a benchmark has fewer examples than its share,
    the surplus is redistributed to the others proportionally.
    """
    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")

    # Caps per benchmark (AIME is small: only 30 questions)
    _AIME_CAP = 30

    base = total_tasks // 3
    remainder = total_tasks % 3
    mmlu_target = base + (1 if remainder >= 1 else 0)
    aime_target = base + (1 if remainder >= 2 else 0)
    olympiad_target = base

    # If AIME target exceeds cap, redistribute surplus
    if aime_target > _AIME_CAP:
        surplus = aime_target - _AIME_CAP
        aime_target = _AIME_CAP
        mmlu_target += (surplus + 1) // 2
        olympiad_target += surplus // 2

    examples: list[BenchmarkExample] = []
    mmlu = load_mmlu_examples(mmlu_target, subjects=mmlu_subjects)
    aime = load_aime_examples(aime_target)
    olympiad = load_olympiad_examples(olympiad_target, config_name=olympiad_config)

    # If any benchmark underflows, redistribute again
    loaded = [("mmlu", mmlu, mmlu_target), ("aime", aime, aime_target), ("olympiad", olympiad, olympiad_target)]
    shortfall = sum(max(0, target - len(ex)) for _, ex, target in loaded)
    if shortfall > 0:
        # Fill from MMLU (largest pool)
        extra_mmlu = load_mmlu_examples(mmlu_target + shortfall, subjects=mmlu_subjects)
        mmlu = extra_mmlu

    examples.extend(mmlu)
    examples.extend(aime)
    examples.extend(olympiad)
    examples = examples[:total_tasks]

    if len(examples) < total_tasks:
        raise ValueError(
            f"Requested {total_tasks} tasks, but only loaded {len(examples)} official benchmark tasks."
        )
    return OfficialBenchmarkSuite(examples)


def load_mmlu_examples(
    limit: int,
    *,
    split: str = "test",
    subjects: Sequence[str] = MMLU_DEFAULT_SUBJECTS,
) -> list[BenchmarkExample]:
    """Load a small deterministic MMLU slice across subjects."""
    if limit <= 0:
        return []

    active_subjects = subjects[: min(limit, len(subjects))]
    datasets = [(subject, load_dataset(MMLU_DATASET, subject, split=split)) for subject in active_subjects]
    examples: list[BenchmarkExample] = []
    row_index = 0
    while len(examples) < limit:
        made_progress = False
        for subject, dataset in datasets:
            if row_index >= len(dataset):
                continue
            row = dataset[row_index]
            prompt = format_mmlu_prompt(row["question"], row["choices"])
            answer_letter = _MMLU_ANSWER_LETTERS[int(row["answer"])]
            examples.append(
                BenchmarkExample(
                    benchmark_name="MMLU",
                    task_id=f"MMLU::{subject}::{row_index}",
                    prompt=prompt,
                    gold_answers=(answer_letter,),
                    answer_type="multiple_choice",
                    metadata={"subject": subject, "choices": tuple(row["choices"])},
                )
            )
            made_progress = True
            if len(examples) >= limit:
                break
        if not made_progress:
            break
        row_index += 1
    return examples


def load_olympiad_examples(
    limit: int,
    *,
    config_name: str = OLYMPIADBENCH_DEFAULT_CONFIG,
    split: str = "test",
) -> list[BenchmarkExample]:
    """Load text-only OlympiadBench examples with machine-checkable answers."""
    if limit <= 0:
        return []

    dataset = load_dataset(OLYMPIADBENCH_DATASET, config_name, split=split)
    examples: list[BenchmarkExample] = []
    for row in dataset:
        if not is_supported_olympiad_row(row):
            continue
        prompt = format_olympiad_prompt(row["question"], row.get("context"))
        examples.append(
            BenchmarkExample(
                benchmark_name="OlympiadBench",
                task_id=f"OlympiadBench::{row['id']}",
                prompt=prompt,
                gold_answers=tuple(answer for answer in row["final_answer"] if answer),
                answer_type=row["answer_type"],
                metadata={
                    "difficulty": row.get("difficulty"),
                    "error": row.get("error"),
                    "subject": row.get("subject"),
                    "subfield": row.get("subfield"),
                },
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_aime_examples(
    limit: int,
    *,
    configs: Sequence[str] = AIME2025_CONFIGS,
    split: str = "test",
) -> list[BenchmarkExample]:
    """Load a small deterministic slice from AIME 2025."""
    if limit <= 0:
        return []

    datasets = [(config, load_dataset(AIME2025_DATASET, config, split=split)) for config in configs]
    examples: list[BenchmarkExample] = []
    row_index = 0
    while len(examples) < limit:
        made_progress = False
        for config, dataset in datasets:
            if row_index >= len(dataset):
                continue
            row = dataset[row_index]
            examples.append(
                BenchmarkExample(
                    benchmark_name="AIME2025",
                    task_id=f"AIME2025::{config}::{row_index}",
                    prompt=row["question"].strip(),
                    gold_answers=(row["answer"].strip(),),
                    answer_type="integer",
                    metadata={"config": config},
                )
            )
            made_progress = True
            if len(examples) >= limit:
                break
        if not made_progress:
            break
        row_index += 1
    return examples


def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    """Format an MMLU question into a prompt the DAG can consume."""
    lines = [question.strip(), "", "Options:"]
    lines.extend(f"({_MMLU_ANSWER_LETTERS[idx]}) {choice}" for idx, choice in enumerate(choices))
    return "\n".join(lines)


def format_olympiad_prompt(question: str, context: str | None) -> str:
    """Format a text-only OlympiadBench question."""
    clean_question = re.sub(r"<image_\d+>", "", question).strip()
    clean_context = (context or "").strip()
    if not clean_context:
        return clean_question
    return f"{clean_question}\n\nContext:\n{clean_context}"


def is_supported_olympiad_row(row: dict[str, Any]) -> bool:
    """Keep only text-only OlympiadBench rows with exact ground truth."""
    if any(row.get(f"image_{idx}") is not None for idx in range(1, 6)):
        return False
    if re.search(r"<image_\d+>", row.get("question", "")):
        return False
    if row.get("answer_type") not in {"Numerical", "Expression"}:
        return False
    return bool(row.get("final_answer"))


def extract_prediction(benchmark_name: str, exec_trace: ExecutionTrace) -> str | None:
    """Extract the benchmark answer from the execution trace."""
    texts = []
    texts.extend(
        step.output_text
        for step in reversed(exec_trace.steps)
        if step.node_id == "Agent_Verifier" and step.output_text
    )
    texts.extend(step.output_text for step in reversed(exec_trace.steps) if step.output_text)

    if benchmark_name == "MMLU":
        for text in texts:
            for boxed in reversed(extract_boxed_segments(text)):
                normalized = normalize_mmlu_answer(boxed)
                if normalized is not None:
                    return normalized
            match = re.search(
                r"(?:selected_option|option|answer)[^A-D]{0,20}([A-D])\b",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                normalized = normalize_mmlu_answer(match.group(1))
                if normalized is not None:
                    return normalized
        return None

    for text in texts:
        boxed = extract_boxed_segments(text)
        if boxed:
            return boxed[-1]
    return None


def extract_boxed_segments(text: str) -> list[str]:
    """Extract every ``\\boxed{...}`` segment, keeping nested braces intact."""
    segments: list[str] = []
    start = 0
    while True:
        idx = text.find(_BOXED_TOKEN, start)
        if idx == -1:
            return segments
        cursor = idx + len(_BOXED_TOKEN)
        depth = 1
        chunk: list[str] = []
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    break
            if depth > 0:
                chunk.append(char)
            cursor += 1
        if depth == 0:
            segments.append("".join(chunk).strip())
        start = cursor


def normalize_mmlu_answer(answer: str) -> str | None:
    """Normalize a multiple-choice answer into A/B/C/D."""
    cleaned = answer.strip().upper().strip("(). ")
    return cleaned if cleaned in _MMLU_ANSWER_LETTERS else None


def matches_math_answer(example: BenchmarkExample, prediction: str | None) -> bool:
    """Match an extracted AIME/Olympiad answer against official gold answers."""
    if prediction is None:
        return False
    normalized_prediction = normalize_simple_answer(prediction)
    tolerance = parse_numeric_tolerance(example.metadata.get("error"))

    for gold in example.gold_answers:
        if normalized_prediction == normalize_simple_answer(gold):
            return True

        pred_expr = parse_math_expression(prediction)
        gold_expr = parse_math_expression(gold)
        if pred_expr is None or gold_expr is None:
            continue

        if tolerance is not None:
            try:
                if abs(float(pred_expr.evalf()) - float(gold_expr.evalf())) <= tolerance:
                    return True
            except (TypeError, ValueError):
                continue
            continue

        try:
            if simplify(pred_expr - gold_expr) == 0:
                return True
        except Exception:
            continue
    return False


def parse_numeric_tolerance(raw_error: Any) -> float | None:
    """Parse the OlympiadBench numeric tolerance field."""
    if raw_error in (None, ""):
        return None
    try:
        return float(raw_error)
    except (TypeError, ValueError):
        return None


def normalize_simple_answer(text: str) -> str:
    """Normalize an answer string for exact string comparison."""
    cleaned = strip_answer_wrappers(text)
    return re.sub(r"\s+", "", cleaned)


def strip_answer_wrappers(text: str) -> str:
    """Remove common answer wrappers such as ``$...$`` and trailing punctuation."""
    cleaned = text.strip()
    cleaned = cleaned.strip("$")
    cleaned = cleaned.rstrip(".")
    return cleaned


def parse_math_expression(text: str):
    """Parse a small LaTeX-like mathematical answer into a SymPy expression."""
    cleaned = strip_answer_wrappers(text)
    cleaned = cleaned.replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace(r"\cdot", "*").replace(r"\times", "*")
    cleaned = cleaned.replace(r"\pi", "pi")
    cleaned = cleaned.replace("^", "**")
    while True:
        updated = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    while True:
        updated = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    try:
        return sympify(cleaned)
    except Exception:
        return None
