from __future__ import annotations

from optpilot.data.benchmarks import (
    BenchmarkExample,
    OfficialBenchmarkSuite,
    extract_boxed_segments,
    matches_math_answer,
)
from optpilot.dag.executor import ExecutionTrace, NodeExecution


def _trace(output_text: str) -> ExecutionTrace:
    return ExecutionTrace(
        dag_id="AG2_MathChat",
        task_prompt="task",
        steps=[
            NodeExecution(
                node_id="Agent_Verifier",
                node_type="agent",
                iteration=1,
                input_text="",
                output_text=output_text,
            )
        ],
        finished=True,
    )


def test_extract_boxed_segments_keeps_nested_braces() -> None:
    text = r"Answer: \boxed{\frac{1}{2}} and later \boxed{A}"

    boxed = extract_boxed_segments(text)

    assert boxed == [r"\frac{1}{2}", "A"]


def test_official_benchmark_suite_scores_mmlu_answers() -> None:
    example = BenchmarkExample(
        benchmark_name="MMLU",
        task_id="mmlu::0",
        prompt="What is 2+2?",
        gold_answers=("B",),
        answer_type="multiple_choice",
    )
    suite = OfficialBenchmarkSuite([example])

    assert suite.score_task("What is 2+2?", None, _trace(r"SOLUTION_FOUND \boxed{B}")) == 1.0
    assert suite.score_task("What is 2+2?", None, _trace(r"SOLUTION_FOUND \boxed{C}")) == 0.0
    assert suite.benchmark_name_for_task("What is 2+2?") == "MMLU"


def test_matches_olympiad_answer_uses_numeric_tolerance() -> None:
    example = BenchmarkExample(
        benchmark_name="OlympiadBench",
        task_id="olympiad::0",
        prompt="Compute x.",
        gold_answers=("79.67",),
        answer_type="Numerical",
        metadata={"error": "1e-1"},
    )

    assert matches_math_answer(example, "79.62") is True
    assert matches_math_answer(example, "79.40") is False


def test_official_benchmark_suite_scores_aime_answers() -> None:
    example = BenchmarkExample(
        benchmark_name="AIME2025",
        task_id="aime::0",
        prompt="Find the answer.",
        gold_answers=("70",),
        answer_type="integer",
    )
    suite = OfficialBenchmarkSuite([example])

    assert suite.score_task("Find the answer.", None, _trace(r"SOLUTION_FOUND \boxed{70}")) == 1.0
    assert suite.score_task("Find the answer.", None, _trace(r"SOLUTION_FOUND \boxed{71}")) == 0.0
