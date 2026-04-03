from __future__ import annotations

from optpilot.data.benchmarks import BenchmarkExample
from optpilot.data.benchmarks_gaia import (
    is_strict_supported_gaia_row,
    load_gaia_examples,
    split_gaia_examples_evenly,
)


def test_is_strict_supported_gaia_row_rejects_files_and_multimodal_cues():
    assert is_strict_supported_gaia_row("Find the capital of France.")
    assert not is_strict_supported_gaia_row(
        "Look at the attached image and answer the question."
    )
    assert not is_strict_supported_gaia_row(
        "Find the answer from the webpage image.",
    )
    assert not is_strict_supported_gaia_row(
        "Summarize this task.",
        file_name="example.pdf",
    )


def test_load_gaia_examples_strict_supported_only_filters_dataset(monkeypatch):
    fake_rows = [
        {
            "task_id": "keep-1",
            "Question": "Find the capital of France.",
            "Level": "1",
            "Final answer": "Paris",
            "file_name": "",
            "file_path": "",
        },
        {
            "task_id": "drop-image",
            "Question": "Look at the attached image and answer the question.",
            "Level": "2",
            "Final answer": "42",
            "file_name": "",
            "file_path": "",
        },
        {
            "task_id": "drop-file",
            "Question": "Read the spreadsheet and compute the sum.",
            "Level": "1",
            "Final answer": "10",
            "file_name": "sheet.xlsx",
            "file_path": "/tmp/sheet.xlsx",
        },
        {
            "task_id": "keep-2",
            "Question": "What is 12 * 12?",
            "Level": "3",
            "Final answer": "144",
            "file_name": "",
            "file_path": "",
        },
    ]

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: fake_rows)

    examples = load_gaia_examples(limit=10, strict_supported_only=True)

    assert [example.task_id for example in examples] == ["keep-1", "keep-2"]


def test_split_gaia_examples_evenly_balances_levels():
    examples = [
        BenchmarkExample("GAIA", "l1-a", "p1", ("a",), "exact", metadata={"level": "1"}),
        BenchmarkExample("GAIA", "l1-b", "p2", ("a",), "exact", metadata={"level": "1"}),
        BenchmarkExample("GAIA", "l1-c", "p3", ("a",), "exact", metadata={"level": "1"}),
        BenchmarkExample("GAIA", "l2-a", "p4", ("a",), "exact", metadata={"level": "2"}),
        BenchmarkExample("GAIA", "l2-b", "p5", ("a",), "exact", metadata={"level": "2"}),
        BenchmarkExample("GAIA", "l3-a", "p6", ("a",), "exact", metadata={"level": "3"}),
    ]

    train_examples, test_examples = split_gaia_examples_evenly(examples)

    assert [example.task_id for example in train_examples] == ["l1-a", "l1-c", "l2-a", "l3-a"]
    assert [example.task_id for example in test_examples] == ["l1-b", "l2-b"]
