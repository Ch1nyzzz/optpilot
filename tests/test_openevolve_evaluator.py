from __future__ import annotations

import importlib.util
from pathlib import Path

from optpilot.data.benchmarks import BenchmarkExample, OfficialBenchmarkSuite


def test_openevolve_evaluator_uses_explicit_eval_prompts(monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_TOTAL_TASKS", "4")
    monkeypatch.setenv("OPENEVOLVE_EVAL_TASKS", "2")
    monkeypatch.setenv("OPENEVOLVE_EVAL_PROMPTS_JSON", '["task-3", "task-1"]')

    module_path = Path(__file__).resolve().parents[1] / "experiments" / "openevolve_evaluator.py"
    spec = importlib.util.spec_from_file_location("tests.openevolve_evaluator", module_path)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator._suite = None
    evaluator._eval_prompts = None
    evaluator._runner = None
    evaluator._diagnoser = None

    def fake_load_online_benchmark_suite(_total_tasks):
        return OfficialBenchmarkSuite([
            BenchmarkExample("AIME2024", "0", "task-0", ("1",), "integer"),
            BenchmarkExample("AIME2024", "1", "task-1", ("1",), "integer"),
            BenchmarkExample("AIME2024", "2", "task-2", ("1",), "integer"),
            BenchmarkExample("AIME2024", "3", "task-3", ("1",), "integer"),
        ])

    class DummyRunner:
        def __init__(self, **_kwargs):
            pass

    class DummyDiagnoser:
        pass

    monkeypatch.setattr(evaluator, "load_online_benchmark_suite", fake_load_online_benchmark_suite)
    monkeypatch.setattr(evaluator, "OptPilotRunner", DummyRunner)
    monkeypatch.setattr(evaluator, "Diagnoser", DummyDiagnoser)

    evaluator._init_globals()

    assert evaluator._eval_prompts == ["task-3", "task-1"]
