"""Tests for the Jacobian-driven single repair loop and repair_loop functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.models import (
    AnalysisResult,
    EvolveResult,
    FMLabel,
    FMProfile,
    ReflectInsight,
    SkillBudget,
    SkillResult,
)
from optpilot.orchestrator import Orchestrator, rank_fm_groups
from optpilot.skills.repair_loop import (
    build_synthetic_insight,
    extract_python_code,
    fm_rate,
    format_negatives,
    has_material_change,
    pass_rate,
)
from optpilot.skills.repair_patterns import PatternCatalog, RepairPattern


# -------------------------------------------------------------------- #
#  repair_loop helper tests                                             #
# -------------------------------------------------------------------- #

def test_fm_rate_with_active_profiles():
    profiles = []
    for i in range(4):
        p = FMProfile(trace_id=i)
        if i < 3:
            p.labels["B"] = FMLabel(fm_id="B", fm_name="Loop", category="B", present=True)
        profiles.append(p)

    assert fm_rate("B", profiles) == 0.75
    assert fm_rate("A", profiles) == 0.0
    assert fm_rate("B", []) == 0.0


def test_has_material_change_detects_dag_difference():
    dag1 = MASDAG(
        dag_id="test", nodes={"a": DAGNode(node_id="a", node_type="literal", prompt="v1")},
        edges=[], metadata={},
    )
    dag2 = MASDAG(
        dag_id="test", nodes={"a": DAGNode(node_id="a", node_type="literal", prompt="v2")},
        edges=[], metadata={},
    )
    assert has_material_change(dag1, dag2) is True
    assert has_material_change(dag1, dag1) is False


def test_format_negatives_empty():
    assert format_negatives([]) == "None yet."


def test_format_negatives_with_entries():
    neg = ReflectInsight(
        round_index=1, fm_id="A", changes_attempted=["fix prompt"],
        before_fm_rate=1.0, after_fm_rate=0.8,
        before_pass_rate=0.0, after_pass_rate=0.1,
        failure_reason="still failing", lesson="try again",
    )
    text = format_negatives([neg])
    assert "fix prompt" in text
    assert "still failing" in text


def test_extract_python_code_fenced():
    response = "```python\ndef build_dag():\n    return {}\n```\nDone."
    code = extract_python_code(response)
    assert "def build_dag" in code


def test_extract_python_code_fallback():
    response = "def build_dag():\n    return {}"
    code = extract_python_code(response)
    assert "def build_dag" in code


def test_extract_python_code_empty():
    code = extract_python_code("No code here.")
    assert code == ""


def test_build_synthetic_insight():
    insight = build_synthetic_insight(
        fm_group="B",
        evolve_result=None,
        before_fm=1.0, after_fm=0.8,
        before_pass=0.0, after_pass=0.1,
        failure_reason="test reason",
        lesson="test lesson",
    )
    assert insight.fm_id == "B"
    assert insight.failure_reason == "test reason"
    assert insight.lesson == "test lesson"


# -------------------------------------------------------------------- #
#  PatternCatalog tests                                                 #
# -------------------------------------------------------------------- #

def test_pattern_catalog_defaults():
    catalog = PatternCatalog()
    assert len(catalog) >= 13
    assert "prompt_add_constraint" in catalog
    p = catalog["prompt_add_constraint"]
    assert p.effective is True


def test_pattern_catalog_add_and_update(tmp_path):
    store = tmp_path / "catalog.json"
    catalog = PatternCatalog(store_path=store)
    original_count = len(catalog)

    catalog.add_pattern(RepairPattern(
        pattern_id="test_new",
        name="Test Pattern",
        description="A test pattern",
        target_components=["agent_prompt"],
    ))
    assert len(catalog) == original_count + 1
    assert "test_new" in catalog

    catalog.update_pattern("test_new", effective=False)
    assert catalog["test_new"].effective is False

    # effective_items should exclude it
    effective_ids = [pid for pid, _ in catalog.effective_items()]
    assert "test_new" not in effective_ids

    # Save and reload
    catalog.save()
    catalog2 = PatternCatalog(store_path=store)
    assert "test_new" in catalog2
    assert catalog2["test_new"].effective is False


def test_pattern_catalog_as_llm_context():
    catalog = PatternCatalog()
    text = catalog.as_llm_context()
    assert "prompt_add_constraint" in text
    assert "Available repair patterns" in text


# -------------------------------------------------------------------- #
#  Orchestrator single-loop test                                        #
# -------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_aoptimize_single_loop_improved(monkeypatch, tmp_path):
    """Test the happy path: diagnose → evolve → evaluate → improved."""
    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    repaired = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="repaired"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    call_count = {"run": 0}

    class _AsyncRunner:
        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            call_count["run"] += 1
            return [_make_trace(i) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        def __init__(self):
            self._call_count = 0

        async def aclassify_batch(self, traces):
            self._call_count += 1
            if self._call_count == 1:
                # First call: FM active
                return [_make_profile(i, fm_active=True) for i in range(len(traces))]
            else:
                # After repair: FM gone
                return [_make_profile(i, fm_active=False) for i in range(len(traces))]

    # Mock analyze and aevolve
    async def fake_aevolve(dag, fm_group, analysis, negatives, history, recommended_pattern=None, traces=None, profiles=None):
        return EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
        )

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr("optpilot.orchestrator.aevolve", fake_aevolve)
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag)
    orchestrator.diagnoser = _AsyncDiagnoser()

    summary = await orchestrator.aoptimize(
        tasks=["t1", "t2"],
        max_rounds=1,
        dag_output_base=tmp_path / "dag_versions",
    )

    assert len(summary["results"]) == 1
    assert summary["results"][0]["success"] is True
    assert orchestrator.dag.nodes["start"].prompt == "repaired"


@pytest.mark.anyio
async def test_aoptimize_no_material_change(monkeypatch, tmp_path):
    """When evolve produces no material change, record failure."""
    dag = MASDAG(
        dag_id="test",
        nodes={"start": DAGNode(node_id="start", node_type="literal", prompt="begin")},
        edges=[], metadata={},
    )

    class _AsyncRunner:
        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            return [_make_trace(i) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

    async def fake_aevolve(dag, fm_group, analysis, negatives, history, recommended_pattern=None, traces=None, profiles=None):
        return EvolveResult(
            dag=dag,  # same DAG = no change
            analysis_text="tried",
            modified_source="source",
            change_description="nothing changed",
        )

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr("optpilot.orchestrator.aevolve", fake_aevolve)
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag)
    orchestrator.diagnoser = _AsyncDiagnoser()

    summary = await orchestrator.aoptimize(tasks=["t1", "t2"], max_rounds=1)

    assert len(summary["results"]) == 1
    assert summary["results"][0]["success"] is False


@pytest.mark.anyio
async def test_aoptimize_uses_fixed_eval_subset(monkeypatch):
    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    repaired = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="repaired"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    seen_task_batches: list[list[str]] = []
    batch_scores = [0.0, 1.0]

    class _AsyncRunner:
        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            seen_task_batches.append(list(tasks))
            score = batch_scores[len(seen_task_batches) - 1]
            return [_make_trace(i, score=score) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

    async def fake_aevolve(dag, fm_group, analysis, negatives, history, recommended_pattern=None, traces=None, profiles=None):
        return EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
            metadata={"observed_pattern_id": "prompt_add_constraint"},
        )

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr("optpilot.orchestrator.aevolve", fake_aevolve)
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag)
    orchestrator.diagnoser = _AsyncDiagnoser()

    await orchestrator.aoptimize(
        tasks=["t0", "t1", "t2", "t3"],
        max_rounds=1,
        eval_tasks_per_round=2,
    )

    assert seen_task_batches == [["t0", "t1"], ["t0", "t1"]]


# -------------------------------------------------------------------- #
#  Helpers                                                              #
# -------------------------------------------------------------------- #

class _FakeTrace:
    def __init__(self, trace_id, score=0.0):
        self.trace_id = trace_id
        self.task_score = score
        self.task_success = score > 0
        self.latency_s = 1.0
        self.task_key = f"task-{trace_id}"
        self.benchmark_name = "test"
        self.trace_path = ""
        self.mas_name = "test"
        self.llm_name = "test"
        self.trajectory = ""


def _make_trace(idx, score=0.0):
    return _FakeTrace(trace_id=idx, score=score)


def _make_profile(idx, fm_active=True, fm_group="B"):
    p = FMProfile(trace_id=idx)
    if fm_active:
        p.labels[fm_group] = FMLabel(
            fm_id=fm_group, fm_name="Test FM", category=fm_group, present=True,
        )
    return p
