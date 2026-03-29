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
    FMLocalization,
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


def test_rank_fm_groups_uses_primary_failure_only():
    p1 = FMProfile(trace_id=1, primary_fm_id="B")
    p1.labels["B"] = FMLabel(fm_id="B", fm_name="Loop", category="B", present=True)
    p1.labels["F"] = FMLabel(fm_id="F", fm_name="Verify", category="F", present=True)

    p2 = FMProfile(trace_id=2, primary_fm_id="F")
    p2.labels["B"] = FMLabel(fm_id="B", fm_name="Loop", category="B", present=True)
    p2.labels["F"] = FMLabel(fm_id="F", fm_name="Verify", category="F", present=True)

    p3 = FMProfile(trace_id=3, primary_fm_id="F")
    p3.labels["F"] = FMLabel(fm_id="F", fm_name="Verify", category="F", present=True)

    assert rank_fm_groups([p1, p2, p3], min_support=1) == [("F", 2), ("B", 1)]


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
    assert p.pattern_id == "prompt_add_constraint"


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
    assert "Available repair patterns" in text
    effective_ids = [pid for pid, _ in catalog.effective_items()]
    if effective_ids:
        assert effective_ids[0] in text


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

    # Mock analyze and candidate generation
    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        return [EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
        )]

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr(
        "optpilot.orchestrator.agenerate_evolve_candidates",
        fake_agenerate_evolve_candidates,
    )
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

    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        return [EvolveResult(
            dag=dag,  # same DAG = no change
            analysis_text="tried",
            modified_source="source",
            change_description="nothing changed",
        )]

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr(
        "optpilot.orchestrator.agenerate_evolve_candidates",
        fake_agenerate_evolve_candidates,
    )
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag)
    orchestrator.diagnoser = _AsyncDiagnoser()

    summary = await orchestrator.aoptimize(tasks=["t1", "t2"], max_rounds=1)

    assert len(summary["results"]) == 1
    assert summary["results"][0]["success"] is False


@pytest.mark.anyio
async def test_aoptimize_samples_balanced_active_subset(monkeypatch):
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
        benchmark_name_resolver = staticmethod(lambda task: task.split(":", 1)[0])

        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            seen_task_batches.append(list(tasks))
            score = batch_scores[len(seen_task_batches) - 1]
            return [_make_trace(i, score=score) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        return [EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
            metadata={"observed_pattern_id": "prompt_add_constraint"},
        )]

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr(
        "optpilot.orchestrator.agenerate_evolve_candidates",
        fake_agenerate_evolve_candidates,
    )
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag)
    orchestrator.diagnoser = _AsyncDiagnoser()

    await orchestrator.aoptimize(
        tasks=["A:0", "A:1", "B:0", "B:1", "C:0", "C:1", "D:0", "D:1"],
        max_rounds=1,
        eval_tasks_per_round=4,
    )

    assert seen_task_batches[0] == seen_task_batches[1]
    assert {task.split(":", 1)[0] for task in seen_task_batches[0]} == {"A", "B", "C", "D"}


@pytest.mark.anyio
async def test_aoptimize_runs_shadow_gate_every_interval(monkeypatch):
    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    seen_task_batches: list[list[str]] = []
    batch_scores = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    candidate_counter = {"value": 0}

    class _AsyncRunner:
        benchmark_name_resolver = staticmethod(lambda task: task.split(":", 1)[0])

        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            seen_task_batches.append(list(tasks))
            score = batch_scores[len(seen_task_batches) - 1]
            return [_make_trace(i, score=score) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        candidate_counter["value"] += 1
        repaired = MASDAG(
            dag_id="test",
            nodes={
                "start": DAGNode(
                    node_id="start",
                    node_type="literal",
                    prompt=f"repaired-{candidate_counter['value']}",
                ),
                "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
            },
            edges=[DAGEdge(source="start", target="FINAL")],
            metadata={"start": ["start"]},
        )
        return [EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
            metadata={"observed_pattern_id": "prompt_add_constraint"},
        )]

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr(
        "optpilot.orchestrator.agenerate_evolve_candidates",
        fake_agenerate_evolve_candidates,
    )
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)
    monkeypatch.setattr("optpilot.orchestrator.SHADOW_EVAL_INTERVAL", 2)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag)
    orchestrator.diagnoser = _AsyncDiagnoser()

    await orchestrator.aoptimize(
        tasks=["A:0", "A:1", "B:0", "B:1", "C:0", "C:1", "D:0", "D:1"],
        max_rounds=2,
        eval_tasks_per_round=4,
    )

    assert len(seen_task_batches) == 6
    assert seen_task_batches[0] == seen_task_batches[1]
    assert seen_task_batches[2] == seen_task_batches[3]
    assert seen_task_batches[4] == seen_task_batches[5]
    assert set(seen_task_batches[4]).isdisjoint(seen_task_batches[2])


@pytest.mark.anyio
async def test_shadow_gate_uses_accuracy_only(monkeypatch, tmp_path):
    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    batch_scores = [0.0, 1.0, 1.0, 1.0]
    classify_calls = {"count": 0}

    class _AsyncRunner:
        benchmark_name_resolver = staticmethod(lambda task: task.split(":", 1)[0])

        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            score = batch_scores.pop(0)
            return [_make_trace(i, score=score) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            classify_calls["count"] += 1
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        repaired = MASDAG(
            dag_id="test",
            nodes={
                "start": DAGNode(node_id="start", node_type="literal", prompt="repaired"),
                "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
            },
            edges=[DAGEdge(source="start", target="FINAL")],
            metadata={"start": ["start"]},
        )
        return [EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
            metadata={"observed_pattern_id": "prompt_add_constraint"},
        )]

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr("optpilot.orchestrator.agenerate_evolve_candidates", fake_agenerate_evolve_candidates)
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)
    monkeypatch.setattr(
        "optpilot.orchestrator.fitness_score",
        lambda traces, profiles, num_agents: sum(float(t.task_score or 0.0) for t in traces) / len(traces),
    )
    monkeypatch.setattr("optpilot.orchestrator.SHADOW_EVAL_INTERVAL", 1)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag, negatives_dir=tmp_path / "neg")
    orchestrator.diagnoser = _AsyncDiagnoser()

    await orchestrator.aoptimize(
        tasks=["A:0", "A:1", "B:0", "B:1", "C:0", "C:1", "D:0", "D:1"],
        max_rounds=1,
        eval_tasks_per_round=4,
    )

    assert classify_calls["count"] == 2


@pytest.mark.anyio
async def test_shadow_rejection_records_diagnosis_metadata(monkeypatch, tmp_path):
    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    batch_scores = [0.0, 1.0, 1.0, 0.0]

    class _AsyncRunner:
        benchmark_name_resolver = staticmethod(lambda task: task.split(":", 1)[0])

        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            score = batch_scores.pop(0)
            traces = [_make_trace(i, score=score) for i in range(len(tasks))]
            for trace, task in zip(traces, tasks):
                trace.task_key = task
                trace.benchmark_name = task.split(":", 1)[0]
            return traces

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

        async def adiagnose_batch(self, traces, target_group=None):
            profiles = []
            for i, trace in enumerate(traces):
                p = _make_profile(i, fm_active=True)
                p.primary_fm_id = "B"
                p.primary_localization = FMLocalization(
                    agent="Agent_Verifier",
                    step="shadow",
                    context="shadow regression",
                    root_cause="Verifier-focused prompt change overfit the active batch.",
                    dag_component="agent_prompt",
                )
                profiles.append(p)
            return profiles

    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        repaired = MASDAG(
            dag_id="test",
            nodes={
                "start": DAGNode(node_id="start", node_type="literal", prompt="repaired"),
                "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
            },
            edges=[DAGEdge(source="start", target="FINAL")],
            metadata={"start": ["start"]},
        )
        return [EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="strengthen verifier instructions",
            actions_taken=["change verifier prompt"],
            metadata={"observed_pattern_id": "prompt_strengthen_verification"},
        )]

    monkeypatch.setattr("optpilot.orchestrator.agenerate_evolve_candidates", fake_agenerate_evolve_candidates)
    monkeypatch.setattr(
        "optpilot.orchestrator.analyze",
        lambda dag, fm_group, traces, profiles, negatives: AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        ),
    )
    monkeypatch.setattr(
        "optpilot.orchestrator.fitness_score",
        lambda traces, profiles, num_agents: sum(float(t.task_score or 0.0) for t in traces) / len(traces),
    )
    monkeypatch.setattr(
        "optpilot.orchestrator.reflect",
        lambda fm_group, dag, evolve_result, before_fm, after_fm, before_pass, after_pass: build_synthetic_insight(
            fm_group=fm_group,
            evolve_result=evolve_result,
            before_fm=before_fm,
            after_fm=after_fm,
            before_pass=before_pass,
            after_pass=after_pass,
            failure_reason="shadow regression",
            lesson="stabilize shadow performance",
        ),
    )
    monkeypatch.setattr("optpilot.orchestrator.SHADOW_EVAL_INTERVAL", 1)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag, negatives_dir=tmp_path / "neg")
    orchestrator.diagnoser = _AsyncDiagnoser()

    await orchestrator.aoptimize(
        tasks=["A:0", "A:1", "B:0", "B:1", "C:0", "C:1", "D:0", "D:1"],
        max_rounds=1,
        eval_tasks_per_round=4,
    )

    saved = json.loads((tmp_path / "neg" / "negatives_B.json").read_text())
    shadow = saved[0]["metadata"]["shadow_gate"]
    assert shadow["candidate_change_description"] == "strengthen verifier instructions"
    assert shadow["observed_pattern_id"] == "prompt_strengthen_verification"
    assert shadow["diagnostics"][0]["root_cause"] == "Verifier-focused prompt change overfit the active batch."


@pytest.mark.anyio
async def test_meta_evolve_triggers_after_three_shadow_rejections(monkeypatch, tmp_path):
    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    batch_scores = [
        0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
    ]
    evolved = []

    class _AsyncRunner:
        benchmark_name_resolver = staticmethod(lambda task: task.split(":", 1)[0])

        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            score = batch_scores.pop(0)
            return [_make_trace(i, score=score) for i in range(len(tasks))]

    class _AsyncDiagnoser:
        async def aclassify_batch(self, traces):
            return [_make_profile(i, fm_active=True) for i in range(len(traces))]

    async def fake_agenerate_evolve_candidates(
        dag, fm_group, analysis, negatives, history,
        recommended_pattern=None, recommended_patterns=None, traces=None, profiles=None,
    ):
        repaired = MASDAG(
            dag_id="test",
            nodes={
                "start": DAGNode(node_id="start", node_type="literal", prompt="repaired"),
                "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
            },
            edges=[DAGEdge(source="start", target="FINAL")],
            metadata={"start": ["start"]},
        )
        return [EvolveResult(
            dag=repaired,
            analysis_text="fixed prompt",
            modified_source="source",
            change_description="fixed prompt",
            actions_taken=["change"],
            metadata={"observed_pattern_id": "prompt_add_constraint"},
        )]

    def fake_analyze(dag, fm_group, traces, profiles, negatives):
        return AnalysisResult(
            fm_id=fm_group, fm_rate=1.0,
            metadata={"dag_components": [], "proposal_traces": [], "failure_signatures": []},
        )

    monkeypatch.setattr("optpilot.orchestrator.agenerate_evolve_candidates", fake_agenerate_evolve_candidates)
    monkeypatch.setattr("optpilot.orchestrator.analyze", fake_analyze)
    monkeypatch.setattr(
        "optpilot.orchestrator.fitness_score",
        lambda traces, profiles, num_agents: sum(float(t.task_score or 0.0) for t in traces) / len(traces),
    )
    monkeypatch.setattr(
        "optpilot.orchestrator.reflect",
        lambda fm_group, dag, evolve_result, before_fm, after_fm, before_pass, after_pass: build_synthetic_insight(
            fm_group=fm_group,
            evolve_result=evolve_result,
            before_fm=before_fm,
            after_fm=after_fm,
            before_pass=before_pass,
            after_pass=after_pass,
            failure_reason="shadow regression",
            lesson="keep shadow accuracy stable",
        ),
    )
    monkeypatch.setattr("optpilot.orchestrator.SHADOW_EVAL_INTERVAL", 1)
    monkeypatch.setattr("optpilot.orchestrator.SHADOW_META_EVOLVE_THRESHOLD", 3)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag, negatives_dir=tmp_path / "neg")
    orchestrator.diagnoser = _AsyncDiagnoser()
    orchestrator.evolver.evolve_catalog = lambda fm_group, negatives: evolved.append((fm_group, len(negatives))) or True

    await orchestrator.aoptimize(
        tasks=["A:0", "A:1", "B:0", "B:1", "C:0", "C:1", "D:0", "D:1"],
        max_rounds=3,
        eval_tasks_per_round=4,
    )

    assert evolved == [("B", 3)]


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
