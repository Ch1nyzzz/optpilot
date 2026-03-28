"""Tests for CatalogEvolver and repair_loop direct generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from optpilot.config import (
    META_EVOLVE_FAILURE_THRESHOLD,
    META_EVOLVE_MAX_TOKENS,
    META_EVOLVE_MAX_TURNS,
)
from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.models import AnalysisResult, ReflectInsight
from optpilot.skills.evolution import CatalogEvolver
from optpilot.skills.jacobian import RepairJacobian, RepairOutcome
from optpilot.skills.repair_loop import (
    aevolve,
    build_performance_summary,
    extract_python_code,
)
from optpilot.skills.repair_patterns import (
    FailureSignature,
    PatternCatalog,
    infer_observed_pattern_from_dags,
)


def test_extract_python_code_fenced():
    response = "Here is the fix:\n```python\ndef build_dag():\n    return {}\n```\nDone."
    code = extract_python_code(response)
    assert "def build_dag" in code
    assert "return {}" in code


def test_extract_python_code_fallback():
    response = "def build_dag():\n    return {'dag_id': 'test'}\n"
    code = extract_python_code(response)
    assert "def build_dag" in code


def test_extract_python_code_empty():
    response = "I cannot fix this."
    code = extract_python_code(response)
    assert code == ""


def test_build_performance_summary():
    from optpilot.models import FMLabel, FMProfile, MASTrace

    traces = [
        MASTrace(trace_id=0, mas_name="t", llm_name="m", benchmark_name="b",
                 trajectory="", task_score=1.0),
        MASTrace(trace_id=1, mas_name="t", llm_name="m", benchmark_name="b",
                 trajectory="", task_score=0.0),
    ]
    profiles = [
        FMProfile(trace_id=0),
        FMProfile(trace_id=1, labels={
            "B": FMLabel(fm_id="B", fm_name="Execution Loop", category="B", present=True),
        }),
    ]
    summary = build_performance_summary(traces, profiles)
    assert "50.0%" in summary
    assert "Group B" in summary


def test_catalog_evolver_failure_threshold():
    catalog = PatternCatalog()
    evolver = CatalogEvolver(catalog=catalog)

    for _ in range(META_EVOLVE_FAILURE_THRESHOLD - 1):
        evolver.record_failure("B")
    assert evolver.should_evolve("B") is False

    evolver.record_failure("B")
    assert evolver.should_evolve("B") is True

    evolver.reset_failures("B")
    assert evolver.should_evolve("B") is False


def test_failure_signature_key_is_fm_group_only():
    sig = FailureSignature(fm_group="D", dag_component="edge_missing", agent="Agent_Verifier")
    assert sig.signature_key() == "D"


def test_infer_observed_pattern_from_dags_detects_carry_data_change():
    original = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "Agent_Verifier": DAGNode(node_id="Agent_Verifier", node_type="agent", prompt="verify"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[
            DAGEdge(source="start", target="Agent_Verifier", carry_data=False),
            DAGEdge(source="Agent_Verifier", target="FINAL"),
        ],
        metadata={"start": ["start"]},
    )
    repaired = MASDAG(
        dag_id="test",
        nodes=original.nodes,
        edges=[
            DAGEdge(source="start", target="Agent_Verifier", carry_data=True),
            DAGEdge(source="Agent_Verifier", target="FINAL"),
        ],
        metadata={"start": ["start"]},
    )

    assert infer_observed_pattern_from_dags(original, repaired) == "edge_fix_carry_data"


def test_jacobian_skips_positive_credit_when_observed_unknown(tmp_path):
    catalog = PatternCatalog(store_path=tmp_path / "catalog.json")
    jacobian = RepairJacobian(catalog=catalog, base_dir=tmp_path / "jacobian")

    jacobian.update(RepairOutcome(
        fm_group="B",
        dag_component="",
        agent="",
        assigned_pattern_id="loop_fix_config",
        observed_pattern_id="",
        success=True,
        fm_delta=-0.2,
        pass_delta=0.1,
    ))

    assert jacobian.matrix == {}

    jacobian.update(RepairOutcome(
        fm_group="B",
        dag_component="",
        agent="",
        assigned_pattern_id="loop_fix_config",
        observed_pattern_id="",
        success=False,
        fm_delta=0.0,
        pass_delta=0.0,
    ))

    entry = jacobian.matrix["B"]["loop_fix_config"]
    assert entry.n_applied == 1
    assert entry.n_success == 0


def test_catalog_evolver_calls_llm_and_saves(monkeypatch, tmp_path):
    store_path = tmp_path / "catalog.json"
    catalog = PatternCatalog(store_path=store_path)
    evolver = CatalogEvolver(catalog=catalog)

    negatives = [
        ReflectInsight(
            round_index=1, fm_id="B",
            changes_attempted=["add loop guard"],
            before_fm_rate=1.0, after_fm_rate=1.0,
            before_pass_rate=0.2, after_pass_rate=0.2,
            failure_reason="loop still never exits",
            lesson="inspect executor loop semantics",
        )
    ]

    captured: dict[str, object] = {}

    async def fake_acall_llm_with_tools(*, messages, tools, tool_executor, max_tokens, max_turns):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        captured["max_turns"] = max_turns
        # Add a new pattern
        result = tool_executor("add_pattern", {
            "pattern_id": "loop_add_explicit_exit",
            "name": "Add explicit exit keyword to loop",
            "description": "Add SOLUTION_FOUND keyword to exit edges.",
            "target_components": ["loop_config", "edge_condition"],
        })
        assert "OK" in result
        return [{"role": "assistant", "content": "added new pattern"}]

    monkeypatch.setattr("optpilot.skills.evolution.acall_llm_with_tools", fake_acall_llm_with_tools)
    monkeypatch.setattr("optpilot.skills.evolution._META_EVOLVE_TRACE_DIR", tmp_path / "traces")

    success = asyncio.run(evolver._aevolve_catalog("B", negatives))

    assert success is True
    assert "loop_add_explicit_exit" in catalog
    assert store_path.exists()
    saved = json.loads(store_path.read_text())
    assert "loop_add_explicit_exit" in saved
    assert captured["max_tokens"] == META_EVOLVE_MAX_TOKENS
    assert captured["max_turns"] == META_EVOLVE_MAX_TURNS


@pytest.mark.anyio
async def test_aevolve_direct_generation(monkeypatch, tmp_path):
    """Test that aevolve uses direct code generation and parses the result."""
    fake_code = '''\
def build_dag():
    """Build the MASDAG configuration."""
    nodes = [
        {"id": "start", "type": "literal", "config": {"content": "repaired"}},
        {"id": "Agent_Verifier", "type": "agent", "role": "verify", "config": {"params": {"temperature": 0.2}}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]
    edges = [
        {"from": "start", "to": "Agent_Verifier", "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Verifier", "to": "FINAL", "trigger": True, "condition": "true", "carry_data": True},
    ]
    return {
        "dag_id": "test",
        "nodes": nodes,
        "edges": edges,
        "metadata": {"start": ["start"], "success_nodes": ["FINAL"]},
    }
'''

    async def fake_acall_llm(messages, max_tokens, temperature):
        return f"```python\n{fake_code}\n```\nFixed prompt issue."

    monkeypatch.setattr("optpilot.skills.repair_loop.acall_llm", fake_acall_llm)
    monkeypatch.setattr("optpilot.skills.repair_loop._SKILL_AGENT_TRACE_ROOT", tmp_path / "traces")

    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "Agent_Verifier": DAGNode(node_id="Agent_Verifier", node_type="agent", prompt="verify"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[
            DAGEdge(source="start", target="Agent_Verifier"),
            DAGEdge(source="Agent_Verifier", target="FINAL"),
        ],
        metadata={"start": ["start"]},
    )
    analysis = AnalysisResult(
        fm_id="B", fm_rate=1.0,
        common_agents=["Agent_Verifier"],
        common_steps=["verification"],
        root_cause_clusters=["prompt issue"],
        metadata={"dag_components": ["node_config"]},
    )

    result = await aevolve(dag, "B", analysis, negatives=[], history=[])

    assert result.metadata.get("invalid_evolve_reason") == ""
    assert result.dag.dag_id == "test"
    assert "Agent_Verifier" in result.dag.nodes
    assert result.modified_source != ""


@pytest.mark.anyio
async def test_aevolve_rejects_no_code(monkeypatch, tmp_path):
    """Test that aevolve handles LLM response with no code block."""

    async def fake_acall_llm(messages, max_tokens, temperature):
        return "I cannot make any changes."

    monkeypatch.setattr("optpilot.skills.repair_loop.acall_llm", fake_acall_llm)
    monkeypatch.setattr("optpilot.skills.repair_loop._SKILL_AGENT_TRACE_ROOT", tmp_path / "traces")

    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    analysis = AnalysisResult(
        fm_id="B", fm_rate=1.0,
        common_agents=[], common_steps=[],
        root_cause_clusters=[],
        metadata={"dag_components": []},
    )

    result = await aevolve(dag, "B", analysis, negatives=[], history=[])

    assert "No Python code block" in result.metadata["invalid_evolve_reason"]


@pytest.mark.anyio
async def test_aevolve_rejects_deleted_verifier(monkeypatch, tmp_path):
    """Test that aevolve rejects code that deletes the verification agent."""
    fake_code = '''\
def build_dag():
    nodes = [
        {"id": "start", "type": "literal", "config": {"content": "hello"}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]
    edges = [
        {"from": "start", "to": "FINAL", "trigger": True, "condition": "true", "carry_data": True},
    ]
    return {
        "dag_id": "test",
        "nodes": nodes,
        "edges": edges,
        "metadata": {"start": ["start"], "success_nodes": ["FINAL"]},
    }
'''

    async def fake_acall_llm(messages, max_tokens, temperature):
        return f"```python\n{fake_code}\n```\nRemoved verifier."

    monkeypatch.setattr("optpilot.skills.repair_loop.acall_llm", fake_acall_llm)
    monkeypatch.setattr("optpilot.skills.repair_loop._SKILL_AGENT_TRACE_ROOT", tmp_path / "traces")

    dag = MASDAG(
        dag_id="test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "Agent_Verifier": DAGNode(node_id="Agent_Verifier", node_type="agent", prompt="verify"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[
            DAGEdge(source="start", target="Agent_Verifier"),
            DAGEdge(source="Agent_Verifier", target="FINAL"),
        ],
        metadata={"start": ["start"]},
    )
    analysis = AnalysisResult(
        fm_id="F", fm_rate=0.5,
        common_agents=[], common_steps=[],
        root_cause_clusters=[],
        metadata={"dag_components": []},
    )

    result = await aevolve(dag, "F", analysis, negatives=[], history=[])

    assert "Verification agent deleted" in result.metadata["invalid_evolve_reason"]
