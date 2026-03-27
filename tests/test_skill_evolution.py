from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from optpilot.config import (
    LIBRARY_DIR,
    META_EVOLVE_FAILURE_THRESHOLD,
    META_EVOLVE_MAX_TOKENS,
    META_EVOLVE_MAX_TURNS,
    NEGATIVES_DIR,
    PROJECT_ROOT,
)
from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.models import AnalysisResult, ReflectInsight
from optpilot.skills.base import GenericSkill, _build_skill_context_index
from optpilot.skills.evolution import SkillEvolver


def test_skill_context_index_exposes_repo_and_experience_paths():
    analysis = AnalysisResult(
        fm_id="B",
        fm_rate=1.0,
        common_agents=["Agent_Verifier"],
        common_steps=["verification"],
        root_cause_clusters=["missing exit condition"],
        metadata={"dag_components": ["loop_config"]},
    )

    text = _build_skill_context_index(
        fm_group="B",
        fm_name="Execution Loop / Stuck",
        fm_description="The system loops without making progress.",
        analysis=analysis,
        trace_file_lines=["- trace_0.txt: runtime trace"],
    )

    assert "agent_context.md" in text
    assert str(PROJECT_ROOT / "memory_bank/architecture.md") in text
    assert str(PROJECT_ROOT / "src/optpilot/dag/executor.py") in text
    assert str(NEGATIVES_DIR / "negatives_B.json") in text
    assert str(LIBRARY_DIR / "subskills" / "B") in text


def test_meta_evolver_prepares_context_files_and_budget(monkeypatch, tmp_path):
    negatives = [
        ReflectInsight(
            round_index=3,
            fm_id="B",
            changes_attempted=["add loop guard"],
            before_fm_rate=1.0,
            after_fm_rate=1.0,
            before_pass_rate=0.2,
            after_pass_rate=0.2,
            failure_reason="loop still never exits",
            lesson="inspect executor loop semantics",
        )
    ]
    captured: dict[str, object] = {}
    skill_trace_dir = tmp_path / "skill_agent_traces" / "B"
    skill_trace_dir.mkdir(parents=True)
    recent_skill_trace = skill_trace_dir / "tool_trace_seed.json"
    recent_skill_trace.write_text('{"messages": []}', encoding="utf-8")

    async def fake_acall_llm_with_tools(*, messages, tools, tool_executor, max_tokens, max_turns):
        captured["messages"] = messages
        captured["tools"] = tools
        captured["max_tokens"] = max_tokens
        captured["max_turns"] = max_turns
        captured["meta_context"] = tool_executor("bash", {"command": "cat meta_context.md"})
        captured["failure_summary"] = tool_executor("bash", {"command": "cat failure_summary.md"})
        captured["failures_json"] = tool_executor("bash", {"command": "cat failures.json"})
        tool_executor("search_and_replace", {
            "old_str": "class DemoSkill:\n    pass\n",
            "new_str": "class DemoSkill:\n    # evolved\n    pass\n",
        })
        return [{"role": "assistant", "content": "updated meta skill"}]

    monkeypatch.setattr("optpilot.skills.evolution.acall_llm_with_tools", fake_acall_llm_with_tools)
    monkeypatch.setattr("optpilot.skills.evolution._SKILL_AGENT_TRACE_DIR", tmp_path / "skill_agent_traces")
    monkeypatch.setattr("optpilot.skills.evolution._META_EVOLVE_TRACE_DIR", tmp_path / "meta_evolve_traces")

    evolver = SkillEvolver(evolved_dir=tmp_path)
    out_path = asyncio.run(
        evolver._aevolve_skill(
            fm_group="B",
            source="class DemoSkill:\n    pass\n",
            negatives=negatives,
            source_path=Path("/tmp/skill_b.py"),
        )
    )

    assert out_path is not None
    assert out_path.exists()
    assert "# evolved" in out_path.read_text(encoding="utf-8")
    assert captured["max_tokens"] == META_EVOLVE_MAX_TOKENS
    assert captured["max_turns"] == META_EVOLVE_MAX_TURNS
    assert "meta_context.md" in captured["messages"][1]["content"]
    assert "failure_summary.md" in captured["messages"][1]["content"]
    assert str(PROJECT_ROOT / "memory_bank/architecture.md") in captured["meta_context"]
    assert str(tmp_path) in captured["meta_context"]
    assert str(skill_trace_dir) in captured["meta_context"]
    assert str(recent_skill_trace) in captured["meta_context"]
    assert "inspect executor loop semantics" in captured["failure_summary"]
    assert '"fm_id": "B"' in captured["failures_json"]
    assert (tmp_path / "meta_evolve_traces" / "B").exists()


def test_meta_evolver_uses_shared_failure_threshold():
    evolver = SkillEvolver()
    for _ in range(META_EVOLVE_FAILURE_THRESHOLD - 1):
        evolver.record_failure("B")

    assert evolver.should_evolve("B") is False

    evolver.record_failure("B")
    assert evolver.should_evolve("B") is True


@pytest.mark.anyio
async def test_generic_skill_persists_tool_trace(monkeypatch, tmp_path):
    class _TraceSkill(GenericSkill):
        FM_GROUP = "B"

    async def fake_acall_llm_with_tools(*, messages, tools, tool_executor, max_tokens, max_turns):
        tool_executor("bash", {"command": "cat \"$DAG_FILE\""})
        tool_executor("search_and_replace", {
            "old_str": "prompt: begin",
            "new_str": "prompt: repaired",
        })
        return [
            *messages,
            {"role": "assistant", "content": "updated the DAG prompt"},
        ]

    monkeypatch.setattr("optpilot.skills.base.acall_llm_with_tools", fake_acall_llm_with_tools)
    monkeypatch.setattr("optpilot.skills.base._SKILL_AGENT_TRACE_ROOT", tmp_path / "skill_agent_traces")

    dag = MASDAG(
        dag_id="trace",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    analysis = AnalysisResult(
        fm_id="B",
        fm_rate=1.0,
        common_agents=["Agent_Verifier"],
        common_steps=["verification"],
        root_cause_clusters=["prompt issue"],
        metadata={"proposal_traces": [], "dag_components": ["node_config"]},
    )

    result = await _TraceSkill().aevolve(dag, analysis, negatives=[], history=[])

    tool_trace_path = Path(result.metadata["tool_trace_path"])
    payload = json.loads(tool_trace_path.read_text(encoding="utf-8"))

    assert tool_trace_path.exists()
    assert tool_trace_path.parent == tmp_path / "skill_agent_traces" / "B"
    assert payload["fm_group"] == "B"
    assert payload["messages"][-1]["content"] == "updated the DAG prompt"
    assert payload["change_records"][0]["source"] == "search_and_replace"
