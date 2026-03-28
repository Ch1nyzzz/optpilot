from __future__ import annotations

import json

from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.library.repair_library import RepairLibrary
from optpilot.models import (
    FMLabel, FMLocalization, FMProfile, MASTrace, RepairAction,
    RepairCandidate, RepairEntry, RepairType,
)
from optpilot.modules._legacy.distiller import Distiller
from optpilot.modules.runner import OptPilotRunner
from optpilot.modules._legacy.wrap_up import WrapUp


def _profile(trace_id: int, fm_id: str | None, cause: str = "loop repeats") -> FMProfile:
    profile = FMProfile(trace_id=trace_id)
    if fm_id is None:
        return profile
    profile.labels[fm_id] = FMLabel(
        fm_id=fm_id,
        fm_name="Execution Loop / Stuck",
        category=fm_id,
        present=True,
    )
    profile.localization[fm_id] = FMLocalization(
        agent="Programmer",
        step="review",
        context="The same review step keeps repeating.",
        root_cause=cause,
    )
    return profile


def _trace(trace_id: int, score: float, latency_s: float = 1.0) -> MASTrace:
    return MASTrace(
        trace_id=trace_id,
        mas_name="AG2",
        llm_name="MiniMaxAI/MiniMax-M2.5",
        benchmark_name="ProgramDev",
        trajectory="trace",
        task_key=f"task-{trace_id}",
        task_score=score,
        task_success=score > 0,
        latency_s=latency_s,
    )


def test_distill_online_requires_pass_improvement(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("optpilot.modules._legacy.distiller.call_llm", lambda *args, **kwargs: "use when loops repeat")

    library = RepairLibrary(tmp_path / "online_library.json")
    distiller = Distiller(library)
    candidate = RepairCandidate(
        fm_id="B",
        description="Add a verifier before loop continuation.",
        actions=[
            RepairAction(
                repair_type=RepairType.NODE_ADD,
                target="verifier",
                description="Add verifier",
                rationale="Break repetition",
            )
        ],
    )

    before_traces = [_trace(1, 0.8, 2.0), _trace(2, 0.4, 2.2)]
    before_profiles = [_profile(1, "B"), _profile(2, None)]
    after_traces = [_trace(1, 0.5, 2.6), _trace(2, 0.3, 2.8)]
    after_profiles = [_profile(1, None), _profile(2, None)]

    entry = distiller.distill_online(
        "B",
        candidate,
        before_traces,
        before_profiles,
        after_traces,
        after_profiles,
    )

    assert entry.status == "failed"
    assert entry.validation_metrics["fm_fixed"] is True
    assert entry.validation_metrics["pass_improved"] is False
    assert entry.validation_metrics["before_pass_rate"] == 1.0
    assert entry.validation_metrics["after_pass_rate"] == 1.0



def test_runner_uses_success_proxy_score() -> None:
    dag = MASDAG(
        dag_id="score_proxy",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    runner = OptPilotRunner(dag=dag, model="test-model")

    trace = runner.run_task("solve task")

    assert trace.task_success is True
    assert trace.task_score == 1.0
    assert trace.latency_s is not None


def test_runner_persists_trace_path_when_output_dir_is_provided(tmp_path) -> None:
    dag = MASDAG(
        dag_id="trace_persist",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    runner = OptPilotRunner(dag=dag, model="test-model")

    trace = runner.run_task("solve task", output_dir=tmp_path / "task_0")

    assert trace.trace_path
    assert trace.trace_path.endswith("trace.txt")
    assert (tmp_path / "task_0" / "trace.txt").exists()
    metadata = json.loads((tmp_path / "task_0" / "trace.json").read_text(encoding="utf-8"))
    assert metadata["task_key"] == "solve task"
    assert metadata["benchmark_name"] == "MathChat"


def test_wrap_up_combines_positive_and_negative_hints(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "optpilot.modules._legacy.wrap_up.call_llm_json",
        lambda *args, **kwargs: {
            "skills": [{
                "when_to_use": "Use when programmer-reviewer loops repeat without a verifier.",
                "when_not_to_use": "Avoid when failures come from task misunderstanding rather than missing verification.",
                "recommended_actions": ["Add a verifier node before loop continuation."],
                "avoid_actions": ["Do not only reduce max_iterations without adding an exit check."],
            }]
        },
    )

    library = RepairLibrary(tmp_path / "wrap_library.json")
    positive = RepairEntry(
        fm_id="B",
        status="validated",
        success_rate=0.8,
        candidate=RepairCandidate(fm_id="B", description="Add verifier"),
        root_cause_pattern="Programmer-reviewer loop repeats without acceptance criteria.",
    )
    negative = RepairEntry(
        fm_id="B",
        status="failed",
        candidate=RepairCandidate(fm_id="B", description="Just lower the loop limit"),
        root_cause_pattern="Loop repeats because the task was misunderstood.",
    )
    library.add(positive)
    library.add(negative)

    wrapped = WrapUp(library).wrap_fm("B", source_mas="AG2")

    assert len(wrapped) == 1
    assert wrapped[0].entry_kind == "wrapped"
    assert wrapped[0].when_not_to_use.startswith("Avoid when")
    assert wrapped[0].counter_entry_ids == [negative.entry_id]
    assert "Do not only reduce max_iterations" in wrapped[0].avoid_actions[0]
    assert library.search("B")[0].entry_kind == "wrapped"
