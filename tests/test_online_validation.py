from __future__ import annotations

from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.library.repair_library import RepairLibrary
from optpilot.models import (
    FMCategory, FMLabel, FMLocalization, FMProfile, MASTrace, RepairAction,
    RepairCandidate, RepairEntry, RepairType,
)
from optpilot.modules.distiller import Distiller
from optpilot.modules.runner import OptPilotRunner
from optpilot.modules.wrap_up import WrapUp
from optpilot.orchestrator import split_proposal_validation_indices


def _profile(trace_id: int, fm_id: str | None, cause: str = "loop repeats") -> FMProfile:
    profile = FMProfile(trace_id=trace_id)
    if fm_id is None:
        return profile
    profile.labels[fm_id] = FMLabel(
        fm_id=fm_id,
        fm_name="Step Repetition",
        category=FMCategory.FC1,
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
    monkeypatch.setattr("optpilot.modules.distiller.call_llm", lambda *args, **kwargs: "use when loops repeat")

    library = RepairLibrary(tmp_path / "online_library.json")
    distiller = Distiller(library)
    candidate = RepairCandidate(
        fm_id="1.3",
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
    before_profiles = [_profile(1, "1.3"), _profile(2, None)]
    after_traces = [_trace(1, 0.5, 2.6), _trace(2, 0.3, 2.8)]
    after_profiles = [_profile(1, None), _profile(2, None)]

    entry = distiller.distill_online(
        "1.3",
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


def test_split_proposal_validation_indices_keeps_holdout_positive() -> None:
    profiles = [
        _profile(0, "1.3"),
        _profile(1, "1.3"),
        _profile(2, None),
        _profile(3, "1.3"),
    ]

    proposal_indices, validation_indices = split_proposal_validation_indices("1.3", profiles) or ([], [])

    assert proposal_indices
    assert validation_indices
    assert set(proposal_indices).isdisjoint(validation_indices)
    assert any("1.3" in profiles[idx].active_fm_ids() for idx in validation_indices)


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


def test_wrap_up_combines_positive_and_negative_hints(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "optpilot.modules.wrap_up.call_llm_json",
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
        fm_id="1.3",
        status="validated",
        success_rate=0.8,
        candidate=RepairCandidate(fm_id="1.3", description="Add verifier"),
        root_cause_pattern="Programmer-reviewer loop repeats without acceptance criteria.",
    )
    negative = RepairEntry(
        fm_id="1.3",
        status="failed",
        candidate=RepairCandidate(fm_id="1.3", description="Just lower the loop limit"),
        root_cause_pattern="Loop repeats because the task was misunderstood.",
    )
    library.add(positive)
    library.add(negative)

    wrapped = WrapUp(library).wrap_fm("1.3", source_mas="AG2")

    assert len(wrapped) == 1
    assert wrapped[0].entry_kind == "wrapped"
    assert wrapped[0].when_not_to_use.startswith("Avoid when")
    assert wrapped[0].counter_entry_ids == [negative.entry_id]
    assert "Do not only reduce max_iterations" in wrapped[0].avoid_actions[0]
    assert library.search("1.3")[0].entry_kind == "wrapped"
