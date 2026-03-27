from __future__ import annotations

import json
from pathlib import Path

import pytest

from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.models import EvolveResult, FMLabel, FMProfile, ReflectInsight, SkillBudget, SkillResult
from optpilot.orchestrator import Orchestrator
from optpilot.skills.base import BaseSkill


class _DummyRunner:
    def run_batch(self, tasks, dag=None):
        return [{"dag": dag, "tasks": tuple(tasks)}]


class _DummyDiagnoser:
    def diagnose_batch(self, traces):
        return traces


class _MetricSkill(BaseSkill):
    FM_GROUP = "B"

    def __init__(self, fm_values=None, pass_values=None):
        self.fm_values = list(fm_values or [])
        self.pass_values = list(pass_values or [])

    def analyze(self, dag, traces, profiles, negatives):
        return object()

    def evolve(self, dag, analysis, negatives, history):
        return EvolveResult(
            dag=dag,
            analysis_text="analysis",
            modified_yaml="yaml",
            change_description="change",
        )

    def reflect(
        self,
        original_dag,
        evolve_history,
        before_val_traces,
        before_val_profiles,
        after_val_traces,
        after_val_profiles,
    ):
        return ReflectInsight(
            round_index=0,
            fm_id=self.FM_GROUP,
            changes_attempted=["change"],
            before_fm_rate=1.0,
            after_fm_rate=1.0,
            before_pass_rate=0.0,
            after_pass_rate=0.0,
            failure_reason="still failing",
            lesson="try a verifier",
        )

    def _fm_rate(self, profiles):
        if self.fm_values:
            return self.fm_values.pop(0)
        return 1.0

    def _pass_rate(self, traces):
        if self.pass_values:
            return self.pass_values.pop(0)
        return 0.0


class _SkillUnderTest(BaseSkill):
    FM_GROUP = "B"
    MAX_INNER_ITERS = 1
    MAX_OUTER_ROUNDS = 1

    def __init__(self):
        self.analyze_negatives = []
        self.evolve_negatives = []

    def analyze(self, dag, traces, profiles, negatives):
        self.analyze_negatives = list(negatives)
        return object()

    def evolve(self, dag, analysis, negatives, history):
        self.evolve_negatives = list(negatives)
        return EvolveResult(
            dag="repaired-dag",
            analysis_text="analysis",
            modified_yaml="yaml",
            change_description="change",
        )

    def judge(self, before_profiles, after_profiles, before_traces, after_traces):
        return False

    def reflect(
        self,
        original_dag,
        evolve_history,
        before_val_traces,
        before_val_profiles,
        after_val_traces,
        after_val_profiles,
    ):
        return ReflectInsight(
            round_index=0,
            fm_id=self.FM_GROUP,
            changes_attempted=["change"],
            before_fm_rate=1.0,
            after_fm_rate=1.0,
            before_pass_rate=0.0,
            after_pass_rate=0.0,
            failure_reason="still failing",
            lesson="try a verifier",
        )

    def _fm_rate(self, profiles):
        return 1.0

    def _pass_rate(self, traces):
        return 0.0


def test_base_skill_uses_prior_negatives_without_rewriting_them():
    seed = ReflectInsight(
        round_index=7,
        fm_id="B",
        changes_attempted=["old change"],
        before_fm_rate=1.0,
        after_fm_rate=0.8,
        before_pass_rate=0.2,
        after_pass_rate=0.2,
        failure_reason="old failure",
        lesson="old lesson",
    )
    skill = _SkillUnderTest()

    result = skill.run(
        original_dag="original-dag",
        proposal_traces=[{"trace": 1}],
        proposal_profiles=[{"profile": 1}],
        proposal_tasks=["proposal-task"],
        validation_tasks=["validation-task"],
        runner=_DummyRunner(),
        diagnoser=_DummyDiagnoser(),
        budget=SkillBudget(max_llm_calls=10, max_batch_runs=10),
        prior_negatives=[seed],
    )

    assert skill.analyze_negatives == [seed]
    assert skill.evolve_negatives == [seed]
    assert result.outer_rounds == 1
    assert len(result.negatives) == 1
    assert result.negatives[0].lesson == "try a verifier"
    assert result.negatives[0].round_index == 8
    assert result.metadata["prior_negative_count"] == 1


def test_orchestrator_passes_prior_negatives_to_skill(monkeypatch):
    seed = ReflectInsight(
        round_index=1,
        fm_id="B",
        changes_attempted=[],
        before_fm_rate=1.0,
        after_fm_rate=1.0,
        before_pass_rate=0.0,
        after_pass_rate=0.0,
        failure_reason="seed",
        lesson="seed lesson",
    )
    captured = {}

    class _InjectedSkill:
        def run(self, **kwargs):
            captured["prior_negatives"] = kwargs["prior_negatives"]
            return SkillResult(success=False, fm_id="B")

    monkeypatch.setattr("optpilot.orchestrator.get_skill", lambda fm_id: _InjectedSkill())

    orchestrator = Orchestrator(runner=_DummyRunner(), dag="dag", parallel=False)
    fm_id, result = orchestrator._run_skill(
        job={
            "fm_id": "B",
            "proposal_idx": [0],
            "validation_idx": [1],
            "prior_negatives": [seed],
        },
        dag="dag",
        tasks=["proposal", "validation"],
        traces=[{"trace": 1}, {"trace": 2}],
        profiles=[{"profile": 1}, {"profile": 2}],
        budget=SkillBudget(),
    )

    assert fm_id == "B"
    assert result.success is False
    assert captured["prior_negatives"] == [seed]


def test_judge_accepts_flat_pass_if_fm_decreases():
    skill = _MetricSkill(fm_values=[1.0, 0.0], pass_values=[0.0, 0.0])
    assert skill.judge([object()], [object()], [object()], [object()]) is True


def test_run_rejects_noop_success_and_records_negative():
    dag = MASDAG(
        dag_id="noop",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    class _NoOpSkill(_MetricSkill):
        MAX_INNER_ITERS = 1
        MAX_OUTER_ROUNDS = 1

        def evolve(self, dag, analysis, negatives, history):
            return EvolveResult(
                dag=dag,
                analysis_text="analysis",
                modified_yaml=self.read_yaml(dag),
                change_description="",
            )

    skill = _NoOpSkill(fm_values=[1.0, 0.0, 1.0], pass_values=[0.0])
    result = skill.run(
        original_dag=dag,
        proposal_traces=[{"trace": 1}],
        proposal_profiles=[{"profile": 1}],
        proposal_tasks=["proposal-task"],
        validation_tasks=["validation-task"],
        runner=_DummyRunner(),
        diagnoser=_DummyDiagnoser(),
        budget=SkillBudget(max_llm_calls=10, max_batch_runs=10),
    )

    assert result.success is False
    assert len(result.negatives) == 1
    assert "No concrete DAG change" in result.negatives[0].failure_reason


def test_run_records_budget_exhaustion_snapshot_before_validation():
    dag = MASDAG(
        dag_id="budget",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    repaired = MASDAG(
        dag_id="budget",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="changed"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    class _BudgetSkill(_MetricSkill):
        MAX_INNER_ITERS = 1
        MAX_OUTER_ROUNDS = 1

        def evolve(self, dag, analysis, negatives, history):
            return EvolveResult(
                dag=repaired,
                analysis_text="analysis",
                modified_yaml=self.read_yaml(repaired),
                change_description="change",
                actions_taken=["change"],
            )

    skill = _BudgetSkill(fm_values=[1.0, 0.8, 1.0, 0.8], pass_values=[0.0, 0.2])
    result = skill.run(
        original_dag=dag,
        proposal_traces=[{"trace": 1}],
        proposal_profiles=[{"profile": 1}],
        proposal_tasks=["proposal-task"],
        validation_tasks=["validation-task"],
        runner=_DummyRunner(),
        diagnoser=_DummyDiagnoser(),
        budget=SkillBudget(max_llm_calls=10, max_batch_runs=2),
    )

    assert result.success is False
    assert len(result.negatives) == 1
    assert "Budget exhausted before holdout validation" in result.negatives[0].failure_reason


@pytest.mark.anyio
async def test_aoptimize_persists_dag_versions(monkeypatch, tmp_path):
    dag = MASDAG(
        dag_id="test_dag",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    repaired = MASDAG(
        dag_id="test_dag_repaired",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="repaired"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )

    class _AsyncRunner:
        async def arun_batch(self, tasks, dag=None, output_base=None, max_concurrency=256):
            return [{"trace": i, "dag": dag} for i, _ in enumerate(tasks)]

    class _AsyncDiagnoser:
        async def adiagnose_batch(self, traces):
            class _Profile:
                def active_fm_ids(self):
                    return ["B"]

            return [_Profile() for _ in traces]

    async def fake_arun_skill(self, job, dag, tasks, traces, profiles, budget, concurrency=256):
        return "B", SkillResult(
            success=True,
            fm_id="B",
            dag=repaired,
            final_fm_rate=0.5,
            final_pass_rate=0.5,
        )

    monkeypatch.setattr("optpilot.orchestrator.rank_fm_groups", lambda profiles, target_fm=None: [("B", 2)])
    monkeypatch.setattr("optpilot.orchestrator.split_proposal_validation_indices", lambda fm_id, profiles: ([0], [1]))
    monkeypatch.setattr(Orchestrator, "_arun_skill", fake_arun_skill)

    orchestrator = Orchestrator(runner=_AsyncRunner(), dag=dag, parallel=False)
    orchestrator.diagnoser = _AsyncDiagnoser()

    summary = await orchestrator.aoptimize(
        tasks=["task-1", "task-2"],
        max_rounds=1,
        dag_output_base=tmp_path / "dag_versions",
    )

    saved_paths = [Path(entry["path"]) for entry in summary["dag_versions"]]
    assert tmp_path.joinpath("dag_versions", "initial.yaml").exists()
    assert tmp_path.joinpath("dag_versions", "round_1", "start.yaml").exists()
    assert tmp_path.joinpath("dag_versions", "round_1", "skill_B_success.yaml").exists()
    assert tmp_path.joinpath("dag_versions", "round_1", "final.yaml").exists()
    assert tmp_path.joinpath("dag_versions", "final.yaml").exists()
    assert all(path.exists() for path in saved_paths)


@pytest.mark.anyio
async def test_aoptimize_from_traces_reuses_persisted_traces_and_saves_diagnosis(monkeypatch, tmp_path):
    dag = MASDAG(
        dag_id="warm_start_dag",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="begin"),
            "FINAL": DAGNode(node_id="FINAL", node_type="passthrough"),
        },
        edges=[DAGEdge(source="start", target="FINAL")],
        metadata={"start": ["start"]},
    )
    trace_root = tmp_path / "persisted_train"
    for idx in range(2):
        task_dir = trace_root / f"task_{idx}"
        task_dir.mkdir(parents=True)
        (task_dir / "trace.txt").write_text(f"trace-{idx}", encoding="utf-8")
        (task_dir / "trace.json").write_text(json.dumps({
            "benchmark_name": "Bench",
            "task_key": f"task-{idx}",
            "task_prompt": f"prompt-{idx}",
            "task_score": 0.0,
            "task_success": False,
            "latency_s": 1.0,
        }), encoding="utf-8")

    class _WarmRunner:
        model = "test-model"

        def _resolve_benchmark_name(self, task_prompt):
            return "Bench"

    class _AsyncDiagnoser:
        async def adiagnose_batch(self, traces):
            profiles = []
            for trace in traces:
                profile = FMProfile(trace_id=trace.trace_id)
                profile.labels["B"] = FMLabel(
                    fm_id="B",
                    fm_name="Execution Loop / Stuck",
                    category="B",
                    present=True,
                )
                profiles.append(profile)
            return profiles

    async def fake_arun_skill(self, job, dag, tasks, traces, profiles, budget, concurrency=256):
        return "B", SkillResult(
            success=False,
            fm_id="B",
            negatives=[],
        )

    monkeypatch.setattr(Orchestrator, "_arun_skill", fake_arun_skill)

    orchestrator = Orchestrator(runner=_WarmRunner(), dag=dag, parallel=False)
    orchestrator.diagnoser = _AsyncDiagnoser()

    summary = await orchestrator.aoptimize_from_traces(
        tasks=["prompt-0", "prompt-1"],
        trace_base=trace_root,
        trace_output_base=tmp_path / "artifacts" / "optimization",
        dag_output_base=tmp_path / "artifacts" / "dag_versions",
    )

    diagnose_dir = tmp_path / "artifacts" / "optimization" / "round_1" / "diagnose"
    assert summary["trace_source"] == str(trace_root)
    assert diagnose_dir.joinpath("profiles.json").exists()
    assert diagnose_dir.joinpath("summary.json").exists()
    assert diagnose_dir.joinpath("fm_groups", "B.json").exists()
    assert diagnose_dir.joinpath("skill_jobs", "B.json").exists()
    assert (tmp_path / "artifacts" / "optimization" / "round_1" / "reused_train_manifest.json").exists()
