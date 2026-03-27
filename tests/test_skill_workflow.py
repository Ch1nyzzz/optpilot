from __future__ import annotations

from optpilot.models import EvolveResult, ReflectInsight, SkillBudget, SkillResult
from optpilot.orchestrator import Orchestrator
from optpilot.skills.base import BaseSkill


class _DummyRunner:
    def run_batch(self, tasks, dag=None):
        return [{"dag": dag, "tasks": tuple(tasks)}]


class _DummyDiagnoser:
    def diagnose_batch(self, traces):
        return traces


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
