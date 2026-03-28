"""BaseSkill — abstract base for FM-group-specific repair workflows.

Each Skill is a complete repair agent: analyze → evolve (inner loop) →
validate → reflect → retry.  The `run()` template method implements the
outer reflection loop and inner convergence loop; subclasses only override
the four hooks: analyze, evolve, judge, reflect.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import unified_diff
from statistics import mean
from typing import TYPE_CHECKING, Any

from optpilot.config import (
    LIBRARY_DIR,
    META_EVOLVE_FAILURE_THRESHOLD,
    NEGATIVES_DIR,
    PROJECT_ROOT,
    SKILL_EVOLVE_MAX_TOKENS,
    SKILL_EVOLVE_MAX_TURNS,
)
from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
from optpilot.llm import acall_llm, acall_llm_json, acall_llm_with_tools, call_llm, call_llm_json
from optpilot.models import (
    AnalysisResult,
    EvolveResult,
    ReflectInsight,
    SkillBudget,
    SkillResult,
)
from optpilot.repair_utils import extract_fenced_block, extract_preface, summarize_faults
from optpilot.skills.subskills import SubSkillStore
from optpilot.skills.tools import TOOL_SCHEMAS, ChangeRecord, ToolContext, dag_to_python, execute_tool

if TYPE_CHECKING:
    from optpilot.dag.core import MASDAG
    from optpilot.models import FMProfile, MASTrace
    from optpilot.modules.base_runner import MASRunner
    from optpilot.modules.diagnoser import Diagnoser


class BaseSkill(ABC):
    """Abstract base for FM-group-specific repair workflows."""

    FM_GROUP: str = ""
    MAX_INNER_ITERS: int = 10
    CONVERGENCE_THRESHOLD: float = 0.2
    NO_IMPROVE_PATIENCE: int = 3
    MAX_OUTER_ROUNDS: int = 3
    META_EVOLVE_THRESHOLD: int = META_EVOLVE_FAILURE_THRESHOLD

    # ------------------------------------------------------------------ #
    #  Template Method — do NOT override in subclasses                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        original_dag: MASDAG,
        proposal_traces: list[MASTrace],
        proposal_profiles: list[FMProfile],
        proposal_tasks: list[str],
        validation_tasks: list[str],
        runner: MASRunner,
        diagnoser: Diagnoser,
        budget: SkillBudget | None = None,
        prior_negatives: list[ReflectInsight] | None = None,
    ) -> SkillResult:
        """Full repair workflow: outer reflection loop + inner convergence loop."""

        budget = budget or SkillBudget()
        budget.start_time = time.time()
        prior_negatives = prior_negatives or []
        prior_round_index = max((neg.round_index for neg in prior_negatives), default=0)
        negatives: list[ReflectInsight] = list(prior_negatives)
        new_negatives: list[ReflectInsight] = []
        fm_info = GROUP_DEFINITIONS.get(self.FM_GROUP, {})
        fm_name = fm_info.get("name", self.FM_GROUP)

        print(f"  [Skill-{self.FM_GROUP}] Starting repair for {fm_name}")

        # Cache validation baseline (original DAG is constant).
        print(f"  [Skill-{self.FM_GROUP}] Running validation baseline ({len(validation_tasks)} tasks)...")
        before_val_traces = runner.run_batch(validation_tasks, dag=original_dag)
        before_val_profiles = diagnoser.diagnose_batch(before_val_traces)
        budget.used_batch_runs += 1

        total_inner = 0
        completed_outer_rounds = 0

        for outer_round in range(self.MAX_OUTER_ROUNDS):
            if not budget.check():
                print(f"  [Skill-{self.FM_GROUP}] Budget exhausted, aborting.")
                break

            current_dag = original_dag  # reset each outer round
            print(f"  [Skill-{self.FM_GROUP}] Outer round {outer_round + 1}/{self.MAX_OUTER_ROUNDS}")

            # --- Analyze ---
            analysis = self.analyze(current_dag, proposal_traces, proposal_profiles, negatives)
            budget.used_llm_calls += 1

            # --- Inner loop: evolve until convergence ---
            evolve_history: list[EvolveResult] = []
            prev_fm_rate = self._fm_rate(proposal_profiles)
            no_improve = 0
            latest_eval_traces: list[MASTrace] = []
            latest_eval_profiles: list[FMProfile] = []

            for inner_iter in range(self.MAX_INNER_ITERS):
                if not budget.check():
                    break

                print(f"    Inner iter {inner_iter + 1}/{self.MAX_INNER_ITERS} (fm_rate={prev_fm_rate:.2f})")

                result = self.evolve(current_dag, analysis, negatives, evolve_history)
                evolve_history.append(result)
                current_dag = result.dag
                budget.used_llm_calls += 1

                invalid_reason = self._invalid_evolve_result_reason(result)
                if invalid_reason:
                    print(f"    Invalid evolve result: {invalid_reason[:120]}")
                    break

                new_traces = runner.run_batch(proposal_tasks, dag=current_dag)
                new_profiles = diagnoser.diagnose_batch(new_traces)
                latest_eval_traces = new_traces
                latest_eval_profiles = new_profiles
                budget.used_batch_runs += 1

                fm_rate = self._fm_rate(new_profiles)
                print(f"    → fm_rate={fm_rate:.2f}, change: {result.change_description[:80]}")

                if fm_rate < self.CONVERGENCE_THRESHOLD:
                    print(f"    Converged (fm_rate < {self.CONVERGENCE_THRESHOLD})")
                    break

                if fm_rate >= prev_fm_rate:
                    no_improve += 1
                    if no_improve >= self.NO_IMPROVE_PATIENCE:
                        print(f"    No improvement for {no_improve} iters, stopping inner loop.")
                        break
                else:
                    no_improve = 0
                prev_fm_rate = fm_rate

            total_inner += len(evolve_history)

            before_fm = self._fm_rate(before_val_profiles)
            before_pass = self._pass_rate(before_val_traces)
            invalid_evolve_reason = ""
            for er in reversed(evolve_history):
                invalid_evolve_reason = self._invalid_evolve_result_reason(er)
                if invalid_evolve_reason:
                    break

            if invalid_evolve_reason:
                print(f"  [Skill-{self.FM_GROUP}] Invalid evolve termination, recording failure snapshot.")
                insight = self._build_synthetic_insight(
                    evolve_history=evolve_history,
                    before_fm=before_fm,
                    after_fm=before_fm,
                    before_pass=before_pass,
                    after_pass=before_pass,
                    failure_reason=invalid_evolve_reason,
                    lesson=(
                        "A repair attempt is only valid if the tool loop ends with a non-empty "
                        "assistant summary after edits and validation. Treat truncated or empty "
                        "assistant endings as failed attempts."
                    ),
                )
                insight.round_index = prior_round_index + len(new_negatives) + 1
                negatives.append(insight)
                new_negatives.append(insight)
                completed_outer_rounds = outer_round + 1
                print(f"    Lesson: {insight.lesson[:120]}")
                continue

            material_change = self._has_material_change(original_dag, current_dag, evolve_history)

            if not material_change:
                print(f"  [Skill-{self.FM_GROUP}] No material DAG changes detected, skipping validation.")
                insight = self._build_synthetic_insight(
                    evolve_history=evolve_history,
                    before_fm=before_fm,
                    after_fm=before_fm,
                    before_pass=before_pass,
                    after_pass=before_pass,
                    failure_reason=(
                        "No concrete DAG change was applied. The evolve loop produced no material "
                        "code diff, so this round cannot count as a repair."
                    ),
                    lesson=(
                        "A valid repair must leave a real DAG diff. Require at least one concrete "
                        "search_and_replace or bash-written code change before evaluating success."
                    ),
                )
                insight.round_index = prior_round_index + len(new_negatives) + 1
                negatives.append(insight)
                new_negatives.append(insight)
                completed_outer_rounds = outer_round + 1
                print(f"    Lesson: {insight.lesson[:120]}")
                continue

            # --- Validate: same tasks, before vs after ---
            if not budget.check():
                after_fm = self._fm_rate(latest_eval_profiles) if latest_eval_profiles else before_fm
                after_pass = self._pass_rate(latest_eval_traces) if latest_eval_traces else before_pass
                insight = self._build_synthetic_insight(
                    evolve_history=evolve_history,
                    before_fm=before_fm,
                    after_fm=after_fm,
                    before_pass=before_pass,
                    after_pass=after_pass,
                    failure_reason=(
                        "Budget exhausted before holdout validation completed. This round ended "
                        "without a validated repair verdict."
                    ),
                    lesson=(
                        "Persist this failed attempt and either reduce per-round work or increase "
                        "budget so the skill can reach holdout validation."
                    ),
                )
                insight.round_index = prior_round_index + len(new_negatives) + 1
                negatives.append(insight)
                new_negatives.append(insight)
                completed_outer_rounds = outer_round + 1
                print(f"    Lesson: {insight.lesson[:120]}")
                break

            print(f"  [Skill-{self.FM_GROUP}] Validating on holdout ({len(validation_tasks)} tasks)...")
            after_val_traces = runner.run_batch(validation_tasks, dag=current_dag)
            after_val_profiles = diagnoser.diagnose_batch(after_val_traces)
            budget.used_batch_runs += 1

            after_fm = self._fm_rate(after_val_profiles)
            after_pass = self._pass_rate(after_val_traces)
            print(
                f"    Validation: FM {before_fm:.2f}→{after_fm:.2f}, "
                f"pass {before_pass:.3f}→{after_pass:.3f}"
            )

            if self.judge(before_val_profiles, after_val_profiles,
                          before_val_traces, after_val_traces):
                print(f"  [Skill-{self.FM_GROUP}] Repair validated!")
                return SkillResult(
                    success=True,
                    fm_id=self.FM_GROUP,
                    dag=current_dag,
                    inner_iterations=total_inner,
                    outer_rounds=outer_round + 1,
                    final_fm_rate=after_fm,
                    final_pass_rate=after_pass,
                    negatives=new_negatives,
                    budget_used=budget,
                    metadata={"prior_negative_count": len(prior_negatives)},
                )

            # --- Reflect ---
            print(f"  [Skill-{self.FM_GROUP}] Validation failed, reflecting...")
            insight = self.reflect(
                original_dag, evolve_history,
                before_val_traces, before_val_profiles,
                after_val_traces, after_val_profiles,
            )
            budget.used_llm_calls += 1
            insight.round_index = prior_round_index + len(new_negatives) + 1
            negatives.append(insight)
            new_negatives.append(insight)
            completed_outer_rounds = outer_round + 1
            print(f"    Lesson: {insight.lesson[:120]}")

        print(f"  [Skill-{self.FM_GROUP}] Repair failed after {completed_outer_rounds} rounds.")
        return SkillResult(
            success=False,
            fm_id=self.FM_GROUP,
            outer_rounds=completed_outer_rounds,
            inner_iterations=total_inner,
            negatives=new_negatives,
            budget_used=budget,
            metadata={"prior_negative_count": len(prior_negatives)},
        )

    async def arun(
        self,
        original_dag: MASDAG,
        proposal_traces: list[MASTrace],
        proposal_profiles: list[FMProfile],
        proposal_tasks: list[str],
        validation_tasks: list[str],
        runner: MASRunner,
        diagnoser: Diagnoser,
        budget: SkillBudget | None = None,
        prior_negatives: list[ReflectInsight] | None = None,
        concurrency: int = 256,
    ) -> SkillResult:
        """Async repair workflow with meta-evolution and sub-skill accumulation."""

        budget = budget or SkillBudget()
        budget.start_time = time.time()
        prior_negatives = prior_negatives or []
        prior_round_index = max((neg.round_index for neg in prior_negatives), default=0)
        negatives: list[ReflectInsight] = list(prior_negatives)
        new_negatives: list[ReflectInsight] = []
        fm_info = GROUP_DEFINITIONS.get(self.FM_GROUP, {})
        fm_name = fm_info.get("name", self.FM_GROUP)
        subskill_store = SubSkillStore()
        all_change_records: list[ChangeRecord] = []

        print(f"  [Skill-{self.FM_GROUP}] Starting repair for {fm_name}")

        print(f"  [Skill-{self.FM_GROUP}] Running validation baseline ({len(validation_tasks)} tasks)...")
        before_val_traces = await runner.arun_batch(validation_tasks, dag=original_dag, max_concurrency=concurrency)
        before_val_profiles = await diagnoser.adiagnose_batch(before_val_traces)
        budget.used_batch_runs += 1

        total_inner = 0
        completed_outer_rounds = 0
        consecutive_inner_failures = 0

        for outer_round in range(self.MAX_OUTER_ROUNDS):
            if not budget.check():
                print(f"  [Skill-{self.FM_GROUP}] Budget exhausted, aborting.")
                break

            current_dag = original_dag
            print(f"  [Skill-{self.FM_GROUP}] Outer round {outer_round + 1}/{self.MAX_OUTER_ROUNDS}")

            analysis = await self.aanalyze(current_dag, proposal_traces, proposal_profiles, negatives)
            budget.used_llm_calls += 1

            evolve_history: list[EvolveResult] = []
            prev_fm_rate = self._fm_rate(proposal_profiles)
            no_improve = 0
            latest_eval_traces: list[MASTrace] = []
            latest_eval_profiles: list[FMProfile] = []

            for inner_iter in range(self.MAX_INNER_ITERS):
                if not budget.check():
                    break

                print(f"    Inner iter {inner_iter + 1}/{self.MAX_INNER_ITERS} (fm_rate={prev_fm_rate:.2f})")

                result = await self.aevolve(current_dag, analysis, negatives, evolve_history)
                evolve_history.append(result)
                current_dag = result.dag
                all_change_records.extend(getattr(result, "change_records", []))
                budget.used_llm_calls += 1

                new_traces = await runner.arun_batch(proposal_tasks, dag=current_dag, max_concurrency=concurrency)
                new_profiles = await diagnoser.adiagnose_batch(new_traces)
                latest_eval_traces = new_traces
                latest_eval_profiles = new_profiles
                budget.used_batch_runs += 1

                fm_rate = self._fm_rate(new_profiles)
                print(f"    → fm_rate={fm_rate:.2f}, change: {result.change_description[:80]}")

                if fm_rate < self.CONVERGENCE_THRESHOLD:
                    print(f"    Converged (fm_rate < {self.CONVERGENCE_THRESHOLD})")
                    consecutive_inner_failures = 0
                    break

                if fm_rate >= prev_fm_rate:
                    no_improve += 1
                    consecutive_inner_failures += 1

                    # Meta-evolution: if inner loop keeps failing, evolve the skill itself
                    if consecutive_inner_failures >= self.META_EVOLVE_THRESHOLD:
                        print(f"    [Meta] {consecutive_inner_failures} consecutive failures, triggering meta-evolution...")
                        from optpilot.skills.evolution import SkillEvolver
                        evolver = SkillEvolver()
                        evolved_path = evolver.evolve_skill(self.FM_GROUP, self, negatives + new_negatives)
                        if evolved_path:
                            print(f"    [Meta] Evolved skill saved: {evolved_path}")
                            from optpilot.skills.registry import load_evolved_skill
                            load_evolved_skill(self.FM_GROUP, evolved_path)
                        consecutive_inner_failures = 0

                    if no_improve >= self.NO_IMPROVE_PATIENCE:
                        print(f"    No improvement for {no_improve} iters, stopping inner loop.")
                        break
                else:
                    no_improve = 0
                    consecutive_inner_failures = 0
                prev_fm_rate = fm_rate

            total_inner += len(evolve_history)

            before_fm = self._fm_rate(before_val_profiles)
            before_pass = self._pass_rate(before_val_traces)
            material_change = self._has_material_change(original_dag, current_dag, evolve_history)

            if not material_change:
                print(f"  [Skill-{self.FM_GROUP}] No material DAG changes detected, skipping validation.")
                insight = self._build_synthetic_insight(
                    evolve_history=evolve_history,
                    before_fm=before_fm,
                    after_fm=before_fm,
                    before_pass=before_pass,
                    after_pass=before_pass,
                    failure_reason=(
                        "No concrete DAG change was applied. The evolve loop produced no material "
                        "code diff, so this round cannot count as a repair."
                    ),
                    lesson=(
                        "A valid repair must leave a real DAG diff. Require at least one concrete "
                        "search_and_replace or bash-written code change before evaluating success."
                    ),
                )
                insight.round_index = prior_round_index + len(new_negatives) + 1
                negatives.append(insight)
                new_negatives.append(insight)
                completed_outer_rounds = outer_round + 1
                print(f"    Lesson: {insight.lesson[:120]}")
                continue

            if not budget.check():
                after_fm = self._fm_rate(latest_eval_profiles) if latest_eval_profiles else before_fm
                after_pass = self._pass_rate(latest_eval_traces) if latest_eval_traces else before_pass
                insight = self._build_synthetic_insight(
                    evolve_history=evolve_history,
                    before_fm=before_fm,
                    after_fm=after_fm,
                    before_pass=before_pass,
                    after_pass=after_pass,
                    failure_reason=(
                        "Budget exhausted before holdout validation completed. This round ended "
                        "without a validated repair verdict."
                    ),
                    lesson=(
                        "Persist this failed attempt and either reduce per-round work or increase "
                        "budget so the skill can reach holdout validation."
                    ),
                )
                insight.round_index = prior_round_index + len(new_negatives) + 1
                negatives.append(insight)
                new_negatives.append(insight)
                completed_outer_rounds = outer_round + 1
                print(f"    Lesson: {insight.lesson[:120]}")
                break

            print(f"  [Skill-{self.FM_GROUP}] Validating on holdout ({len(validation_tasks)} tasks)...")
            after_val_traces = await runner.arun_batch(validation_tasks, dag=current_dag, max_concurrency=concurrency)
            after_val_profiles = await diagnoser.adiagnose_batch(after_val_traces)
            budget.used_batch_runs += 1

            after_fm = self._fm_rate(after_val_profiles)
            after_pass = self._pass_rate(after_val_traces)
            print(
                f"    Validation: FM {before_fm:.2f}→{after_fm:.2f}, "
                f"pass {before_pass:.3f}→{after_pass:.3f}"
            )

            if self.judge(before_val_profiles, after_val_profiles,
                          before_val_traces, after_val_traces):
                print(f"  [Skill-{self.FM_GROUP}] Repair validated!")

                # Save as sub-skill
                subskill = SubSkillStore.from_evolve_result(
                    fm_group=self.FM_GROUP,
                    change_records=all_change_records,
                    root_causes=analysis.root_cause_clusters,
                    before_fm=before_fm, after_fm=after_fm,
                    before_pass=before_pass, after_pass=after_pass,
                    summary=evolve_history[-1].change_description if evolve_history else "",
                )
                path = subskill_store.save(subskill)
                print(f"  [Skill-{self.FM_GROUP}] Saved sub-skill: {path}")

                return SkillResult(
                    success=True,
                    fm_id=self.FM_GROUP,
                    dag=current_dag,
                    inner_iterations=total_inner,
                    outer_rounds=outer_round + 1,
                    final_fm_rate=after_fm,
                    final_pass_rate=after_pass,
                    negatives=new_negatives,
                    budget_used=budget,
                    metadata={
                        "prior_negative_count": len(prior_negatives),
                        "change_records": all_change_records,
                    },
                )

            print(f"  [Skill-{self.FM_GROUP}] Validation failed, reflecting...")
            insight = await self.areflect(
                original_dag, evolve_history,
                before_val_traces, before_val_profiles,
                after_val_traces, after_val_profiles,
            )
            budget.used_llm_calls += 1
            insight.round_index = prior_round_index + len(new_negatives) + 1
            negatives.append(insight)
            new_negatives.append(insight)
            completed_outer_rounds = outer_round + 1
            print(f"    Lesson: {insight.lesson[:120]}")

        print(f"  [Skill-{self.FM_GROUP}] Repair failed after {completed_outer_rounds} rounds.")
        return SkillResult(
            success=False,
            fm_id=self.FM_GROUP,
            outer_rounds=completed_outer_rounds,
            inner_iterations=total_inner,
            negatives=new_negatives,
            budget_used=budget,
            metadata={
                "prior_negative_count": len(prior_negatives),
                "change_records": all_change_records,
            },
        )

    # ------------------------------------------------------------------ #
    #  Hooks — subclasses MUST override analyze, evolve, reflect          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def analyze(
        self,
        dag: MASDAG,
        traces: list[MASTrace],
        profiles: list[FMProfile],
        negatives: list[ReflectInsight],
    ) -> AnalysisResult:
        """Analyze the current system + a batch of traces with this FM."""

    @abstractmethod
    def evolve(
        self,
        dag: MASDAG,
        analysis: AnalysisResult,
        negatives: list[ReflectInsight],
        history: list[EvolveResult],
    ) -> EvolveResult:
        """Generate one repair iteration. Returns modified DAG + evidence."""

    def judge(
        self,
        before_profiles: list[FMProfile],
        after_profiles: list[FMProfile],
        before_traces: list[MASTrace],
        after_traces: list[MASTrace],
    ) -> bool:
        """Decide whether the repair improved the system.

        Success if FM rate decreased. Material DAG change is enforced separately.
        """
        before_fm = self._fm_rate(before_profiles)
        after_fm = self._fm_rate(after_profiles)
        return after_fm < before_fm

    @abstractmethod
    def reflect(
        self,
        original_dag: MASDAG,
        evolve_history: list[EvolveResult],
        before_val_traces: list[MASTrace],
        before_val_profiles: list[FMProfile],
        after_val_traces: list[MASTrace],
        after_val_profiles: list[FMProfile],
    ) -> ReflectInsight:
        """Explain why validation failed. Returns negative example."""

    # ------------------------------------------------------------------ #
    #  Async hooks — default implementations call sync versions via thread #
    # ------------------------------------------------------------------ #

    async def aanalyze(
        self, dag: MASDAG, traces: list[MASTrace],
        profiles: list[FMProfile], negatives: list[ReflectInsight],
    ) -> AnalysisResult:
        import asyncio
        return await asyncio.to_thread(self.analyze, dag, traces, profiles, negatives)

    async def aevolve(
        self, dag: MASDAG, analysis: AnalysisResult,
        negatives: list[ReflectInsight], history: list[EvolveResult],
    ) -> EvolveResult:
        """Async evolve using tool-calling agent loop. Override in subclasses."""
        import asyncio
        return await asyncio.to_thread(self.evolve, dag, analysis, negatives, history)

    async def areflect(
        self, original_dag: MASDAG, evolve_history: list[EvolveResult],
        before_val_traces: list[MASTrace], before_val_profiles: list[FMProfile],
        after_val_traces: list[MASTrace], after_val_profiles: list[FMProfile],
    ) -> ReflectInsight:
        import asyncio
        return await asyncio.to_thread(
            self.reflect, original_dag, evolve_history,
            before_val_traces, before_val_profiles,
            after_val_traces, after_val_profiles,
        )

    # ------------------------------------------------------------------ #
    #  Tool methods — available to all subclasses                          #
    # ------------------------------------------------------------------ #

    def read_source(self, dag: MASDAG) -> str:
        """Serialize DAG to build_dag() Python source string."""
        return dag_to_python(dag)

    def write_source(self, source: str) -> MASDAG:
        """Parse a build_dag() Python source into a MASDAG."""
        from optpilot.skills.tools import python_source_to_dag

        return python_source_to_dag(source)

    def diff_source(self, before: str, after: str) -> list[str]:
        """Compute human-readable diff between two source strings."""
        diff_lines = list(unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            n=1,
        ))
        changes: list[str] = []
        for line in diff_lines:
            line = line.rstrip("\n")
            if line.startswith("+ ") or line.startswith("- "):
                changes.append(line)
        return changes[:20]  # cap for prompt size

    def format_negatives(self, negatives: list[ReflectInsight]) -> str:
        """Format accumulated negative examples for prompt injection."""
        if not negatives:
            return "None yet."
        lines: list[str] = []
        for i, neg in enumerate(negatives, 1):
            lines.append(
                f"Round {i}: tried [{', '.join(neg.changes_attempted[:3])}] "
                f"→ FM {neg.before_fm_rate:.2f}→{neg.after_fm_rate:.2f}, "
                f"pass {neg.before_pass_rate:.3f}→{neg.after_pass_rate:.3f}. "
                f"Failure: {neg.failure_reason}. Lesson: {neg.lesson}"
            )
        return "\n".join(lines)

    def format_history(self, history: list[EvolveResult]) -> str:
        """Format evolve history for prompt injection."""
        if not history:
            return "No prior modifications in this round."
        lines: list[str] = []
        for i, er in enumerate(history, 1):
            lines.append(f"Iter {i}: {er.change_description[:200]}")
        return "\n".join(lines)

    def _fm_rate(self, profiles: list[FMProfile]) -> float:
        if not profiles:
            return 0.0
        return sum(1 for p in profiles if self.FM_GROUP in p.active_fm_ids()) / len(profiles)

    def _pass_rate(self, traces: list[MASTrace]) -> float:
        # Prefer task_score (ground-truth benchmark accuracy) over task_success
        # (mere completion). task_success ≈ True for most finished runs, making
        # judge comparisons meaningless; task_score reflects actual correctness.
        scores = [t.task_score for t in traces if t.task_score is not None]
        if scores:
            return mean(scores)
        fallback = [1.0 if t.task_success else 0.0 for t in traces if t.task_success is not None]
        return mean(fallback) if fallback else 0.0

    def _has_material_change(
        self,
        original_dag: MASDAG,
        candidate_dag: MASDAG,
        evolve_history: list[EvolveResult],
    ) -> bool:
        if hasattr(original_dag, "canonical_dict") and hasattr(candidate_dag, "canonical_dict"):
            return original_dag.canonical_dict() != candidate_dag.canonical_dict()
        if hasattr(original_dag, "to_dict") and hasattr(candidate_dag, "to_dict"):
            return original_dag.to_dict() != candidate_dag.to_dict()
        elif original_dag != candidate_dag:
            return True
        for er in evolve_history:
            if er.change_records or er.actions_taken:
                return True
        return False

    def _invalid_evolve_result_reason(self, result: EvolveResult) -> str:
        reason = result.metadata.get("invalid_evolve_reason", "")
        return reason.strip() if isinstance(reason, str) else ""

    def _extract_final_assistant_summary(self, final_msgs: list[dict[str, Any]]) -> tuple[str, str]:
        """Return (summary, invalid_reason)."""
        if not final_msgs:
            return "", "Tool loop ended without any final assistant summary."

        last_msg = final_msgs[-1]
        if last_msg.get("role") != "assistant":
            return "", (
                "Tool loop ended after a tool result instead of a final assistant summary. "
                "The repair did not complete its validation-and-summary phase."
            )
        if last_msg.get("tool_calls"):
            return "", (
                "Tool loop ended while the assistant was still issuing tool calls. "
                "The repair did not reach a final summary."
            )

        content = last_msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return "", (
                "Tool loop ended with an empty assistant message instead of a final repair summary."
            )
        return content.strip(), ""

    def _build_synthetic_insight(
        self,
        evolve_history: list[EvolveResult],
        before_fm: float,
        after_fm: float,
        before_pass: float,
        after_pass: float,
        failure_reason: str,
        lesson: str,
    ) -> ReflectInsight:
        changes = [er.change_description for er in evolve_history if er.change_description]
        if not changes:
            changes = ["No concrete DAG changes recorded."]
        return ReflectInsight(
            round_index=0,
            fm_id=self.FM_GROUP,
            changes_attempted=changes,
            before_fm_rate=before_fm,
            after_fm_rate=after_fm,
            before_pass_rate=before_pass,
            after_pass_rate=after_pass,
            failure_reason=failure_reason,
            lesson=lesson,
            timestamp=datetime.now().isoformat(),
        )


# -------------------------------------------------------------------- #
#  GenericSkill — concrete implementation shared by all 6 FM groups     #
# -------------------------------------------------------------------- #

_EXECUTOR_PATH = PROJECT_ROOT / "src/optpilot/dag/executor.py"
_DAG_CORE_PATH = PROJECT_ROOT / "src/optpilot/dag/core.py"
_SKILL_BASE_PATH = PROJECT_ROOT / "src/optpilot/skills/base.py"
_SKILL_TOOLS_PATH = PROJECT_ROOT / "src/optpilot/skills/tools.py"
_MODELS_PATH = PROJECT_ROOT / "src/optpilot/models.py"
_ARCHITECTURE_PATH = PROJECT_ROOT / "memory_bank/architecture.md"
_PROJECT_GOAL_PATH = PROJECT_ROOT / "memory_bank/project_goal.md"
_PROGRESS_PATH = PROJECT_ROOT / "memory_bank/progress.md"
_SUBSKILLS_ROOT = LIBRARY_DIR / "subskills"
_SKILL_AGENT_TRACE_ROOT = LIBRARY_DIR / "skill_agent_traces"


def _build_skill_context_index(
    fm_group: str,
    fm_name: str,
    fm_description: str,
    analysis: AnalysisResult,
    trace_file_lines: list[str],
) -> str:
    negatives_path = NEGATIVES_DIR / f"negatives_{fm_group}.json"
    subskills_path = _SUBSKILLS_ROOT / fm_group
    skill_agent_trace_path = _SKILL_AGENT_TRACE_ROOT / fm_group
    root_causes = ", ".join(analysis.root_cause_clusters) or "unknown"
    agents = ", ".join(analysis.common_agents) or "unknown"
    steps = ", ".join(analysis.common_steps) or "unknown"
    dag_components = ", ".join(analysis.metadata.get("dag_components", [])) or "unknown"

    return "\n".join([
        "# Skill Agent Context Index",
        "",
        "Read this file first, then use bash to inspect only the files you need.",
        "",
        "## Current Diagnosis Summary",
        f"- FM group: {fm_group} ({fm_name})",
        f"- Description: {fm_description}",
        f"- Root causes: {root_causes}",
        f"- Affected agents: {agents}",
        f"- Affected steps: {steps}",
        f"- DAG components: {dag_components}",
        "",
        "## Prepared Local Files",
        "- $DAG_FILE: current build_dag() Python source under repair",
        "- agent_context.md: this index file",
        *trace_file_lines,
        "",
        "## Repository Files To Inspect With Bash",
        f"- {_EXECUTOR_PATH}: runtime executor semantics for prompts, carry_data, loop control, and edge conditions",
        f"- {_DAG_CORE_PATH}: MASDAG, DAGNode, DAGEdge schema",
        f"- {_SKILL_BASE_PATH}: shared analyze/evolve/reflect workflow and prompt templates",
        f"- {_SKILL_TOOLS_PATH}: bash/search_and_replace tool semantics and Python source sync behavior",
        f"- {_MODELS_PATH}: AnalysisResult, EvolveResult, ReflectInsight, SkillBudget",
        f"- {_ARCHITECTURE_PATH}: project-level architecture and workflow notes",
        f"- {_PROJECT_GOAL_PATH}: project mission and scope",
        f"- {_PROGRESS_PATH}: current project progress and latest milestones",
        "",
        "## Persistent Experience",
        f"- {negatives_path}: accumulated failed repair reflections for this FM group",
        f"- {subskills_path}: successful reusable repairs for this FM group",
        f"- {skill_agent_trace_path}: persisted skill-agent tool traces for this FM group",
        "",
        "Use bash (`cat`, `sed -n`, `rg`) to inspect these files before making edits.",
    ])

_ANALYZE_PROMPT = """\
You are a multi-agent system (MAS) failure analyst.

Analyze the following MAS for **{fm_name}** problems: {fm_description}

## Current MAS Configuration (Python)
```python
{python_source}
```

## Fault Evidence from {n_traces} traces
{fault_summary}

## Prior Failed Repair Attempts (avoid repeating these)
{negatives_text}

{analyze_hint}

Respond with ONLY a JSON object:
{{
    "common_agents": ["<agent names involved>"],
    "common_steps": ["<steps where the fault occurs>"],
    "root_cause_clusters": ["<distinct root cause patterns>"],
    "evidence_snippets": ["<key evidence from traces>"]
}}"""

_EVOLVE_SYSTEM_PROMPT = """\
You are a multi-agent system (MAS) architect performing targeted repair.

You have two tools:
- **search_and_replace**: modify the build_dag() Python source by replacing exact text segments. \
The old_str must match exactly including whitespace. Use multiple calls for multiple changes.
- **bash**: run shell commands. Read files, run Python, validate the DAG, explore code, etc.

## Key Files
- `$DAG_FILE` — the current build_dag() Python source you are modifying
- `agent_context.md` — prepared context index with diagnosis summary, trace file locations, \
project memory, and repository file addresses
- `{executor_path}` — the DAG executor source code. Read this to understand how nodes, \
edges, carry_data, loops, and conditions work at runtime.
- `{dag_core_path}` — the DAG data model (MASDAG, DAGNode, DAGEdge).

## Workflow
1. Read the prepared index: `cat agent_context.md`
2. Read the current Python source: `cat $DAG_FILE`
3. Inspect only the most relevant repository files and trace files with `bash`.
4. Analyze the diagnosis and runtime evidence to identify the root cause.
5. Apply targeted fixes with `search_and_replace`.
6. Validate: `python3 -c "exec(open('$DAG_FILE').read()); d=build_dag(); print('OK:', len(d['nodes']), 'nodes')" && echo VALID`
7. Summarize what you changed and why.

## Rules
- Fix the ROOT CAUSE, not just symptoms.
- Preserve dag_id and metadata unchanged.
- Each search_and_replace old_str must be unique in the file.
- Do not spend the entire tool budget only reading files. After the minimum necessary
investigation, either apply at least one concrete code change or explicitly state why
no safe repair can be made.
- Leave enough turns for validation and a final summary after making edits.
- Your repair must not degrade task accuracy — fixing a structural issue is only \
valuable if the system still produces correct answers.
- The build_dag() function must return a dict parseable by MASDAG.from_dict(). \
You may add helper variables, modify prompts, change parameters, add/remove agents, \
or restructure routing — all through Python code modifications."""

_EVOLVE_USER_PROMPT = """\
The system suffers from **{fm_name}**: {fm_description}

## What this failure looks like
{failure_examples}

## Diagnosis
Root causes: {root_causes}
Affected agents: {agents}
Affected steps: {steps}
DAG components involved: {dag_components}

## Prepared Context
- Read `{context_file}` first. It contains the key file paths, project memory paths, \
and persistent experience locations for this FM group.

## Failed Traces
The following trace files show what went wrong at runtime. \
Use `bash` to read them and understand the actual data flow:
{trace_files}

## Prior Successful Repairs (try these first if relevant)
{subskills_text}

## Modification History (this round)
{history_text}

## Failed Approaches (from prior rounds — DO NOT repeat)
{negatives_text}

Fix the diagnosed problems. Start by reading the build_dag() Python source and the executor source \
to understand the data flow, then examine the trace files to see the actual failure."""

_REFLECT_PROMPT = """\
You are analyzing why a MAS repair attempt failed to resolve **{fm_name}** problems.

## Original MAS (before repair)
```python
{original_python}
```

## Final MAS (after repair)
```python
{final_python}
```

## Changes attempted in this round
{changes_text}

## Validation results
- FM occurrence rate: {before_fm:.2f} → {after_fm:.2f}
- Task pass rate: {before_pass:.3f} → {after_pass:.3f}

Analyze why the repair did not work and what lesson should be learned.

Respond with ONLY a JSON object:
{{
    "failure_reason": "<why the repair failed, max 100 words>",
    "lesson": "<what to avoid or try differently next time, max 100 words>"
}}"""


class GenericSkill(BaseSkill):
    """Concrete skill with shared analyze/evolve/reflect logic.

    Subclasses only need to set FM_GROUP and optionally ANALYZE_HINT.
    """

    ANALYZE_HINT: str = ""  # extra domain guidance injected into analyze prompt

    def analyze(
        self, dag: MASDAG, traces: list[MASTrace],
        profiles: list[FMProfile], negatives: list[ReflectInsight],
    ) -> AnalysisResult:
        fm_info = GROUP_DEFINITIONS[self.FM_GROUP]
        prompt = _ANALYZE_PROMPT.format(
            fm_name=fm_info["name"], fm_description=fm_info["description"],
            python_source=self.read_source(dag), n_traces=len(traces),
            fault_summary=summarize_faults(self.FM_GROUP, profiles, traces),
            negatives_text=self.format_negatives(negatives),
            analyze_hint=self.ANALYZE_HINT,
        )
        result = call_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)

        # Collect dag_components from localization
        dag_components: set[str] = set()
        for profile in profiles:
            loc = profile.localization.get(self.FM_GROUP)
            if loc and loc.dag_component and loc.dag_component != "other":
                dag_components.add(loc.dag_component)

        return AnalysisResult(
            fm_id=self.FM_GROUP, fm_rate=self._fm_rate(profiles),
            common_agents=result.get("common_agents", []),
            common_steps=result.get("common_steps", []),
            root_cause_clusters=result.get("root_cause_clusters", []),
            dag_summary=dag.summary() if hasattr(dag, "summary") else "",
            evidence_snippets=result.get("evidence_snippets", []),
            metadata={
                "dag_components": sorted(dag_components),
                "proposal_traces": traces,
            },
        )

    def evolve(
        self, dag: MASDAG, analysis: AnalysisResult,
        negatives: list[ReflectInsight], history: list[EvolveResult],
    ) -> EvolveResult:
        """Sync evolve — falls back to async via event loop."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.aevolve(dag, analysis, negatives, history)
            )
        finally:
            loop.close()

    async def aevolve(
        self, dag: MASDAG, analysis: AnalysisResult,
        negatives: list[ReflectInsight], history: list[EvolveResult],
    ) -> EvolveResult:
        """Async evolve via tool-calling agent loop."""
        fm_info = GROUP_DEFINITIONS[self.FM_GROUP]
        ctx = ToolContext.from_dag(dag)

        # Write trace files to tmpdir so the LLM can read them
        import os
        trace_file_lines: list[str] = []
        traces_from_analysis = analysis.metadata.get("proposal_traces", [])
        for i, trace in enumerate(traces_from_analysis[:5]):
            if trace.trace_path and os.path.exists(trace.trace_path):
                trace_file_lines.append(
                    f"- {trace.trace_path}: persisted runtime trace "
                    f"(score={trace.task_score}, benchmark={trace.benchmark_name})"
                )
                continue

            trace_name = f"trace_{i}.txt"
            trace_path = os.path.join(ctx.tmpdir, trace_name)
            with open(trace_path, "w") as f:
                f.write(trace.trajectory)
            trace_file_lines.append(
                f"- {trace_name}: temporary runtime trace "
                f"(score={trace.task_score}, benchmark={trace.benchmark_name})"
            )
        if not trace_file_lines:
            trace_file_lines.append("No trace files available.")

        context_file = os.path.join(ctx.tmpdir, "agent_context.md")
        with open(context_file, "w") as f:
            f.write(_build_skill_context_index(
                fm_group=self.FM_GROUP,
                fm_name=fm_info["name"],
                fm_description=fm_info["description"],
                analysis=analysis,
                trace_file_lines=trace_file_lines,
            ))

        subskill_store = SubSkillStore()
        user_prompt = _EVOLVE_USER_PROMPT.format(
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            failure_examples=fm_info.get("failure_examples", "No examples available."),
            root_causes=", ".join(analysis.root_cause_clusters) or "unknown",
            agents=", ".join(analysis.common_agents) or "unknown",
            steps=", ".join(analysis.common_steps) or "unknown",
            dag_components=", ".join(analysis.metadata.get("dag_components", [])) or "unknown",
            trace_files="\n".join(trace_file_lines),
            subskills_text=subskill_store.format_for_prompt(self.FM_GROUP),
            history_text=self.format_history(history),
            negatives_text=self.format_negatives(negatives),
            context_file=os.path.basename(context_file),
        )

        def tool_executor(name: str, arguments: dict) -> str:
            return execute_tool(name, arguments, ctx)

        messages = [
            {"role": "system", "content": _EVOLVE_SYSTEM_PROMPT.format(
                executor_path=_EXECUTOR_PATH,
                dag_core_path=_DAG_CORE_PATH,
            )},
            {"role": "user", "content": user_prompt},
        ]

        final_msgs = await acall_llm_with_tools(
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_executor=tool_executor,
            max_tokens=SKILL_EVOLVE_MAX_TOKENS,
            max_turns=SKILL_EVOLVE_MAX_TURNS,
        )
        tool_trace_path = self._persist_tool_trace(
            analysis=analysis,
            negatives=negatives,
            history=history,
            ctx=ctx,
            context_file=context_file,
            trace_file_lines=trace_file_lines,
            final_msgs=final_msgs,
        )

        analysis_text, invalid_evolve_reason = self._extract_final_assistant_summary(final_msgs)

        # Build result
        modified_source = ctx.python_source
        try:
            new_dag = ctx.to_dag()
        except Exception:
            new_dag = dag  # fallback if parsing fails

        change_description = analysis_text[:200] if analysis_text else ""
        if not change_description and invalid_evolve_reason:
            change_description = invalid_evolve_reason[:200]
        if not change_description:
            change_description = "; ".join(ctx.change_previews())[:200]

        return EvolveResult(
            dag=new_dag,
            analysis_text=analysis_text[:500],
            modified_source=modified_source,
            change_description=change_description,
            actions_taken=ctx.change_previews()[:20],
            change_records=ctx.changes,
            metadata={
                "tool_trace_path": tool_trace_path,
                "invalid_evolve_reason": invalid_evolve_reason,
                "final_assistant_summary_valid": not bool(invalid_evolve_reason),
            },
        )

    def _persist_tool_trace(
        self,
        analysis: AnalysisResult,
        negatives: list[ReflectInsight],
        history: list[EvolveResult],
        ctx: ToolContext,
        context_file: str,
        trace_file_lines: list[str],
        final_msgs: list[dict],
    ) -> str:
        trace_dir = _SKILL_AGENT_TRACE_ROOT / self.FM_GROUP
        trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_path = trace_dir / f"tool_trace_{timestamp}.json"
        payload = {
            "created_at": datetime.now().isoformat(),
            "fm_group": self.FM_GROUP,
            "agent_context_path": context_file,
            "dag_file": os.path.join(ctx.tmpdir, "dag.py"),
            "trace_files": trace_file_lines,
            "analysis": {
                "fm_id": analysis.fm_id,
                "fm_rate": analysis.fm_rate,
                "common_agents": analysis.common_agents,
                "common_steps": analysis.common_steps,
                "root_cause_clusters": analysis.root_cause_clusters,
                "evidence_snippets": analysis.evidence_snippets,
                "metadata": analysis.metadata,
            },
            "prior_negatives": [
                {
                    "round_index": neg.round_index,
                    "failure_reason": neg.failure_reason,
                    "lesson": neg.lesson,
                    "changes_attempted": neg.changes_attempted,
                    "before_fm_rate": neg.before_fm_rate,
                    "after_fm_rate": neg.after_fm_rate,
                    "before_pass_rate": neg.before_pass_rate,
                    "after_pass_rate": neg.after_pass_rate,
                    "timestamp": neg.timestamp,
                }
                for neg in negatives
            ],
            "history": [
                {
                    "change_description": er.change_description,
                    "actions_taken": er.actions_taken,
                    "metadata": er.metadata,
                }
                for er in history
            ],
            "messages": final_msgs,
            "change_records": [
                {
                    "old_str": change.old_str,
                    "new_str": change.new_str,
                    "source": change.source,
                }
                if isinstance(change, ChangeRecord)
                else change
                for change in ctx.changes
            ],
            "change_previews": ctx.change_previews(),
            "final_source": ctx.python_source,
        }
        trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(trace_path)

    def reflect(
        self, original_dag: MASDAG, evolve_history: list[EvolveResult],
        before_val_traces: list[MASTrace], before_val_profiles: list[FMProfile],
        after_val_traces: list[MASTrace], after_val_profiles: list[FMProfile],
    ) -> ReflectInsight:
        fm_info = GROUP_DEFINITIONS[self.FM_GROUP]
        original_python = self.read_source(original_dag)
        final_python = evolve_history[-1].modified_source if evolve_history else original_python
        changes = [er.change_description for er in evolve_history]
        before_fm = self._fm_rate(before_val_profiles)
        after_fm = self._fm_rate(after_val_profiles)
        before_pass = self._pass_rate(before_val_traces)
        after_pass = self._pass_rate(after_val_traces)

        prompt = _REFLECT_PROMPT.format(
            fm_name=fm_info["name"],
            original_python=original_python[:3000], final_python=final_python[:3000],
            changes_text="\n".join(f"- {c}" for c in changes) or "No changes recorded.",
            before_fm=before_fm, after_fm=after_fm,
            before_pass=before_pass, after_pass=after_pass,
        )
        result = call_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)
        return ReflectInsight(
            round_index=0, fm_id=self.FM_GROUP, changes_attempted=changes,
            before_fm_rate=before_fm, after_fm_rate=after_fm,
            before_pass_rate=before_pass, after_pass_rate=after_pass,
            failure_reason=result.get("failure_reason", "Unknown"),
            lesson=result.get("lesson", "Unknown"),
            timestamp=datetime.now().isoformat(),
        )
