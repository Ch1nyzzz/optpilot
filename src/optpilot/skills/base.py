"""BaseSkill — abstract base for FM-group-specific repair workflows.

Each Skill is a complete repair agent: analyze → evolve (inner loop) →
validate → reflect → retry.  The `run()` template method implements the
outer reflection loop and inner convergence loop; subclasses only override
the four hooks: analyze, evolve, judge, reflect.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import unified_diff
from statistics import mean
from typing import TYPE_CHECKING

import yaml

from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
from optpilot.llm import call_llm, call_llm_json
from optpilot.models import (
    AnalysisResult,
    EvolveResult,
    ReflectInsight,
    SkillBudget,
    SkillResult,
)
from optpilot.repair_utils import extract_fenced_block, extract_preface, summarize_faults

if TYPE_CHECKING:
    from optpilot.dag.core import MASDAG
    from optpilot.models import FMProfile, MASTrace
    from optpilot.modules.base_runner import MASRunner
    from optpilot.modules.diagnoser import Diagnoser


class BaseSkill(ABC):
    """Abstract base for FM-group-specific repair workflows."""

    FM_GROUP: str = ""
    MAX_INNER_ITERS: int = 5
    CONVERGENCE_THRESHOLD: float = 0.2
    NO_IMPROVE_PATIENCE: int = 2
    MAX_OUTER_ROUNDS: int = 3

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

            for inner_iter in range(self.MAX_INNER_ITERS):
                if not budget.check():
                    break

                print(f"    Inner iter {inner_iter + 1}/{self.MAX_INNER_ITERS} (fm_rate={prev_fm_rate:.2f})")

                result = self.evolve(current_dag, analysis, negatives, evolve_history)
                evolve_history.append(result)
                current_dag = result.dag
                budget.used_llm_calls += 1

                new_traces = runner.run_batch(proposal_tasks, dag=current_dag)
                new_profiles = diagnoser.diagnose_batch(new_traces)
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

            # --- Validate: same tasks, before vs after ---
            if not budget.check():
                break

            print(f"  [Skill-{self.FM_GROUP}] Validating on holdout ({len(validation_tasks)} tasks)...")
            after_val_traces = runner.run_batch(validation_tasks, dag=current_dag)
            after_val_profiles = diagnoser.diagnose_batch(after_val_traces)
            budget.used_batch_runs += 1

            before_fm = self._fm_rate(before_val_profiles)
            after_fm = self._fm_rate(after_val_profiles)
            before_pass = self._pass_rate(before_val_traces)
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

        Default: FM rate decreased AND pass rate improved.
        Both comparisons are on the SAME validation tasks (before vs after).
        """
        before_fm = self._fm_rate(before_profiles)
        after_fm = self._fm_rate(after_profiles)
        before_pass = self._pass_rate(before_traces)
        after_pass = self._pass_rate(after_traces)
        return after_fm < before_fm and after_pass > before_pass

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
    #  Tool methods — available to all subclasses                          #
    # ------------------------------------------------------------------ #

    def read_yaml(self, dag: MASDAG) -> str:
        """Serialize DAG to YAML string."""
        return yaml.dump(dag.to_dict(), allow_unicode=True, sort_keys=False)

    def write_yaml(self, yaml_str: str) -> MASDAG:
        """Parse a YAML string into a MASDAG."""
        from optpilot.dag.core import MASDAG as _MASDAG

        parsed = yaml.safe_load(yaml_str)
        return _MASDAG.from_dict(parsed)

    def diff_yaml(self, before: str, after: str) -> list[str]:
        """Compute human-readable diff between two YAML strings."""
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


# -------------------------------------------------------------------- #
#  GenericSkill — concrete implementation shared by all 6 FM groups     #
# -------------------------------------------------------------------- #

_ANALYZE_PROMPT = """\
You are a multi-agent system (MAS) failure analyst.

Analyze the following MAS for **{fm_name}** problems: {fm_description}

## Current MAS Configuration (YAML)
```yaml
{yaml_content}
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

_EVOLVE_PROMPT = """\
You are a multi-agent system (MAS) architect performing targeted repair.

The system below suffers from **{fm_name}**: {fm_description}

## Current MAS Configuration (YAML)
```yaml
{yaml_content}
```

## Diagnosis
Root causes: {root_causes}
Affected agents: {agents}
Affected steps: {steps}

## Modification History (this round)
{history_text}

## Failed Approaches (from prior rounds — DO NOT repeat)
{negatives_text}

Modify the YAML to fix the diagnosed problems. You may change ANY part of the \
configuration: agent prompts, roles, model parameters, loop limits, edge conditions, \
add/remove agents, rewire edges, etc. Make the minimal effective change.

Output format:
1. Analysis: explain your reasoning and proposed changes (max 200 words)
2. Modified YAML: output the COMPLETE modified YAML wrapped in ```yaml ... ```
   - Must be valid, engine-executable YAML
   - Preserve dag_id and metadata unchanged"""

_REFLECT_PROMPT = """\
You are analyzing why a MAS repair attempt failed to resolve **{fm_name}** problems.

## Original MAS (before repair)
```yaml
{original_yaml}
```

## Final MAS (after repair)
```yaml
{final_yaml}
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
            yaml_content=self.read_yaml(dag), n_traces=len(traces),
            fault_summary=summarize_faults(self.FM_GROUP, profiles, traces),
            negatives_text=self.format_negatives(negatives),
            analyze_hint=self.ANALYZE_HINT,
        )
        result = call_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)
        return AnalysisResult(
            fm_id=self.FM_GROUP, fm_rate=self._fm_rate(profiles),
            common_agents=result.get("common_agents", []),
            common_steps=result.get("common_steps", []),
            root_cause_clusters=result.get("root_cause_clusters", []),
            dag_summary=dag.summary() if hasattr(dag, "summary") else "",
            evidence_snippets=result.get("evidence_snippets", []),
        )

    def evolve(
        self, dag: MASDAG, analysis: AnalysisResult,
        negatives: list[ReflectInsight], history: list[EvolveResult],
    ) -> EvolveResult:
        fm_info = GROUP_DEFINITIONS[self.FM_GROUP]
        current_yaml = self.read_yaml(dag)
        prompt = _EVOLVE_PROMPT.format(
            fm_name=fm_info["name"], fm_description=fm_info["description"],
            yaml_content=current_yaml,
            root_causes=", ".join(analysis.root_cause_clusters) or "unknown",
            agents=", ".join(analysis.common_agents) or "unknown",
            steps=", ".join(analysis.common_steps) or "unknown",
            history_text=self.format_history(history),
            negatives_text=self.format_negatives(negatives),
        )
        response = call_llm([{"role": "user", "content": prompt}], max_tokens=16384)
        analysis_text = extract_preface(response, "yaml")
        modified_yaml = extract_fenced_block(response, "yaml")
        new_dag = self.write_yaml(modified_yaml)
        return EvolveResult(
            dag=new_dag, analysis_text=analysis_text, modified_yaml=modified_yaml,
            change_description=analysis_text[:200],
            actions_taken=self.diff_yaml(current_yaml, modified_yaml),
        )

    def reflect(
        self, original_dag: MASDAG, evolve_history: list[EvolveResult],
        before_val_traces: list[MASTrace], before_val_profiles: list[FMProfile],
        after_val_traces: list[MASTrace], after_val_profiles: list[FMProfile],
    ) -> ReflectInsight:
        fm_info = GROUP_DEFINITIONS[self.FM_GROUP]
        original_yaml = self.read_yaml(original_dag)
        final_yaml = evolve_history[-1].modified_yaml if evolve_history else original_yaml
        changes = [er.change_description for er in evolve_history]
        before_fm = self._fm_rate(before_val_profiles)
        after_fm = self._fm_rate(after_val_profiles)
        before_pass = self._pass_rate(before_val_traces)
        after_pass = self._pass_rate(after_val_traces)

        prompt = _REFLECT_PROMPT.format(
            fm_name=fm_info["name"],
            original_yaml=original_yaml[:3000], final_yaml=final_yaml[:3000],
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
