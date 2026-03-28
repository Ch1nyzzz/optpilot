"""Orchestrator — Jacobian-driven single repair loop.

Each round: run → diagnose → Jacobian recommend → evolve → evaluate → update.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from optpilot.config import LIBRARY_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy_6group import GROUP_IDS, GROUP_NAMES
from optpilot.models import FMLabel, FMLocalization, FMProfile, MASTrace, SkillBudget, SkillResult
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.skills.evolution import CatalogEvolver
from optpilot.skills.jacobian import RepairJacobian, RepairOutcome
from optpilot.skills.negatives import NegativesStore
from optpilot.skills.repair_loop import (
    aevolve,
    analyze,
    build_synthetic_insight,
    fm_rate,
    format_negatives,
    has_material_change,
    pass_rate,
    reflect,
)
from optpilot.skills.repair_patterns import (
    PatternCatalog,
    dominant_signature,
    extract_failure_signatures,
    FailureSignature,
)
from optpilot.tracking import Tracker


def rank_fm_groups(
    profiles: list[FMProfile],
    target_fm: str | None = None,
    min_support: int = 2,
) -> list[tuple[str, int]]:
    """Rank FM groups by frequency (descending). Returns [(fm_id, count), ...]."""
    fm_counts: Counter[str] = Counter()
    for p in profiles:
        for fm_id in p.active_fm_ids():
            fm_counts[fm_id] += 1

    if target_fm:
        count = fm_counts.get(target_fm, 0)
        return [(target_fm, count)] if count >= min_support else []

    eligible = [(fm_id, count) for fm_id, count in fm_counts.items() if count >= min_support]
    eligible.sort(key=lambda item: item[1], reverse=True)
    return eligible


def fitness_score(
    traces: list[MASTrace],
    profiles: list[FMProfile],
    num_agents: int,
) -> float:
    """MAST-blog fitness: mean of per-task scores.

    per_task = 1/(1+trace_failures) * (1.2 if correct) - 0.01*(agents-4 if >4)
    """
    if not traces:
        return 0.0
    scores: list[float] = []
    for i, trace in enumerate(traces):
        is_correct = bool(trace.task_score and trace.task_score > 0)
        trace_failures = 0
        if i < len(profiles):
            trace_failures = sum(
                1 for gid in GROUP_IDS
                if gid in profiles[i].labels and profiles[i].labels[gid].present
            )
        s = 1.0 / (1.0 + trace_failures)
        if is_correct:
            s *= 1.2
        if num_agents > 4:
            s -= 0.01 * (num_agents - 4)
        scores.append(max(0.0, s))
    return sum(scores) / len(scores)


def _save_dag_if_possible(dag: object, path: str | Path) -> str:
    path_obj = Path(path)
    if hasattr(dag, "save"):
        dag.save(path_obj)
        return str(path_obj)
    return ""


def _trace_to_dict(trace) -> dict[str, object]:
    return {
        "trace_id": trace.trace_id,
        "mas_name": trace.mas_name,
        "llm_name": trace.llm_name,
        "benchmark_name": trace.benchmark_name,
        "trace_path": trace.trace_path,
        "task_key": trace.task_key,
        "task_success": trace.task_success,
        "task_score": trace.task_score,
        "latency_s": trace.latency_s,
    }


def _profile_to_dict(profile: FMProfile) -> dict[str, object]:
    return {
        "trace_id": profile.trace_id,
        "active_fm_ids": profile.active_fm_ids(),
        "labels": {
            fm_id: {
                "fm_name": label.fm_name,
                "category": label.category,
                "present": label.present,
                "confidence": label.confidence,
            }
            for fm_id, label in profile.labels.items()
        },
        "localization": {
            fm_id: {
                "agent": loc.agent,
                "step": loc.step,
                "context": loc.context,
                "root_cause": loc.root_cause,
                "dag_component": loc.dag_component,
            }
            for fm_id, loc in profile.localization.items()
        },
    }


def _profile_from_dict(data: dict[str, object]) -> FMProfile:
    profile = FMProfile(trace_id=int(data["trace_id"]))

    labels = data.get("labels", {})
    if isinstance(labels, dict):
        for fm_id, raw in labels.items():
            if not isinstance(raw, dict):
                continue
            profile.labels[str(fm_id)] = FMLabel(
                fm_id=str(fm_id),
                fm_name=str(raw.get("fm_name", fm_id)),
                category=str(raw.get("category", "")),
                present=bool(raw.get("present", False)),
                confidence=float(raw.get("confidence", 1.0)),
            )

    localization = data.get("localization", {})
    if isinstance(localization, dict):
        for fm_id, raw in localization.items():
            if not isinstance(raw, dict):
                continue
            profile.localization[str(fm_id)] = FMLocalization(
                agent=str(raw.get("agent", "")),
                step=str(raw.get("step", "")),
                context=str(raw.get("context", "")),
                root_cause=str(raw.get("root_cause", "")),
                dag_component=str(raw.get("dag_component", "other")),
            )

    return profile


def _write_json(path: str | Path, data: object) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path_obj)


def _task_dir_index(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


class Orchestrator:
    """Jacobian-driven single repair loop."""

    def __init__(
        self,
        runner: MASRunner,
        dag: MASDAG,
        use_wandb: bool = False,
        negatives_dir: str | Path | None = None,
    ):
        self.runner = runner
        self.dag = dag
        self.diagnoser = Diagnoser()
        self.tracker = Tracker("skill_optimization", use_wandb=use_wandb)
        self.negatives_store = NegativesStore(Path(negatives_dir or LIBRARY_DIR / "negatives"))
        self.catalog = PatternCatalog()
        self.jacobian = RepairJacobian(catalog=self.catalog)
        self.evolver = CatalogEvolver(catalog=self.catalog, jacobian=self.jacobian)

    # ---------------------------------------------------------------- #
    #  Main optimization loop                                           #
    # ---------------------------------------------------------------- #

    async def aoptimize(
        self,
        tasks: list[str],
        max_rounds: int = 10,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        trace_output_base: str | Path | None = None,
        dag_output_base: str | Path | None = None,
        eval_tasks_per_round: int | None = None,
    ) -> dict:
        """Single-loop optimization: diagnose → recommend → evolve → evaluate.

        Each round targets the most severe FM group, applies one Jacobian-
        recommended repair pattern, evaluates the candidate once on a fixed
        online eval set, and updates the matrix.

        If *eval_tasks_per_round* is set, use the same fixed prefix subset for
        every round (aligned with the OpenEvolve baseline evaluator).
        """
        current_dag = self.dag
        budget = budget or SkillBudget()

        eval_label = (
            f"{eval_tasks_per_round} sampled per round"
            if eval_tasks_per_round else "all"
        )
        print("=== OptPilot Jacobian-Driven Optimization ===")
        print(f"Tasks: {len(tasks)} (eval: {eval_label}), Max rounds: {max_rounds}")
        print()

        results: list[dict] = []
        dag_versions: list[dict[str, str | int]] = []
        if dag_output_base:
            initial_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "initial.yaml")
            if initial_path:
                dag_versions.append({"stage": "initial", "path": initial_path})

        # Fixed online evaluation set, matching the OpenEvolve-style protocol.
        if eval_tasks_per_round and eval_tasks_per_round < len(tasks):
            eval_tasks = tasks[:eval_tasks_per_round]
        else:
            eval_tasks = tasks

        print(f"Initial incumbent evaluation on {len(eval_tasks)} task(s)...")
        incumbent_trace_dir = (
            Path(trace_output_base) / "round_0"
            if trace_output_base else None
        )
        incumbent_traces = await self.runner.arun_batch(
            eval_tasks,
            dag=current_dag,
            output_base=incumbent_trace_dir,
            max_concurrency=concurrency,
        )
        incumbent_profiles = await self.diagnoser.aclassify_batch(incumbent_traces)
        incumbent_score = fitness_score(
            incumbent_traces,
            incumbent_profiles,
            len(current_dag.agent_nodes),
        )
        budget.used_batch_runs += 1
        print(f"  Incumbent fitness: {incumbent_score:.4f}")
        print()

        for round_i in range(max_rounds):
            print(f"=== Round {round_i + 1}/{max_rounds} ===")

            round_trace_dir = (
                Path(trace_output_base) / f"round_{round_i + 1}"
                if trace_output_base else None
            )
            round_dag_dir = (
                Path(dag_output_base) / f"round_{round_i + 1}"
                if dag_output_base else None
            )

            # 1. Extract dominant failure from the cached incumbent eval.
            traces = incumbent_traces
            profiles = incumbent_profiles
            fm_ranking = rank_fm_groups(profiles, target_fm)
            if not fm_ranking:
                print("  No active FM groups. Optimization complete!")
                break

            top_fm, top_count = fm_ranking[0]
            fm_name = GROUP_NAMES.get(top_fm, top_fm)
            print(f"  Active: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")
            print(f"  Targeting: Group-{top_fm} ({fm_name}), count={top_count}")

            # Extract failure signature for Jacobian
            failure_sigs = extract_failure_signatures(top_fm, profiles)
            dom_sig = dominant_signature(failure_sigs) if failure_sigs else FailureSignature(
                fm_group=top_fm, dag_component="other",
            )

            # 2. Jacobian recommend
            recommended = self.jacobian.recommend(dom_sig, top_k=1)
            pattern = recommended[0][0] if recommended else None
            pattern_label = pattern.pattern_id if pattern else "none"
            if pattern:
                print(f"  Jacobian recommendation: {pattern.name} ({pattern.pattern_id})")

            # Load negatives for this FM group
            negatives = self.negatives_store.load(top_fm)

            # 3. Analyze + Evolve
            print(f"  [1/3] Analyzing + evolving (pattern={pattern_label})...")
            analysis = analyze(current_dag, top_fm, traces, profiles, negatives)
            budget.used_llm_calls += 1

            evolve_result = await aevolve(
                current_dag, top_fm, analysis, negatives, [],
                recommended_pattern=pattern,
                traces=traces,
                profiles=profiles,
            )
            budget.used_llm_calls += 1

            # Check for invalid evolve
            invalid_reason = evolve_result.metadata.get("invalid_evolve_reason", "")
            if invalid_reason:
                print(f"  Invalid evolve: {invalid_reason[:120]}")
                insight = build_synthetic_insight(
                    fm_group=top_fm,
                    evolve_result=evolve_result,
                    before_fm=fm_rate(top_fm, profiles),
                    after_fm=fm_rate(top_fm, profiles),
                    before_pass=pass_rate(traces),
                    after_pass=pass_rate(traces),
                    failure_reason=invalid_reason,
                    lesson="Ensure evolve completes with a valid summary after edits.",
                )
                self.negatives_store.extend(top_fm, [insight])
                self.evolver.record_failure(top_fm)
                results.append(self._round_entry(round_i, top_fm, False, evolve_result))
                self._maybe_catalog_evolve(top_fm)
                continue

            # Check for material change
            if not has_material_change(current_dag, evolve_result.dag, evolve_result):
                print("  No material DAG change, skipping evaluation.")
                insight = build_synthetic_insight(
                    fm_group=top_fm,
                    evolve_result=evolve_result,
                    before_fm=fm_rate(top_fm, profiles),
                    after_fm=fm_rate(top_fm, profiles),
                    before_pass=pass_rate(traces),
                    after_pass=pass_rate(traces),
                    failure_reason="No concrete DAG change was applied.",
                    lesson="Require at least one concrete code change before evaluating.",
                )
                self.negatives_store.extend(top_fm, [insight])
                self.evolver.record_failure(top_fm)
                results.append(self._round_entry(round_i, top_fm, False, evolve_result))
                self._maybe_catalog_evolve(top_fm)
                continue

            # 4. Evaluate candidate once on the fixed online eval set.
            print(f"  [2/3] Evaluating candidate DAG on {len(eval_tasks)} tasks...")
            new_traces = await self.runner.arun_batch(
                eval_tasks,
                dag=evolve_result.dag,
                output_base=round_trace_dir,
                max_concurrency=concurrency,
            )
            new_profiles = await self.diagnoser.aclassify_batch(new_traces)
            budget.used_batch_runs += 1

            before_fm_val = fm_rate(top_fm, profiles)
            after_fm_val = fm_rate(top_fm, new_profiles)
            before_pass_val = pass_rate(traces)
            after_pass_val = pass_rate(new_traces)

            num_agents_after = len(evolve_result.dag.agent_nodes)
            before_fitness = incumbent_score
            after_fitness = fitness_score(new_traces, new_profiles, num_agents_after)
            improved = after_fitness > before_fitness

            print(f"  Fitness: {before_fitness:.4f} → {after_fitness:.4f}  "
                  f"FM-{top_fm}: {before_fm_val:.2f} → {after_fm_val:.2f}  "
                  f"pass: {before_pass_val:.3f} → {after_pass_val:.3f}")

            # 6. Update Jacobian
            if pattern:
                observed = str(evolve_result.metadata.get("observed_pattern_id", ""))
                outcome = RepairOutcome(
                    fm_group=dom_sig.fm_group,
                    dag_component=dom_sig.dag_component,
                    agent=dom_sig.agent,
                    assigned_pattern_id=pattern.pattern_id,
                    observed_pattern_id=observed,
                    success=improved,
                    fm_delta=after_fm_val - before_fm_val,
                    pass_delta=after_pass_val - before_pass_val,
                )
                self.jacobian.update(outcome)

            # 7. Adopt or reflect
            if improved:
                current_dag = evolve_result.dag
                self.dag = current_dag
                incumbent_traces = new_traces
                incumbent_profiles = new_profiles
                incumbent_score = after_fitness
                self.evolver.reset_failures(top_fm)
                print(f"  [3/3] IMPROVED — adopting new DAG.")

                if round_dag_dir:
                    dag_path = _save_dag_if_possible(
                        current_dag, round_dag_dir / "improved.yaml",
                    )
                    if dag_path:
                        dag_versions.append({
                            "stage": "improved",
                            "round": round_i + 1,
                            "fm_id": top_fm,
                            "path": dag_path,
                        })

                results.append(self._round_entry(
                    round_i, top_fm, True, evolve_result,
                    final_fm_rate=after_fm_val, final_pass_rate=after_pass_val,
                        final_fitness=after_fitness,
                ))
            else:
                print(f"  [3/3] No improvement, reflecting...")
                insight = reflect(
                    top_fm, current_dag, evolve_result,
                    before_fm_val, after_fm_val, before_pass_val, after_pass_val,
                )
                self.negatives_store.extend(top_fm, [insight])
                self.evolver.record_failure(top_fm)
                print(f"    Lesson: {insight.lesson[:120]}")

                results.append(self._round_entry(
                    round_i, top_fm, False, evolve_result,
                    final_fm_rate=after_fm_val, final_pass_rate=after_pass_val,
                    final_fitness=after_fitness,
                ))

                self._maybe_catalog_evolve(top_fm)

            print()

        # Save final state
        if dag_output_base:
            final_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "final.yaml")
            if final_path:
                dag_versions.append({"stage": "final", "path": final_path})

        self.jacobian.save()
        print(f"  Jacobian saved. {self.jacobian.format_matrix_summary()}")

        summary = {
            "total_rounds": len(results),
            "results": results,
            "dag_versions": dag_versions,
        }
        self.tracker.save_local("skill_optimization.json")
        return summary

    # ---------------------------------------------------------------- #
    #  Warm-start variants                                              #
    # ---------------------------------------------------------------- #

    async def aoptimize_from_traces(
        self,
        tasks: list[str],
        trace_base: str | Path,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        trace_output_base: str | Path | None = None,
        dag_output_base: str | Path | None = None,
    ) -> dict:
        """Warm-start: load persisted traces, re-diagnose, then run one repair round."""
        print("=== OptPilot Warm Start (from traces) ===")
        print(f"Tasks: {len(tasks)}, Trace source: {trace_base}")
        print()

        traces = self._load_persisted_traces(trace_base, tasks)
        profiles = await self.diagnoser.aclassify_batch(traces)

        return await self._run_single_repair_round(
            tasks, traces, profiles,
            target_fm=target_fm,
            budget=budget,
            concurrency=concurrency,
            dag_output_base=dag_output_base,
        )

    async def aoptimize_from_diagnose(
        self,
        tasks: list[str],
        diagnose_dir: str | Path,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        dag_output_base: str | Path | None = None,
    ) -> dict:
        """Warm-start: load persisted diagnose artifacts, then run one repair round."""
        print("=== OptPilot Warm Start (from diagnose) ===")
        print(f"Tasks: {len(tasks)}, Diagnose source: {diagnose_dir}")
        print()

        traces, profiles, _, _ = self._load_persisted_diagnosis(
            diagnose_dir=diagnose_dir, tasks=tasks, target_fm=target_fm,
        )

        return await self._run_single_repair_round(
            tasks, traces, profiles,
            target_fm=target_fm,
            budget=budget,
            concurrency=concurrency,
            dag_output_base=dag_output_base,
        )

    async def _run_single_repair_round(
        self,
        tasks: list[str],
        traces: list,
        profiles: list[FMProfile],
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        dag_output_base: str | Path | None = None,
    ) -> dict:
        """Run one analyze → evolve → evaluate round using pre-loaded traces/profiles."""
        current_dag = self.dag
        budget = budget or SkillBudget()
        results: list[dict] = []
        dag_versions: list[dict[str, str | int]] = []

        fm_ranking = rank_fm_groups(profiles, target_fm)
        if not fm_ranking:
            print("  No active FM groups. Done!")
            return {"total_rounds": 0, "results": [], "dag_versions": dag_versions}

        top_fm, top_count = fm_ranking[0]
        fm_name = GROUP_NAMES.get(top_fm, top_fm)
        print(f"  Active: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")
        print(f"  Targeting: Group-{top_fm} ({fm_name})")

        failure_sigs = extract_failure_signatures(top_fm, profiles)
        dom_sig = dominant_signature(failure_sigs) if failure_sigs else FailureSignature(
            fm_group=top_fm, dag_component="other",
        )

        recommended = self.jacobian.recommend(dom_sig, top_k=1)
        pattern = recommended[0][0] if recommended else None
        if pattern:
            print(f"  Jacobian recommendation: {pattern.name}")

        negatives = self.negatives_store.load(top_fm)

        print("  Analyzing + evolving...")
        analysis = analyze(current_dag, top_fm, traces, profiles, negatives)
        evolve_result = await aevolve(
            current_dag, top_fm, analysis, negatives, [],
            recommended_pattern=pattern,
            traces=traces,
            profiles=profiles,
        )

        invalid_reason = evolve_result.metadata.get("invalid_evolve_reason", "")
        if invalid_reason or not has_material_change(current_dag, evolve_result.dag, evolve_result):
            reason = invalid_reason or "No material DAG change."
            print(f"  Failed: {reason[:120]}")
            insight = build_synthetic_insight(
                fm_group=top_fm, evolve_result=evolve_result,
                before_fm=fm_rate(top_fm, profiles), after_fm=fm_rate(top_fm, profiles),
                before_pass=pass_rate(traces), after_pass=pass_rate(traces),
                failure_reason=reason, lesson="Ensure concrete repairs are applied.",
            )
            self.negatives_store.extend(top_fm, [insight])
            self.evolver.record_failure(top_fm)
            results.append(self._round_entry(0, top_fm, False, evolve_result))
        else:
            print("  Evaluating repaired DAG...")
            new_traces = await self.runner.arun_batch(
                tasks, dag=evolve_result.dag, max_concurrency=concurrency,
            )
            new_profiles = await self.diagnoser.aclassify_batch(new_traces)

            before_fm_val = fm_rate(top_fm, profiles)
            after_fm_val = fm_rate(top_fm, new_profiles)
            before_pass_val = pass_rate(traces)
            after_pass_val = pass_rate(new_traces)

            num_agents_before = len(current_dag.agent_nodes)
            num_agents_after = len(evolve_result.dag.agent_nodes)
            before_fitness = fitness_score(traces, profiles, num_agents_before)
            after_fitness = fitness_score(new_traces, new_profiles, num_agents_after)
            improved = after_fitness > before_fitness

            print(f"  Fitness: {before_fitness:.4f} → {after_fitness:.4f}  "
                  f"FM-{top_fm}: {before_fm_val:.2f} → {after_fm_val:.2f}  "
                  f"pass: {before_pass_val:.3f} → {after_pass_val:.3f}")

            if pattern:
                observed = str(evolve_result.metadata.get("observed_pattern_id", ""))
                self.jacobian.update(RepairOutcome(
                    fm_group=dom_sig.fm_group, dag_component=dom_sig.dag_component,
                    agent=dom_sig.agent, assigned_pattern_id=pattern.pattern_id,
                    observed_pattern_id=observed, success=improved,
                    fm_delta=after_fm_val - before_fm_val,
                    pass_delta=after_pass_val - before_pass_val,
                ))

            if improved:
                current_dag = evolve_result.dag
                self.dag = current_dag
                self.evolver.reset_failures(top_fm)
                print("  IMPROVED — adopting new DAG.")
                results.append(self._round_entry(
                    0, top_fm, True, evolve_result,
                    final_fm_rate=after_fm_val, final_pass_rate=after_pass_val,
                    final_fitness=after_fitness,
                ))
                if dag_output_base:
                    p = _save_dag_if_possible(current_dag, Path(dag_output_base) / "improved.yaml")
                    if p:
                        dag_versions.append({"stage": "improved", "path": p})
            else:
                insight = reflect(
                    top_fm, current_dag, evolve_result,
                    before_fm_val, after_fm_val, before_pass_val, after_pass_val,
                )
                self.negatives_store.extend(top_fm, [insight])
                self.evolver.record_failure(top_fm)
                print(f"  No improvement. Lesson: {insight.lesson[:120]}")
                results.append(self._round_entry(
                    0, top_fm, False, evolve_result,
                    final_fm_rate=after_fm_val, final_pass_rate=after_pass_val,
                    final_fitness=after_fitness,
                ))

        if dag_output_base:
            final_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "final.yaml")
            if final_path:
                dag_versions.append({"stage": "final", "path": final_path})

        self.jacobian.save()
        self.tracker.save_local("skill_optimization.json")
        return {"total_rounds": 1, "results": results, "dag_versions": dag_versions}

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _round_entry(
        self,
        round_i: int,
        fm_id: str,
        success: bool,
        evolve_result: Any = None,
        final_fm_rate: float = 1.0,
        final_pass_rate: float = 0.0,
        final_fitness: float | None = None,
    ) -> dict:
        fm_name = GROUP_NAMES.get(fm_id, fm_id)
        entry = {
            "round": round_i + 1,
            "fm_id": fm_id,
            "fm_name": fm_name,
            "success": success,
            "final_fm_rate": final_fm_rate,
            "final_pass_rate": final_pass_rate,
            "final_fitness": final_fitness,
        }
        self.tracker.log(entry, step=round_i)
        return entry

    def _maybe_catalog_evolve(self, fm_group: str) -> None:
        """Trigger catalog meta-evolution if failure threshold reached."""
        if self.evolver.should_evolve(fm_group):
            print(f"  Triggering catalog meta-evolution for Group-{fm_group}...")
            all_neg = self.negatives_store.load(fm_group)
            self.evolver.evolve_catalog(fm_group, all_neg)

    # ---------------------------------------------------------------- #
    #  Persistence loaders (warm-start)                                 #
    # ---------------------------------------------------------------- #

    def _load_persisted_traces(
        self,
        trace_base: str | Path,
        tasks: list[str],
    ) -> list:
        trace_root = Path(trace_base)
        if not trace_root.exists():
            raise FileNotFoundError(f"Persisted trace directory does not exist: {trace_root}")

        loaded: list[MASTrace | None] = [None] * len(tasks)
        for task_dir in sorted(trace_root.glob("task_*"), key=_task_dir_index):
            idx = _task_dir_index(task_dir)
            if idx >= len(tasks):
                continue
            trace_file = task_dir / "trace.txt"
            if not trace_file.exists():
                continue
            metadata_path = task_dir / "trace.json"
            metadata: dict[str, object] = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}

            task_prompt = str(metadata.get("task_prompt", tasks[idx]))
            benchmark_name = str(
                metadata.get(
                    "benchmark_name",
                    self.runner._resolve_benchmark_name(task_prompt)
                    if hasattr(self.runner, "_resolve_benchmark_name")
                    else "",
                )
            )
            loaded[idx] = MASTrace(
                trace_id=idx,
                mas_name=str(metadata.get("mas_name", getattr(self.dag, "dag_id", "OptPilot"))),
                llm_name=str(metadata.get("llm_name", getattr(self.runner, "model", ""))),
                benchmark_name=benchmark_name,
                trajectory=trace_file.read_text(encoding="utf-8"),
                trace_path=str(trace_file),
                task_key=str(metadata.get("task_key", task_prompt[:50])),
                task_success=metadata.get("task_success"),
                task_score=metadata.get("task_score"),
                latency_s=metadata.get("latency_s"),
            )

        missing = [i for i, trace in enumerate(loaded) if trace is None]
        if missing:
            raise ValueError(
                f"Missing persisted traces for task indices: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )
        return loaded  # type: ignore[return-value]

    def _load_persisted_diagnosis(
        self,
        diagnose_dir: str | Path,
        tasks: list[str],
        target_fm: str | None = None,
    ) -> tuple[list, list[FMProfile], list[dict], list[tuple[str, int]]]:
        base = Path(diagnose_dir)
        if not base.exists():
            raise FileNotFoundError(f"Persisted diagnose directory does not exist: {base}")

        profiles_path = base / "profiles.json"
        if not profiles_path.exists():
            raise FileNotFoundError(f"Persisted diagnose profiles are missing: {profiles_path}")

        raw_entries = json.loads(profiles_path.read_text(encoding="utf-8"))
        if not isinstance(raw_entries, list):
            raise ValueError(f"Persisted diagnose profiles must be a JSON list: {profiles_path}")

        traces: list[MASTrace | None] = [None] * len(tasks)
        profiles: list[FMProfile | None] = [None] * len(tasks)
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            idx = int(raw.get("task_index", -1))
            if idx < 0 or idx >= len(tasks):
                continue
            trace_path = str(raw.get("trace_path", ""))
            if not trace_path:
                raise ValueError(f"Persisted diagnose entry missing trace_path for task index {idx}")
            trace_file = Path(trace_path)
            if not trace_file.exists():
                raise FileNotFoundError(
                    f"Persisted trace file referenced by diagnose artifacts is missing: {trace_file}"
                )

            traces[idx] = MASTrace(
                trace_id=int(raw.get("trace_id", idx)),
                mas_name=str(raw.get("mas_name", getattr(self.dag, "dag_id", "OptPilot"))),
                llm_name=str(raw.get("llm_name", getattr(self.runner, "model", ""))),
                benchmark_name=str(raw.get("benchmark_name", "")),
                trajectory=trace_file.read_text(encoding="utf-8"),
                trace_path=str(trace_file),
                task_key=str(raw.get("task_key", tasks[idx][:50])),
                task_success=raw.get("task_success"),
                task_score=raw.get("task_score"),
                latency_s=raw.get("latency_s"),
            )
            profiles[idx] = _profile_from_dict(raw)

        missing = [i for i, trace in enumerate(traces) if trace is None or profiles[i] is None]
        if missing:
            raise ValueError(
                f"Persisted diagnose artifacts missing task indices: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )

        summary_path = base / "summary.json"
        summary_data: dict[str, object] = {}
        if summary_path.exists():
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary_data = loaded

        summary_ranking = summary_data.get("fm_ranking", [])
        fm_ranking_list = [
            (str(item.get("fm_id", "")), int(item.get("count", 0)))
            for item in summary_ranking
            if isinstance(item, dict) and item.get("fm_id")
        ]
        if target_fm:
            fm_ranking_list = [item for item in fm_ranking_list if item[0] == target_fm]
        if not fm_ranking_list:
            fm_ranking_list = rank_fm_groups(profiles, target_fm)

        return traces, profiles, [], fm_ranking_list  # type: ignore[return-value]
