"""Orchestrator — Jacobian-driven single repair loop.

Each round: run → diagnose → Jacobian recommend → evolve → evaluate → update.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from optpilot.config import LIBRARY_DIR
from optpilot.config import (
    FAILURE_EXAMPLE_TARGET,
    JACOBIAN_TOP_K_PATTERNS,
    ONLINE_EVAL_RANDOM_SEED,
    SHADOW_EVAL_INTERVAL,
    SHADOW_META_EVOLVE_THRESHOLD,
    SKILL_EVOLVE_NUM_CANDIDATES,
    topology_recipes_dir,
)
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy_6group import GROUP_IDS, GROUP_NAMES
from optpilot.models import FMLabel, FMLocalization, FMProfile, MASTrace, SkillBudget, SkillResult
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.skills.evolution import CatalogEvolver
from optpilot.skills.jacobian import RepairJacobian, RepairOutcome
from optpilot.skills.negatives import NegativesStore
from optpilot.skills.repair_loop import (
    agenerate_evolve_candidates,
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
    min_support: int = 1,
) -> list[tuple[str, int]]:
    """Rank FM groups by active issue frequency (descending)."""
    fm_counts: Counter[str] = Counter()
    for p in profiles:
        fm_counts.update(p.active_fm_ids())

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


def _resolve_task_benchmark_name(runner: MASRunner, task: str) -> str:
    resolver = getattr(runner, "benchmark_name_resolver", None)
    if callable(resolver):
        try:
            return str(resolver(task))
        except Exception:
            pass

    fallback = getattr(runner, "_resolve_benchmark_name", None)
    if callable(fallback):
        try:
            return str(fallback(task))
        except Exception:
            pass

    return str(getattr(runner, "benchmark_name", "default"))


def _sample_balanced_tasks(
    runner: MASRunner,
    tasks: list[str],
    sample_size: int | None,
    *,
    seed: int,
    exclude: set[str] | None = None,
) -> list[str]:
    if sample_size is None or sample_size >= len(tasks):
        return list(tasks)

    exclude = exclude or set()
    available = [task for task in tasks if task not in exclude]
    if sample_size >= len(available):
        return list(available)

    groups: dict[str, list[str]] = defaultdict(list)
    for task in available:
        groups[_resolve_task_benchmark_name(runner, task)].append(task)

    rng = random.Random(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    for bucket in groups.values():
        rng.shuffle(bucket)

    selected: list[str] = []
    while len(selected) < sample_size:
        progressed = False
        for name in group_names:
            bucket = groups[name]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break
        rng.shuffle(group_names)

    return selected


def _save_dag_if_possible(dag: object, path: str | Path) -> str:
    path_obj = Path(path)
    if hasattr(dag, "save"):
        dag.save(path_obj)
        return str(path_obj)
    return ""


def _trace_failed(trace: MASTrace) -> bool:
    score = getattr(trace, "task_score", None)
    if score is not None:
        return float(score) <= 0.0
    success = getattr(trace, "task_success", None)
    if success is not None:
        return not bool(success)
    return True


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
    elif isinstance(data.get("active_fm_ids"), list):
        for fm_id in data["active_fm_ids"]:
            profile.labels[str(fm_id)] = FMLabel(
                fm_id=str(fm_id),
                fm_name=str(fm_id),
                category=str(fm_id),
                present=True,
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
        topology: str | None = None,
    ):
        self.runner = runner
        self.dag = dag
        self.topology = topology
        self.diagnoser = Diagnoser()
        self.tracker = Tracker("skill_optimization", use_wandb=use_wandb)

        if topology:
            from optpilot.config import (
                topology_catalog_path,
                topology_jacobian_dir,
                topology_negatives_dir,
            )
            neg_dir = Path(negatives_dir) if negatives_dir else topology_negatives_dir(topology)
            jac_dir = topology_jacobian_dir(topology)
            cat_path = topology_catalog_path(topology)
            self.negatives_store = NegativesStore(neg_dir)
            self.catalog = PatternCatalog(store_path=cat_path)
            self.jacobian = RepairJacobian(catalog=self.catalog, base_dir=jac_dir)
        else:
            self.negatives_store = NegativesStore(Path(negatives_dir or LIBRARY_DIR / "negatives"))
            self.catalog = PatternCatalog()
            self.jacobian = RepairJacobian(catalog=self.catalog)

        self.evolver = CatalogEvolver(
            catalog=self.catalog,
            jacobian=self.jacobian,
            topology=topology,
        )

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
        recommended repair pattern, evaluates the candidate on a balanced
        active online minibatch, and periodically checks a shadow minibatch
        before adoption.
        """
        current_dag = self.dag
        budget = budget or SkillBudget()

        eval_label = (
            f"{eval_tasks_per_round} balanced active samples per round"
            if eval_tasks_per_round else "all"
        )
        print("=== OptPilot Jacobian-Driven Optimization ===")
        print(f"Tasks: {len(tasks)} (eval: {eval_label}), Max rounds: {max_rounds}")
        if eval_tasks_per_round and SHADOW_EVAL_INTERVAL > 0:
            print(
                "Shadow gate: "
                f"{eval_tasks_per_round} balanced samples every {SHADOW_EVAL_INTERVAL} round(s)"
            )
        print()

        results: list[dict] = []
        dag_versions: list[dict[str, str | int]] = []
        seeded_failure_tasks: list[str] | None = None
        if dag_output_base:
            initial_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "initial.yaml")
            if initial_path:
                dag_versions.append({"stage": "initial", "path": initial_path})

        # Track successfully applied patterns for diminishing-returns decay
        session_applied_patterns: set[str] = set()

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

            if round_i == 0 and eval_tasks_per_round:
                (
                    seeded_failure_tasks,
                    incumbent_traces,
                    incumbent_profiles,
                ) = await self._collect_failure_examples(
                    tasks=tasks,
                    dag=current_dag,
                    concurrency=concurrency,
                    seed=ONLINE_EVAL_RANDOM_SEED,
                    target_failures=FAILURE_EXAMPLE_TARGET,
                    batch_size=eval_tasks_per_round,
                    budget=budget,
                )
                active_tasks = list(seeded_failure_tasks)
                print(f"  Seeded failure set: {len(active_tasks)} task(s)")
            else:
                active_tasks = (
                    list(seeded_failure_tasks)
                    if seeded_failure_tasks
                    else _sample_balanced_tasks(
                        self.runner,
                        tasks,
                        eval_tasks_per_round,
                        seed=ONLINE_EVAL_RANDOM_SEED + round_i,
                    )
                )

                incumbent_traces = await self.runner.arun_batch(
                    active_tasks,
                    dag=current_dag,
                    output_base=None,
                    max_concurrency=concurrency,
                )
                incumbent_profiles = await self.diagnoser.aclassify_batch(incumbent_traces)
                budget.used_batch_runs += 1

            active_counts = Counter(
                _resolve_task_benchmark_name(self.runner, task) for task in active_tasks
            )
            print(
                f"  Active batch ({len(active_tasks)}): "
                + ", ".join(f"{bench}={count}" for bench, count in sorted(active_counts.items()))
            )

            incumbent_score = fitness_score(
                incumbent_traces,
                incumbent_profiles,
                len(current_dag.agent_nodes),
            )
            print(f"  Incumbent active fitness: {incumbent_score:.4f}")

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
            if SKILL_EVOLVE_NUM_CANDIDATES <= 1:
                top_k_patterns = 1
            else:
                top_k_patterns = max(
                    0, min(JACOBIAN_TOP_K_PATTERNS, SKILL_EVOLVE_NUM_CANDIDATES - 1)
                )
            recommended = self.jacobian.recommend(
                dom_sig,
                top_k=top_k_patterns,
                applied_patterns=session_applied_patterns,
            )
            recommended_patterns = [pattern for pattern, _ in recommended]
            exploratory_slots = (
                0
                if SKILL_EVOLVE_NUM_CANDIDATES <= 1
                else max(0, SKILL_EVOLVE_NUM_CANDIDATES - len(recommended_patterns))
            )
            candidate_pattern_plan = recommended_patterns + [None] * exploratory_slots
            pattern = recommended_patterns[0] if recommended_patterns else None
            pattern_label = ",".join(
                [
                    *(p.pattern_id for p in recommended_patterns),
                    *(["explore"] * exploratory_slots),
                ]
            ) or "none"
            if recommended_patterns:
                print(
                    "  Jacobian recommendations: "
                    + ", ".join(f"{p.name} ({p.pattern_id})" for p in recommended_patterns)
                )
            if session_applied_patterns:
                print(f"  Applied patterns (decayed): {session_applied_patterns}")
            if exploratory_slots:
                print(f"  Exploratory candidates: {exploratory_slots}")

            # Load negatives for this FM group
            negatives = self.negatives_store.load(top_fm)

            # 3. Analyze + Evolve
            print(f"  [1/3] Analyzing + generating candidates (pattern={pattern_label})...")
            analysis = analyze(current_dag, top_fm, traces, profiles, negatives)
            budget.used_llm_calls += 1

            candidates = await agenerate_evolve_candidates(
                current_dag, top_fm, analysis, negatives, [],
                recommended_pattern=pattern,
                recommended_patterns=candidate_pattern_plan,
                traces=traces,
                profiles=profiles,
                recipe_dir=topology_recipes_dir(self.topology) if self.topology else None,
            )
            budget.used_llm_calls += max(1, len(candidates))

            valid_candidates = [
                candidate for candidate in candidates
                if not candidate.metadata.get("invalid_evolve_reason", "")
            ]
            changed_candidates = [
                candidate for candidate in valid_candidates
                if has_material_change(current_dag, candidate.dag, candidate)
            ]

            if not changed_candidates:
                failed_candidate = candidates[0] if candidates else None
                invalid_reason = ""
                if failed_candidate is not None:
                    invalid_reason = str(failed_candidate.metadata.get("invalid_evolve_reason", ""))
                reason = invalid_reason or "No concrete DAG change was applied."
                print(f"  Candidate generation failed: {reason[:120]}")
                insight = build_synthetic_insight(
                    fm_group=top_fm,
                    evolve_result=failed_candidate,
                    before_fm=fm_rate(top_fm, profiles),
                    after_fm=fm_rate(top_fm, profiles),
                    before_pass=pass_rate(traces),
                    after_pass=pass_rate(traces),
                    failure_reason=reason,
                    lesson="Generate at least one valid, materially different candidate before evaluation.",
                )
                self.negatives_store.extend(top_fm, [insight])
                results.append(self._round_entry(round_i, top_fm, False, failed_candidate))
                continue

            # 4. Evaluate candidate pool on the active online minibatch.
            print(
                f"  [2/3] Evaluating {len(changed_candidates)} candidate DAG(s) "
                f"on {len(active_tasks)} active task(s)..."
            )
            before_fm_val = fm_rate(top_fm, profiles)
            before_pass_val = pass_rate(traces)
            before_fitness = incumbent_score
            best_candidate = None
            best_traces = None
            best_profiles = None
            best_after_fm = before_fm_val
            best_after_pass = before_pass_val
            best_after_fitness = before_fitness

            for candidate_index, candidate in enumerate(changed_candidates, start=1):
                candidate_trace_dir = (
                    round_trace_dir / f"candidate_{candidate_index}"
                    if round_trace_dir else None
                )
                candidate_traces = await self.runner.arun_batch(
                    active_tasks,
                    dag=candidate.dag,
                    output_base=candidate_trace_dir,
                    max_concurrency=concurrency,
                )
                candidate_profiles = await self.diagnoser.aclassify_batch(candidate_traces)
                budget.used_batch_runs += 1

                after_fm_val = fm_rate(top_fm, candidate_profiles)
                after_pass_val = pass_rate(candidate_traces)
                after_fitness = fitness_score(
                    candidate_traces,
                    candidate_profiles,
                    len(candidate.dag.agent_nodes),
                )
                print(
                    f"    Candidate {candidate_index}: fitness {after_fitness:.4f}, "
                    f"FM-{top_fm} {before_fm_val:.2f}->{after_fm_val:.2f}, "
                    f"pass {before_pass_val:.3f}->{after_pass_val:.3f}"
                )

                if (
                    best_candidate is None
                    or after_fitness > best_after_fitness
                    or (
                        after_fitness == best_after_fitness
                        and after_fm_val < best_after_fm
                    )
                ):
                    best_candidate = candidate
                    best_traces = candidate_traces
                    best_profiles = candidate_profiles
                    best_after_fm = after_fm_val
                    best_after_pass = after_pass_val
                    best_after_fitness = after_fitness

            assert best_candidate is not None
            assert best_traces is not None
            assert best_profiles is not None
            improved = best_after_fitness > before_fitness

            print(f"  Best candidate fitness: {before_fitness:.4f} → {best_after_fitness:.4f}  "
                  f"FM-{top_fm}: {before_fm_val:.2f} → {best_after_fm:.2f}  "
                  f"pass: {before_pass_val:.3f} → {best_after_pass:.3f}")

            shadow_checked = False
            shadow_passed = True
            shadow_failure_metadata: dict[str, Any] | None = None
            if (
                improved
                and eval_tasks_per_round
                and SHADOW_EVAL_INTERVAL > 0
                and (round_i + 1) % SHADOW_EVAL_INTERVAL == 0
            ):
                shadow_tasks = _sample_balanced_tasks(
                    self.runner,
                    tasks,
                    eval_tasks_per_round,
                    seed=ONLINE_EVAL_RANDOM_SEED + 10_000 + round_i,
                    exclude=set(active_tasks),
                )
                if shadow_tasks:
                    shadow_checked = True
                else:
                    print("  Shadow gate skipped: no held-out tasks available beyond active batch.")
                if shadow_checked:
                    shadow_counts = Counter(
                        _resolve_task_benchmark_name(self.runner, task) for task in shadow_tasks
                    )
                    print(
                        "  Shadow gate: "
                        + ", ".join(f"{bench}={count}" for bench, count in sorted(shadow_counts.items()))
                    )

                    incumbent_shadow_traces = await self.runner.arun_batch(
                        shadow_tasks,
                        dag=current_dag,
                        output_base=None,
                        max_concurrency=concurrency,
                    )
                    incumbent_shadow_accuracy = pass_rate(incumbent_shadow_traces)

                    candidate_shadow_traces = await self.runner.arun_batch(
                        shadow_tasks,
                        dag=best_candidate.dag,
                        output_base=round_trace_dir / "shadow_candidate" if round_trace_dir else None,
                        max_concurrency=concurrency,
                    )
                    budget.used_batch_runs += 2
                    candidate_shadow_accuracy = pass_rate(candidate_shadow_traces)

                    shadow_passed = candidate_shadow_accuracy >= incumbent_shadow_accuracy
                    print(
                        "  Shadow accuracy: "
                        f"{incumbent_shadow_accuracy:.4f} → {candidate_shadow_accuracy:.4f}"
                    )
                    if not shadow_passed:
                        improved = False
                        print("  Shadow gate rejected candidate due to accuracy regression.")
                        shadow_failure_metadata = await self._collect_shadow_rejection_metadata(
                            fm_group=top_fm,
                            shadow_tasks=shadow_tasks,
                            incumbent_shadow_traces=incumbent_shadow_traces,
                            candidate_shadow_traces=candidate_shadow_traces,
                            candidate=best_candidate,
                            incumbent_accuracy=incumbent_shadow_accuracy,
                            candidate_accuracy=candidate_shadow_accuracy,
                        )

            # 6. Update Jacobian
            if pattern:
                observed = str(best_candidate.metadata.get("observed_pattern_id", ""))
                outcome = RepairOutcome(
                    fm_group=dom_sig.fm_group,
                    dag_component=dom_sig.dag_component,
                    agent=dom_sig.agent,
                    assigned_pattern_id=pattern.pattern_id,
                    observed_pattern_id=observed,
                    success=improved,
                    fm_delta=best_after_fm - before_fm_val,
                    pass_delta=best_after_pass - before_pass_val,
                )
                self.jacobian.update(outcome)

            # 7. Adopt or reflect
            if improved:
                current_dag = best_candidate.dag
                self.dag = current_dag
                self.evolver.reset_failures(top_fm)
                # Track applied pattern for diminishing-returns decay
                observed = str(best_candidate.metadata.get("observed_pattern_id", ""))
                if observed:
                    session_applied_patterns.add(observed)
                elif pattern:
                    session_applied_patterns.add(pattern.pattern_id)
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
                    round_i, top_fm, True, best_candidate,
                    final_fm_rate=best_after_fm, final_pass_rate=best_after_pass,
                    final_fitness=best_after_fitness,
                ))
            else:
                print(f"  [3/3] No improvement, reflecting...")
                insight = reflect(
                    top_fm, current_dag, best_candidate,
                    before_fm_val, best_after_fm, before_pass_val, best_after_pass,
                )
                if shadow_checked and not shadow_passed:
                    insight.failure_reason = (
                        f"{insight.failure_reason} Shadow eval regressed at round {round_i + 1}."
                    ).strip()
                    insight.lesson = (
                        "Preserve gains on the active minibatch without regressing on a fresh "
                        "balanced shadow batch."
                    )
                    if shadow_failure_metadata:
                        insight.metadata.update(shadow_failure_metadata)
                self.negatives_store.extend(top_fm, [insight])
                if shadow_checked and not shadow_passed:
                    self.evolver.record_failure(top_fm)
                print(f"    Lesson: {insight.lesson[:120]}")

                results.append(self._round_entry(
                    round_i, top_fm, False, best_candidate,
                    final_fm_rate=best_after_fm, final_pass_rate=best_after_pass,
                    final_fitness=best_after_fitness,
                ))

                self._maybe_catalog_evolve(
                    top_fm,
                    shadow_failure=(shadow_checked and not shadow_passed),
                )

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

        if SKILL_EVOLVE_NUM_CANDIDATES <= 1:
            top_k_patterns = 1
        else:
            top_k_patterns = max(
                0, min(JACOBIAN_TOP_K_PATTERNS, SKILL_EVOLVE_NUM_CANDIDATES - 1)
            )
        recommended = self.jacobian.recommend(dom_sig, top_k=top_k_patterns)
        recommended_patterns = [pattern for pattern, _ in recommended]
        exploratory_slots = (
            0
            if SKILL_EVOLVE_NUM_CANDIDATES <= 1
            else max(0, SKILL_EVOLVE_NUM_CANDIDATES - len(recommended_patterns))
        )
        candidate_pattern_plan = recommended_patterns + [None] * exploratory_slots
        pattern = recommended_patterns[0] if recommended_patterns else None
        if recommended_patterns:
            print(
                "  Jacobian recommendations: "
                + ", ".join(pattern.name for pattern in recommended_patterns)
            )
        if exploratory_slots:
            print(f"  Exploratory candidates: {exploratory_slots}")

        negatives = self.negatives_store.load(top_fm)

        print("  Analyzing + generating candidates...")
        analysis = analyze(current_dag, top_fm, traces, profiles, negatives)
        candidates = await agenerate_evolve_candidates(
            current_dag, top_fm, analysis, negatives, [],
            recommended_pattern=pattern,
            recommended_patterns=candidate_pattern_plan,
            traces=traces,
            profiles=profiles,
            recipe_dir=topology_recipes_dir(self.topology) if self.topology else None,
        )
        valid_candidates = [
            candidate for candidate in candidates
            if not candidate.metadata.get("invalid_evolve_reason", "")
        ]
        changed_candidates = [
            candidate for candidate in valid_candidates
            if has_material_change(current_dag, candidate.dag, candidate)
        ]

        if not changed_candidates:
            failed_candidate = candidates[0] if candidates else None
            invalid_reason = ""
            if failed_candidate is not None:
                invalid_reason = str(failed_candidate.metadata.get("invalid_evolve_reason", ""))
            reason = invalid_reason or "No material DAG change."
            print(f"  Failed: {reason[:120]}")
            insight = build_synthetic_insight(
                fm_group=top_fm, evolve_result=failed_candidate,
                before_fm=fm_rate(top_fm, profiles), after_fm=fm_rate(top_fm, profiles),
                before_pass=pass_rate(traces), after_pass=pass_rate(traces),
                failure_reason=reason, lesson="Ensure concrete repairs are applied.",
            )
            self.negatives_store.extend(top_fm, [insight])
            results.append(self._round_entry(0, top_fm, False, failed_candidate))
        else:
            before_fm_val = fm_rate(top_fm, profiles)
            before_pass_val = pass_rate(traces)
            num_agents_before = len(current_dag.agent_nodes)
            before_fitness = fitness_score(traces, profiles, num_agents_before)
            best_candidate = None
            best_traces = None
            best_profiles = None
            best_after_fm = before_fm_val
            best_after_pass = before_pass_val
            best_after_fitness = before_fitness

            print(f"  Evaluating {len(changed_candidates)} repaired DAG candidate(s)...")
            for candidate_index, candidate in enumerate(changed_candidates, start=1):
                candidate_traces = await self.runner.arun_batch(
                    tasks, dag=candidate.dag, max_concurrency=concurrency,
                )
                candidate_profiles = await self.diagnoser.aclassify_batch(candidate_traces)
                after_fm_val = fm_rate(top_fm, candidate_profiles)
                after_pass_val = pass_rate(candidate_traces)
                after_fitness = fitness_score(
                    candidate_traces,
                    candidate_profiles,
                    len(candidate.dag.agent_nodes),
                )
                print(
                    f"    Candidate {candidate_index}: fitness {after_fitness:.4f}, "
                    f"FM-{top_fm} {before_fm_val:.2f}->{after_fm_val:.2f}, "
                    f"pass {before_pass_val:.3f}->{after_pass_val:.3f}"
                )
                if (
                    best_candidate is None
                    or after_fitness > best_after_fitness
                    or (
                        after_fitness == best_after_fitness
                        and after_fm_val < best_after_fm
                    )
                ):
                    best_candidate = candidate
                    best_traces = candidate_traces
                    best_profiles = candidate_profiles
                    best_after_fm = after_fm_val
                    best_after_pass = after_pass_val
                    best_after_fitness = after_fitness

            assert best_candidate is not None
            assert best_traces is not None
            assert best_profiles is not None
            improved = best_after_fitness > before_fitness

            print(f"  Best candidate fitness: {before_fitness:.4f} → {best_after_fitness:.4f}  "
                  f"FM-{top_fm}: {before_fm_val:.2f} → {best_after_fm:.2f}  "
                  f"pass: {before_pass_val:.3f} → {best_after_pass:.3f}")

            if pattern:
                observed = str(best_candidate.metadata.get("observed_pattern_id", ""))
                self.jacobian.update(RepairOutcome(
                    fm_group=dom_sig.fm_group, dag_component=dom_sig.dag_component,
                    agent=dom_sig.agent, assigned_pattern_id=pattern.pattern_id,
                    observed_pattern_id=observed, success=improved,
                    fm_delta=best_after_fm - before_fm_val,
                    pass_delta=best_after_pass - before_pass_val,
                ))

            if improved:
                current_dag = best_candidate.dag
                self.dag = current_dag
                self.evolver.reset_failures(top_fm)
                print("  IMPROVED — adopting new DAG.")
                results.append(self._round_entry(
                    0, top_fm, True, best_candidate,
                    final_fm_rate=best_after_fm, final_pass_rate=best_after_pass,
                    final_fitness=best_after_fitness,
                ))
                if dag_output_base:
                    p = _save_dag_if_possible(current_dag, Path(dag_output_base) / "improved.yaml")
                    if p:
                        dag_versions.append({"stage": "improved", "path": p})
            else:
                insight = reflect(
                    top_fm, current_dag, best_candidate,
                    before_fm_val, best_after_fm, before_pass_val, best_after_pass,
                )
                self.negatives_store.extend(top_fm, [insight])
                print(f"  No improvement. Lesson: {insight.lesson[:120]}")
                results.append(self._round_entry(
                    0, top_fm, False, best_candidate,
                    final_fm_rate=best_after_fm, final_pass_rate=best_after_pass,
                    final_fitness=best_after_fitness,
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

    async def _collect_failure_examples(
        self,
        *,
        tasks: list[str],
        dag: MASDAG,
        concurrency: int,
        seed: int,
        target_failures: int,
        batch_size: int,
        budget: SkillBudget,
    ) -> tuple[list[str], list[MASTrace], list[FMProfile]]:
        """Scan task batches until enough failed examples are accumulated."""
        seen_tasks: set[str] = set()
        failed_tasks: list[str] = []
        failed_traces: list[MASTrace] = []
        attempt = 0

        while len(seen_tasks) < len(tasks) and len(failed_tasks) < target_failures:
            batch = _sample_balanced_tasks(
                self.runner,
                tasks,
                batch_size,
                seed=seed + attempt,
                exclude=seen_tasks,
            )
            if not batch:
                break

            seen_tasks.update(batch)
            batch_traces = await self.runner.arun_batch(
                batch,
                dag=dag,
                output_base=None,
                max_concurrency=concurrency,
            )
            budget.used_batch_runs += 1

            for task, trace in zip(batch, batch_traces, strict=False):
                if _trace_failed(trace):
                    failed_tasks.append(task)
                    failed_traces.append(trace)
                    if len(failed_tasks) >= target_failures:
                        break
            attempt += 1

        if not failed_tasks:
            fallback_tasks = _sample_balanced_tasks(
                self.runner,
                tasks,
                batch_size,
                seed=seed,
            )
            failed_tasks = list(fallback_tasks)
            failed_traces = await self.runner.arun_batch(
                fallback_tasks,
                dag=dag,
                output_base=None,
                max_concurrency=concurrency,
            )
            budget.used_batch_runs += 1
            seen_tasks.update(fallback_tasks)

        failed_profiles = await self.diagnoser.aclassify_batch(failed_traces)
        print(
            f"  Collected {len(failed_tasks)} failed example(s) from {len(seen_tasks)} scanned task(s)"
        )
        return failed_tasks, failed_traces, failed_profiles

    async def _collect_shadow_rejection_metadata(
        self,
        *,
        fm_group: str,
        shadow_tasks: list[str],
        incumbent_shadow_traces: list[MASTrace],
        candidate_shadow_traces: list[MASTrace],
        candidate: Any,
        incumbent_accuracy: float,
        candidate_accuracy: float,
        max_diagnostics: int = 5,
    ) -> dict[str, Any]:
        regressions: list[tuple[str, MASTrace, bool]] = []
        candidate_failures: list[tuple[str, MASTrace, bool]] = []

        for task, incumbent_trace, candidate_trace in zip(
            shadow_tasks, incumbent_shadow_traces, candidate_shadow_traces
        ):
            incumbent_success = bool(getattr(incumbent_trace, "task_success", False))
            candidate_success = bool(getattr(candidate_trace, "task_success", False))
            if incumbent_success and not candidate_success:
                regressions.append((task, candidate_trace, True))
            elif not candidate_success:
                candidate_failures.append((task, candidate_trace, False))

        selected = regressions[:max_diagnostics]
        if len(selected) < max_diagnostics:
            selected.extend(candidate_failures[: max_diagnostics - len(selected)])

        traces_to_diagnose = [trace for _, trace, _ in selected]
        if traces_to_diagnose and hasattr(self.diagnoser, "aclassify_batch"):
            diagnosed_profiles = await self.diagnoser.aclassify_batch(traces_to_diagnose)
        elif traces_to_diagnose and hasattr(self.diagnoser, "adiagnose_batch"):
            diagnosed_profiles = await self.diagnoser.adiagnose_batch(traces_to_diagnose)
        else:
            diagnosed_profiles = [FMProfile(trace_id=getattr(trace, "trace_id", i)) for i, trace in enumerate(traces_to_diagnose)]

        diagnostics: list[dict[str, Any]] = []
        for (task, trace, is_regression), profile in zip(selected, diagnosed_profiles):
            diagnostics.append({
                "task": task,
                "task_key": str(getattr(trace, "task_key", "")),
                "benchmark": str(getattr(trace, "benchmark_name", "")),
                "trace_path": str(getattr(trace, "trace_path", "")),
                "is_regression": is_regression,
                "candidate_success": bool(getattr(trace, "task_success", False)),
                "candidate_score": getattr(trace, "task_score", None),
                "active_fm_ids": profile.active_fm_ids(),
            })

        return {
            "shadow_gate": {
                "fm_group": fm_group,
                "incumbent_accuracy": incumbent_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "num_regressions": len(regressions),
                "num_candidate_failures": sum(
                    1 for trace in candidate_shadow_traces if not bool(getattr(trace, "task_success", False))
                ),
                "candidate_change_description": str(getattr(candidate, "change_description", "")),
                "candidate_actions": list(getattr(candidate, "actions_taken", []) or []),
                "observed_pattern_id": str(getattr(candidate, "metadata", {}).get("observed_pattern_id", "")),
                "diagnostics": diagnostics,
            },
        }

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

    def _maybe_catalog_evolve(self, fm_group: str, *, shadow_failure: bool = False) -> None:
        """Trigger catalog meta-evolution after repeated shadow-gate rejections."""
        if not shadow_failure:
            return
        if self.evolver.should_evolve(fm_group, threshold=SHADOW_META_EVOLVE_THRESHOLD):
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
