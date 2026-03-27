"""Orchestrator — controls the online optimization loop.

Coordinates: Runner → Diagnoser → dispatch to Skill Workflows (parallel).
"""

from __future__ import annotations

import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

from optpilot.config import LIBRARY_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy_6group import GROUP_NAMES
from optpilot.models import FMProfile, SkillBudget, SkillResult
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.skills.evolution import SkillEvolver
from optpilot.skills.negatives import NegativesStore
from optpilot.skills.registry import get_skill, load_evolved_skill
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


T = TypeVar("T")


def _take_by_index(items: list[T], indices: list[int]) -> list[T]:
    return [items[i] for i in indices]


def split_proposal_validation_indices(
    target_fm: str,
    profiles: list[FMProfile],
) -> tuple[list[int], list[int]] | None:
    """Split tasks into disjoint proposal and validation subsets.

    Proposal uses only a subset of traces that actually exhibit the target FM.
    Validation uses every remaining task, ensuring at least one holdout example
    still contains the target FM.
    """
    matched_indices = [i for i, profile in enumerate(profiles) if target_fm in profile.active_fm_ids()]
    if len(matched_indices) < 2:
        return None

    proposal_count = max(1, len(matched_indices) // 2)
    if proposal_count >= len(matched_indices):
        proposal_count = len(matched_indices) - 1

    proposal_indices = matched_indices[:proposal_count]
    proposal_set = set(proposal_indices)
    validation_indices = [i for i in range(len(profiles)) if i not in proposal_set]
    if not any(i in matched_indices for i in validation_indices):
        return None
    return proposal_indices, validation_indices


class Orchestrator:
    """Online optimization loop controller — dispatches to Skill Workflows."""

    def __init__(
        self,
        runner: MASRunner,
        dag: MASDAG,
        use_wandb: bool = False,
        negatives_dir: str | Path | None = None,
        evolved_dir: str | Path | None = None,
        parallel: bool = True,
    ):
        self.runner = runner
        self.dag = dag
        self.diagnoser = Diagnoser()
        self.tracker = Tracker("skill_optimization", use_wandb=use_wandb)
        self.negatives_store = NegativesStore(Path(negatives_dir or LIBRARY_DIR / "negatives"))
        self.evolver = SkillEvolver(Path(evolved_dir or LIBRARY_DIR / "evolved_skills"))
        self.parallel = parallel

    def optimize(
        self,
        tasks: list[str],
        max_rounds: int = 5,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
    ) -> dict:
        """Run the full online optimization loop with Skill Workflows.

        Skills for different FM groups can run in parallel since they each
        work on independent copies of the DAG.
        """
        current_dag = self.dag

        print(f"=== OptPilot Skill Workflow Optimization ===")
        print(f"Tasks: {len(tasks)}, Max rounds: {max_rounds}, Parallel: {self.parallel}")
        print()

        results: list[dict] = []

        for round_i in range(max_rounds):
            print(f"=== Round {round_i + 1}/{max_rounds} ===")

            # 1. Run MAS + Diagnose
            print(f"  [1/2] Running MAS on {len(tasks)} tasks + diagnosing...")
            traces = self.runner.run_batch(tasks, dag=current_dag)
            profiles = self.diagnoser.diagnose_batch(traces)

            # 2. Rank FM groups
            fm_ranking = rank_fm_groups(profiles, target_fm)
            if not fm_ranking:
                print(f"  No active FM groups with ≥2 traces. Done!")
                break

            print(f"  Active FM groups: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")

            # 3. Prepare skill dispatches
            skill_jobs: list[dict] = []
            for fm_id, _fm_count in fm_ranking:
                split = split_proposal_validation_indices(fm_id, profiles)
                if split is None:
                    print(f"  Skipping Group-{fm_id}: cannot split proposal/validation.")
                    continue

                proposal_idx, validation_idx = split
                prior_negatives = self.negatives_store.load(fm_id)

                skill_jobs.append({
                    "fm_id": fm_id,
                    "proposal_idx": proposal_idx,
                    "validation_idx": validation_idx,
                    "prior_negatives": prior_negatives,
                })

            if not skill_jobs:
                print(f"  No dispatchable skills. Done!")
                break

            # 4. Dispatch skills (parallel or sequential)
            if self.parallel and len(skill_jobs) > 1:
                skill_results = self._run_parallel(
                    skill_jobs, current_dag, tasks, traces, profiles, budget,
                )
            else:
                skill_results = self._run_sequential(
                    skill_jobs, current_dag, tasks, traces, profiles, budget,
                )

            # 5. Collect results — pick the best successful skill
            best_result: SkillResult | None = None
            for fm_id, result in skill_results:
                fm_name = GROUP_NAMES.get(fm_id, fm_id)
                round_entry = {
                    "round": round_i + 1,
                    "fm_id": fm_id,
                    "fm_name": fm_name,
                    "success": result.success,
                    "inner_iterations": result.inner_iterations,
                    "outer_rounds": result.outer_rounds,
                    "final_fm_rate": result.final_fm_rate,
                    "final_pass_rate": result.final_pass_rate,
                    "n_negatives": len(result.negatives),
                }
                results.append(round_entry)
                self.tracker.log(round_entry, step=len(results) - 1)

                if result.success:
                    print(f"  Group-{fm_id} ({fm_name}): REPAIRED "
                          f"(fm={result.final_fm_rate:.2f}, pass={result.final_pass_rate:.3f})")
                    if best_result is None or result.final_pass_rate > best_result.final_pass_rate:
                        best_result = result
                else:
                    print(f"  Group-{fm_id} ({fm_name}): FAILED after {result.outer_rounds} rounds")
                    # Persist negatives for future runs
                    self.negatives_store.extend(fm_id, result.negatives)
                    self.evolver.record_failure(fm_id)

                    # Meta-evolution check
                    if self.evolver.should_evolve(fm_id):
                        print(f"  Triggering meta-evolution for Skill-{fm_id}...")
                        all_neg = self.negatives_store.load(fm_id)
                        skill = get_skill(fm_id)
                        evolved_path = self.evolver.evolve_skill(fm_id, skill, all_neg)
                        if evolved_path:
                            print(f"    Evolved skill saved: {evolved_path}")
                            load_evolved_skill(fm_id, evolved_path)

            # 6. Update DAG if any skill succeeded
            if best_result and best_result.dag:
                current_dag = best_result.dag
                self.evolver.reset_failures(best_result.fm_id)
                print(f"  Adopting DAG from Skill-{best_result.fm_id}.")
            else:
                print(f"  No successful repairs this round.")

            print()

        # Summary
        summary = {
            "total_rounds": len(set(r["round"] for r in results)),
            "results": results,
        }
        self.tracker.save_local("skill_optimization.json")
        return summary

    def _run_skill(
        self,
        job: dict,
        dag: MASDAG,
        tasks: list[str],
        traces: list,
        profiles: list[FMProfile],
        budget: SkillBudget | None,
    ) -> tuple[str, SkillResult]:
        """Run a single skill job."""
        fm_id = job["fm_id"]
        proposal_idx = job["proposal_idx"]
        validation_idx = job["validation_idx"]
        prior_negatives = job["prior_negatives"]

        # Each skill gets its own DAG copy
        skill_dag = copy.deepcopy(dag)

        try:
            skill = get_skill(fm_id)
        except KeyError:
            return fm_id, SkillResult(success=False, fm_id=fm_id)

        # Inject prior negatives into the skill run
        skill_budget = budget or SkillBudget()

        result = skill.run(
            original_dag=skill_dag,
            proposal_traces=_take_by_index(traces, proposal_idx),
            proposal_profiles=_take_by_index(profiles, proposal_idx),
            proposal_tasks=_take_by_index(tasks, proposal_idx),
            validation_tasks=_take_by_index(tasks, validation_idx),
            runner=self.runner,
            diagnoser=self.diagnoser,
            budget=SkillBudget(
                max_llm_calls=skill_budget.max_llm_calls,
                max_batch_runs=skill_budget.max_batch_runs,
                max_wall_time_s=skill_budget.max_wall_time_s,
            ),
            prior_negatives=prior_negatives,
        )

        return fm_id, result

    def _run_parallel(
        self, jobs, dag, tasks, traces, profiles, budget,
    ) -> list[tuple[str, SkillResult]]:
        """Run multiple skills in parallel using threads."""
        results: list[tuple[str, SkillResult]] = []

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                executor.submit(self._run_skill, job, dag, tasks, traces, profiles, budget): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    fm_id = job["fm_id"]
                    print(f"  Skill-{fm_id} failed with error: {e}")
                    results.append((fm_id, SkillResult(success=False, fm_id=fm_id)))

        return results

    def _run_sequential(
        self, jobs, dag, tasks, traces, profiles, budget,
    ) -> list[tuple[str, SkillResult]]:
        """Run skills sequentially."""
        results: list[tuple[str, SkillResult]] = []
        for job in jobs:
            result = self._run_skill(job, dag, tasks, traces, profiles, budget)
            results.append(result)
        return results
