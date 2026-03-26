"""Orchestrator - controls the online optimization loop.

Coordinates: Runner → Diagnoser → Optimizer → Runner (verify) → Distiller
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import TypeVar

from optpilot.config import LIBRARY_DIR
from optpilot.dag.core import MASDAG
from optpilot.library.repair_library import RepairLibrary
from optpilot.models import FMProfile
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.distiller import Distiller
from optpilot.modules.optimizer import Optimizer
from optpilot.tracking import Tracker


def get_highest_frequency_fm(profiles: list[FMProfile], min_support: int = 1) -> str | None:
    """Find the most frequent active FM across profiles."""
    fm_counts: Counter[str] = Counter()
    for p in profiles:
        for fm_id in p.active_fm_ids():
            fm_counts[fm_id] += 1
    eligible = [(fm_id, count) for fm_id, count in fm_counts.items() if count >= min_support]
    if not eligible:
        return None
    eligible.sort(key=lambda item: item[1], reverse=True)
    return eligible[0][0]


def _mean_metric(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return mean(present) if present else 0.0


def _pass_rate(traces: list) -> float:
    values = [1.0 if trace.task_success else 0.0 for trace in traces if trace.task_success is not None]
    return mean(values) if values else 0.0


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
    """Online optimization loop controller."""

    def __init__(
        self,
        runner: MASRunner,
        dag: MASDAG,
        library_path: str | Path | None = None,
        use_wandb: bool = False,
    ):
        self.runner = runner
        self.dag = dag
        self.library = RepairLibrary(library_path or LIBRARY_DIR / "online_library.json")
        self.diagnoser = Diagnoser()
        self.optimizer = Optimizer(self.library)
        self.distiller = Distiller(self.library)
        self.tracker = Tracker("online_optimization", use_wandb=use_wandb)
        self.touched_fms: set[str] = set()

    def optimize(
        self,
        tasks: list[str],
        max_rounds: int = 5,
        target_fm: str | None = None,
    ) -> dict:
        """Run the full online optimization loop.

        Args:
            tasks: List of task prompts to run.
            max_rounds: Maximum optimization rounds.
            target_fm: If set, only target this specific FM.

        Returns:
            Summary dict with results.
        """
        current_dag = self.dag

        print(f"=== OptPilot Online Optimization ===")
        print(f"Tasks: {len(tasks)}, Max rounds: {max_rounds}")
        print()

        results = []
        for round_i in range(max_rounds):
            print(f"=== Round {round_i + 1}/{max_rounds} ===")

            # 1. Run MAS
            print(f"  [1/6] Running MAS on {len(tasks)} tasks...")
            traces = self.runner.run_batch(tasks, dag=current_dag)

            # 2. Diagnose
            print(f"  [2/6] Diagnosing traces...")
            profiles = self.diagnoser.diagnose_batch(traces)

            # 3. Find target FM
            if target_fm:
                top_fm = target_fm
                fm_count = sum(1 for p in profiles if top_fm in p.active_fm_ids())
            else:
                top_fm = get_highest_frequency_fm(profiles, min_support=2)
                fm_count = sum(1 for p in profiles if top_fm and top_fm in p.active_fm_ids())

            if not top_fm or fm_count == 0:
                print(f"  No active FMs found. Optimization complete!")
                break
            if fm_count < 2:
                print(f"  FM-{top_fm} appears in only {fm_count} trace. Need at least 2 for holdout validation.")
                break

            print(f"  Target FM: FM-{top_fm} ({fm_count}/{len(profiles)} traces)")

            split = split_proposal_validation_indices(top_fm, profiles)
            if split is None:
                print(f"  Unable to build disjoint proposal/validation splits for FM-{top_fm}.")
                break
            proposal_indices, validation_indices = split

            proposal_traces = _take_by_index(traces, proposal_indices)
            proposal_profiles = _take_by_index(profiles, proposal_indices)
            validation_tasks = _take_by_index(tasks, validation_indices)
            before_validation_traces = _take_by_index(traces, validation_indices)
            before_validation_profiles = _take_by_index(profiles, validation_indices)

            validation_fm_count = sum(
                1 for profile in before_validation_profiles if top_fm in profile.active_fm_ids()
            )

            # 4. Generate repair from proposal subset
            print(
                f"  [3/6] Generating repair from {len(proposal_traces)} proposal traces "
                f"(holdout positives: {validation_fm_count})..."
            )
            candidate = self.optimizer.generate_repair(
                top_fm,
                proposal_profiles,
                proposal_traces,
                current_dag,
            )
            self.touched_fms.add(top_fm)
            print(f"    Repair: {candidate.description}")

            # 5. Apply repair and re-run validation split only
            print(f"  [4/6] Applying repair on holdout validation tasks...")
            repaired_dag = current_dag
            for action in candidate.actions:
                repaired_dag = repaired_dag.apply_repair(action)

            new_traces = self.runner.run_batch(validation_tasks, dag=repaired_dag)
            for idx, trace in zip(validation_indices, new_traces, strict=False):
                trace.trace_id = idx
            new_profiles = self.diagnoser.diagnose_batch(new_traces)

            # 6. Distill on holdout results
            print(f"  [5/6] Distilling holdout results...")
            entry = self.distiller.distill_online(
                top_fm,
                candidate,
                before_validation_traces,
                before_validation_profiles,
                new_traces,
                new_profiles,
            )

            new_fm_count = sum(1 for p in new_profiles if top_fm in p.active_fm_ids())
            before_pass_rate = _pass_rate(before_validation_traces)
            after_pass_rate = _pass_rate(new_traces)
            before_runtime = _mean_metric([trace.latency_s for trace in before_validation_traces])
            after_runtime = _mean_metric([trace.latency_s for trace in new_traces])
            print(
                f"  [6/6] Holdout result: FM-{top_fm} {validation_fm_count} → {new_fm_count}, "
                f"pass {before_pass_rate:.3f} → {after_pass_rate:.3f}, "
                f"runtime {before_runtime:.2f}s → {after_runtime:.2f}s ({entry.status})"
            )

            round_result = {
                "round": round_i + 1,
                "target_fm": top_fm,
                "proposal_size": len(proposal_indices),
                "validation_size": len(validation_indices),
                "before_count": validation_fm_count,
                "after_count": new_fm_count,
                "before_pass_rate": before_pass_rate,
                "after_pass_rate": after_pass_rate,
                "before_runtime_s": before_runtime,
                "after_runtime_s": after_runtime,
                "repair_status": entry.status,
                "repair_description": candidate.description,
                "validation_metrics": entry.validation_metrics,
            }
            results.append(round_result)
            self.tracker.log(round_result, step=round_i)

            # Update DAG if repair was successful
            if entry.status == "validated":
                current_dag = repaired_dag
                print(f"  Repair applied successfully! Using updated DAG.")
            else:
                print(f"  Repair failed. Keeping original DAG.")

            print()

        # Summary
        if self.touched_fms:
            from optpilot.modules.wrap_up import WrapUp

            wrap_up = WrapUp(self.library)
            print(f"=== Wrap-up ===")
            for fm_id in sorted(self.touched_fms):
                wrapped = wrap_up.wrap_fm(fm_id, source_mas=current_dag.dag_id or "OptPilot")
                print(f"  FM-{fm_id}: wrapped {len(wrapped)} canonical skills")
        self.library.flush()
        summary = {
            "total_rounds": len(results),
            "library_stats": self.library.get_stats(),
            "results": results,
        }
        self.tracker.save_local("online_optimization.json")
        return summary
