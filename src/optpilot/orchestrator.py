"""Orchestrator - controls the online optimization loop.

Coordinates: Runner → Diagnoser → Optimizer → Runner (verify) → Distiller
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from optpilot.config import LIBRARY_DIR
from optpilot.dag.core import MASDAG
from optpilot.library.repair_library import RepairLibrary
from optpilot.models import FMProfile
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.distiller import Distiller
from optpilot.modules.optimizer import Optimizer
from optpilot.tracking import Tracker


def get_highest_frequency_fm(profiles: list[FMProfile]) -> str | None:
    """Find the most frequent active FM across profiles."""
    fm_counts: Counter[str] = Counter()
    for p in profiles:
        for fm_id in p.active_fm_ids():
            fm_counts[fm_id] += 1
    if not fm_counts:
        return None
    return fm_counts.most_common(1)[0][0]


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
            print(f"  [1/5] Running MAS on {len(tasks)} tasks...")
            traces = self.runner.run_batch(tasks, dag=current_dag)

            # 2. Diagnose
            print(f"  [2/5] Diagnosing traces...")
            profiles = self.diagnoser.diagnose_batch(traces)

            # 3. Find target FM
            if target_fm:
                top_fm = target_fm
                fm_count = sum(1 for p in profiles if top_fm in p.active_fm_ids())
            else:
                top_fm = get_highest_frequency_fm(profiles)
                fm_count = sum(1 for p in profiles if top_fm and top_fm in p.active_fm_ids())

            if not top_fm or fm_count == 0:
                print(f"  No active FMs found. Optimization complete!")
                break

            print(f"  Target FM: FM-{top_fm} ({fm_count}/{len(profiles)} traces)")

            # 4. Generate repair
            print(f"  [3/5] Generating repair...")
            fm_trace = next(t for t, p in zip(traces, profiles) if top_fm in p.active_fm_ids())
            fm_profile = next(p for p in profiles if top_fm in p.active_fm_ids())

            candidate = self.optimizer.generate_repair(top_fm, fm_profile, fm_trace, current_dag)
            print(f"    Repair: {candidate.description}")

            # 5. Apply repair and re-run
            print(f"  [4/5] Applying repair and re-running...")
            repaired_dag = current_dag
            for action in candidate.actions:
                repaired_dag = repaired_dag.apply_repair(action)

            new_traces = self.runner.run_batch(tasks, dag=repaired_dag)
            new_profiles = self.diagnoser.diagnose_batch(new_traces)

            # 6. Distill
            print(f"  [5/5] Distilling results...")
            entry = self.distiller.distill_online(
                top_fm, candidate, profiles, new_profiles
            )

            new_fm_count = sum(1 for p in new_profiles if top_fm in p.active_fm_ids())
            print(f"  Result: FM-{top_fm} {fm_count} → {new_fm_count} ({entry.status})")

            round_result = {
                "round": round_i + 1,
                "target_fm": top_fm,
                "before_count": fm_count,
                "after_count": new_fm_count,
                "repair_status": entry.status,
                "repair_description": candidate.description,
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
        summary = {
            "total_rounds": len(results),
            "library_stats": self.library.get_stats(),
            "results": results,
        }
        self.tracker.save_local("online_optimization.json")
        return summary
