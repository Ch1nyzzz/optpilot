"""Exp 1: Targeted Repair vs Baselines.

Compares: Original, Random Repair, OptPilot cold.

Usage:
    python -m experiments.exp1_targeted_repair --dag dags/chatdev.yaml --tasks 3 --rounds 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, LIBRARY_DIR, RESULTS_DIR
from optpilot.dag.core import MASDAG
from optpilot.models import RepairAction, RepairType
from optpilot.modules.runner import OptPilotRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.orchestrator import Orchestrator, get_highest_frequency_fm
from optpilot.tracking import Tracker

SAMPLE_TASKS = [
    "Write a Python program that implements a simple calculator with GUI using tkinter.",
    "Create a Python to-do list application with file-based storage.",
    "Build a Python text-based adventure game with rooms and items.",
]


def run_baseline_original(runner: OptPilotRunner, tasks: list[str], diagnoser: Diagnoser) -> dict:
    """Baseline: run original MAS without modification."""
    print("=== Baseline: Original ===")
    traces = runner.run_batch(tasks)
    profiles = diagnoser.diagnose_batch(traces)

    fm_counts = {}
    for p in profiles:
        for fm_id in p.active_fm_ids():
            fm_counts[fm_id] = fm_counts.get(fm_id, 0) + 1

    return {"method": "original", "fm_counts": fm_counts, "n_traces": len(traces)}


def run_baseline_random(
    runner: OptPilotRunner, dag: MASDAG, tasks: list[str], diagnoser: Diagnoser, n_rounds: int = 3
) -> dict:
    """Baseline: random DAG modifications with same budget."""
    print("=== Baseline: Random Repair ===")
    current_dag = dag

    random_actions = [
        RepairAction(
            repair_type=RepairType.CONFIG_CHANGE,
            target="Code Review Phase Loop Counter",
            description="Randomly change code review max iterations",
            details={"max_iterations": random.choice([3, 5, 8, 15])},
        ),
        RepairAction(
            repair_type=RepairType.CONFIG_CHANGE,
            target="Test Phase Loop Counter",
            description="Randomly change test max iterations",
            details={"max_iterations": random.choice([1, 2, 5])},
        ),
    ]

    for _ in range(n_rounds):
        action = random.choice(random_actions)
        current_dag = current_dag.apply_repair(action)

    traces = runner.run_batch(tasks, dag=current_dag)
    profiles = diagnoser.diagnose_batch(traces)

    fm_counts = {}
    for p in profiles:
        for fm_id in p.active_fm_ids():
            fm_counts[fm_id] = fm_counts.get(fm_id, 0) + 1

    return {"method": "random", "fm_counts": fm_counts, "n_traces": len(traces), "n_rounds": n_rounds}


def run_optpilot_cold(dag: MASDAG, tasks: list[str], n_rounds: int = 3, target_group: str | None = None) -> dict:
    """OptPilot with empty library (cold start)."""
    print("=== OptPilot Cold ===")
    runner = OptPilotRunner(dag=dag)
    orchestrator = Orchestrator(runner=runner, dag=dag)
    summary = orchestrator.optimize(tasks=tasks, max_rounds=n_rounds, target_fm=target_group)
    return {"method": "optpilot_cold", **summary}


def run_experiment(dag_path: str | None = None, n_tasks: int = 3, n_rounds: int = 3, target_group: str | None = None):
    """Run all methods and compare."""
    dag_file = Path(dag_path) if dag_path else DAG_DIR / "chatdev.yaml"
    dag = MASDAG.load(dag_file)

    tasks = SAMPLE_TASKS[:n_tasks]
    runner = OptPilotRunner(dag=dag)
    diagnoser = Diagnoser()

    results = []

    # 1. Original
    r1 = run_baseline_original(runner, tasks, diagnoser)
    results.append(r1)
    print(f"  Group counts: {r1['fm_counts']}\n")

    # 2. Random
    r2 = run_baseline_random(runner, dag, tasks, diagnoser, n_rounds)
    results.append(r2)
    print(f"  Group counts: {r2['fm_counts']}\n")

    # 3. OptPilot cold
    r3 = run_optpilot_cold(dag, tasks, n_rounds, target_group)
    results.append(r3)
    print()

    # Save
    out_path = RESULTS_DIR / "exp1_targeted_repair.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML file")
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--group", default=None)
    args = parser.parse_args()
    run_experiment(args.dag, args.tasks, args.rounds, args.group)
