"""Online Pipeline - Phase B: Run MAS → Diagnose → Optimize → Verify → Distill.

Usage:
    python -m experiments.online_pipeline --dag dags/chatdev.yaml --tasks 3 --rounds 3 --fm 1.3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR
from optpilot.dag.core import MASDAG
from optpilot.modules.runner import OptPilotRunner
from optpilot.orchestrator import Orchestrator

# Sample ProgramDev tasks (from MAST-Data ChatDev traces)
SAMPLE_TASKS = [
    "Write a Python program that implements a simple calculator with GUI using tkinter. Support add, subtract, multiply, divide.",
    "Create a Python program that reads a CSV file, processes the data, and generates a summary report with statistics.",
    "Build a Python program that implements a to-do list application with file-based storage. Support add, delete, list, and mark-complete.",
    "Write a Python program that implements a simple text-based adventure game with multiple rooms and items.",
    "Create a Python program that implements a basic web scraper that extracts article titles and URLs from a news website.",
]


def run_online_pipeline(
    dag_path: str | None = None,
    n_tasks: int = 3,
    max_rounds: int = 3,
    target_fm: str | None = None,
    use_wandb: bool = False,
):
    """Run the full online optimization pipeline."""
    tasks = SAMPLE_TASKS[:n_tasks]

    dag_file = Path(dag_path) if dag_path else DAG_DIR / "chatdev.yaml"
    dag = MASDAG.load(dag_file)

    runner = OptPilotRunner(dag=dag)
    orchestrator = Orchestrator(runner=runner, dag=dag, use_wandb=use_wandb)

    print(f"=== OptPilot Online Pipeline ===")
    print(f"DAG: {dag.dag_id}, Tasks: {n_tasks}, Max rounds: {max_rounds}")
    if target_fm:
        print(f"Target FM: FM-{target_fm}")
    print()

    summary = orchestrator.optimize(
        tasks=tasks,
        max_rounds=max_rounds,
        target_fm=target_fm,
    )

    print(f"\n=== Summary ===")
    print(f"Rounds completed: {summary['total_rounds']}")
    print(f"Library: {summary['library_stats']}")
    for r in summary["results"]:
        print(
            f"  Round {r['round']}: FM-{r['target_fm']} {r['before_count']}→{r['after_count']}, "
            f"pass {r['before_pass_rate']:.3f}→{r['after_pass_rate']:.3f}, "
            f"runtime {r['before_runtime_s']:.2f}s→{r['after_runtime_s']:.2f}s "
            f"({r['repair_status']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptPilot Online Pipeline")
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML file")
    parser.add_argument("--tasks", type=int, default=3, help="Number of tasks")
    parser.add_argument("--rounds", type=int, default=3, help="Max optimization rounds")
    parser.add_argument("--fm", default=None, help="Target specific FM id")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()

    run_online_pipeline(
        dag_path=args.dag, n_tasks=args.tasks, max_rounds=args.rounds,
        target_fm=args.fm, use_wandb=args.wandb,
    )
