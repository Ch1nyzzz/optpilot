"""AG2 MathChat × Official Benchmarks — Skill Workflow Optimization.

┌─────────────────────────────────────────────────────────────────────┐
│  Framework:   AG2 MathChat (3-agent GroupChat)                      │
│               Agent_Problem_Solver + Agent_Code_Executor            │
│               + Agent_Verifier                                      │
│  DAG:         dags/ag2_mathchat.yaml                                │
│  Model:       MiniMax M2.5 (via Together AI)                        │
│  Taxonomy:    6-group FM (A-F), merged from 14 MAST failure modes   │
│  Benchmarks:  MMLU · AIME 2025 · OlympiadBench                     │
│  Pipeline:    Skill Workflows (analyze → evolve → validate →        │
│               reflect), one per FM group, parallel execution        │
└─────────────────────────────────────────────────────────────────────┘

Usage:
    # Official benchmarks (MMLU + AIME + OlympiadBench, auto-scored)
    python -m experiments.run_ag2_mathchat_skill --tasks 9 --rounds 3

    # Target a specific FM group
    python -m experiments.run_ag2_mathchat_skill --tasks 9 --rounds 5 --group B

    # Sequential execution (for debugging)
    python -m experiments.run_ag2_mathchat_skill --tasks 6 --rounds 2 --no-parallel
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, RESULTS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import load_online_benchmark_suite
from optpilot.models import SkillBudget
from optpilot.modules.runner import OptPilotRunner
from optpilot.orchestrator import Orchestrator

import optpilot.skills  # noqa: F401  — trigger @register_skill for A-F

# ── Experiment configuration ──────────────────────────────────────────

DAG_FILE = "ag2_mathchat.yaml"          # 3-agent GroupChat from MAST Appendix L
RESULT_PREFIX = "ag2_mathchat_skill"    # result filename prefix


def run(
    dag_path: str | None = None,
    n_tasks: int = 9,
    max_rounds: int = 5,
    target_group: str | None = None,
    use_wandb: bool = False,
    parallel: bool = True,
):
    """Run AG2 MathChat optimization with Skill Workflows on official benchmarks.

    Benchmark composition (deterministic 3-way split):
      - MMLU:          multi-choice knowledge questions
      - AIME 2025:     competition math (integer answers)
      - OlympiadBench: open-ended math (numerical / expression)

    Each benchmark has a ground-truth scorer so `pass_rate` is real accuracy,
    not a proxy.
    """
    # ── Load benchmarks ──
    suite = load_online_benchmark_suite(n_tasks)
    tasks = suite.tasks()

    # ── Load DAG ──
    dag_file = Path(dag_path) if dag_path else DAG_DIR / DAG_FILE
    dag = MASDAG.load(dag_file)

    # ── Build runner with benchmark scorer ──
    runner = OptPilotRunner(
        dag=dag,
        benchmark_name="AG2_MathChat",
        score_fn=suite.score_task,
        benchmark_name_resolver=suite.benchmark_name_for_task,
    )

    # ── Build orchestrator ──
    orchestrator = Orchestrator(
        runner=runner,
        dag=dag,
        use_wandb=use_wandb,
        parallel=parallel,
    )

    # ── Print experiment config ──
    bench_counts = suite.benchmark_counts()
    print("=" * 65)
    print("  AG2 MathChat × Official Benchmarks — Skill Workflow")
    print("=" * 65)
    print(f"  DAG:        {dag.dag_id}  ({dag_file.name})")
    print(f"  Agents:     {', '.join(dag.agent_nodes.keys())}")
    print(f"  Tasks:      {n_tasks} total")
    for bname, bcount in bench_counts.items():
        print(f"              · {bname}: {bcount}")
    print(f"  Rounds:     {max_rounds}")
    print(f"  Parallel:   {parallel}")
    print(f"  Taxonomy:   6-group (A-F)")
    if target_group:
        print(f"  Target FM:  Group-{target_group}")
    print("=" * 65)
    print()

    # ── Run optimization ──
    summary = orchestrator.optimize(
        tasks=tasks,
        max_rounds=max_rounds,
        target_fm=target_group,
        budget=SkillBudget(max_batch_runs=10, max_wall_time_s=600),
    )

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_tag = f"_group{target_group}" if target_group else ""
    result_path = RESULTS_DIR / f"{RESULT_PREFIX}{group_tag}_{timestamp}.json"

    output = {
        "experiment": "ag2_mathchat_skill_workflow",
        "dag": dag.dag_id,
        "dag_file": str(dag_file),
        "benchmarks": bench_counts,
        "n_tasks": n_tasks,
        "max_rounds": max_rounds,
        "target_group": target_group,
        "parallel": parallel,
        **summary,
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {result_path}")

    # ── Print summary ──
    print(f"\n{'=' * 65}")
    print(f"  Summary — {summary['total_rounds']} round(s)")
    print(f"{'=' * 65}")
    for r in summary["results"]:
        status = "✓ REPAIRED" if r["success"] else "✗ FAILED"
        print(
            f"  Group-{r['fm_id']} ({r['fm_name']}): {status}  "
            f"fm={r['final_fm_rate']:.2f}  pass={r['final_pass_rate']:.3f}  "
            f"iters={r['inner_iterations']}  rounds={r['outer_rounds']}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AG2 MathChat Skill Workflow on MMLU / AIME / OlympiadBench",
    )
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML (default: ag2_mathchat.yaml)")
    parser.add_argument("--tasks", type=int, default=9, help="Total tasks (split across 3 benchmarks)")
    parser.add_argument("--rounds", type=int, default=5, help="Max optimization rounds")
    parser.add_argument("--group", default=None, help="Target a specific FM group (A-F)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    parser.add_argument("--no-parallel", action="store_true", help="Run skills sequentially")
    args = parser.parse_args()

    run(
        dag_path=args.dag,
        n_tasks=args.tasks,
        max_rounds=args.rounds,
        target_group=args.group,
        use_wandb=args.wandb,
        parallel=not args.no_parallel,
    )
