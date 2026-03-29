"""AG2 MathChat × Official Benchmarks — Skill Workflow Optimization.

Optimizes on train set, evaluates final DAG on held-out test set.

Usage:
    python -m experiments.run_ag2_mathchat_skill --model openai/gpt-oss-120b
    python -m experiments.run_ag2_mathchat_skill --model openai/gpt-oss-120b --group E
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, RESULTS_DIR, SHADOW_EVAL_INTERVAL
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import load_online_benchmark_suite, OfficialBenchmarkSuite
from optpilot.models import SkillBudget
from optpilot.modules.runner import OptPilotRunner
from optpilot.orchestrator import Orchestrator

DAG_FILE = "ag2_mathchat.yaml"
RESULT_PREFIX = "ag2_mathchat_skill"


def _split_suite(suite: OfficialBenchmarkSuite, n_train: int) -> tuple[list, list]:
    """Split examples into train/test proportionally per benchmark."""
    by_bench: dict[str, list] = defaultdict(list)
    for ex in suite.examples:
        by_bench[ex.benchmark_name].append(ex)

    total = len(suite.examples)
    train_examples, test_examples = [], []
    for bname, examples in sorted(by_bench.items()):
        bench_train = round(len(examples) * n_train / total)
        bench_train = min(bench_train, len(examples))
        train_examples.extend(examples[:bench_train])
        test_examples.extend(examples[bench_train:])
    return train_examples, test_examples


async def _eval_on_test(
    runner: OptPilotRunner,
    dag: MASDAG,
    test_suite: OfficialBenchmarkSuite,
    concurrency: int,
    label: str,
    output_base: str | Path | None = None,
):
    """Evaluate a DAG on the test set and print per-benchmark results."""
    tasks = test_suite.tasks()
    print(f"\n--- {label}: evaluating on {len(tasks)} test tasks ---")
    t0 = time.time()
    traces = await runner.arun_batch(
        tasks,
        dag=dag,
        output_base=output_base,
        max_concurrency=concurrency,
    )
    elapsed = time.time() - t0

    per_bench: dict[str, list[float]] = defaultdict(list)
    details = []
    for trace in traces:
        per_bench[trace.benchmark_name].append(trace.task_score)
        details.append({
            "benchmark": trace.benchmark_name,
            "score": trace.task_score,
            "success": trace.task_success,
            "latency_s": trace.latency_s,
            "task_key": trace.task_key,
            "trace_path": trace.trace_path,
        })

    all_scores = [t.task_score for t in traces]
    overall_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"  {label} Test Results:")
    for bname, scores in sorted(per_bench.items()):
        acc = sum(scores) / len(scores) if scores else 0.0
        correct = sum(1 for s in scores if s > 0)
        print(f"    {bname:20s}  {correct}/{len(scores)}  acc={acc:.3f}")
    total_correct = sum(1 for s in all_scores if s > 0)
    print(f"    {'OVERALL':20s}  {total_correct}/{len(all_scores)}  acc={overall_acc:.3f}")
    print(f"    Elapsed: {elapsed:.1f}s")

    return {
        "overall_accuracy": overall_acc,
        "elapsed_s": elapsed,
        "per_benchmark": {
            bname: {
                "n": len(scores),
                "correct": sum(1 for s in scores if s > 0),
                "accuracy": sum(scores) / len(scores) if scores else 0.0,
            }
            for bname, scores in sorted(per_bench.items())
        },
        "details": details,
    }


async def run(
    dag_path: str | None = None,
    n_train: int = 100,
    n_test: int = 100,
    max_rounds: int = 5,
    eval_tasks: int | None = None,
    target_group: str | None = None,
    use_wandb: bool = False,
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 512,
    timeout: int = 600,
    reuse_traces_dir: str | None = None,
    reuse_diagnose_dir: str | None = None,
    clear_negatives: bool = False,
):
    total = n_train + n_test
    model_short = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_tag = f"_group{target_group}" if target_group else ""
    result_stem = f"{RESULT_PREFIX}_{model_short}{group_tag}_{timestamp}"
    artifact_dir = RESULTS_DIR / f"{result_stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dag_versions_dir = artifact_dir / "dag_versions"
    dag_versions_dir.mkdir(parents=True, exist_ok=True)

    # Load and split benchmarks
    print("Loading benchmarks...")
    full_suite = load_online_benchmark_suite(total)
    train_examples, test_examples = _split_suite(full_suite, n_train)
    train_suite = OfficialBenchmarkSuite(train_examples)
    test_suite = OfficialBenchmarkSuite(test_examples)

    dag_file = Path(dag_path) if dag_path else DAG_DIR / DAG_FILE
    dag = MASDAG.load(dag_file)
    dag.save(dag_versions_dir / "input.yaml")

    # Runner scored against train suite (for optimization)
    train_runner = OptPilotRunner(
        dag=dag,
        model=model,
        benchmark_name="AG2_MathChat",
        score_fn=train_suite.score_task,
        benchmark_name_resolver=train_suite.benchmark_name_for_task,
        timeout=timeout,
    )

    # Separate runner scored against test suite (for evaluation)
    test_runner = OptPilotRunner(
        dag=dag,
        model=model,
        benchmark_name="AG2_MathChat",
        score_fn=test_suite.score_task,
        benchmark_name_resolver=test_suite.benchmark_name_for_task,
        timeout=timeout,
    )

    orchestrator = Orchestrator(
        runner=train_runner,
        dag=dag,
        use_wandb=use_wandb,
    )
    if clear_negatives:
        removed = orchestrator.negatives_store.clear_all()
        print(f"Cleared {removed} persisted negatives before optimization.")

    print("=" * 65)
    print("  AG2 MathChat — Skill Workflow Optimization")
    print("=" * 65)
    print(f"  Model:       {model}")
    print(f"  DAG:         {dag.dag_id}  ({dag_file.name})")
    print(f"  Agents:      {', '.join(dag.agent_nodes.keys())}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Timeout:     {timeout}s/task")
    print(f"  Train:       {len(train_examples)} tasks  {dict(train_suite.benchmark_counts())}")
    print(f"  Test:        {len(test_examples)} tasks  {dict(test_suite.benchmark_counts())}")
    print(f"  Rounds:      {max_rounds}")
    if eval_tasks:
        print(f"  Eval/round:  {eval_tasks} tasks (balanced sampled)")
        if SHADOW_EVAL_INTERVAL > 0:
            print(f"  Shadow gate: every {SHADOW_EVAL_INTERVAL} round(s)")
    if target_group:
        print(f"  Target FM:   Group-{target_group}")
    if reuse_diagnose_dir:
        print(f"  Reuse Diagnose: {reuse_diagnose_dir}")
    elif reuse_traces_dir:
        print(f"  Reuse Traces:   {reuse_traces_dir}")
    print("=" * 65)
    print()

    # Run optimization on train set
    train_tasks = train_suite.tasks()
    if reuse_diagnose_dir:
        summary = await orchestrator.aoptimize_from_diagnose(
            tasks=train_tasks,
            diagnose_dir=reuse_diagnose_dir,
            target_fm=target_group,
            budget=SkillBudget(max_llm_calls=100, max_batch_runs=100, max_wall_time_s=3600),
            concurrency=concurrency,
            dag_output_base=dag_versions_dir / "optimization",
        )
    elif reuse_traces_dir:
        summary = await orchestrator.aoptimize_from_traces(
            tasks=train_tasks,
            trace_base=reuse_traces_dir,
            target_fm=target_group,
            budget=SkillBudget(max_llm_calls=100, max_batch_runs=100, max_wall_time_s=3600),
            concurrency=concurrency,
            trace_output_base=artifact_dir / "optimization",
            dag_output_base=dag_versions_dir / "optimization",
        )
    else:
        summary = await orchestrator.aoptimize(
            tasks=train_tasks,
            max_rounds=max_rounds,
            target_fm=target_group,
            budget=SkillBudget(max_llm_calls=100, max_batch_runs=100, max_wall_time_s=3600),
            concurrency=concurrency,
            trace_output_base=artifact_dir / "optimization",
            dag_output_base=dag_versions_dir / "optimization",
            eval_tasks_per_round=eval_tasks,
        )

    # Evaluate final DAG on held-out test set
    final_dag = orchestrator.dag  # updated by aoptimize if any repair succeeded
    final_dag.save(dag_versions_dir / "optimized_final.yaml")
    test_stats = await _eval_on_test(
        test_runner,
        final_dag,
        test_suite,
        concurrency,
        "FINAL",
        output_base=artifact_dir / "test_final",
    )

    # Also evaluate original DAG on test for comparison
    original_dag = MASDAG.load(dag_file)
    original_dag.save(dag_versions_dir / "baseline_reference.yaml")
    baseline_test_stats = await _eval_on_test(
        test_runner,
        original_dag,
        test_suite,
        concurrency,
        "BASELINE",
        output_base=artifact_dir / "test_baseline",
    )

    # Save results
    result_path = RESULTS_DIR / f"{result_stem}.json"

    output = {
        "experiment": "ag2_mathchat_skill_workflow",
        "model": model,
        "dag": dag.dag_id,
        "artifacts_dir": str(artifact_dir),
        "dag_versions_dir": str(dag_versions_dir),
        "n_train": n_train,
        "n_test": n_test,
        "max_rounds": max_rounds,
        "eval_tasks_per_round": eval_tasks,
        "shadow_eval_interval": SHADOW_EVAL_INTERVAL,
        "target_group": target_group,
        "concurrency": concurrency,
        "timeout_s": timeout,
        "reuse_traces_dir": reuse_traces_dir,
        "reuse_diagnose_dir": reuse_diagnose_dir,
        "clear_negatives": clear_negatives,
        "optimization": summary,
        "test_baseline": baseline_test_stats,
        "test_final": test_stats,
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {result_path}")

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  Summary — {summary['total_rounds']} optimization round(s)")
    print(f"{'=' * 65}")
    for r in summary["results"]:
        status = "REPAIRED" if r["success"] else "FAILED"
        print(
            f"  Group-{r['fm_id']} ({r['fm_name']}): {status}  "
            f"fm={r['final_fm_rate']:.2f}  pass={r['final_pass_rate']:.3f}"
        )
    print(f"\n  Test accuracy: {baseline_test_stats['overall_accuracy']:.3f} → {test_stats['overall_accuracy']:.3f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AG2 MathChat Skill Workflow on AIME2024/2025 / OlympiadBench / MMLU",
    )
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML")
    parser.add_argument("--train", type=int, default=100, help="Train set size")
    parser.add_argument("--test", type=int, default=100, help="Test set size")
    parser.add_argument("--rounds", type=int, default=50, help="Max optimization rounds")
    parser.add_argument("--eval-tasks", type=int, default=None, help="Tasks per round (subsample from train set, default: all)")
    parser.add_argument("--group", default=None, help="Target a specific FM group (A-F)")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model ID on Together AI")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    parser.add_argument("--reuse-traces-dir", default=None, help="Reuse persisted train traces instead of rerunning train")
    parser.add_argument("--reuse-diagnose-dir", default=None, help="Reuse persisted diagnose artifacts instead of rerunning train/diagnose")
    parser.add_argument("--clear-negatives", action="store_true", help="Delete persisted negatives before optimization")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()

    if args.reuse_traces_dir and args.reuse_diagnose_dir:
        parser.error("--reuse-traces-dir and --reuse-diagnose-dir are mutually exclusive")

    asyncio.run(run(
        dag_path=args.dag,
        n_train=args.train,
        n_test=args.test,
        max_rounds=args.rounds,
        eval_tasks=args.eval_tasks,
        target_group=args.group,
        use_wandb=args.wandb,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        reuse_traces_dir=args.reuse_traces_dir,
        reuse_diagnose_dir=args.reuse_diagnose_dir,
        clear_negatives=args.clear_negatives,
    ))
