"""AG2 MathChat × OpenEvolve — Baseline comparison experiment.

Uses SkyDiscover's OpenEvolve (native MAP-Elites) to evolve the AG2 MathChat
YAML DAG, with MAST-style FM rates as the fitness signal.  This is the
"general evolver" baseline to compare against OptPilot's targeted Skill
Workflows.

Aligned with the MAST+OpenEvolve blog:
  - 50 iterations
  - 20 tasks per evaluation
  - Fitness = 1/(1+total_failures), ×1.2 if correct, -0.01 per agent > 4
  - No diagnosis-driven repair — only FM rates as feedback

Usage:
    python -m experiments.run_ag2_mathchat_openevolve \\
        --model openai/gpt-oss-120b \\
        --train 100 --test 100 \\
        --iterations 50 \\
        --concurrency 512 \\
        --timeout 600

    # With pre-computed diagnose results (skips initial diagnose):
    python -m experiments.run_ag2_mathchat_openevolve \\
        --model openai/gpt-oss-120b \\
        --train 100 --test 100 \\
        --iterations 50 \\
        --reuse-diagnose-dir results/.../diagnose
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, RESULTS_DIR, TOGETHER_API_KEY
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import load_online_benchmark_suite, OfficialBenchmarkSuite
from optpilot.modules.runner import OptPilotRunner

# SkyDiscover
from skydiscover.api import run_discovery

DAG_FILE = "ag2_mathchat.yaml"
RESULT_PREFIX = "ag2_mathchat_openevolve"
CONFIG_FILE = Path(__file__).parent / "openevolve_config.yaml"
EVALUATOR_FILE = Path(__file__).parent / "openevolve_evaluator.py"
INITIAL_PROGRAM_FILE = Path(__file__).parent / "openevolve_initial_dag.py"


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


def run(
    dag_path: str | None = None,
    n_train: int = 100,
    n_test: int = 100,
    iterations: int = 50,
    eval_tasks: int = 20,
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 512,
    timeout: int = 600,
    reuse_diagnose_dir: str | None = None,
):
    total = n_train + n_test
    model_short = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_stem = f"{RESULT_PREFIX}_{model_short}_{timestamp}"
    artifact_dir = RESULTS_DIR / f"{result_stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dag_versions_dir = artifact_dir / "dag_versions"
    dag_versions_dir.mkdir(parents=True, exist_ok=True)
    openevolve_output_dir = str(artifact_dir / "openevolve_output")

    # ------------------------------------------------------------------
    # Load benchmarks and split
    # ------------------------------------------------------------------
    print("Loading benchmarks...")
    full_suite = load_online_benchmark_suite(total)
    train_examples, test_examples = _split_suite(full_suite, n_train)
    train_suite = OfficialBenchmarkSuite(train_examples)
    test_suite = OfficialBenchmarkSuite(test_examples)

    # ------------------------------------------------------------------
    # Load baseline DAG
    # ------------------------------------------------------------------
    dag_file = Path(dag_path) if dag_path else DAG_DIR / DAG_FILE
    original_dag = MASDAG.load(dag_file)
    original_dag.save(dag_versions_dir / "input.yaml")

    # Read the Python initial program for OpenEvolve
    initial_program_path = str(INITIAL_PROGRAM_FILE)

    # ------------------------------------------------------------------
    # Set environment variables for the evaluator
    # ------------------------------------------------------------------
    os.environ["OPENEVOLVE_MODEL"] = model
    os.environ["OPENEVOLVE_CONCURRENCY"] = str(concurrency)
    os.environ["OPENEVOLVE_TIMEOUT"] = str(timeout)
    os.environ["OPENEVOLVE_EVAL_TASKS"] = str(eval_tasks)
    os.environ["OPENEVOLVE_TOTAL_TASKS"] = str(total)
    selected_eval_prompts = [example.prompt for example in train_examples[: min(eval_tasks, len(train_examples))]]
    os.environ["OPENEVOLVE_EVAL_PROMPTS_JSON"] = json.dumps(selected_eval_prompts, ensure_ascii=False)

    # Ensure Together AI API key is visible to SkyDiscover
    # SkyDiscover's "together" provider reads: TOGETHER_API_KEY, TOGETHER_AI_API_KEY, together_ai_api
    if TOGETHER_API_KEY and not os.environ.get("TOGETHER_API_KEY"):
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    # ------------------------------------------------------------------
    # Print experiment info
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  AG2 MathChat — OpenEvolve Baseline Experiment")
    print("=" * 65)
    print(f"  Model:          {model}")
    print(f"  DAG:            {original_dag.dag_id}  ({dag_file.name})")
    print(f"  Agents:         {', '.join(original_dag.agent_nodes.keys())}")
    print(f"  Iterations:     {iterations}")
    print(f"  Eval tasks:     {eval_tasks} per iteration")
    print(f"  Concurrency:    {concurrency}")
    print(f"  Timeout:        {timeout}s/task")
    print(f"  Train:          {len(train_examples)} tasks  {dict(train_suite.benchmark_counts())}")
    print(f"  Test:           {len(test_examples)} tasks  {dict(test_suite.benchmark_counts())}")
    print(f"  Output:         {artifact_dir}")
    print("=" * 65)
    print()

    # ------------------------------------------------------------------
    # Run OpenEvolve
    # ------------------------------------------------------------------
    print("Starting OpenEvolve evolution...")
    t0 = time.time()

    # Prefix model with "together/" for SkyDiscover provider resolution
    skydiscover_model = model if "/" not in model or model.startswith("together/") else f"together/{model}"

    result = run_discovery(
        evaluator=str(EVALUATOR_FILE),
        initial_program=initial_program_path,
        model=skydiscover_model,
        iterations=iterations,
        search="openevolve_native",
        config=str(CONFIG_FILE),
        output_dir=openevolve_output_dir,
        cleanup=False,
    )

    evolution_time = time.time() - t0
    print(f"\nOpenEvolve completed in {evolution_time:.1f}s")
    print(f"  Initial score: {result.initial_score}")
    print(f"  Best score:    {result.best_score}")
    print(f"  Best metrics:  {result.metrics}")

    # ------------------------------------------------------------------
    # Parse the best solution (Python code) back into a MASDAG
    # ------------------------------------------------------------------
    best_code = result.best_solution
    # Save the Python source for inspection
    (dag_versions_dir / "openevolve_best.py").write_text(best_code, encoding="utf-8")
    try:
        # Execute the best code to get the DAG dict
        import importlib.util
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(best_code)
            tmp_path = f.name
        spec = importlib.util.spec_from_file_location("_best_dag", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        os.unlink(tmp_path)
        best_dag = MASDAG.from_dict(mod.build_dag())
        best_dag.save(dag_versions_dir / "openevolve_best.yaml")
        print(f"\n  Best DAG agents: {', '.join(best_dag.agent_nodes.keys())}")
    except Exception as e:
        print(f"\nWarning: Could not parse best solution as MASDAG: {e}")
        print("Using original DAG for test evaluation.")
        best_dag = original_dag

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    test_runner = OptPilotRunner(
        dag=original_dag,
        model=model,
        benchmark_name="AG2_MathChat",
        score_fn=test_suite.score_task,
        benchmark_name_resolver=test_suite.benchmark_name_for_task,
        timeout=timeout,
    )

    async def _run_test_evals():
        final = await _eval_on_test(
            test_runner, best_dag, test_suite, concurrency,
            "OPENEVOLVE BEST", output_base=artifact_dir / "test_final",
        )
        baseline = await _eval_on_test(
            test_runner, original_dag, test_suite, concurrency,
            "BASELINE", output_base=artifact_dir / "test_baseline",
        )
        return final, baseline

    test_stats, baseline_test_stats = asyncio.run(_run_test_evals())

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result_path = RESULTS_DIR / f"{result_stem}.json"
    output = {
        "experiment": "ag2_mathchat_openevolve",
        "model": model,
        "dag": original_dag.dag_id,
        "artifacts_dir": str(artifact_dir),
        "dag_versions_dir": str(dag_versions_dir),
        "n_train": n_train,
        "n_test": n_test,
        "iterations": iterations,
        "eval_tasks_per_iteration": eval_tasks,
        "concurrency": concurrency,
        "timeout_s": timeout,
        "evolution": {
            "initial_score": result.initial_score,
            "best_score": result.best_score,
            "best_metrics": result.metrics,
            "elapsed_s": evolution_time,
            "output_dir": openevolve_output_dir,
        },
        "test_baseline": baseline_test_stats,
        "test_final": test_stats,
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {result_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  Summary — OpenEvolve ({iterations} iterations)")
    print(f"{'=' * 65}")
    print(f"  Evolution time:   {evolution_time:.1f}s")
    print(f"  Initial score:    {result.initial_score}")
    print(f"  Best score:       {result.best_score:.4f}")
    baseline_acc = baseline_test_stats["overall_accuracy"]
    final_acc = test_stats["overall_accuracy"]
    print(f"  Test accuracy:    {baseline_acc:.3f} → {final_acc:.3f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AG2 MathChat OpenEvolve Baseline on AIME2024/2025 / OlympiadBench / MMLU",
    )
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML")
    parser.add_argument("--train", type=int, default=100, help="Train set size")
    parser.add_argument("--test", type=int, default=100, help="Test set size")
    parser.add_argument("--iterations", type=int, default=50, help="OpenEvolve iterations")
    parser.add_argument("--eval-tasks", type=int, default=20, help="Tasks per evaluation (default 20, aligned with MAST blog)")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model ID on Together AI")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    parser.add_argument("--reuse-diagnose-dir", default=None, help="(unused in OpenEvolve mode, kept for CLI compat)")
    args = parser.parse_args()

    run(
        dag_path=args.dag,
        n_train=args.train,
        n_test=args.test,
        iterations=args.iterations,
        eval_tasks=args.eval_tasks,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        reuse_diagnose_dir=args.reuse_diagnose_dir,
    )
