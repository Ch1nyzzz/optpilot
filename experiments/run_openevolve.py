"""Multi-topology OpenEvolve cold-start experiment.

Evolves MAS DAG topologies via SkyDiscover's OpenEvolve (MAP-Elites).
Supports multiple topologies and their corresponding benchmarks:

  - ag2:        AG2 MathChat (3-agent linear) on MMLU/AIME/OlympiadBench
  - appworld:   AppWorld Star (Supervisor + specialists) on multi-step API tasks
  - hyperagent: HyperAgent Hierarchical (Planner → sub-agents) on SWE-bench Lite
  - magentic:   Magentic-One Star (Orchestrator + 4 agents) on GAIA

Usage:
    # AG2 MathChat (linear, math)
    python -m experiments.run_openevolve --topology ag2 --iterations 50

    # AppWorld (star, API tasks)
    python -m experiments.run_openevolve --topology appworld --iterations 50

    # HyperAgent (hierarchical, code fixing)
    python -m experiments.run_openevolve --topology hyperagent --iterations 50

    # Magentic-One (complex star, general tasks)
    python -m experiments.run_openevolve --topology magentic --iterations 50
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

CONFIG_FILE = Path(__file__).parent / "openevolve_config.yaml"
EVALUATOR_FILE = Path(__file__).parent / "openevolve_evaluator.py"
EVALUATOR_MULTI_FILE = Path(__file__).parent / "openevolve_evaluator_multi.py"

# Topology → (DAG file, initial program, result prefix)
TOPOLOGIES: dict[str, tuple[str, str, str]] = {
    "ag2": (
        "ag2_mathchat.yaml",
        "openevolve_initial_dag.py",
        "ag2_mathchat_openevolve",
    ),
    "appworld": (
        "appworld_star.yaml",
        "openevolve_initial_dag_appworld.py",
        "appworld_star_openevolve",
    ),
    "hyperagent": (
        "hyperagent_hierarchical.yaml",
        "openevolve_initial_dag_hyperagent.py",
        "hyperagent_hierarchical_openevolve",
    ),
    "magentic": (
        "magentic_one_star.yaml",
        "openevolve_initial_dag_magentic.py",
        "magentic_one_star_openevolve",
    ),
}


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
    topology: str = "ag2",
    n_train: int = 100,
    n_test: int = 100,
    iterations: int = 50,
    eval_tasks: int = 20,
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 512,
    timeout: int = 600,
    reuse_diagnose_dir: str | None = None,
):
    # Resolve topology config
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {topology}. Available: {list(TOPOLOGIES.keys())}")
    dag_filename, initial_program_filename, result_prefix = TOPOLOGIES[topology]
    initial_program_path = str(Path(__file__).parent / initial_program_filename)

    total = n_train + n_test
    model_short = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_stem = f"{result_prefix}_{model_short}_{timestamp}"
    artifact_dir = RESULTS_DIR / f"{result_stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dag_versions_dir = artifact_dir / "dag_versions"
    dag_versions_dir.mkdir(parents=True, exist_ok=True)
    openevolve_output_dir = str(artifact_dir / "openevolve_output")

    # ------------------------------------------------------------------
    # Load benchmarks per topology
    # ------------------------------------------------------------------
    print(f"Loading benchmarks for topology={topology}...")
    custom_score_fn = None  # None = use test_suite.score_task (works for ag2)

    if topology == "ag2":
        full_suite = load_online_benchmark_suite(total)
        train_examples, test_examples = _split_suite(full_suite, n_train)
        benchmark_label = "AG2_MathChat"
        tool_setup_fn = None
    elif topology == "appworld":
        from optpilot.data.benchmarks_appworld import load_appworld_examples, score_appworld
        from optpilot.tools.appworld_tools import AppWorldWrapper, build_tools as aw_build
        all_examples = load_appworld_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        benchmark_label = "AppWorld"
        _aw_lookup = {ex.prompt: ex for ex in all_examples}
        def tool_setup_fn(task_prompt):
            ex = _aw_lookup.get(task_prompt)
            if ex is None:
                return None
            wrapper = AppWorldWrapper(task_id=ex.task_id, experiment_name="openevolve")
            return aw_build(wrapper)
        def custom_score_fn(task_prompt, _dag, exec_trace):
            ex = _aw_lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_appworld(pred, ex.gold_answers[0])
    elif topology == "hyperagent":
        from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench
        all_examples = load_swebench_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        benchmark_label = "SWE-bench-Lite"
        _ha_lookup = {ex.prompt: ex for ex in all_examples}
        def tool_setup_fn(task_prompt):
            from optpilot.tools.hyperagent_tools import CodeEnvironment, build_tools as ha_build
            ex = _ha_lookup.get(task_prompt)
            repo = ex.metadata.get("repo", "") if ex else ""
            base_commit = ex.metadata.get("base_commit", "") if ex else ""
            return ha_build(CodeEnvironment(repo=repo, base_commit=base_commit))
        def custom_score_fn(task_prompt, _dag, exec_trace):
            ex = _ha_lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_swebench(pred, ex.gold_answers[0])
    elif topology == "magentic":
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        from optpilot.tools.magentic_tools import GeneralEnvironment, build_tools as mg_build
        all_examples = load_gaia_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        benchmark_label = "GAIA"
        _mg_lookup = {ex.prompt: ex for ex in all_examples}
        def tool_setup_fn(task_prompt):
            ex = _mg_lookup.get(task_prompt)
            docs = ex.metadata.get("context_docs", {}) if ex else {}
            return mg_build(GeneralEnvironment(docs))
        def custom_score_fn(task_prompt, _dag, exec_trace):
            ex = _mg_lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_gaia(pred, ex.gold_answers[0])
    else:
        raise ValueError(f"Unknown topology: {topology}")

    train_suite = OfficialBenchmarkSuite(train_examples)
    test_suite = OfficialBenchmarkSuite(test_examples)

    # ------------------------------------------------------------------
    # Load baseline DAG
    # ------------------------------------------------------------------
    dag_file = Path(dag_path) if dag_path else DAG_DIR / dag_filename
    original_dag = MASDAG.load(dag_file)
    original_dag.save(dag_versions_dir / "input.yaml")

    # ------------------------------------------------------------------
    # Baseline run on train to find failure tasks for eval
    # ------------------------------------------------------------------
    print("Running baseline to identify failure tasks...")
    baseline_runner = OptPilotRunner(
        dag=original_dag,
        model=model,
        benchmark_name=benchmark_label,
        score_fn=custom_score_fn or train_suite.score_task,
        benchmark_name_resolver=train_suite.benchmark_name_for_task,
        timeout=timeout,
        tool_setup_fn=tool_setup_fn,
    )
    baseline_traces = asyncio.run(
        baseline_runner.arun_batch(
            train_suite.tasks(),
            dag=original_dag,
            max_concurrency=concurrency,
        )
    )
    # Collect failed tasks (score == 0), use as eval set for evolution
    # Note: task_key is truncated to 50 chars, use original prompts instead
    train_prompts = train_suite.tasks()
    failed_prompts = [
        train_prompts[i] for i, t in enumerate(baseline_traces)
        if not t.task_score or t.task_score <= 0
    ]
    n_baseline_correct = len(baseline_traces) - len(failed_prompts)
    print(f"  Baseline: {n_baseline_correct}/{len(baseline_traces)} correct, "
          f"{len(failed_prompts)} failures")
    selected_eval_prompts = failed_prompts[:eval_tasks]
    if len(selected_eval_prompts) < eval_tasks:
        # Not enough failures, pad with random train tasks
        remaining = [
            ex.prompt for ex in train_examples
            if ex.prompt not in set(selected_eval_prompts)
        ]
        selected_eval_prompts.extend(remaining[:eval_tasks - len(selected_eval_prompts)])
    print(f"  Eval set: {len(selected_eval_prompts)} tasks (all baseline failures)")

    # ------------------------------------------------------------------
    # Set environment variables for the evaluator
    # ------------------------------------------------------------------
    os.environ["OPENEVOLVE_MODEL"] = model
    os.environ["OPENEVOLVE_CONCURRENCY"] = str(concurrency)
    os.environ["OPENEVOLVE_TIMEOUT"] = str(timeout)
    os.environ["OPENEVOLVE_EVAL_TASKS"] = str(len(selected_eval_prompts))
    os.environ["OPENEVOLVE_TOTAL_TASKS"] = str(total)
    os.environ["OPENEVOLVE_TOPOLOGY"] = topology
    os.environ["OPENEVOLVE_EVAL_PROMPTS_JSON"] = json.dumps(selected_eval_prompts, ensure_ascii=False)

    # Ensure Together AI API key is visible to SkyDiscover
    # SkyDiscover's "together" provider reads: TOGETHER_API_KEY, TOGETHER_AI_API_KEY, together_ai_api
    if TOGETHER_API_KEY and not os.environ.get("TOGETHER_API_KEY"):
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    # ------------------------------------------------------------------
    # Print experiment info
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"  OpenEvolve Baseline — Topology: {topology}")
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

    evaluator_file = EVALUATOR_FILE if topology == "ag2" else EVALUATOR_MULTI_FILE
    result = run_discovery(
        evaluator=str(evaluator_file),
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
        benchmark_name=benchmark_label,
        score_fn=custom_score_fn or test_suite.score_task,
        benchmark_name_resolver=test_suite.benchmark_name_for_task,
        timeout=timeout,
        tool_setup_fn=tool_setup_fn,
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
        "experiment": f"{topology}_openevolve",
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
        description="OpenEvolve cold-start experiment for multiple MAS topologies",
    )
    parser.add_argument("--topology", default="ag2", choices=list(TOPOLOGIES.keys()),
                        help=f"MAS topology to evolve (choices: {list(TOPOLOGIES.keys())})")
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML (overrides topology default)")
    parser.add_argument("--train", type=int, default=100, help="Train set size")
    parser.add_argument("--test", type=int, default=100, help="Test set size")
    parser.add_argument("--iterations", type=int, default=50, help="OpenEvolve iterations")
    parser.add_argument("--eval-tasks", type=int, default=20, help="Tasks per evaluation")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model ID on Together AI")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    args = parser.parse_args()

    run(
        dag_path=args.dag,
        topology=args.topology,
        n_train=args.train,
        n_test=args.test,
        iterations=args.iterations,
        eval_tasks=args.eval_tasks,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
