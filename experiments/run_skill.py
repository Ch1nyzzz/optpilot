"""Multi-topology Skill Workflow (Jacobian-driven repair) optimization.

Supports multiple MAS topologies and their corresponding benchmarks:

  - ag2:        AG2 MathChat (3-agent linear) on MMLU/AIME/OlympiadBench
  - appworld:   AppWorld Star (Supervisor + specialists) on multi-step API tasks
  - hyperagent: HyperAgent Hierarchical (Planner → sub-agents) on SWE-bench Lite
  - magentic:   Magentic-One Star (Orchestrator + 4 agents) on GAIA

Usage:
    python -m experiments.run_skill --topology ag2 --rounds 50
    python -m experiments.run_skill --topology hyperagent --train 100 --test 100 --rounds 50
    python -m experiments.run_skill --topology magentic --train 80 --test 80 --rounds 50
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
from optpilot.data.benchmarks import OfficialBenchmarkSuite, BenchmarkExample
from optpilot.models import SkillBudget
from optpilot.modules.runner import OptPilotRunner
from optpilot.orchestrator import Orchestrator

# Topology → (DAG file, result prefix)
TOPOLOGIES: dict[str, tuple[str, str]] = {
    "ag2": ("ag2_mathchat.yaml", "ag2_mathchat_skill"),
    "appworld": ("appworld_star.yaml", "appworld_star_skill"),
    "hyperagent": ("hyperagent_hierarchical.yaml", "hyperagent_hierarchical_skill"),
    "magentic": ("magentic_one_star.yaml", "magentic_one_star_skill"),
    "simple_star": ("simple_star_gaia.yaml", "simple_star_gaia_skill"),
    "simple_hier": ("simple_hierarchical_swebench.yaml", "simple_hier_swebench_skill"),
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


def _load_topology(
    topology: str, n_train: int, n_test: int,
) -> tuple[list[BenchmarkExample], list[BenchmarkExample], str, object, object]:
    """Load benchmark data for a topology.

    Returns (train_examples, test_examples, benchmark_label, score_fn, tool_setup_fn).
    score_fn and tool_setup_fn may be None (for AG2 which uses the suite scorer).
    """
    total = n_train + n_test

    if topology == "ag2":
        from optpilot.data.benchmarks import load_online_benchmark_suite
        full_suite = load_online_benchmark_suite(total)
        train_examples, test_examples = _split_suite(full_suite, n_train)
        return train_examples, test_examples, "AG2_MathChat", None, None

    elif topology == "appworld":
        from optpilot.data.benchmarks_appworld import load_appworld_examples, score_appworld
        from optpilot.tools.appworld_tools import AppWorldWrapper, build_tools
        all_examples = load_appworld_examples(
            total, splits=("train", "dev", "test_normal", "test_challenge"),
        )
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            ex = lookup.get(task_prompt)
            if ex is None:
                return None
            wrapper = AppWorldWrapper(task_id=ex.task_id, experiment_name="skill_opt")
            return build_tools(wrapper)

        def score_fn(task_prompt, _dag, exec_trace):
            ex = lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_appworld(pred, ex.gold_answers[0])

        return train_examples, test_examples, "AppWorld", score_fn, tool_setup_fn

    elif topology == "hyperagent":
        from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench
        all_examples = load_swebench_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            from optpilot.tools.hyperagent_tools import CodeEnvironment, build_tools
            ex = lookup.get(task_prompt)
            repo = ex.metadata.get("repo", "") if ex else ""
            base_commit = ex.metadata.get("base_commit", "") if ex else ""
            return build_tools(CodeEnvironment(repo=repo, base_commit=base_commit))

        def score_fn(task_prompt, _dag, exec_trace):
            ex = lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_swebench(pred, ex.gold_answers[0])

        return train_examples, test_examples, "SWE-bench-Lite", score_fn, tool_setup_fn

    elif topology == "magentic":
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        from optpilot.tools.magentic_tools import GeneralEnvironment, build_tools
        all_examples = load_gaia_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            ex = lookup.get(task_prompt)
            docs = ex.metadata.get("context_docs", {}) if ex else {}
            return build_tools(GeneralEnvironment(docs))

        def score_fn(task_prompt, _dag, exec_trace):
            ex = lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_gaia(pred, ex.gold_answers[0])

        return train_examples, test_examples, "GAIA", score_fn, tool_setup_fn

    elif topology == "simple_star":
        # Reuses GAIA benchmark + magentic tools
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        from optpilot.tools.magentic_tools import GeneralEnvironment, build_tools
        all_examples = load_gaia_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            ex = lookup.get(task_prompt)
            docs = ex.metadata.get("context_docs", {}) if ex else {}
            return build_tools(GeneralEnvironment(docs))

        def score_fn(task_prompt, _dag, exec_trace):
            ex = lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_gaia(pred, ex.gold_answers[0])

        return train_examples, test_examples, "GAIA", score_fn, tool_setup_fn

    elif topology == "simple_hier":
        # Reuses SWE-bench benchmark + hyperagent tools
        from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench
        all_examples = load_swebench_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            from optpilot.tools.hyperagent_tools import CodeEnvironment, build_tools
            ex = lookup.get(task_prompt)
            repo = ex.metadata.get("repo", "") if ex else ""
            base_commit = ex.metadata.get("base_commit", "") if ex else ""
            return build_tools(CodeEnvironment(repo=repo, base_commit=base_commit))

        def score_fn(task_prompt, _dag, exec_trace):
            ex = lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_swebench(pred, ex.gold_answers[0])

        return train_examples, test_examples, "SWE-bench-Lite", score_fn, tool_setup_fn

    else:
        raise ValueError(f"Unknown topology: {topology}. Available: {list(TOPOLOGIES.keys())}")


async def run(
    topology: str = "ag2",
    dag_path: str | None = None,
    n_train: int = 100,
    n_test: int = 100,
    max_rounds: int = 50,
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
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {topology}. Available: {list(TOPOLOGIES.keys())}")
    dag_filename, result_prefix = TOPOLOGIES[topology]

    model_short = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_tag = f"_group{target_group}" if target_group else ""
    result_stem = f"{result_prefix}_{model_short}{group_tag}_{timestamp}"
    artifact_dir = RESULTS_DIR / f"{result_stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dag_versions_dir = artifact_dir / "dag_versions"
    dag_versions_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmarks
    print(f"Loading benchmarks for topology={topology}...")
    train_examples, test_examples, benchmark_label, custom_score_fn, tool_setup_fn = (
        _load_topology(topology, n_train, n_test)
    )
    train_suite = OfficialBenchmarkSuite(train_examples)
    test_suite = OfficialBenchmarkSuite(test_examples)

    # Load DAG
    dag_file = Path(dag_path) if dag_path else DAG_DIR / dag_filename
    dag = MASDAG.load(dag_file)
    dag.save(dag_versions_dir / "input.yaml")

    # Build runners
    train_score_fn = custom_score_fn or train_suite.score_task
    test_score_fn = custom_score_fn or test_suite.score_task

    train_runner = OptPilotRunner(
        dag=dag,
        model=model,
        benchmark_name=benchmark_label,
        score_fn=train_score_fn,
        benchmark_name_resolver=train_suite.benchmark_name_for_task,
        timeout=timeout,
        tool_setup_fn=tool_setup_fn,
    )
    test_runner = OptPilotRunner(
        dag=dag,
        model=model,
        benchmark_name=benchmark_label,
        score_fn=test_score_fn,
        benchmark_name_resolver=test_suite.benchmark_name_for_task,
        timeout=timeout,
        tool_setup_fn=tool_setup_fn,
    )

    # Build orchestrator with topology-isolated storage
    orchestrator = Orchestrator(
        runner=train_runner,
        dag=dag,
        use_wandb=use_wandb,
        topology=topology,
    )
    if clear_negatives:
        removed = orchestrator.negatives_store.clear_all()
        print(f"Cleared {removed} persisted negatives before optimization.")

    # Print experiment info
    print("=" * 65)
    print(f"  {benchmark_label} — Skill Workflow Optimization (Jacobian-driven)")
    print("=" * 65)
    print(f"  Topology:    {topology}")
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
    final_dag = orchestrator.dag
    final_dag.save(dag_versions_dir / "optimized_final.yaml")
    test_stats = await _eval_on_test(
        test_runner, final_dag, test_suite, concurrency,
        "FINAL", output_base=artifact_dir / "test_final",
    )

    # Evaluate original DAG on test for comparison
    original_dag = MASDAG.load(dag_file)
    original_dag.save(dag_versions_dir / "baseline_reference.yaml")
    baseline_test_stats = await _eval_on_test(
        test_runner, original_dag, test_suite, concurrency,
        "BASELINE", output_base=artifact_dir / "test_baseline",
    )

    # Save results
    result_path = RESULTS_DIR / f"{result_stem}.json"
    output = {
        "experiment": f"{topology}_skill_workflow",
        "topology": topology,
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
    baseline_acc = baseline_test_stats["overall_accuracy"]
    final_acc = test_stats["overall_accuracy"]
    print(f"\n  Test accuracy: {baseline_acc:.3f} → {final_acc:.3f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-topology Skill Workflow (Jacobian-driven) optimization",
    )
    parser.add_argument("--topology", default="ag2", choices=list(TOPOLOGIES.keys()),
                        help="MAS topology to optimize")
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML (overrides topology default)")
    parser.add_argument("--train", type=int, default=100, help="Train set size")
    parser.add_argument("--test", type=int, default=100, help="Test set size")
    parser.add_argument("--rounds", type=int, default=50, help="Max optimization rounds")
    parser.add_argument("--eval-tasks", type=int, default=None,
                        help="Tasks per round (subsample from train set, default: all)")
    parser.add_argument("--group", default=None, help="Target a specific FM group (A-F)")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model ID on Together AI")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    parser.add_argument("--reuse-traces-dir", default=None,
                        help="Reuse persisted train traces instead of rerunning train")
    parser.add_argument("--reuse-diagnose-dir", default=None,
                        help="Reuse persisted diagnose artifacts instead of rerunning train/diagnose")
    parser.add_argument("--clear-negatives", action="store_true",
                        help="Delete persisted negatives before optimization")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()

    if args.reuse_traces_dir and args.reuse_diagnose_dir:
        parser.error("--reuse-traces-dir and --reuse-diagnose-dir are mutually exclusive")

    asyncio.run(run(
        topology=args.topology,
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
