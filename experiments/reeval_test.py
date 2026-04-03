"""Re-evaluate completed OpenEvolve experiments on the full test set (3 runs, averaged).

Loads the shadow-selected (or openevolve-best) DAG and the original baseline DAG
from completed experiment artifacts, then runs test evaluation 3 times to get
stable accuracy estimates.

Usage:
    # Re-eval a single experiment
    python -m experiments.reeval_test --result-json results/star_humaneval_openevolve_..._blind.json

    # Re-eval all completed experiments
    python -m experiments.reeval_test --all
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import RESULTS_DIR, TOGETHER_API_KEY
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import load_online_benchmark_suite, OfficialBenchmarkSuite
from optpilot.modules.runner import OptPilotRunner

N_RUNS = 3
MODEL = "openai/gpt-oss-120b"
CONCURRENCY = 512
TIMEOUT = 600


def _split_suite(suite: OfficialBenchmarkSuite, n_train: int):
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


def _load_benchmark(benchmark: str, n_train: int, n_test: int):
    """Load test examples and return (test_examples, benchmark_label, score_fn, tool_setup_fn)."""
    total = n_train + n_test

    if benchmark == "math":
        full_suite = load_online_benchmark_suite(total)
        _, test_examples = _split_suite(full_suite, n_train)
        return test_examples, "Math", None, None

    elif benchmark == "humaneval":
        from optpilot.data.benchmarks_humaneval import load_humaneval_examples, score_humaneval
        from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools as ac_build
        all_examples = load_humaneval_examples(total)
        test_examples = all_examples[n_train:n_train + n_test]
        _lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            return ac_build(CodeExecutionEnvironment())

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_humaneval(pred, ex)

        return test_examples, "HumanEval", score_fn, tool_setup_fn

    elif benchmark == "gaia":
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        all_examples = load_gaia_examples(total)
        test_examples = all_examples[n_train:n_train + n_test]
        _lookup = {ex.prompt: ex for ex in all_examples}

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_gaia(pred, ex.gold_answers[0])

        return test_examples, "GAIA", score_fn, None

    elif benchmark == "swebench":
        from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench
        all_examples = load_swebench_examples(total)
        test_examples = all_examples[n_train:n_train + n_test]
        _lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            from optpilot.tools.hyperagent_tools import CodeEnvironment, build_tools as ha_build
            ex = _lookup.get(task_prompt)
            repo = ex.metadata.get("repo", "") if ex else ""
            base_commit = ex.metadata.get("base_commit", "") if ex else ""
            return ha_build(CodeEnvironment(repo=repo, base_commit=base_commit))

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_swebench(pred, ex.gold_answers[0])

        return test_examples, "SWE-bench-Lite", score_fn, tool_setup_fn

    elif benchmark == "livecodebench":
        from optpilot.data.benchmarks_humaneval import load_humaneval_examples, score_humaneval
        from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools as ac_build
        # LiveCodeBench uses same loader as humaneval for now
        all_examples = load_humaneval_examples(total)
        test_examples = all_examples[n_train:n_train + n_test]
        _lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            return ac_build(CodeExecutionEnvironment())

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_humaneval(pred, ex)

        return test_examples, "LiveCodeBench", score_fn, tool_setup_fn

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


async def _eval_on_test(runner, dag, test_suite, concurrency, label, output_base=None):
    """Evaluate a DAG on the full test set."""
    tasks = test_suite.tasks()
    print(f"\n--- {label}: evaluating on {len(tasks)} test tasks ---")
    t0 = time.time()
    traces = await runner.arun_batch(
        tasks, dag=dag, output_base=output_base, max_concurrency=concurrency,
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


def _average_stats(runs: list[dict]) -> dict:
    avg_acc = sum(r["overall_accuracy"] for r in runs) / len(runs)
    avg_elapsed = sum(r["elapsed_s"] for r in runs) / len(runs)
    all_benches = set()
    for r in runs:
        all_benches.update(r["per_benchmark"].keys())
    avg_per_bench = {}
    for bname in sorted(all_benches):
        bench_runs = [r["per_benchmark"][bname] for r in runs if bname in r["per_benchmark"]]
        avg_per_bench[bname] = {
            "n": bench_runs[0]["n"],
            "correct": sum(b["correct"] for b in bench_runs) / len(bench_runs),
            "accuracy": sum(b["accuracy"] for b in bench_runs) / len(bench_runs),
        }
    return {
        "overall_accuracy": avg_acc,
        "elapsed_s": avg_elapsed,
        "per_benchmark": avg_per_bench,
        "individual_runs": runs,
    }


def _detect_benchmark(target_mas: str) -> str:
    """Detect benchmark from target_mas name."""
    # topology_benchmark format (e.g., linear_humaneval, star_loop_math)
    for bench in ("humaneval", "math", "gaia", "swebench", "livecodebench"):
        if target_mas.endswith(f"_{bench}"):
            return bench
    # Legacy presets
    legacy_map = {
        "ag2": "math",
        "agentcoder": "humaneval",
        "appworld": "swebench",
        "hyperagent": "swebench",
        "simple_hier": "swebench",
        "magentic": "gaia",
        "simple_star": "gaia",
    }
    return legacy_map.get(target_mas, "math")


def reeval_single(result_json: Path):
    """Re-evaluate a single completed experiment."""
    with open(result_json) as f:
        result = json.load(f)

    target_mas = result["target_mas"]
    artifact_dir = Path(result["artifacts_dir"])
    dag_versions_dir = Path(result["dag_versions_dir"])
    n_train = result["n_train"]
    n_test = result["n_test"]

    print(f"\n{'=' * 65}")
    print(f"  Re-evaluating: {target_mas}")
    print(f"  Source: {result_json.name}")
    print(f"{'=' * 65}")

    # Load DAGs
    baseline_dag_path = dag_versions_dir / "input.yaml"
    baseline_dag = MASDAG.load(baseline_dag_path)

    # Check if shadow gate rejected all candidates → best = original_dag
    shadow = result.get("shadow_selection", {})
    shadow_rejected = shadow.get("selected_candidate") == "original_dag"

    if shadow_rejected:
        print(f"  Shadow gate rejected all candidates → best DAG = original DAG (skip)")
        return None

    # Load best DAG from Python program (shadow_selected.py or openevolve_best.py)
    best_py = dag_versions_dir / "shadow_selected.py"
    if not best_py.exists():
        best_py = dag_versions_dir / "openevolve_best.py"
    if not best_py.exists():
        print(f"  SKIP: no best program found in {dag_versions_dir}")
        return None

    import importlib.util
    spec = importlib.util.spec_from_file_location("_best", str(best_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    best_dag = MASDAG.from_dict(mod.build_dag())
    print(f"  Loaded best DAG from {best_py.name}")

    # Detect benchmark
    benchmark = _detect_benchmark(target_mas)
    print(f"  Benchmark: {benchmark}")
    print(f"  n_train={n_train}, n_test={n_test}")

    # Load test set
    test_examples, benchmark_label, custom_score_fn, tool_setup_fn = \
        _load_benchmark(benchmark, n_train, n_test)
    test_suite = OfficialBenchmarkSuite(test_examples)
    print(f"  Test tasks: {len(test_examples)}")

    # Create runner
    runner = OptPilotRunner(
        dag=baseline_dag,
        model=MODEL,
        benchmark_name=benchmark_label,
        score_fn=custom_score_fn or test_suite.score_task,
        benchmark_name_resolver=test_suite.benchmark_name_for_task,
        timeout=TIMEOUT,
        tool_setup_fn=tool_setup_fn,
    )

    # Run 3x test evaluation
    async def _run():
        final_runs = []
        baseline_runs = []
        for run_i in range(1, N_RUNS + 1):
            print(f"\n=== Test run {run_i}/{N_RUNS} ===")
            final = await _eval_on_test(
                runner, best_dag, test_suite, CONCURRENCY,
                f"BEST (run {run_i})",
                output_base=artifact_dir / f"reeval_final_run{run_i}",
            )
            baseline = await _eval_on_test(
                runner, baseline_dag, test_suite, CONCURRENCY,
                f"BASELINE (run {run_i})",
                output_base=artifact_dir / f"reeval_baseline_run{run_i}",
            )
            final_runs.append(final)
            baseline_runs.append(baseline)
        return final_runs, baseline_runs

    final_runs, baseline_runs = asyncio.run(_run())

    final_avg = _average_stats(final_runs)
    baseline_avg = _average_stats(baseline_runs)

    accs_final = [r["overall_accuracy"] for r in final_runs]
    accs_baseline = [r["overall_accuracy"] for r in baseline_runs]
    print(f"\n  BEST     acc across {N_RUNS} runs: {accs_final} → mean={final_avg['overall_accuracy']:.4f}")
    print(f"  BASELINE acc across {N_RUNS} runs: {accs_baseline} → mean={baseline_avg['overall_accuracy']:.4f}")

    # Save re-eval results
    reeval_output = {
        "source_result": str(result_json),
        "target_mas": target_mas,
        "benchmark": benchmark,
        "n_test": n_test,
        "n_runs": N_RUNS,
        "test_final": final_avg,
        "test_baseline": baseline_avg,
    }
    reeval_path = result_json.parent / result_json.name.replace(".json", "_reeval.json")
    with open(reeval_path, "w") as f:
        json.dump(reeval_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {reeval_path}")

    return reeval_output


def main():
    import os
    if TOGETHER_API_KEY and not os.environ.get("TOGETHER_API_KEY"):
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    parser = argparse.ArgumentParser(description="Re-evaluate completed OpenEvolve experiments")
    parser.add_argument("--result-json", type=str, help="Path to a single result JSON")
    parser.add_argument("--all", action="store_true", help="Re-eval all completed experiments")
    parser.add_argument("--filter", type=str, default="", help="Filter experiments by substring")
    args = parser.parse_args()

    if args.result_json:
        reeval_single(Path(args.result_json))
    elif args.all:
        result_files = sorted(RESULTS_DIR.glob("*_blind.json"))
        if args.filter:
            result_files = [f for f in result_files if args.filter in f.name]
        print(f"Found {len(result_files)} completed experiments to re-evaluate")
        for f in result_files:
            # Skip livecodebench (uses same loader as humaneval, may not be valid)
            if "livecodebench" in f.name:
                print(f"\n  SKIP: {f.name} (livecodebench)")
                continue
            try:
                reeval_single(f)
            except Exception as e:
                print(f"\n  ERROR re-evaluating {f.name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
