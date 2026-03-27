"""AG2 MathChat Baseline — train/test split, trace + score only.

Usage:
    python -m experiments.run_ag2_mathchat_baseline --model openai/gpt-oss-120b
    python -m experiments.run_ag2_mathchat_baseline --model openai/gpt-oss-20b
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, RESULTS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import load_online_benchmark_suite, OfficialBenchmarkSuite
from optpilot.modules.runner import OptPilotRunner

DAG_FILE = "ag2_mathchat.yaml"


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


def _run_split(name: str, suite: OfficialBenchmarkSuite, runner: OptPilotRunner, dag: MASDAG):
    """Run baseline on a split, return (traces, stats)."""
    tasks = suite.tasks()
    print(f"\n--- {name}: {len(tasks)} tasks ---")
    t0 = time.time()
    traces = runner.run_batch(tasks, dag=dag)
    elapsed = time.time() - t0

    per_bench: dict[str, list[float]] = defaultdict(list)
    details = []
    for i, trace in enumerate(traces):
        per_bench[trace.benchmark_name].append(trace.task_score)
        details.append({
            "task_id": i,
            "benchmark": trace.benchmark_name,
            "score": trace.task_score,
            "success": trace.task_success,
            "latency_s": trace.latency_s,
            "task_key": trace.task_key,
        })

    all_scores = [t.task_score for t in traces]
    overall_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"  {name} Results:")
    for bname, scores in sorted(per_bench.items()):
        acc = sum(scores) / len(scores) if scores else 0.0
        correct = sum(1 for s in scores if s > 0)
        print(f"    {bname:20s}  {correct}/{len(scores)}  acc={acc:.3f}")
    total_correct = sum(1 for s in all_scores if s > 0)
    print(f"    {'OVERALL':20s}  {total_correct}/{len(all_scores)}  acc={overall_acc:.3f}")
    print(f"    Elapsed: {elapsed:.1f}s  ({elapsed / len(tasks):.1f}s/task)")

    stats = {
        "n_tasks": len(tasks),
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
    return traces, stats


def run_baseline(n_train: int = 100, n_test: int = 100, model: str = "MiniMaxAI/MiniMax-M2.5"):
    total = n_train + n_test
    model_short = model.split("/")[-1]

    print("Loading benchmarks...")
    full_suite = load_online_benchmark_suite(total)
    train_examples, test_examples = _split_suite(full_suite, n_train)
    train_suite = OfficialBenchmarkSuite(train_examples)
    test_suite = OfficialBenchmarkSuite(test_examples)

    dag_file = DAG_DIR / DAG_FILE
    dag = MASDAG.load(dag_file)

    train_runner = OptPilotRunner(
        dag=dag, model=model,
        benchmark_name="AG2_MathChat",
        score_fn=train_suite.score_task,
        benchmark_name_resolver=train_suite.benchmark_name_for_task,
    )
    test_runner = OptPilotRunner(
        dag=dag, model=model,
        benchmark_name="AG2_MathChat",
        score_fn=test_suite.score_task,
        benchmark_name_resolver=test_suite.benchmark_name_for_task,
    )

    print("=" * 65)
    print(f"  AG2 MathChat Baseline — {model_short}")
    print("=" * 65)
    print(f"  Model:    {model}")
    print(f"  DAG:      {dag.dag_id}  ({dag_file.name})")
    print(f"  Agents:   {', '.join(dag.agent_nodes.keys())}")
    print(f"  Train:    {len(train_examples)} tasks  {dict(train_suite.benchmark_counts())}")
    print(f"  Test:     {len(test_examples)} tasks  {dict(test_suite.benchmark_counts())}")
    print("=" * 65)

    train_traces, train_stats = _run_split("TRAIN", train_suite, train_runner, dag)
    test_traces, test_stats = _run_split("TEST", test_suite, test_runner, dag)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"baseline_{model_short}_{total}tasks_{ts}.json"

    output = {
        "experiment": "ag2_mathchat_baseline",
        "model": model,
        "dag": dag.dag_id,
        "timestamp": ts,
        "train": train_stats,
        "test": test_stats,
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 65}")
    print(f"  Summary — {model_short}")
    print(f"{'=' * 65}")
    print(f"  TRAIN acc: {train_stats['overall_accuracy']:.3f}")
    print(f"  TEST  acc: {test_stats['overall_accuracy']:.3f}")
    print(f"{'=' * 65}")
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG2 MathChat Baseline")
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-M2.5", help="Model ID on Together AI")
    args = parser.parse_args()
    run_baseline(n_train=args.train, n_test=args.test, model=args.model)
