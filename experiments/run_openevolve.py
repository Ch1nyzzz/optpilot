"""Target-MAS OpenEvolve cold-start experiment.

Evolves target MAS DAGs via SkyDiscover's OpenEvolve (MAP-Elites).
Supports multiple target MAS presets and their corresponding benchmarks:

  - ag2:         AG2 MathChat (3-agent linear) on MMLU/AIME/OlympiadBench
  - appworld:    AppWorld Star (Supervisor + specialists) on multi-step API tasks
  - hyperagent:  HyperAgent Hierarchical (Planner → sub-agents) on SWE-bench Lite
  - magentic:    Magentic-One Star (Orchestrator + 4 agents) on GAIA
  - agentcoder:  AgentCoder (Programmer + TestDesigner + TestExecutor) on HumanEval

Usage:
    # AG2 MathChat (linear, math)
    python -m experiments.run_openevolve --target-mas ag2 --iterations 50

    # AppWorld (star, API tasks)
    python -m experiments.run_openevolve --target-mas appworld --iterations 50

    # HyperAgent (hierarchical, code fixing)
    python -m experiments.run_openevolve --target-mas hyperagent --iterations 50

    # Magentic-One (complex star, general tasks)
    python -m experiments.run_openevolve --target-mas magentic --iterations 50

    # AgentCoder (pipeline, code generation)
    python -m experiments.run_openevolve --target-mas agentcoder --iterations 50
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import RESULTS_DIR, TOGETHER_API_KEY
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import load_online_benchmark_suite, OfficialBenchmarkSuite
from optpilot.modules.runner import OptPilotRunner

# SkyDiscover
from skydiscover.api import run_discovery

CONFIG_FILE = Path(__file__).parent / "openevolve_config.yaml"
EVALUATOR_FILE = Path(__file__).parent / "openevolve_evaluator.py"
EVALUATOR_MULTI_FILE = Path(__file__).parent / "openevolve_evaluator_multi.py"

# Target MAS → (initial program, result prefix) — legacy presets
TARGET_MASES: dict[str, tuple[str, str]] = {
    "ag2": (
        "openevolve_initial_dag.py",
        "ag2_mathchat_openevolve",
    ),
    "appworld": (
        "openevolve_initial_dag_appworld.py",
        "appworld_star_openevolve",
    ),
    "hyperagent": (
        "openevolve_initial_dag_hyperagent.py",
        "hyperagent_hierarchical_openevolve",
    ),
    "magentic": (
        "openevolve_initial_dag_magentic.py",
        "magentic_one_star_openevolve",
    ),
    "simple_star": (
        "openevolve_initial_dag_simple_star.py",
        "simple_star_gaia_openevolve",
    ),
    "simple_hier": (
        "openevolve_initial_dag_simple_hier.py",
        "simple_hier_swebench_openevolve",
    ),
    "agentcoder": (
        "openevolve_initial_dag_agentcoder.py",
        "agentcoder_humaneval_openevolve",
    ),
}

# Topology → initial program filename (decoupled from benchmark)
TOPOLOGIES: dict[str, str] = {
    "linear": "openevolve_initial_dag_linear.py",
    "linear_loop": "openevolve_initial_dag_linear_loop.py",
    "star": "openevolve_initial_dag_star.py",
    "star_loop": "openevolve_initial_dag_star_loop.py",
}

# Benchmark IDs supported for topology × benchmark experiments
BENCHMARKS = ("math", "livecodebench", "gaia", "swebench")


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


@dataclass
class ShadowCandidate:
    name: str
    code: str
    train_combined_score: float
    train_accuracy: float
    iteration: int
    source_path: Path


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


def _dag_from_python_source(code: str) -> MASDAG:
    """Build a MASDAG from Python source and enforce structural validity."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("_candidate_dag", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        dag = MASDAG.from_dict(mod.build_dag())
    finally:
        os.unlink(tmp_path)

    structural_errors = dag.structural_errors()
    if structural_errors:
        raise ValueError("; ".join(structural_errors))
    return dag


def _load_shadow_candidates(openevolve_output_dir: Path) -> list[ShadowCandidate]:
    """Load a small candidate shortlist for post-search shadow validation.

    We use the final best plus each checkpoint best. This is much cheaper than
    replaying every discovered program, while still giving the gate multiple
    train-improving candidates to choose from.
    """
    candidates: list[ShadowCandidate] = []

    def _add_candidate(name: str, code_path: Path, info_path: Path) -> None:
        if not code_path.exists():
            return
        info = {}
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        candidates.append(ShadowCandidate(
            name=name,
            code=code_path.read_text(encoding="utf-8"),
            train_combined_score=float(metrics.get("combined_score", 0.0) or 0.0),
            train_accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
            iteration=int(info.get("iteration", info.get("current_iteration", 0)) or 0),
            source_path=code_path,
        ))

    best_dir = openevolve_output_dir / "best"
    _add_candidate("final_best", best_dir / "best_program.py", best_dir / "best_program_info.json")

    checkpoints_dir = openevolve_output_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0,
        )
        for cp in checkpoints:
            _add_candidate(
                cp.name,
                cp / "best_program.py",
                cp / "best_program_info.json",
            )

    deduped: list[ShadowCandidate] = []
    seen_code: set[str] = set()
    for cand in sorted(candidates, key=lambda c: (c.train_combined_score, c.train_accuracy, c.iteration), reverse=True):
        if cand.code in seen_code:
            continue
        seen_code.add(cand.code)
        deduped.append(cand)
    return deduped


def _load_benchmark_config(benchmark: str, total: int):
    """Load benchmark examples, score function, tool setup, and label.

    Returns (all_examples, benchmark_label, custom_score_fn, tool_setup_fn).
    """
    if benchmark == "math":
        full_suite = load_online_benchmark_suite(total)
        return list(full_suite.examples), "Math", None, None

    elif benchmark == "livecodebench":
        from optpilot.data.benchmarks_livecodebench import load_livecodebench_examples, score_livecodebench
        from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools as ac_build
        all_examples = load_livecodebench_examples(total)
        _lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            return ac_build(CodeExecutionEnvironment())

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_livecodebench(pred, ex)

        return all_examples, "LiveCodeBench", score_fn, tool_setup_fn

    elif benchmark == "gaia":
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        all_examples = load_gaia_examples(total)
        _lookup = {ex.prompt: ex for ex in all_examples}

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_gaia(pred, ex.gold_answers[0])

        return all_examples, "GAIA", score_fn, None

    elif benchmark == "swebench":
        from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench
        all_examples = load_swebench_examples(total)
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

        return all_examples, "SWE-bench-Lite", score_fn, tool_setup_fn

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def run(
    dag_path: str | None = None,
    target_mas: str = "ag2",
    topology: str | None = None,
    benchmark: str | None = None,
    n_train: int = 100,
    n_test: int = 100,
    iterations: int = 50,
    eval_tasks: int = 20,
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 512,
    timeout: int = 600,
    reuse_diagnose_dir: str | None = None,
    with_priors: bool = False,
):
    # Resolve topology × benchmark (new) or legacy target_mas
    if topology and benchmark:
        if topology not in TOPOLOGIES:
            raise ValueError(f"Unknown topology: {topology}. Available: {list(TOPOLOGIES.keys())}")
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARKS)}")
        initial_program_filename = TOPOLOGIES[topology]
        result_prefix = f"{topology}_{benchmark}_openevolve"
        # For evaluator, combine as target_mas
        target_mas = f"{topology}_{benchmark}"
    elif target_mas in TARGET_MASES:
        initial_program_filename, result_prefix = TARGET_MASES[target_mas]
    else:
        raise ValueError(f"Unknown target_mas: {target_mas}. Use --topology + --benchmark or a legacy preset.")
    initial_program_path = str(Path(__file__).parent / initial_program_filename)

    total = n_train + n_test
    model_short = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prior_tag = "_priors" if with_priors else "_blind"
    result_stem = f"{result_prefix}_{model_short}_{timestamp}{prior_tag}"
    artifact_dir = RESULTS_DIR / f"{result_stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dag_versions_dir = artifact_dir / "dag_versions"
    dag_versions_dir.mkdir(parents=True, exist_ok=True)
    openevolve_output_dir = str(artifact_dir / "openevolve_output")

    # ------------------------------------------------------------------
    # Load benchmarks
    # ------------------------------------------------------------------
    print(f"Loading benchmarks for target_mas={target_mas}...")

    if benchmark:
        # New topology × benchmark mode
        all_examples, benchmark_label, custom_score_fn, tool_setup_fn = \
            _load_benchmark_config(benchmark, total)
        if benchmark == "math":
            # Math suite uses _split_suite for proportional split
            full_suite = load_online_benchmark_suite(total)
            train_examples, test_examples = _split_suite(full_suite, n_train)
        else:
            train_examples = all_examples[:n_train]
            test_examples = all_examples[n_train:n_train + n_test]
    elif target_mas == "ag2":
        full_suite = load_online_benchmark_suite(total)
        train_examples, test_examples = _split_suite(full_suite, n_train)
        benchmark_label = "AG2_MathChat"
        custom_score_fn = None
        tool_setup_fn = None
    elif target_mas == "appworld":
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
    elif target_mas in ("hyperagent", "simple_hier"):
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
    elif target_mas in ("magentic", "simple_star"):
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
    elif target_mas == "agentcoder":
        from optpilot.data.benchmarks_humaneval import load_humaneval_examples, score_humaneval
        from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools as ac_build
        all_examples = load_humaneval_examples(total)
        train_examples = all_examples[:n_train]
        test_examples = all_examples[n_train:n_train + n_test]
        benchmark_label = "HumanEval"
        _ac_lookup = {ex.prompt: ex for ex in all_examples}
        def tool_setup_fn(task_prompt):
            return ac_build(CodeExecutionEnvironment())
        def custom_score_fn(task_prompt, _dag, exec_trace):
            ex = _ac_lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_humaneval(pred, ex)
    else:
        raise ValueError(f"Unknown target_mas: {target_mas}")

    train_suite = OfficialBenchmarkSuite(train_examples)
    test_suite = OfficialBenchmarkSuite(test_examples)

    # ------------------------------------------------------------------
    # Load baseline DAG
    # ------------------------------------------------------------------
    if dag_path:
        original_dag = MASDAG.load(dag_path)
    else:
        original_dag = MASDAG.from_initial_program(initial_program_path)
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
    shadow_examples = [ex for ex in train_examples if ex.prompt not in set(selected_eval_prompts)]
    shadow_suite = OfficialBenchmarkSuite(shadow_examples)
    print(f"  Shadow set: {len(shadow_examples)} held-out train tasks")

    # ------------------------------------------------------------------
    # Set environment variables for the evaluator
    # ------------------------------------------------------------------
    os.environ["OPENEVOLVE_MODEL"] = model
    os.environ["OPENEVOLVE_CONCURRENCY"] = str(concurrency)
    os.environ["OPENEVOLVE_TIMEOUT"] = str(timeout)
    os.environ["OPENEVOLVE_EVAL_TASKS"] = str(len(selected_eval_prompts))
    os.environ["OPENEVOLVE_TOTAL_TASKS"] = str(total)
    os.environ["OPENEVOLVE_TARGET_MAS"] = target_mas
    os.environ["OPENEVOLVE_BENCHMARK"] = benchmark or ""
    os.environ["OPENEVOLVE_EVAL_PROMPTS_JSON"] = json.dumps(selected_eval_prompts, ensure_ascii=False)
    os.environ["OPENEVOLVE_USE_PRIORS"] = "alternate" if with_priors else ""
    os.environ["OPTPILOT_PROJECT_ROOT"] = str(Path(__file__).resolve().parents[1])

    # Ensure Together AI API key is visible to SkyDiscover
    # SkyDiscover's "together" provider reads: TOGETHER_API_KEY, TOGETHER_AI_API_KEY, together_ai_api
    if TOGETHER_API_KEY and not os.environ.get("TOGETHER_API_KEY"):
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    # ------------------------------------------------------------------
    # Print experiment info
    # ------------------------------------------------------------------
    print("=" * 65)
    mode_label = "WITH PRIORS" if with_priors else "BLIND"
    print(f"  OpenEvolve {mode_label} — Target MAS: {target_mas}")
    print("=" * 65)
    print(f"  Priors:         {'Yes' if with_priors else 'No (blind baseline)'}")
    print(f"  Model:          {model}")
    print(f"  DAG:            {original_dag.dag_id}")
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

    evaluator_file = EVALUATOR_FILE if target_mas == "ag2" else EVALUATOR_MULTI_FILE
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
    # Shadow gate: select a structurally valid discovered candidate that
    # does not regress on held-out train tasks before touching the test set.
    # ------------------------------------------------------------------
    eval_runner = OptPilotRunner(
        dag=original_dag,
        model=model,
        benchmark_name=benchmark_label,
        score_fn=custom_score_fn or train_suite.score_task,
        benchmark_name_resolver=train_suite.benchmark_name_for_task,
        timeout=timeout,
        tool_setup_fn=tool_setup_fn,
    )

    shadow_selection = {
        "shadow_used": bool(shadow_examples),
        "baseline_shadow_accuracy": None,
        "selected_candidate": "original_dag",
        "selected_shadow_accuracy": None,
        "selected_train_combined_score": None,
        "candidates_evaluated": [],
    }
    selected_dag = original_dag
    selected_code = ""

    final_best_code = result.best_solution
    (dag_versions_dir / "openevolve_best.py").write_text(final_best_code, encoding="utf-8")

    shortlist = _load_shadow_candidates(Path(openevolve_output_dir))
    if not shortlist:
        shortlist = [
            ShadowCandidate(
                name="final_best",
                code=final_best_code,
                train_combined_score=float(result.best_score or 0.0),
                train_accuracy=float(result.metrics.get("accuracy", 0.0) or 0.0),
                iteration=iterations,
                source_path=dag_versions_dir / "openevolve_best.py",
            )
        ]

    if shadow_examples:
        baseline_shadow_stats = asyncio.run(
            _eval_on_test(
                eval_runner,
                original_dag,
                shadow_suite,
                concurrency,
                "SHADOW BASELINE",
                output_base=artifact_dir / "shadow_baseline",
            )
        )
        baseline_shadow_acc = baseline_shadow_stats["overall_accuracy"]
        shadow_selection["baseline_shadow_accuracy"] = baseline_shadow_acc

        best_shadow_acc = baseline_shadow_acc
        best_train_score = float("-inf")
        for cand in shortlist:
            cand_record = {
                "name": cand.name,
                "iteration": cand.iteration,
                "train_combined_score": cand.train_combined_score,
                "train_accuracy": cand.train_accuracy,
                "valid": False,
                "shadow_accuracy": None,
            }
            try:
                cand_dag = _dag_from_python_source(cand.code)
            except Exception as e:
                cand_record["error"] = str(e)
                shadow_selection["candidates_evaluated"].append(cand_record)
                continue

            cand_record["valid"] = True
            shadow_stats = asyncio.run(
                _eval_on_test(
                    eval_runner,
                    cand_dag,
                    shadow_suite,
                    concurrency,
                    f"SHADOW {cand.name}",
                    output_base=artifact_dir / "shadow_candidates" / cand.name,
                )
            )
            cand_shadow_acc = shadow_stats["overall_accuracy"]
            cand_record["shadow_accuracy"] = cand_shadow_acc
            shadow_selection["candidates_evaluated"].append(cand_record)

            if (
                cand_shadow_acc > best_shadow_acc
                or (cand_shadow_acc == best_shadow_acc and cand.train_combined_score > best_train_score)
            ):
                best_shadow_acc = cand_shadow_acc
                best_train_score = cand.train_combined_score
                selected_dag = cand_dag
                selected_code = cand.code
                shadow_selection["selected_candidate"] = cand.name
                shadow_selection["selected_shadow_accuracy"] = cand_shadow_acc
                shadow_selection["selected_train_combined_score"] = cand.train_combined_score

        if shadow_selection["selected_candidate"] == "original_dag":
            print("\nShadow gate rejected all discovered candidates. Using original DAG for test evaluation.")
        else:
            print(
                f"\nShadow gate selected {shadow_selection['selected_candidate']} "
                f"(shadow acc {baseline_shadow_acc:.3f} -> {shadow_selection['selected_shadow_accuracy']:.3f})"
            )
    else:
        print("\nShadow gate skipped: no held-out train tasks beyond the OpenEvolve eval set.")
        try:
            selected_dag = _dag_from_python_source(final_best_code)
            selected_code = final_best_code
            shadow_selection["selected_candidate"] = "final_best_no_shadow"
            shadow_selection["selected_train_combined_score"] = float(result.best_score or 0.0)
        except Exception as e:
            print(f"Warning: Could not parse best solution as MASDAG: {e}")
            print("Using original DAG for test evaluation.")
            selected_dag = original_dag

    if selected_code:
        (dag_versions_dir / "shadow_selected.py").write_text(selected_code, encoding="utf-8")
        selected_dag.save(dag_versions_dir / "shadow_selected.yaml")
    print(f"\n  Final selected DAG agents: {', '.join(selected_dag.agent_nodes.keys())}")

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
            test_runner, selected_dag, test_suite, concurrency,
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
        "experiment": f"{target_mas}_openevolve{'_priors' if with_priors else '_blind'}",
        "target_mas": target_mas,
        "target_mas_name": target_mas,
        "with_priors": with_priors,
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
        "shadow_selection": shadow_selection,
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
        description="OpenEvolve cold-start experiment for multiple target MAS presets",
    )
    parser.add_argument("--target-mas", dest="target_mas", default="ag2",
                        help=f"Legacy target MAS preset (choices: {list(TARGET_MASES.keys())})")
    parser.add_argument("--topology", default=None, choices=list(TOPOLOGIES.keys()),
                        help=f"Topology to evolve (choices: {list(TOPOLOGIES.keys())}). Use with --benchmark.")
    parser.add_argument("--benchmark", default=None, choices=list(BENCHMARKS),
                        help=f"Benchmark to evaluate on (choices: {list(BENCHMARKS)}). Use with --topology.")
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML (overrides target_mas default)")
    parser.add_argument("--train", type=int, default=100, help="Train set size")
    parser.add_argument("--test", type=int, default=100, help="Test set size")
    parser.add_argument("--iterations", type=int, default=50, help="OpenEvolve iterations")
    parser.add_argument("--eval-tasks", type=int, default=20, help="Tasks per evaluation")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model ID on Together AI")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    parser.add_argument("--with-priors", dest="with_priors", action="store_true",
                        help="Inject prior experience (Jacobian/recipes/negatives) into evaluator feedback")
    args = parser.parse_args()

    run(
        dag_path=args.dag,
        target_mas=args.target_mas,
        topology=args.topology,
        benchmark=args.benchmark,
        n_train=args.train,
        n_test=args.test,
        iterations=args.iterations,
        eval_tasks=args.eval_tasks,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        with_priors=args.with_priors,
    )
