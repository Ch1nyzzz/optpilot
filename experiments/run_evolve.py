"""Target-MAS evolutionary search experiment.

Evolves target MAS DAGs via SkyDiscover search backends (OpenEvolve, AdaEvolve, etc.).
Supports multiple target MAS presets and topology × benchmark combinations.

Usage:
    # Linear topology on GAIA with AdaEvolve
    python -m experiments.run_evolve --topology linear --benchmark gaia --search adaevolve --iterations 50

    # Star topology on GAIA with OpenEvolve (default)
    python -m experiments.run_evolve --topology star --benchmark gaia --iterations 50

    # Legacy presets
    python -m experiments.run_evolve --target-mas ag2 --iterations 50
    python -m experiments.run_evolve --target-mas magentic --search adaevolve --iterations 50
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

# Config files per search backend
CONFIG_DIR = Path(__file__).parent
CONFIG_FILES: dict[str, Path] = {
    "openevolve_native": CONFIG_DIR / "openevolve_config.yaml",
    "adaevolve": CONFIG_DIR / "evolve_config_adaevolve.yaml",
}
EVALUATOR_FILE = CONFIG_DIR / "openevolve_evaluator.py"
EVALUATOR_MULTI_FILE = CONFIG_DIR / "openevolve_evaluator_multi.py"

# Target MAS → (initial program, result prefix) — legacy presets
TARGET_MASES: dict[str, tuple[str, str]] = {
    "ag2": (
        "openevolve_initial_dag.py",
        "ag2_mathchat",
    ),
    "appworld": (
        "openevolve_initial_dag_appworld.py",
        "appworld_star",
    ),
    "hyperagent": (
        "openevolve_initial_dag_hyperagent.py",
        "hyperagent_hierarchical",
    ),
    "magentic": (
        "openevolve_initial_dag_magentic.py",
        "magentic_one_star",
    ),
    "simple_star": (
        "openevolve_initial_dag_simple_star.py",
        "simple_star_gaia",
    ),
    "simple_hier": (
        "openevolve_initial_dag_simple_hier.py",
        "simple_hier_swebench",
    ),
    "agentcoder": (
        "openevolve_initial_dag_agentcoder.py",
        "agentcoder_humaneval",
    ),
}

# Topology → initial program filename (decoupled from benchmark)
# Only two topologies: linear (no hub) and star (hub).
# Evolution adds loops autonomously, so we don't pre-distinguish loop variants.
TOPOLOGIES: dict[str, str] = {
    "linear": "openevolve_initial_dag_linear.py",
    "star": "openevolve_initial_dag_star.py",
}

# Benchmark IDs supported for topology × benchmark experiments
BENCHMARKS = ("math", "humaneval", "gaia", "swebench", "livecodebench")

# Supported search backends
SEARCH_BACKENDS = ("openevolve_native", "adaevolve", "gepa_native")


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

    elif benchmark == "humaneval":
        from optpilot.data.benchmarks_humaneval import load_humaneval_examples, score_humaneval
        from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools as ac_build
        all_examples = load_humaneval_examples(total)
        _lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup_fn(task_prompt):
            return ac_build(CodeExecutionEnvironment())

        def score_fn(task_prompt, _dag, exec_trace):
            ex = _lookup.get(task_prompt)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_humaneval(pred, ex)

        return all_examples, "HumanEval", score_fn, tool_setup_fn

    elif benchmark == "gaia":
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        all_examples = load_gaia_examples(total, strict_supported_only=True)
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

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def run(
    dag_path: str | None = None,
    target_mas: str = "ag2",
    topology: str | None = None,
    benchmark: str | None = None,
    search: str = "openevolve_native",
    config_path: str | None = None,
    n_train: int = 100,
    n_test: int = 100,
    iterations: int = 50,
    eval_tasks: int = 20,
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 512,
    timeout: int = 600,
    reuse_diagnose_dir: str | None = None,
    with_priors: bool = False,
    skip_diagnose: bool = False,
):
    # Resolve config file: explicit > per-search-backend default > openevolve fallback
    if config_path:
        config_file = Path(config_path)
    elif search in CONFIG_FILES:
        config_file = CONFIG_FILES[search]
    else:
        config_file = CONFIG_FILES["openevolve_native"]

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Resolve topology × benchmark (new) or legacy target_mas
    if topology and benchmark:
        if topology not in TOPOLOGIES:
            raise ValueError(f"Unknown topology: {topology}. Available: {list(TOPOLOGIES.keys())}")
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARKS)}")
        initial_program_filename = TOPOLOGIES[topology]
        result_prefix = f"{topology}_{benchmark}_{search}"
        # For evaluator, combine as target_mas
        target_mas = f"{topology}_{benchmark}"
    elif target_mas in TARGET_MASES:
        initial_program_filename, result_prefix = TARGET_MASES[target_mas]
        result_prefix = f"{result_prefix}_{search}"
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
    evolve_output_dir = str(artifact_dir / "evolve_output")

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
            wrapper = AppWorldWrapper(task_id=ex.task_id, experiment_name="evolve")
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
        all_examples = load_gaia_examples(total, strict_supported_only=True)
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
    # Baseline run on train to find failure tasks for eval (early-stop)
    # ------------------------------------------------------------------
    print("Running baseline to identify failure tasks (early-stop)...")
    baseline_runner = OptPilotRunner(
        dag=original_dag,
        model=model,
        benchmark_name=benchmark_label,
        score_fn=custom_score_fn or train_suite.score_task,
        benchmark_name_resolver=train_suite.benchmark_name_for_task,
        timeout=timeout,
        tool_setup_fn=tool_setup_fn,
    )
    # Early-stop: run tasks one-by-one until we collect enough failures,
    # instead of running all 100 train tasks.
    train_prompts = train_suite.tasks()
    failed_prompts: list[str] = []
    evaluated_prompts: set[str] = set()
    n_evaluated = 0
    n_correct = 0

    async def _collect_failures():
        nonlocal n_evaluated, n_correct
        sem = asyncio.Semaphore(concurrency)
        pending: list[asyncio.Task] = []

        async def _run_one(prompt: str) -> tuple[str, float]:
            async with sem:
                trace = await baseline_runner.arun_task(prompt, dag=original_dag)
                score = trace.task_score if trace.task_score else 0.0
                return prompt, score

        # Launch tasks in batches, stop when we have enough failures
        batch_size = min(concurrency, len(train_prompts))
        i = 0
        while len(failed_prompts) < eval_tasks and i < len(train_prompts):
            # Launch a batch
            batch_end = min(i + batch_size, len(train_prompts))
            batch = [_run_one(train_prompts[j]) for j in range(i, batch_end)]
            results = await asyncio.gather(*batch)
            for prompt, score in results:
                n_evaluated += 1
                evaluated_prompts.add(prompt)
                if score <= 0:
                    failed_prompts.append(prompt)
                else:
                    n_correct += 1
                status = f"failures={len(failed_prompts)}/{eval_tasks}"
                print(f"  Evaluated {n_evaluated}/{len(train_prompts)}: "
                      f"{prompt[:50]}... [score={score:.1f}] ({status})")
                if len(failed_prompts) >= eval_tasks:
                    break
            i = batch_end

    asyncio.run(_collect_failures())

    print(f"  Baseline: {n_correct}/{n_evaluated} correct, "
          f"{len(failed_prompts)} failures (early-stopped)")
    selected_eval_prompts = failed_prompts[:eval_tasks]
    if len(selected_eval_prompts) < eval_tasks:
        # Not enough failures after all train tasks, pad with random unevaluated tasks
        remaining = [p for p in train_prompts if p not in evaluated_prompts]
        selected_eval_prompts.extend(remaining[:eval_tasks - len(selected_eval_prompts)])
    print(f"  Eval set: {len(selected_eval_prompts)} tasks")
    # Shadow: random sample from unevaluated train tasks (not used in eval)
    eval_set = set(selected_eval_prompts)
    unevaluated = [ex for ex in train_examples if ex.prompt not in eval_set and ex.prompt not in evaluated_prompts]
    import random
    random.seed(42)
    n_shadow = min(eval_tasks, len(unevaluated))  # same size as eval set
    shadow_examples = random.sample(unevaluated, n_shadow) if unevaluated else []
    # Also include evaluated non-failure tasks in shadow
    evaluated_nonfail = [ex for ex in train_examples if ex.prompt in evaluated_prompts and ex.prompt not in eval_set]
    shadow_examples.extend(evaluated_nonfail)
    shadow_suite = OfficialBenchmarkSuite(shadow_examples)
    print(f"  Shadow set: {len(shadow_examples)} tasks ({n_shadow} unevaluated + {len(evaluated_nonfail)} evaluated)")

    # Persist eval prompts so analyze_openevolve_traces can reconstruct the shadow set
    eval_prompts_path = artifact_dir / "eval_prompts.json"
    eval_prompts_path.write_text(
        json.dumps(selected_eval_prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
    os.environ["OPENEVOLVE_GAIA_SUPPORTED_ONLY"] = "strict" if benchmark == "gaia" else ""
    os.environ["OPENEVOLVE_EVAL_PROMPTS_JSON"] = json.dumps(selected_eval_prompts, ensure_ascii=False)
    os.environ["OPENEVOLVE_USE_PRIORS"] = "alternate" if with_priors else ""
    os.environ["OPENEVOLVE_SKIP_DIAGNOSE"] = "1" if skip_diagnose else ""
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
    print(f"  Evolve ({search}) {mode_label} — Target MAS: {target_mas}")
    print("=" * 65)
    print(f"  Search:         {search}")
    print(f"  Config:         {config_file}")
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
    # Run evolutionary search
    # ------------------------------------------------------------------
    print(f"Starting evolution ({search})...")
    t0 = time.time()

    evaluator_file = EVALUATOR_FILE if target_mas == "ag2" else EVALUATOR_MULTI_FILE
    result = run_discovery(
        evaluator=str(evaluator_file),
        initial_program=initial_program_path,
        iterations=iterations,
        search=search,
        config=str(config_file),
        output_dir=evolve_output_dir,
        cleanup=False,
    )

    evolution_time = time.time() - t0
    print(f"\nEvolution ({search}) completed in {evolution_time:.1f}s")
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
    (dag_versions_dir / "evolve_best.py").write_text(final_best_code, encoding="utf-8")

    shortlist = _load_shadow_candidates(Path(evolve_output_dir))
    if not shortlist:
        shortlist = [
            ShadowCandidate(
                name="final_best",
                code=final_best_code,
                train_combined_score=float(result.best_score or 0.0),
                train_accuracy=float(result.metrics.get("accuracy", 0.0) or 0.0),
                iteration=iterations,
                source_path=dag_versions_dir / "evolve_best.py",
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
        print("\nShadow gate skipped: no held-out train tasks beyond the eval set.")
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
        n_runs = 3
        final_runs = []
        baseline_runs = []
        for run_i in range(1, n_runs + 1):
            print(f"\n=== Test evaluation run {run_i}/{n_runs} ===")
            final = await _eval_on_test(
                test_runner, selected_dag, test_suite, concurrency,
                f"EVOLVED BEST (run {run_i})",
                output_base=artifact_dir / f"test_final_run{run_i}",
            )
            baseline = await _eval_on_test(
                test_runner, original_dag, test_suite, concurrency,
                f"BASELINE (run {run_i})",
                output_base=artifact_dir / f"test_baseline_run{run_i}",
            )
            final_runs.append(final)
            baseline_runs.append(baseline)

        # Average across runs
        def _average_stats(runs: list[dict]) -> dict:
            avg_acc = sum(r["overall_accuracy"] for r in runs) / len(runs)
            avg_elapsed = sum(r["elapsed_s"] for r in runs) / len(runs)
            # Average per-benchmark accuracies
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

        final_avg = _average_stats(final_runs)
        baseline_avg = _average_stats(baseline_runs)
        accs = [r["overall_accuracy"] for r in final_runs]
        print(f"\n  Test accuracy across {n_runs} runs: {accs} → mean={final_avg['overall_accuracy']:.4f}")
        return final_avg, baseline_avg

    test_stats, baseline_test_stats = asyncio.run(_run_test_evals())

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result_path = RESULTS_DIR / f"{result_stem}.json"
    output = {
        "experiment": f"{target_mas}_{search}{'_priors' if with_priors else '_blind'}",
        "target_mas": target_mas,
        "search_backend": search,
        "config_file": str(config_file),
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
            "output_dir": evolve_output_dir,
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
    print(f"  Summary — Evolve [{search}] ({iterations} iterations)")
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
        description="Evolutionary search experiment for MAS DAGs (supports OpenEvolve, AdaEvolve, etc.)",
    )
    parser.add_argument("--target-mas", dest="target_mas", default="ag2",
                        help=f"Legacy target MAS preset (choices: {list(TARGET_MASES.keys())})")
    parser.add_argument("--topology", default=None, choices=list(TOPOLOGIES.keys()),
                        help=f"Topology to evolve (choices: {list(TOPOLOGIES.keys())}). Use with --benchmark.")
    parser.add_argument("--benchmark", default=None, choices=list(BENCHMARKS),
                        help=f"Benchmark to evaluate on (choices: {list(BENCHMARKS)}). Use with --topology.")
    parser.add_argument("--search", default="openevolve_native", choices=list(SEARCH_BACKENDS),
                        help=f"Search backend (choices: {list(SEARCH_BACKENDS)}). Default: openevolve_native")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (overrides per-backend default)")
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML (overrides target_mas default)")
    parser.add_argument("--train", type=int, default=100, help="Train set size")
    parser.add_argument("--test", type=int, default=100, help="Test set size")
    parser.add_argument("--iterations", type=int, default=50, help="Evolution iterations")
    parser.add_argument("--eval-tasks", type=int, default=20, help="Tasks per evaluation")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model ID on Together AI")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    parser.add_argument("--with-priors", dest="with_priors", action="store_true",
                        help="Inject prior experience (Jacobian/recipes/negatives) into evaluator feedback")
    parser.add_argument("--skip-diagnose", dest="skip_diagnose", action="store_true",
                        help="Skip FM diagnosis, use pure accuracy as fitness score")
    args = parser.parse_args()

    run(
        dag_path=args.dag,
        target_mas=args.target_mas,
        topology=args.topology,
        benchmark=args.benchmark,
        search=args.search,
        config_path=args.config,
        n_train=args.train,
        n_test=args.test,
        iterations=args.iterations,
        eval_tasks=args.eval_tasks,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        with_priors=args.with_priors,
        skip_diagnose=args.skip_diagnose,
    )
