"""Multi-topology OpenEvolve evaluator.

Supports all topologies (ag2, appworld, hyperagent, magentic) by loading
the appropriate benchmark and tool registry based on OPENEVOLVE_TOPOLOGY env var.

Implements the evaluate(program_path) -> dict interface for SkyDiscover.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import TOGETHER_API_KEY  # noqa: E402
from optpilot.dag.core import MASDAG  # noqa: E402
from optpilot.data.benchmarks import OfficialBenchmarkSuite, BenchmarkExample  # noqa: E402
from optpilot.data.fm_taxonomy_6group import GROUP_IDS  # noqa: E402
from optpilot.modules.diagnoser import Diagnoser  # noqa: E402
from optpilot.modules.runner import OptPilotRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
_MODEL = os.environ.get("OPENEVOLVE_MODEL", "openai/gpt-oss-120b")
_CONCURRENCY = int(os.environ.get("OPENEVOLVE_CONCURRENCY", "512"))
_TIMEOUT = int(os.environ.get("OPENEVOLVE_TIMEOUT", "600"))
_EVAL_TASKS = int(os.environ.get("OPENEVOLVE_EVAL_TASKS", "20"))
_TOTAL_TASKS = int(os.environ.get("OPENEVOLVE_TOTAL_TASKS", "200"))
_EVAL_PROMPTS_JSON = os.environ.get("OPENEVOLVE_EVAL_PROMPTS_JSON", "")
_TOPOLOGY = os.environ.get("OPENEVOLVE_TOPOLOGY", "ag2")

# Lazy globals
_suite: OfficialBenchmarkSuite | None = None
_eval_prompts: list[str] | None = None
_runner: OptPilotRunner | None = None
_diagnoser: Diagnoser | None = None
_tool_setup_fn = None
_score_fn_custom = None


def _load_ag2():
    """Load AG2 MathChat benchmark (original)."""
    from optpilot.data.benchmarks import load_online_benchmark_suite
    full_suite = load_online_benchmark_suite(_TOTAL_TASKS)
    return full_suite, full_suite.score_task, full_suite.benchmark_name_for_task, None


def _load_appworld():
    """Load AppWorld benchmark with official AppWorld API server."""
    from optpilot.data.benchmarks_appworld import load_appworld_examples, score_appworld
    from optpilot.tools.appworld_tools import AppWorldWrapper, build_tools

    examples = load_appworld_examples(_TOTAL_TASKS)
    suite = OfficialBenchmarkSuite(examples)
    example_lookup = {ex.prompt: ex for ex in examples}

    def tool_setup(task_prompt: str):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return None
        wrapper = AppWorldWrapper(task_id=ex.task_id, experiment_name="openevolve_eval")
        return build_tools(wrapper)

    def score_fn(task_prompt, dag, exec_trace):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return 0.0
        pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
        return score_appworld(pred, ex.gold_answers[0])

    def bench_resolver(task_prompt):
        return "AppWorld"

    return suite, score_fn, bench_resolver, tool_setup


def _load_hyperagent():
    """Load SWE-bench Lite benchmark with code tools."""
    from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench

    examples = load_swebench_examples(_TOTAL_TASKS)
    suite = OfficialBenchmarkSuite(examples)
    example_lookup = {ex.prompt: ex for ex in examples}

    def tool_setup(task_prompt: str):
        from optpilot.tools.hyperagent_tools import CodeEnvironment, build_tools
        ex = example_lookup.get(task_prompt)
        repo = ex.metadata.get("repo", "") if ex else ""
        base_commit = ex.metadata.get("base_commit", "") if ex else ""
        env = CodeEnvironment(repo=repo, base_commit=base_commit)
        return build_tools(env)

    def score_fn(task_prompt, dag, exec_trace):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return 0.0
        pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
        return score_swebench(pred, ex.gold_answers[0])

    def bench_resolver(task_prompt):
        return "SWE-bench-Lite"

    return suite, score_fn, bench_resolver, tool_setup


def _load_magentic():
    """Load GAIA benchmark with general tools."""
    from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
    from optpilot.tools.magentic_tools import GeneralEnvironment, build_tools

    examples = load_gaia_examples(_TOTAL_TASKS)
    suite = OfficialBenchmarkSuite(examples)
    example_lookup = {ex.prompt: ex for ex in examples}

    def tool_setup(task_prompt: str):
        ex = example_lookup.get(task_prompt)
        context_docs = ex.metadata.get("context_docs", {}) if ex else {}
        env = GeneralEnvironment(context_docs)
        return build_tools(env)

    def score_fn(task_prompt, dag, exec_trace):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return 0.0
        pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
        return score_gaia(pred, ex.gold_answers[0])

    def bench_resolver(task_prompt):
        return "GAIA"

    return suite, score_fn, bench_resolver, tool_setup


_LOADERS = {
    "ag2": _load_ag2,
    "appworld": _load_appworld,
    "hyperagent": _load_hyperagent,
    "magentic": _load_magentic,
    "simple_star": _load_magentic,    # reuses GAIA benchmark + magentic tools
    "simple_hier": _load_hyperagent,  # reuses SWE-bench + hyperagent tools
}


def _init_globals() -> None:
    global _suite, _eval_prompts, _runner, _diagnoser, _tool_setup_fn, _score_fn_custom

    if _suite is not None:
        return

    loader = _LOADERS.get(_TOPOLOGY, _load_ag2)
    suite, score_fn, bench_resolver, tool_setup = loader()

    # Filter to eval prompts if specified
    eval_examples = suite.examples[:_EVAL_TASKS]
    if _EVAL_PROMPTS_JSON:
        try:
            requested_prompts = json.loads(_EVAL_PROMPTS_JSON)
        except json.JSONDecodeError:
            requested_prompts = []
        if isinstance(requested_prompts, list) and requested_prompts:
            requested_set = {str(p) for p in requested_prompts}
            eval_examples = [ex for ex in suite.examples if ex.prompt in requested_set]
            order = {str(p): i for i, p in enumerate(requested_prompts)}
            eval_examples.sort(key=lambda ex: order.get(ex.prompt, len(order)))

    _suite = OfficialBenchmarkSuite(eval_examples)
    _eval_prompts = _suite.tasks()
    _tool_setup_fn = tool_setup
    _score_fn_custom = score_fn

    _runner = OptPilotRunner(
        dag=None,
        model=_MODEL,
        benchmark_name=_TOPOLOGY.upper(),
        score_fn=score_fn,
        benchmark_name_resolver=bench_resolver,
        timeout=_TIMEOUT,
        tool_setup_fn=tool_setup,
    )
    _diagnoser = Diagnoser()


async def _classify_batch(traces: list) -> list:
    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
    from optpilot.models import FMLabel, FMProfile
    assert _diagnoser is not None

    sem = asyncio.Semaphore(64)

    async def _classify_one(idx, trace):
        async with sem:
            labels_dict = await _diagnoser._aclassify(trace)
            profile = FMProfile(trace_id=trace.trace_id)
            for gid in GROUP_IDS:
                gdef = GROUP_DEFINITIONS[gid]
                profile.labels[gid] = FMLabel(
                    fm_id=gid, fm_name=gdef["name"], category=gid,
                    present=labels_dict.get(gid, False),
                )
            return idx, profile

    results = {}
    tasks = [_classify_one(i, t) for i, t in enumerate(traces)]
    for coro in asyncio.as_completed(tasks):
        idx, profile = await coro
        results[idx] = profile
    return [results[i] for i in range(len(traces))]


def _score_task(is_correct: bool, total_failures: int, num_agents: int) -> float:
    s = 1.0 / (1.0 + total_failures)
    if is_correct:
        s *= 1.2
    if num_agents > 4:
        s -= 0.01 * (num_agents - 4)
    return max(0.0, s)


def evaluate(program_path: str) -> dict:
    """Evaluate a candidate DAG."""
    _init_globals()
    assert _runner is not None and _diagnoser is not None
    assert _eval_prompts is not None and _suite is not None

    # 1. Load candidate DAG
    try:
        spec = importlib.util.spec_from_file_location("_candidate_dag", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_dag"):
            raise AttributeError("No build_dag() function found")
        dag = MASDAG.from_dict(module.build_dag())
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Code execution error: {e}",
            "artifacts": {"feedback": f"DAG build failed:\n{e}\n{traceback.format_exc()}"},
        }

    num_agents = len(dag.agent_nodes)

    # 2. Run MAS
    try:
        traces = asyncio.run(
            _runner.arun_batch(_eval_prompts, dag=dag, max_concurrency=_CONCURRENCY)
        )
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Execution error: {e}",
            "artifacts": {"feedback": f"MAS execution failed:\n{e}\n{traceback.format_exc()}"},
        }

    # 3. Classify FM
    try:
        profiles = asyncio.run(_classify_batch(traces))
    except Exception:
        profiles = []

    # 4. Score
    n = len(traces)
    correct_count = sum(1 for t in traces if t.task_score and t.task_score > 0)
    accuracy = correct_count / n if n > 0 else 0.0

    fm_counts = {g: 0 for g in GROUP_IDS}
    for profile in profiles:
        for gid in GROUP_IDS:
            if gid in profile.labels and profile.labels[gid].present:
                fm_counts[gid] += 1
    fm_rates = {g: fm_counts[g] / n for g in GROUP_IDS} if n > 0 else {g: 0.0 for g in GROUP_IDS}
    total_failures = sum(fm_counts.values())

    task_scores = []
    for i, trace in enumerate(traces):
        is_correct = bool(trace.task_score and trace.task_score > 0)
        trace_failures = 0
        if i < len(profiles):
            trace_failures = sum(
                1 for gid in GROUP_IDS
                if gid in profiles[i].labels and profiles[i].labels[gid].present
            )
        task_scores.append(_score_task(is_correct, trace_failures, num_agents))
    combined_score = sum(task_scores) / len(task_scores) if task_scores else 0.0

    # 5. Feedback
    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
    sorted_fms = sorted(fm_rates.items(), key=lambda x: x[1], reverse=True)
    top_failures = [
        f"  Group {gid} ({GROUP_DEFINITIONS.get(gid, {}).get('name', gid)}): {rate:.0%}"
        for gid, rate in sorted_fms if rate > 0
    ]

    feedback_lines = [
        f"Topology: {_TOPOLOGY}",
        f"Accuracy: {accuracy:.1%} ({correct_count}/{n} correct)",
        f"Total failures: {total_failures} across {n} traces",
        f"Agents: {num_agents}",
        "", "FM rates:",
    ]
    feedback_lines.extend(top_failures or ["  No failures detected."])
    feedback_lines.append(f"\nCombined score: {combined_score:.4f}")

    metrics = {
        "combined_score": round(combined_score, 4),
        "accuracy": round(accuracy, 4),
        "total_failures": total_failures,
        "num_agents": num_agents,
    }
    for gid in GROUP_IDS:
        metrics[f"fm_{gid}"] = round(fm_rates[gid], 4)
    metrics["artifacts"] = {"feedback": "\n".join(feedback_lines)}
    return metrics
