"""OpenEvolve evaluator for AG2 MathChat DAG evolution.

Implements the evaluate(program_path) -> dict interface required by
SkyDiscover / OpenEvolve.  Each candidate is a Python file containing
a ``build_dag()`` function that returns a MASDAG dict.  The evaluator
executes the function, runs the MAS on benchmark tasks, diagnoses
failure modes using our 6-group taxonomy, and returns a MAST-style
fitness score plus per-group FM rates as artifacts.

Fitness formula (aligned with MAST blog):
    base  = 1.0 / (1.0 + total_failures)
    bonus = 1.2x  if task is correct
    penalty = -0.01 per agent above 4
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

# Ensure optpilot importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import TOGETHER_API_KEY, TOGETHER_BASE_URL  # noqa: E402
from optpilot.dag.core import MASDAG  # noqa: E402
from optpilot.data.benchmarks import (  # noqa: E402
    OfficialBenchmarkSuite,
    load_online_benchmark_suite,
)
from optpilot.data.fm_taxonomy_6group import GROUP_IDS  # noqa: E402
from optpilot.modules.diagnoser import Diagnoser  # noqa: E402
from optpilot.modules.runner import OptPilotRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration from environment (set by the experiment entry script)
# ---------------------------------------------------------------------------
_MODEL = os.environ.get("OPENEVOLVE_MODEL", "openai/gpt-oss-120b")
_CONCURRENCY = int(os.environ.get("OPENEVOLVE_CONCURRENCY", "512"))
_TIMEOUT = int(os.environ.get("OPENEVOLVE_TIMEOUT", "600"))
_EVAL_TASKS = int(os.environ.get("OPENEVOLVE_EVAL_TASKS", "20"))
_TOTAL_TASKS = int(os.environ.get("OPENEVOLVE_TOTAL_TASKS", "200"))

# Lazy-initialized globals (expensive to rebuild every call)
_suite: OfficialBenchmarkSuite | None = None
_eval_prompts: list[str] | None = None
_runner: OptPilotRunner | None = None
_diagnoser: Diagnoser | None = None


def _init_globals() -> None:
    """Initialize benchmark suite, runner, and diagnoser once."""
    global _suite, _eval_prompts, _runner, _diagnoser

    if _suite is not None:
        return

    full_suite = load_online_benchmark_suite(_TOTAL_TASKS)
    # Use the first _EVAL_TASKS examples as the per-iteration eval set
    eval_examples = full_suite.examples[:_EVAL_TASKS]
    _suite = OfficialBenchmarkSuite(eval_examples)
    _eval_prompts = _suite.tasks()

    _runner = OptPilotRunner(
        dag=None,
        model=_MODEL,
        benchmark_name="AG2_MathChat",
        score_fn=_suite.score_task,
        benchmark_name_resolver=_suite.benchmark_name_for_task,
        timeout=_TIMEOUT,
    )
    _diagnoser = Diagnoser()


async def _classify_batch(traces: list) -> list:
    """Classify traces into 6-group labels without localization (faster)."""
    assert _diagnoser is not None
    import asyncio as _aio
    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
    from optpilot.models import FMLabel, FMProfile

    sem = _aio.Semaphore(64)

    async def _classify_one(idx: int, trace) -> tuple[int, FMProfile]:
        async with sem:
            labels_dict = await _diagnoser._aclassify(trace)
            profile = FMProfile(trace_id=trace.trace_id)
            for gid in GROUP_IDS:
                gdef = GROUP_DEFINITIONS[gid]
                profile.labels[gid] = FMLabel(
                    fm_id=gid,
                    fm_name=gdef["name"],
                    category=gid,
                    present=labels_dict.get(gid, False),
                )
            return idx, profile

    results: dict[int, FMProfile] = {}
    tasks = [_classify_one(i, t) for i, t in enumerate(traces)]
    for coro in _aio.as_completed(tasks):
        idx, profile = await coro
        results[idx] = profile
    return [results[i] for i in range(len(traces))]


def _score_task(is_correct: bool, total_failures: int, num_agents: int) -> float:
    """MAST blog scoring formula."""
    s = 1.0 / (1.0 + total_failures)
    if is_correct:
        s *= 1.2
    if num_agents > 4:
        s -= 0.01 * (num_agents - 4)
    return max(0.0, s)


def evaluate(program_path: str) -> dict:
    """Evaluate a candidate DAG defined as Python code.

    Args:
        program_path: Path to a Python file containing a ``build_dag()``
            function that returns a MASDAG-compatible dict.

    Returns:
        Dictionary with ``combined_score``, per-group FM rates, and
        an ``artifacts`` dict containing human-readable feedback.
    """
    _init_globals()
    assert _runner is not None and _diagnoser is not None
    assert _eval_prompts is not None and _suite is not None

    # ------------------------------------------------------------------
    # 1. Execute the candidate Python code to get a MASDAG dict
    # ------------------------------------------------------------------
    try:
        spec = importlib.util.spec_from_file_location("_candidate_dag", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "build_dag"):
            raise AttributeError("No build_dag() function found in candidate code")

        dag_dict = module.build_dag()
        dag = MASDAG.from_dict(dag_dict)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Code execution error: {e}",
            "artifacts": {
                "feedback": (
                    f"The candidate Python code failed to produce a valid DAG.\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}\n\n"
                    f"The code must define a build_dag() function that returns "
                    f"a dict with dag_id, nodes, edges, and metadata fields."
                ),
            },
        }

    num_agents = len(dag.agent_nodes)

    # Enforce locked-verifier constraint (prevent reward hacking)
    has_verifier = any(
        "verif" in nid.lower() for nid in dag.agent_nodes
    )
    if not has_verifier:
        return {
            "combined_score": 0.0,
            "error": "Verification agent was deleted (constraint violation)",
            "artifacts": {
                "feedback": (
                    "CONSTRAINT VIOLATION: You deleted the verification agent.\n"
                    "The DAG must keep at least one verification/verifier agent.\n"
                    "This is a hard constraint to prevent reward hacking.\n"
                    "Restore the verifier and try a different optimization strategy."
                ),
            },
        }

    # ------------------------------------------------------------------
    # 2. Run the MAS on benchmark tasks
    # ------------------------------------------------------------------
    try:
        traces = asyncio.run(
            _runner.arun_batch(
                _eval_prompts,
                dag=dag,
                max_concurrency=_CONCURRENCY,
            )
        )
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Execution error: {e}",
            "artifacts": {
                "feedback": (
                    f"The MAS DAG failed during execution.\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                ),
            },
        }

    # ------------------------------------------------------------------
    # 3. Classify failure modes (no localization — OpenEvolve only needs FM rates)
    # ------------------------------------------------------------------
    try:
        profiles = asyncio.run(_classify_batch(traces))
    except Exception as e:
        # Fall back to no-diagnosis scoring
        profiles = []

    # ------------------------------------------------------------------
    # 4. Compute MAST-style fitness
    # ------------------------------------------------------------------
    n = len(traces)
    correct_count = sum(1 for t in traces if t.task_score and t.task_score > 0)
    accuracy = correct_count / n if n > 0 else 0.0

    # Per-group FM rates
    fm_counts: dict[str, int] = {g: 0 for g in GROUP_IDS}
    for profile in profiles:
        for gid in GROUP_IDS:
            if gid in profile.labels and profile.labels[gid].present:
                fm_counts[gid] += 1
    fm_rates = {g: fm_counts[g] / n for g in GROUP_IDS} if n > 0 else {g: 0.0 for g in GROUP_IDS}

    total_failures = sum(fm_counts.values())

    # Per-task score then average (aligned with blog)
    task_scores: list[float] = []
    for i, trace in enumerate(traces):
        is_correct = bool(trace.task_score and trace.task_score > 0)
        # Count failures for this trace
        trace_failures = 0
        if i < len(profiles):
            trace_failures = sum(
                1 for gid in GROUP_IDS
                if gid in profiles[i].labels and profiles[i].labels[gid].present
            )
        task_scores.append(_score_task(is_correct, trace_failures, num_agents))
    combined_score = sum(task_scores) / len(task_scores) if task_scores else 0.0

    # ------------------------------------------------------------------
    # 5. Build feedback artifacts for the LLM
    # ------------------------------------------------------------------
    # Identify most common failure groups
    sorted_fms = sorted(fm_rates.items(), key=lambda x: x[1], reverse=True)
    top_failures = [
        f"  Group {gid} ({_fm_name(gid)}): {rate:.0%} of traces"
        for gid, rate in sorted_fms if rate > 0
    ]

    feedback_lines = [
        f"Accuracy: {accuracy:.1%} ({correct_count}/{n} correct)",
        f"Total failure instances: {total_failures} across {n} traces",
        f"Number of agents: {num_agents}",
        "",
        "Failure mode rates:",
    ]
    if top_failures:
        feedback_lines.extend(top_failures)
    else:
        feedback_lines.append("  No failures detected.")

    feedback_lines.extend([
        "",
        "Scoring: base = 1/(1+failures), ×1.2 if correct, -0.01 per agent above 4",
        f"Combined score: {combined_score:.4f}",
    ])

    # Per-task detail for first few failures
    failure_details = []
    for i, trace in enumerate(traces):
        if trace.task_score and trace.task_score > 0:
            continue
        if len(failure_details) >= 3:
            break
        active_fms = []
        if i < len(profiles):
            active_fms = profiles[i].active_fm_ids()
        failure_details.append(
            f"  Task {i}: incorrect, FM groups: {active_fms or ['none']}, "
            f"key: {trace.task_key[:60]}"
        )
    if failure_details:
        feedback_lines.extend(["", "Sample failures:"] + failure_details)

    metrics = {
        "combined_score": round(combined_score, 4),
        "accuracy": round(accuracy, 4),
        "total_failures": total_failures,
        "num_agents": num_agents,
    }
    for gid in GROUP_IDS:
        metrics[f"fm_{gid}"] = round(fm_rates[gid], 4)

    metrics["artifacts"] = {
        "feedback": "\n".join(feedback_lines),
    }

    return metrics


def _fm_name(gid: str) -> str:
    """Human-readable FM group name."""
    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
    return GROUP_DEFINITIONS.get(gid, {}).get("name", gid)
