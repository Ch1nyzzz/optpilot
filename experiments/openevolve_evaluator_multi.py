"""Target-MAS OpenEvolve evaluator.

Supports all target MAS presets (ag2, appworld, hyperagent, magentic) by loading
the appropriate benchmark and tool registry based on OPENEVOLVE_TARGET_MAS.

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
_TARGET_MAS = os.environ.get("OPENEVOLVE_TARGET_MAS", "ag2")
_BENCHMARK = os.environ.get("OPENEVOLVE_BENCHMARK", "")  # topology×benchmark mode
_GAIA_SUPPORTED_ONLY = os.environ.get("OPENEVOLVE_GAIA_SUPPORTED_ONLY", "").lower() == "strict"
_USE_PRIORS_RAW = os.environ.get("OPENEVOLVE_USE_PRIORS", "").lower()
_USE_PRIORS_ALWAYS = _USE_PRIORS_RAW in ("1", "true", "yes")
_USE_PRIORS_ALTERNATE = _USE_PRIORS_RAW == "alternate"
_SKIP_DIAGNOSE = os.environ.get("OPENEVOLVE_SKIP_DIAGNOSE", "").lower() in ("1", "true", "yes")
_eval_counter = 0

# Meta-diagnoser state
# Threshold is per-evaluation call, not per-iteration.
# With max_parallel_iterations=3, 15 evals ≈ 5 iterations of stagnation.
_META_STAGNATION_THRESHOLD = 15
# Direct Together client calls expect bare model ids, not SkyDiscover's
# provider-prefixed "together/..." form.
_META_MODEL = "zai-org/GLM-5"
_meta_best_score: float = 0.0
_meta_best_code: str = ""  # source code of the best program
_meta_best_task_results: dict[str, dict] = {}  # task_hash -> {score, output_snippet}
_meta_stagnation_count: int = 0
_meta_recent_mutations: list[dict] = []  # last few mutations' task results

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

    examples = load_gaia_examples(
        _TOTAL_TASKS,
        strict_supported_only=_GAIA_SUPPORTED_ONLY,
    )
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


def _load_agentcoder():
    """Load HumanEval benchmark with code execution tools."""
    from optpilot.data.benchmarks_humaneval import load_humaneval_examples, score_humaneval
    from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools

    examples = load_humaneval_examples(_TOTAL_TASKS)
    suite = OfficialBenchmarkSuite(examples)
    example_lookup = {ex.prompt: ex for ex in examples}

    def tool_setup(task_prompt: str):
        return build_tools(CodeExecutionEnvironment())

    def score_fn(task_prompt, dag, exec_trace):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return 0.0
        pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
        return score_humaneval(pred, ex)

    def bench_resolver(task_prompt):
        return "HumanEval"

    return suite, score_fn, bench_resolver, tool_setup


def _load_livecodebench():
    """Load LiveCodeBench with code-execution scoring."""
    from optpilot.data.benchmarks_livecodebench import load_livecodebench_examples, score_livecodebench
    from optpilot.tools.agentcoder_tools import CodeExecutionEnvironment, build_tools

    examples = load_livecodebench_examples(_TOTAL_TASKS)
    suite = OfficialBenchmarkSuite(examples)
    example_lookup = {ex.prompt: ex for ex in examples}

    def tool_setup(task_prompt: str):
        return build_tools(CodeExecutionEnvironment())

    def score_fn(task_prompt, dag, exec_trace):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return 0.0
        pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
        return score_livecodebench(pred, ex)

    def bench_resolver(task_prompt):
        return "LiveCodeBench"

    return suite, score_fn, bench_resolver, tool_setup


def _load_hotpotqa():
    """Load HotpotQA benchmark (no special tools)."""
    from optpilot.data.benchmarks_hotpotqa import load_hotpotqa_examples, score_hotpotqa

    examples = load_hotpotqa_examples(_TOTAL_TASKS)
    suite = OfficialBenchmarkSuite(examples)
    example_lookup = {ex.prompt: ex for ex in examples}

    def score_fn(task_prompt, dag, exec_trace):
        ex = example_lookup.get(task_prompt)
        if ex is None:
            return 0.0
        pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
        return score_hotpotqa(pred, ex.gold_answers[0])

    def bench_resolver(task_prompt):
        return "HotpotQA"

    return suite, score_fn, bench_resolver, None


_LOADERS = {
    "ag2": _load_ag2,
    "appworld": _load_appworld,
    "hyperagent": _load_hyperagent,
    "magentic": _load_magentic,
    "simple_star": _load_magentic,    # reuses GAIA benchmark + magentic tools
    "simple_hier": _load_hyperagent,  # reuses SWE-bench + hyperagent tools
    "agentcoder": _load_agentcoder,   # HumanEval + code execution tools
    "livecodebench": _load_livecodebench,  # LiveCodeBench + code execution
}

# Benchmark-only loaders (for topology × benchmark mode)
_BENCHMARK_LOADERS = {
    "math": _load_ag2,
    "humaneval": _load_agentcoder,
    "gaia": _load_magentic,
    "swebench": _load_hyperagent,
    "livecodebench": _load_livecodebench,
}


# ---------------------------------------------------------------------------
# Prior-guided feedback
# ---------------------------------------------------------------------------
_priors_cache: dict | None = None


def _load_priors() -> dict:
    """Load all prior layers from global experience store."""
    global _priors_cache
    if _priors_cache is not None:
        return _priors_cache

    from optpilot.config import CATALOG_PATH, JACOBIAN_DIR, LIBRARY_DIR, NEGATIVES_DIR, RECIPES_DIR

    priors: dict = {}

    # Layer 1: Jacobian matrix (action effectiveness per FM group)
    matrix_path = JACOBIAN_DIR / "matrix.json"
    if matrix_path.exists():
        priors["jacobian"] = json.loads(matrix_path.read_text(encoding="utf-8"))

    # Layer 2: Data-driven priors (cold-start scores)
    ddp_path = JACOBIAN_DIR / "data_driven_priors.json"
    if ddp_path.exists():
        priors["data_driven_priors"] = json.loads(ddp_path.read_text(encoding="utf-8"))

    # Layer 3: Recipes (concrete repair actions per FM group)
    recipes: dict[str, list] = {}
    if RECIPES_DIR.exists():
        for rfile in RECIPES_DIR.glob("*.json"):
            group_id = rfile.stem  # e.g. "B", "F"
            if group_id.startswith("_"):
                continue
            try:
                recipes[group_id] = json.loads(rfile.read_text(encoding="utf-8"))
            except Exception:
                pass
    priors["recipes"] = recipes

    # Layer 4: Pattern catalog (effective / deprecated patterns)
    if CATALOG_PATH.exists():
        priors["pattern_catalog"] = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    # Layer 5: Negatives (lessons from failed repairs)
    negatives: dict[str, list] = {}
    if NEGATIVES_DIR.exists():
        for nfile in NEGATIVES_DIR.glob("negatives_*.json"):
            group_id = nfile.stem.replace("negatives_", "")
            try:
                negatives[group_id] = json.loads(nfile.read_text(encoding="utf-8"))
            except Exception:
                pass
    priors["negatives"] = negatives

    _priors_cache = priors
    return priors


def _build_prior_guidance(fm_rates: dict[str, float]) -> str:
    """Build actionable repair guidance from priors for top FM groups."""
    priors = _load_priors()
    if not priors:
        return ""

    # Sort FM groups by rate, only include those with non-zero rate
    active_fms = sorted(
        [(gid, rate) for gid, rate in fm_rates.items() if rate > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    if not active_fms:
        return ""

    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS

    jacobian = priors.get("jacobian", {})
    ddp = priors.get("data_driven_priors", {})
    recipes = priors.get("recipes", {})
    catalog = priors.get("pattern_catalog", {})
    negatives = priors.get("negatives", {})

    lines = [
        "",
        "=== REPAIR GUIDANCE (from prior experience on similar systems) ===",
        f"Target MAS: {_TARGET_MAS}",
        "",
    ]

    # Focus on top 3 FM groups
    for gid, rate in active_fms[:3]:
        gname = GROUP_DEFINITIONS.get(gid, {}).get("name", gid)
        lines.append(f"## FM Group {gid} ({gname}) — {rate:.0%}")

        # Jacobian: rank actions by success rate
        j_data = jacobian.get(gid, {})
        if j_data:
            action_stats = []
            for action, stats in j_data.items():
                if isinstance(stats, dict) and stats.get("n_applied", 0) > 0:
                    n_app = stats["n_applied"]
                    n_suc = stats["n_success"]
                    avg_fm_delta = stats.get("total_fm_delta", 0) / n_app
                    avg_pass_delta = stats.get("total_pass_delta", 0) / n_app
                    action_stats.append((action, n_suc / n_app, n_app, n_suc, avg_fm_delta, avg_pass_delta))

            # Sort by success rate descending
            action_stats.sort(key=lambda x: x[1], reverse=True)

            if action_stats:
                lines.append("  Recommended actions (by empirical success rate):")
                for action, srate, n_app, n_suc, avg_fm, avg_pass in action_stats[:3]:
                    if srate > 0:
                        lines.append(
                            f"    ✓ {action}: {srate:.0%} success ({n_suc}/{n_app}), "
                            f"avg FM delta: {avg_fm:+.3f}, avg pass delta: {avg_pass:+.3f}"
                        )

                # Warn about ineffective actions
                bad_actions = [a for a in action_stats if a[1] == 0 and a[2] >= 2]
                if bad_actions:
                    lines.append("  AVOID (0% success in prior runs):")
                    for action, _, n_app, _, _, _ in bad_actions[:3]:
                        lines.append(f"    ✗ {action} (0/{n_app} attempts)")

        # Data-driven priors as fallback
        elif ddp.get(gid):
            lines.append("  Prior action scores (no empirical data yet):")
            sorted_actions = sorted(ddp[gid].items(), key=lambda x: x[1], reverse=True)
            for action, score in sorted_actions[:3]:
                lines.append(f"    {action}: prior score {score:.2f}")

        # Best recipe (concrete repair action)
        group_recipes = recipes.get(gid, [])
        if group_recipes:
            best = max(group_recipes, key=lambda r: r.get("n_effective", 0))
            if best.get("n_effective", 0) > 0:
                action_text = best.get("action", "")
                # Truncate to keep feedback manageable
                if len(action_text) > 400:
                    action_text = action_text[:400] + "..."
                lines.append(f"  Best recipe ({best['change_type']}, effective {best['n_effective']}x):")
                lines.append(f"    {action_text}")

        # Key lessons from negatives
        group_negatives = negatives.get(gid, [])
        if group_negatives:
            # Get unique lessons
            lessons = list(dict.fromkeys(
                n.get("lesson", "") for n in group_negatives
                if n.get("lesson") and n.get("failure_reason") != "No concrete DAG change was applied."
            ))
            if lessons:
                lines.append(f"  Key lessons from failed repairs:")
                for lesson in lessons[:2]:
                    if len(lesson) > 200:
                        lesson = lesson[:200] + "..."
                    lines.append(f"    • {lesson}")

        lines.append("")

    # Deprecated patterns from catalog
    deprecated = [
        (pid, p) for pid, p in catalog.items()
        if isinstance(p, dict) and not p.get("effective", True)
    ]
    if deprecated:
        lines.append("## Deprecated patterns (DO NOT USE):")
        for pid, p in deprecated[:5]:
            desc = p.get("description", "")
            if len(desc) > 120:
                desc = desc[:120] + "..."
            lines.append(f"  ✗ {pid}: {desc}")
        lines.append("")

    return "\n".join(lines)


def _init_globals() -> None:
    global _suite, _eval_prompts, _runner, _diagnoser, _tool_setup_fn, _score_fn_custom

    if _suite is not None:
        return

    # Prefer benchmark-specific loader when in topology×benchmark mode
    if _BENCHMARK and _BENCHMARK in _BENCHMARK_LOADERS:
        loader = _BENCHMARK_LOADERS[_BENCHMARK]
    else:
        loader = _LOADERS.get(_TARGET_MAS, _load_ag2)
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
        benchmark_name=_TARGET_MAS.upper(),
        score_fn=score_fn,
        benchmark_name_resolver=bench_resolver,
        timeout=_TIMEOUT,
        tool_setup_fn=tool_setup,
    )
    _diagnoser = None if _SKIP_DIAGNOSE else Diagnoser()


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


def _meta_diagnose_and_repair(
    current_task_results: dict[str, dict],
    current_score: float,
    current_dag: "MASDAG",
    current_traces: list,
) -> tuple[str, bool]:
    """Meta-diagnoser: analyze stagnation, then directly repair the best code.

    Returns (feedback_text, repaired_bool).
    If repair succeeds and scores better, updates _meta_best_* globals.
    """
    global _meta_best_score, _meta_best_code, _meta_best_task_results

    if not _meta_best_code:
        return "", False

    # --- 1. Collect diagnosis context ---
    regressions = []
    for task_hash, best_info in _meta_best_task_results.items():
        curr_info = current_task_results.get(task_hash)
        if curr_info and best_info["score"] > 0 and curr_info["score"] <= 0:
            regressions.append((task_hash, best_info, curr_info))

    best_correct = sum(1 for v in _meta_best_task_results.values() if v["score"] > 0)

    # Frequently regressed tasks across recent mutations
    regression_counts: dict[str, int] = {}
    for mutation in _meta_recent_mutations[-10:]:
        for task_hash in mutation.get("task_results", {}):
            best_info = _meta_best_task_results.get(task_hash)
            curr_info = mutation["task_results"][task_hash]
            if best_info and best_info["score"] > 0 and curr_info["score"] <= 0:
                snippet = best_info["prompt_snippet"][:60]
                regression_counts[snippet] = regression_counts.get(snippet, 0) + 1

    # Build failure trace excerpts for wrong tasks in the best
    failure_excerpts = []
    for task_hash, info in _meta_best_task_results.items():
        if info["score"] <= 0 and info.get("trace_excerpt"):
            failure_excerpts.append(
                f"Task: {info['prompt_snippet']}\n"
                f"Output: {info['output_snippet']}\n"
                f"FM: {', '.join(info.get('fm_labels', []))}\n"
                f"Trace:\n{info['trace_excerpt']}"
            )
    # Keep top 3 failures by length (more trace = more informative)
    failure_excerpts.sort(key=len, reverse=True)
    failure_context = "\n---\n".join(failure_excerpts[:3])

    recent_scores = [round(m["score"], 3) for m in _meta_recent_mutations[-10:]]

    # --- 2. Ask GLM-5 to directly modify the code ---
    system_prompt = """\
You are a meta-diagnoser for a multi-agent system (MAS) evolution loop.

The evolution has STAGNATED. You are given:
1. The current best `_build_core()` source code
2. Failure traces from tasks the best DAG got wrong
3. Regression data showing which tasks recent mutations keep breaking

Your job: DIRECTLY MODIFY the `_build_core()` function to fix the most impactful issue.

Rules:
- Output the COMPLETE modified `_build_core()` function (including the EVOLVE-BLOCK markers)
- Keep changes MINIMAL and TARGETED — do not rewrite everything
- Focus on fixing tasks the best DAG got WRONG, without breaking tasks it got RIGHT
- The frozen wrapper (outside EVOLVE-BLOCK) handles: ANSWER_INSTRUCTION injection,
  TERMINATION_CONDITION (SOLUTION_FOUND keyword), USER/FINAL nodes, and the edge to FINAL.
  Do NOT add these yourself.
- `terminal_agent` must be a valid agent id — the wrapper appends answer format to it
- All edge from/to must reference agent ids in `agents` list, or `USER`, `Introduction`,
  or ids in `extra_nodes`

Output format:
```python
# EVOLVE-BLOCK-START
def _build_core():
    ... your modified code ...
# EVOLVE-BLOCK-END
```

Think step by step about what to change, then output the complete function."""

    freq_regressions = dict(sorted(regression_counts.items(), key=lambda x: x[1], reverse=True)[:3]) if regression_counts else "none"
    failure_ctx = failure_context[:3000] if failure_context else "(no trace excerpts available)"

    user_msg = (
        f"## Current best code (score: {_meta_best_score:.4f}, {best_correct}/20 correct):\n\n"
        f"```python\n{_meta_best_code}\n```\n\n"
        f"## Stagnation data:\n"
        f"- {_meta_stagnation_count} consecutive non-improving evaluations\n"
        f"- Recent scores: {recent_scores}\n"
        f"- Frequently regressed tasks across mutations: {freq_regressions}\n\n"
        f"## Failure traces from tasks the best DAG got WRONG:\n"
        f"{failure_ctx}\n\n"
        f"Now modify the `_build_core()` function to improve accuracy. Output the COMPLETE function."
    )

    try:
        from optpilot.llm import call_llm
        llm_response = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            model=_META_MODEL,
            max_tokens=4096,
            temperature=0.3,
        )
    except Exception as e:
        return f"\n=== META-DIAGNOSIS: LLM call failed: {e} ===", False

    # --- 3. Extract code and evaluate ---
    import re
    code_match = re.search(
        r"```python\s*\n(# EVOLVE-BLOCK-START.*?# EVOLVE-BLOCK-END)",
        llm_response,
        re.DOTALL,
    )
    if not code_match:
        # Try without markers
        code_match = re.search(r"```python\s*\n(.*?)```", llm_response, re.DOTALL)

    if not code_match:
        return "\n=== META-DIAGNOSIS: Could not extract code from LLM response ===", False

    repaired_code = code_match.group(1).strip()

    # Reconstruct full file: frozen prefix + repaired evolve block + frozen suffix
    # Split original best code at EVOLVE-BLOCK markers
    best = _meta_best_code
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start_idx = best.find(start_marker)
    end_idx = best.find(end_marker)

    if start_idx < 0 or end_idx < 0:
        return "\n=== META-DIAGNOSIS: Could not find EVOLVE-BLOCK in best code ===", False

    frozen_prefix = best[:start_idx]
    frozen_suffix = best[end_idx + len(end_marker):]

    # Ensure repaired code has markers
    if start_marker not in repaired_code:
        repaired_code = f"{start_marker}\n{repaired_code}"
    if end_marker not in repaired_code:
        repaired_code = f"{repaired_code}\n{end_marker}"

    full_repaired = frozen_prefix + repaired_code + frozen_suffix

    # Write to temp file and evaluate
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
        f.write(full_repaired)
        repaired_path = f.name

    try:
        # Validate: can we build the DAG?
        spec = importlib.util.spec_from_file_location("_meta_repair", repaired_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        repaired_dag = MASDAG.from_dict(mod.build_dag())
        structural_errors = repaired_dag.structural_errors()
        if structural_errors:
            raise ValueError("; ".join(structural_errors))

        # Run evaluation on the repaired DAG
        repaired_traces = asyncio.run(
            _runner.arun_batch(_eval_prompts, dag=repaired_dag, max_concurrency=_CONCURRENCY)
        )
        repaired_correct = sum(1 for t in repaired_traces if t.task_score and t.task_score > 0)
        repaired_accuracy = repaired_correct / len(repaired_traces) if repaired_traces else 0.0

        # Simple scoring for comparison (use accuracy for now)
        if not _SKIP_DIAGNOSE:
            try:
                repaired_profiles = asyncio.run(_classify_batch(repaired_traces))
            except Exception:
                repaired_profiles = []
            repaired_fm_total = 0
            for profile in repaired_profiles:
                for gid in GROUP_IDS:
                    if gid in profile.labels and profile.labels[gid].present:
                        repaired_fm_total += 1
            repaired_task_scores = []
            for i, trace in enumerate(repaired_traces):
                is_correct = bool(trace.task_score and trace.task_score > 0)
                trace_failures = 0
                if i < len(repaired_profiles):
                    trace_failures = sum(
                        1 for gid in GROUP_IDS
                        if gid in repaired_profiles[i].labels and repaired_profiles[i].labels[gid].present
                    )
                repaired_task_scores.append(_score_task(is_correct, trace_failures, len(repaired_dag.agent_nodes)))
            repaired_score = sum(repaired_task_scores) / len(repaired_task_scores) if repaired_task_scores else 0.0
        else:
            repaired_score = repaired_accuracy

        # Did it improve?
        if repaired_score > _meta_best_score:
            _meta_best_score = repaired_score
            _meta_best_code = full_repaired
            # Update best task results
            new_results: dict[str, dict] = {}
            for i, trace in enumerate(repaired_traces):
                task_hash = str(hash(trace.task_key))[:12]
                output_snippet = ""
                if trace.trajectory:
                    for line in reversed(trace.trajectory.split("\n")):
                        if line.startswith("Output:") and len(line) > 10:
                            output_snippet = line[8:120].strip()
                            break
                new_results[task_hash] = {
                    "score": float(trace.task_score or 0.0),
                    "prompt_snippet": trace.task_key[:80],
                    "output_snippet": output_snippet[:120],
                    "fm_labels": [],
                    "trace_excerpt": "",
                }
            _meta_best_task_results = new_results

            feedback = (
                f"\n=== META-REPAIR SUCCESS ==="
                f"\nMeta-diagnoser improved score: {current_score:.4f} → {repaired_score:.4f}"
                f"\nAccuracy: {best_correct}/20 → {repaired_correct}/20"
                f"\n=== END META-REPAIR ==="
            )
            # Overwrite the program file so SkyDiscover picks up the repaired version
            Path(program_path).write_text(full_repaired, encoding="utf-8")
            return feedback, True
        else:
            feedback = (
                f"\n=== META-REPAIR ATTEMPTED ==="
                f"\nMeta-diagnoser repair did not improve: "
                f"{_meta_best_score:.4f} (best) vs {repaired_score:.4f} (repaired)"
                f"\nRepaired accuracy: {repaired_correct}/20"
                f"\n=== END META-REPAIR ==="
            )
            return feedback, False

    except Exception as e:
        feedback = f"\n=== META-REPAIR FAILED: {e} ==="
        return feedback, False
    finally:
        Path(repaired_path).unlink(missing_ok=True)


def _score_task(is_correct: bool, total_failures: int, num_agents: int) -> float:
    """Score a single task trace.

    Accuracy is the primary signal: correct = 1.0 base, incorrect = 0.0 base.
    Failure reduction is a secondary bonus/penalty on top (max ±0.3).
    This ensures that getting the answer right is always more valuable than
    reducing failure modes on a wrong answer.

    Score ranges:
      correct + 0 failures  → 1.0 + 0.3 = 1.3  (best)
      correct + 3 failures  → 1.0 + 0.0 = 1.0
      correct + 6 failures  → 1.0 - 0.3 = 0.7
      wrong   + 0 failures  → 0.0 + 0.3 = 0.3
      wrong   + 3 failures  → 0.0 + 0.0 = 0.0
      wrong   + 6 failures  → 0.0 - 0.3 = -0.3 → clamped to 0.0
    """
    base = 1.0 if is_correct else 0.0
    # FM bonus: 0 failures → +0.3, 3 failures → 0, 6 failures → -0.3
    fm_bonus = 0.3 * (1.0 - total_failures / 3.0)
    s = base + fm_bonus
    if num_agents > 4:
        s -= 0.01 * (num_agents - 4)
    return max(0.0, s)


def evaluate(program_path: str) -> dict:
    """Evaluate a candidate DAG."""
    _init_globals()
    assert _runner is not None
    assert _eval_prompts is not None and _suite is not None

    # 1. Load candidate DAG
    try:
        spec = importlib.util.spec_from_file_location("_candidate_dag", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_dag"):
            raise AttributeError("No build_dag() function found")
        dag = MASDAG.from_dict(module.build_dag())
        structural_errors = dag.structural_errors()
        if structural_errors:
            raise ValueError("; ".join(structural_errors))
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

    # 3. Score
    n = len(traces)
    correct_count = sum(1 for t in traces if t.task_score and t.task_score > 0)
    accuracy = correct_count / n if n > 0 else 0.0

    profiles: list = []

    if _SKIP_DIAGNOSE:
        # Pure accuracy mode — no FM diagnosis
        combined_score = accuracy
        fm_rates = {g: 0.0 for g in GROUP_IDS}
        total_failures = 0
    else:
        # Full mode with FM diagnosis
        try:
            profiles = asyncio.run(_classify_batch(traces))
        except Exception:
            profiles = []

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

    # 4. Collect per-task results for meta-diagnoser
    current_task_results: dict[str, dict] = {}
    for i, trace in enumerate(traces):
        task_hash = str(hash(trace.task_key))[:12]
        output_snippet = ""
        if hasattr(trace, "trajectory") and trace.trajectory:
            # Extract last agent output from trajectory
            traj_lines = trace.trajectory.split("\n")
            for line in reversed(traj_lines):
                if line.startswith("Output:") and len(line) > 10:
                    output_snippet = line[8:120].strip()
                    break
        fm_labels = []
        if not _SKIP_DIAGNOSE and i < len(profiles):
            fm_labels = [
                gid for gid in GROUP_IDS
                if gid in profiles[i].labels and profiles[i].labels[gid].present
            ]
        # Extract trace excerpt (last 2 agent steps) for failed tasks
        trace_excerpt = ""
        if trace.task_score is not None and trace.task_score <= 0 and trace.trajectory:
            traj_lines = trace.trajectory.split("\n")
            agent_blocks = []
            current_block = []
            for line in traj_lines:
                if line.startswith("--- [agent]"):
                    if current_block:
                        agent_blocks.append("\n".join(current_block))
                    current_block = [line]
                elif current_block:
                    current_block.append(line)
            if current_block:
                agent_blocks.append("\n".join(current_block))
            # Keep last 2 agent blocks, truncated
            for block in agent_blocks[-2:]:
                trace_excerpt += block[:300] + "\n...\n"

        current_task_results[task_hash] = {
            "score": float(trace.task_score or 0.0),
            "prompt_snippet": trace.task_key[:80],
            "output_snippet": output_snippet[:120],
            "fm_labels": fm_labels,
            "trace_excerpt": trace_excerpt[:600] if trace_excerpt else "",
        }

    # 5. Meta-diagnoser: track stagnation and trigger direct code repair
    global _meta_best_score, _meta_best_code, _meta_best_task_results
    global _meta_stagnation_count, _meta_recent_mutations, _eval_counter
    _eval_counter += 1

    meta_guidance = ""
    meta_repaired = False
    if combined_score > _meta_best_score:
        _meta_best_score = combined_score
        _meta_best_code = Path(program_path).read_text(encoding="utf-8")
        _meta_best_task_results = current_task_results.copy()
        _meta_stagnation_count = 0
        _meta_recent_mutations.clear()
    else:
        _meta_stagnation_count += 1
        _meta_recent_mutations.append({
            "score": combined_score,
            "task_results": current_task_results,
        })
        if len(_meta_recent_mutations) > 10:
            _meta_recent_mutations = _meta_recent_mutations[-10:]

        if _meta_stagnation_count >= _META_STAGNATION_THRESHOLD and _meta_best_code:
            meta_guidance, meta_repaired = _meta_diagnose_and_repair(
                current_task_results, combined_score, dag, traces,
            )
            _meta_stagnation_count = 0

    # 6. Feedback
    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
    sorted_fms = sorted(fm_rates.items(), key=lambda x: x[1], reverse=True)
    top_failures = [
        f"  Group {gid} ({GROUP_DEFINITIONS.get(gid, {}).get('name', gid)}): {rate:.0%}"
        for gid, rate in sorted_fms if rate > 0
    ]

    feedback_lines = [
        f"Target MAS: {_TARGET_MAS}",
        f"Accuracy: {accuracy:.1%} ({correct_count}/{n} correct)",
        f"Total failures: {total_failures} across {n} traces",
        f"Agents: {num_agents}",
        "", "FM rates:",
    ]
    feedback_lines.extend(top_failures or ["  No failures detected."])
    feedback_lines.append(f"\nCombined score: {combined_score:.4f}")

    # Inject meta-diagnosis if triggered
    if meta_guidance:
        feedback_lines.append(meta_guidance)

    # Inject prior-guided repair hints
    use_priors_this_eval = _USE_PRIORS_ALWAYS or (_USE_PRIORS_ALTERNATE and _eval_counter % 2 == 0)

    if use_priors_this_eval:
        prior_guidance = _build_prior_guidance(fm_rates)
        if prior_guidance:
            feedback_lines.append(prior_guidance)

    metrics = {
        "combined_score": round(combined_score, 4),
        "accuracy": round(accuracy, 4),
        "total_failures": total_failures,
        "num_agents": num_agents,
        "prior_guided": use_priors_this_eval,
        "meta_diagnosed": bool(meta_guidance),
        "meta_repaired": meta_repaired,
    }
    for gid in GROUP_IDS:
        metrics[f"fm_{gid}"] = round(fm_rates[gid], 4)
    metrics["artifacts"] = {"feedback": "\n".join(feedback_lines)}
    return metrics
