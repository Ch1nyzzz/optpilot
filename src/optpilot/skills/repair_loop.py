"""Repair loop — stateless functions for MAS DAG repair.

Extracted from the former BaseSkill/GenericSkill classes.  Each function
operates on explicit arguments rather than instance state.

Core functions:
- ``analyze()``  — LLM-based failure analysis
- ``aevolve()``  — async multi-candidate DAG mutation generator
- ``reflect()``  — post-validation failure analysis returning lessons
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

try:
    from skydiscover.utils.code_utils import (
        apply_diff as _apply_skydiscover_diff,
        extract_diffs as _extract_skydiscover_diffs,
        parse_full_rewrite as _parse_skydiscover_full_rewrite,
    )
except ImportError:
    def _apply_skydiscover_diff(original_solution: str, diff_text: str) -> str:
        original_lines = original_solution.split("\n")
        result_lines = original_lines.copy()
        for search_text, replace_text in _extract_skydiscover_diffs(diff_text):
            search_lines = search_text.split("\n")
            replace_lines = replace_text.split("\n")
            for i in range(len(result_lines) - len(search_lines) + 1):
                if result_lines[i : i + len(search_lines)] == search_lines:
                    result_lines[i : i + len(search_lines)] = replace_lines
                    break
        return "\n".join(result_lines)

    def _extract_skydiscover_diffs(diff_text: str) -> list[tuple[str, str]]:
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
        return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]

    def _parse_skydiscover_full_rewrite(llm_response: str, language: str = "python") -> str:
        solution_block_pattern = r"```" + language + r"\n(.*?)```"
        matches = re.findall(solution_block_pattern, llm_response, re.DOTALL)
        if matches:
            return matches[0].strip()
        solution_block_pattern = r"```(.*?)```"
        matches = re.findall(solution_block_pattern, llm_response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return llm_response

from optpilot.config import (
    LIBRARY_DIR,
    NEGATIVES_DIR,
    PROJECT_ROOT,
    SKILL_EVOLVE_MAX_TOKENS,
    SKILL_EVOLVE_NUM_CANDIDATES,
)
from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS, GROUP_IDS, GROUP_NAMES
from optpilot.llm import acall_llm, call_llm_json
from optpilot.models import (
    AnalysisResult,
    EvolveResult,
    ReflectInsight,
)
from optpilot.repair_utils import summarize_faults
from optpilot.skills.repair_patterns import infer_observed_pattern_from_dags
from optpilot.skills.tools import dag_to_python, python_source_to_dag

if TYPE_CHECKING:
    from optpilot.dag.core import MASDAG
    from optpilot.models import FMProfile, MASTrace


_SKILL_AGENT_TRACE_ROOT = LIBRARY_DIR / "skill_agent_traces"
_GENERATION_RETRY_TIMES = 3
_FAILED_ATTEMPT_HISTORY = 2


# -------------------------------------------------------------------- #
#  Prompt templates                                                     #
# -------------------------------------------------------------------- #

_ANALYZE_PROMPT = """\
You are a multi-agent system (MAS) failure analyst.

Analyze the following MAS for **{fm_name}** problems: {fm_description}

## Current MAS Configuration (Python)
```python
{python_source}
```

## Fault Evidence from {n_traces} traces
{fault_summary}

## Prior Failed Repair Attempts (avoid repeating these)
{negatives_text}

{analyze_hint}

Respond with ONLY a JSON object:
{{
    "common_agents": ["<agent names involved>"],
    "common_steps": ["<steps where the fault occurs>"],
    "root_cause_clusters": ["<distinct root cause patterns>"],
    "evidence_snippets": ["<key evidence from traces>"]
}}"""

_DIRECT_GEN_SYSTEM_PROMPT = """\
You are an expert multi-agent system (MAS) architect.
Your task is to fix a diagnosed failure in a Python function `build_dag()` that
constructs a multi-agent DAG for solving mathematical reasoning problems.

Generate a SMALL diff-based mutation against the current `build_dag()` source.
Prefer exact search/replace edits over full rewrites.

## How the Code Works
The `build_dag()` function returns a dictionary that defines a Directed Acyclic
Graph (DAG) of agents. The dictionary has these top-level keys:
- `dag_id`: string identifier
- `nodes`: list of node dicts
- `edges`: list of edge dicts
- `metadata`: dict with `start`, `success_nodes`, `description`

## Node Types
Each node is a dict with `id`, `type`, and type-specific fields:
- `agent`: LLM-powered node. Fields: `role` (system prompt string), `config.params.temperature`, `config.params.max_tokens`
- `literal`: Fixed text. Fields: `config.content`, `config.role`
- `loop_counter`: Iteration tracker. Fields: `config.max_iterations`, `config.reset_on_emit`, `config.message`
- `passthrough`: Routes data without processing. Fields: `config` (empty dict)

## Edge Semantics
Each edge is a dict with:
- `from`, `to`: node IDs (must exist in nodes list)
- `trigger`: bool — if True, activates the target node to execute
- `carry_data`: bool — if True, passes source output to target's input
- `condition`: either the string `"true"` (always fire) or a keyword-matching dict:
  `{{"type": "keyword", "config": {{"any": ["SOLUTION_FOUND"], "none": [], "regex": [], "case_sensitive": True}}}}`
- `loop`: optional, `"exit"` or `"continue"` for loop counter edges

## Constraints
- You MUST keep `dag_id` and `metadata` fields unchanged.
- You MUST keep at least one verification agent (do NOT delete Agent_Verifier).
- The DAG must have a valid path from start nodes to FINAL.
- All edge `from`/`to` must reference existing node IDs.

## What You Can Modify
1. Agent prompts (the `role` string) — improve instructions, add constraints
2. Agent parameters — temperature, max_tokens
3. **Add new agents** — append to nodes list + add edges
4. Remove non-essential agents — remove from nodes + clean up edges
5. Edge routing — change triggers, conditions, carry_data
6. Termination conditions — keyword matching in edge conditions
7. Loop limits — max_iterations in loop_counter config
8. Introduction content — update to mention new agents

## Output Format
Preferred format: output one or more exact SEARCH/REPLACE blocks:
<<<<<<< SEARCH
... exact old text ...
=======
... exact new text ...
>>>>>>> REPLACE

Rules for SEARCH/REPLACE blocks:
- SEARCH text must match the current code exactly.
- Keep edits local and minimal.
- Use multiple blocks if needed.

Fallback format: if a local diff is impossible, output the complete modified
`build_dag()` function inside a ```python code block.

After the edits, write one short summary line describing the mutation."""

_DIRECT_GEN_USER_PROMPT = """\
## Current Solution Information
- Main Metrics:
{metrics_text}
- Focus areas: {improvement_areas}

## Program Generation History
### Previous Attempts
{history_text}

### Other Context Programs
{other_context_programs}

## Current Code
```python
{current_code}
```

## Current Performance Summary
{performance_summary}

## Diagnosis: Top Issue — {fm_name} ({fm_rate:.0%} occurrence)
{fm_description}

Observed issue patterns: {issue_patterns}
Affected DAG components: {dag_components}
Affected agents: {agents}

## Recommended Repair Direction (from Jacobian experience matrix)
{recommended_pattern_text}

## Failed Approaches — compressed
{negatives_text}

## Recent Failed Candidate Attempts
{failed_attempts_text}

Fix the diagnosed problem with a local mutation first. Prefer SEARCH/REPLACE
blocks over full rewrites. Keep the mutation targeted and preserve original task
context downstream unless removing a path is clearly necessary."""

_REFLECT_PROMPT = """\
You are analyzing why a MAS repair attempt failed to resolve **{fm_name}** problems.

## Original MAS (before repair)
```python
{original_python}
```

## Final MAS (after repair)
```python
{final_python}
```

## Changes attempted in this round
{changes_text}

## Validation results
- FM occurrence rate: {before_fm:.2f} → {after_fm:.2f}
- Task pass rate: {before_pass:.3f} → {after_pass:.3f}

Analyze why the repair did not work and what lesson should be learned.

Respond with ONLY a JSON object:
{{
    "failure_reason": "<why the repair failed, max 100 words>",
    "lesson": "<what to avoid or try differently next time, max 100 words>"
}}"""


# -------------------------------------------------------------------- #
#  Performance summary builder                                          #
# -------------------------------------------------------------------- #

def build_performance_summary(
    traces: list[MASTrace],
    profiles: list[FMProfile],
) -> str:
    """Build a concise performance summary (OpenEvolve-style feedback)."""
    n = len(traces)
    correct = sum(1 for t in traces if t.task_score and t.task_score > 0)
    accuracy = correct / n if n > 0 else 0.0

    fm_counts: dict[str, int] = {g: 0 for g in GROUP_IDS}
    for profile in profiles:
        for gid in GROUP_IDS:
            if gid in profile.labels and profile.labels[gid].present:
                fm_counts[gid] += 1

    lines = [
        f"Accuracy: {accuracy:.1%} ({correct}/{n} correct)",
    ]

    fm_lines = []
    for gid in GROUP_IDS:
        rate = fm_counts[gid] / n if n > 0 else 0.0
        if rate > 0:
            fm_lines.append(f"  Group {gid} ({GROUP_NAMES.get(gid, gid)}): {rate:.0%}")
    if fm_lines:
        lines.append("Failure mode rates:")
        lines.extend(fm_lines)
    else:
        lines.append("No failures detected.")

    return "\n".join(lines)


# -------------------------------------------------------------------- #
#  Helper functions                                                     #
# -------------------------------------------------------------------- #

def fm_rate(fm_group: str, profiles: list[FMProfile]) -> float:
    """Fraction of profiles that exhibit ``fm_group``."""
    if not profiles:
        return 0.0
    return sum(1 for p in profiles if fm_group in p.active_fm_ids()) / len(profiles)


def pass_rate(traces: list[MASTrace]) -> float:
    """Mean task score (ground-truth benchmark accuracy)."""
    scores = [t.task_score for t in traces if t.task_score is not None]
    if scores:
        return mean(scores)
    fallback = [1.0 if t.task_success else 0.0 for t in traces if t.task_success is not None]
    return mean(fallback) if fallback else 0.0


def has_material_change(
    original_dag: MASDAG,
    candidate_dag: MASDAG,
    evolve_result: EvolveResult | None = None,
) -> bool:
    """Check if the candidate DAG differs meaningfully from the original."""
    if hasattr(original_dag, "canonical_dict") and hasattr(candidate_dag, "canonical_dict"):
        return original_dag.canonical_dict() != candidate_dag.canonical_dict()
    if hasattr(original_dag, "to_dict") and hasattr(candidate_dag, "to_dict"):
        return original_dag.to_dict() != candidate_dag.to_dict()
    elif original_dag != candidate_dag:
        return True
    if evolve_result and (evolve_result.change_records or evolve_result.actions_taken):
        return True
    return False


def _load_recipe_text(fm_group: str, recipe_dir: Path | None = None) -> str:
    """Load recipe text for a FM group from the recipe library."""
    try:
        from optpilot.skills.recipes import RecipeLibrary
        library = RecipeLibrary(base_dir=recipe_dir)
        return library.format_for_prompt(fm_group, top_k=3)
    except Exception:
        return ""


def format_negatives(negatives: list[ReflectInsight]) -> str:
    """Format accumulated negative examples for prompt injection."""
    if not negatives:
        return "None yet."
    lines: list[str] = []
    recent = negatives[-3:]
    start_index = len(negatives) - len(recent) + 1
    for offset, neg in enumerate(recent, start_index):
        lines.append(
            f"- Round {offset}: [{', '.join(neg.changes_attempted[:2])}] "
            f"FM {neg.before_fm_rate:.2f}→{neg.after_fm_rate:.2f}, "
            f"pass {neg.before_pass_rate:.3f}→{neg.after_pass_rate:.3f}; "
            f"failure={neg.failure_reason[:120]}; lesson={neg.lesson[:120]}"
        )
    return "\n".join(lines)


def format_history(history: list[EvolveResult]) -> str:
    """Format evolve history for prompt injection."""
    if not history:
        return "No prior modifications in this round."
    lines: list[str] = []
    recent = history[-3:]
    start_index = len(history) - len(recent) + 1
    for offset, er in enumerate(recent, start_index):
        lines.append(f"- Iter {offset}: {er.change_description[:160]}")
    return "\n".join(lines)


def build_metrics_text(
    traces: list[MASTrace] | None,
    profiles: list[FMProfile] | None,
) -> str:
    """Build a compact metrics block in the OpenEvolve prompt style."""
    if not traces or not profiles:
        return "No evaluation metrics available."

    n = len(traces)
    correct = sum(1 for t in traces if t.task_score and t.task_score > 0)
    accuracy = correct / n if n > 0 else 0.0
    fm_pairs: list[str] = []
    for gid in GROUP_IDS:
        rate = fm_rate(gid, profiles)
        if rate > 0:
            fm_pairs.append(f"{gid}={rate:.0%}")
    fm_text = ", ".join(fm_pairs) if fm_pairs else "none"
    return f"- Accuracy: {accuracy:.1%} ({correct}/{n})\n- FM rates: {fm_text}"


def build_improvement_areas(
    analysis: AnalysisResult,
    profiles: list[FMProfile] | None,
) -> str:
    """Describe concise focus areas for the next repair attempt."""
    areas: list[str] = []
    if analysis.common_agents:
        areas.append(f"stabilize {'/'.join(analysis.common_agents[:3])}")
    if analysis.metadata.get("dag_components"):
        areas.append(
            f"inspect {', '.join(analysis.metadata.get('dag_components', [])[:3])}"
        )
    if analysis.common_steps:
        areas.append(f"tighten step flow around {analysis.common_steps[0]}")
    if profiles:
        active = [
            f"{gid}={fm_rate(gid, profiles):.0%}"
            for gid in GROUP_IDS
            if fm_rate(gid, profiles) > 0
        ]
        if active:
            areas.append(f"reduce recurring FM mix ({', '.join(active[:4])})")
    return "; ".join(areas) if areas else "improve combined score with one targeted repair."


def build_other_context_programs() -> str:
    """Skill mode currently repairs one incumbent DAG at a time."""
    return "No alternate context programs are tracked in skill mode."


def build_synthetic_insight(
    fm_group: str,
    evolve_result: EvolveResult | None,
    before_fm: float,
    after_fm: float,
    before_pass: float,
    after_pass: float,
    failure_reason: str,
    lesson: str,
    metadata: dict[str, Any] | None = None,
) -> ReflectInsight:
    """Create a ReflectInsight without calling the LLM."""
    changes = []
    if evolve_result and evolve_result.change_description:
        changes = [evolve_result.change_description]
    if not changes:
        changes = ["No concrete DAG changes recorded."]
    return ReflectInsight(
        round_index=0,
        fm_id=fm_group,
        changes_attempted=changes,
        before_fm_rate=before_fm,
        after_fm_rate=after_fm,
        before_pass_rate=before_pass,
        after_pass_rate=after_pass,
        failure_reason=failure_reason,
        lesson=lesson,
        timestamp=datetime.now().isoformat(),
        metadata=metadata or {},
    )


# -------------------------------------------------------------------- #
#  Core functions                                                       #
# -------------------------------------------------------------------- #

def analyze(
    dag: MASDAG,
    fm_group: str,
    traces: list[MASTrace],
    profiles: list[FMProfile],
    negatives: list[ReflectInsight],
) -> AnalysisResult:
    """LLM-based failure analysis for a specific FM group."""
    fm_info = GROUP_DEFINITIONS[fm_group]
    prompt = _ANALYZE_PROMPT.format(
        fm_name=fm_info["name"],
        fm_description=fm_info["description"],
        python_source=dag_to_python(dag),
        n_traces=len(traces),
        fault_summary=summarize_faults(fm_group, profiles, traces),
        negatives_text=format_negatives(negatives),
        analyze_hint=fm_info.get("analyze_hint", ""),
    )
    result = call_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)

    dag_components: set[str] = set()
    for profile in profiles:
        loc = profile.localization.get(fm_group)
        if loc and loc.dag_component and loc.dag_component != "other":
            dag_components.add(loc.dag_component)

    from optpilot.skills.repair_patterns import extract_failure_signatures
    failure_signatures = extract_failure_signatures(fm_group, profiles)

    return AnalysisResult(
        fm_id=fm_group,
        fm_rate=fm_rate(fm_group, profiles),
        common_agents=result.get("common_agents", []),
        common_steps=result.get("common_steps", []),
        root_cause_clusters=result.get("root_cause_clusters", []),
        dag_summary=dag.summary() if hasattr(dag, "summary") else "",
        evidence_snippets=result.get("evidence_snippets", []),
        metadata={
            "dag_components": sorted(dag_components),
            "proposal_traces": traces,
            "failure_signatures": failure_signatures,
        },
    )


def extract_python_code(response: str) -> str:
    """Extract a Python code block from LLM response.

    The generated code may itself contain nested ``` markers (e.g. agent
    prompts that include markdown code fences).  We find the opening
    ```python marker, then locate ``def build_dag`` inside it, and take
    everything until the function's return dict is complete.
    """
    import re as _re

    # Strategy 1: find "def build_dag" and extract the complete function.
    # This is the most robust approach because it doesn't depend on ``` delimiters.
    idx = response.find("def build_dag")
    if idx >= 0:
        code = response[idx:]
        # Find the end of the function: look for a top-level return { ... }
        # followed by a line that is not indented (or end of text / closing ```)
        lines = code.splitlines()
        func_lines: list[str] = []
        found_return = False
        brace_depth = 0
        for line in lines:
            func_lines.append(line)
            # Track brace depth after we see 'return {'
            if not found_return and line.strip().startswith("return {"):
                found_return = True
                brace_depth += line.count("{") - line.count("}")
                continue
            if found_return:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    break
            # If we hit a top-level line (no indent) that isn't part of the function, stop
            if func_lines and line and not line[0].isspace() and not line.startswith("def "):
                if line.startswith("```"):
                    func_lines.pop()  # remove the closing ``` line
                    break
        return "\n".join(func_lines).strip()

    return ""


def _extract_change_summary(response: str) -> str:
    """Extract the summary line after the code block."""
    # Find text after the last ``` or after the function body
    idx = response.rfind("```")
    if idx >= 0:
        after = response[idx + 3:].strip()
        for line in after.splitlines():
            line = line.strip()
            if line:
                return line[:200]

    # Fallback: look for text after the function
    idx = response.rfind("return {")
    if idx >= 0:
        # Find end of return statement, then get text after
        rest = response[idx:]
        brace_depth = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth <= 0:
                    after = rest[i + 1:].strip()
                    for line in after.splitlines():
                        line = line.strip()
                        if line and not line.startswith("```"):
                            return line[:200]
                    break
    return ""


def extract_search_replace_blocks(response: str) -> list[tuple[str, str]]:
    """Extract SEARCH/REPLACE blocks using SkyDiscover-compatible parsing."""
    return _extract_skydiscover_diffs(response)


def apply_search_replace_blocks(source: str, blocks: list[tuple[str, str]]) -> str:
    """Apply SEARCH/REPLACE blocks using SkyDiscover-compatible semantics."""
    diff_text = "\n".join(
        f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"
        for search, replace in blocks
    )
    return _apply_skydiscover_diff(source, diff_text)


def _format_failed_attempts(failed_attempts: list[dict[str, Any]]) -> str:
    """Summarize recent failed generation attempts for retry prompts."""
    if not failed_attempts:
        return "None."

    lines: list[str] = []
    for attempt in failed_attempts[-_FAILED_ATTEMPT_HISTORY:]:
        attempt_no = attempt.get("attempt_number", "?")
        error = str(attempt.get("error", "Unknown error")).strip()
        response_preview = str(attempt.get("response_preview", "")).strip()
        lines.append(f"- Attempt {attempt_no}: {error}")
        if response_preview:
            lines.append(f"  Response preview: {response_preview}")
    return "\n".join(lines)


def _response_to_evolve_result(
    *,
    response: str,
    current_dag: MASDAG,
    current_code: str,
    fm_group: str,
    analysis: AnalysisResult,
    negatives: list[ReflectInsight],
    prompt: str,
    recommended_pattern: Any | None = None,
    candidate_index: int = 0,
    candidate_style: str = "",
) -> EvolveResult:
    """Convert one LLM response into an EvolveResult."""
    code = ""
    actions_taken: list[str] = []

    blocks = extract_search_replace_blocks(response)
    if blocks:
        code = apply_search_replace_blocks(current_code, blocks)
        actions_taken = [f"Applied SEARCH/REPLACE block {i + 1}" for i in range(len(blocks))]
        if code == current_code:
            return EvolveResult(
                dag=current_dag,
                analysis_text="",
                modified_source="",
                change_description="Diff mutation failed to apply: no changes matched the current code.",
                actions_taken=[],
                metadata={
                    "invalid_evolve_reason": "Diff SEARCH blocks did not match current code - no changes applied.",
                    "assigned_pattern_id": recommended_pattern.pattern_id if recommended_pattern else "",
                    "candidate_index": candidate_index,
                    "candidate_style": candidate_style,
                },
            )
    else:
        code = _parse_skydiscover_full_rewrite(response, language="python")
        if "def build_dag" not in code:
            return EvolveResult(
                dag=current_dag,
                analysis_text="",
                modified_source="",
                change_description="LLM response contained no diff blocks or Python code.",
                actions_taken=[],
                metadata={
                    "invalid_evolve_reason": "No Python code block found in LLM response (and no SEARCH/REPLACE blocks were provided).",
                    "assigned_pattern_id": recommended_pattern.pattern_id if recommended_pattern else "",
                    "candidate_index": candidate_index,
                    "candidate_style": candidate_style,
                },
            )
        actions_taken = ["Applied full build_dag rewrite fallback"]

    try:
        new_dag = python_source_to_dag(code)
    except ValueError as e:
        return EvolveResult(
            dag=current_dag,
            analysis_text="",
            modified_source=code,
            change_description=f"Generated code failed to parse: {e}",
            actions_taken=actions_taken,
            metadata={
                "invalid_evolve_reason": f"Code parse error: {e}",
                "assigned_pattern_id": recommended_pattern.pattern_id if recommended_pattern else "",
                "candidate_index": candidate_index,
                "candidate_style": candidate_style,
            },
        )

    has_verifier = any("verif" in nid.lower() for nid in new_dag.agent_nodes)
    if not has_verifier:
        return EvolveResult(
            dag=current_dag,
            analysis_text="",
            modified_source=code,
            change_description="Verification agent was deleted (constraint violation).",
            actions_taken=actions_taken,
            metadata={
                "invalid_evolve_reason": "Verification agent deleted.",
                "assigned_pattern_id": recommended_pattern.pattern_id if recommended_pattern else "",
                "candidate_index": candidate_index,
                "candidate_style": candidate_style,
            },
        )

    change_summary = _extract_change_summary(response)
    if not change_summary:
        change_summary = f"Candidate {candidate_index + 1} mutation for FM group {fm_group}"

    trace_path = _persist_generation_trace(
        fm_group=fm_group,
        analysis=analysis,
        negatives=negatives,
        prompt=prompt,
        response=response,
        code=code,
    )

    return EvolveResult(
        dag=new_dag,
        analysis_text=change_summary,
        modified_source=code,
        change_description=change_summary,
        actions_taken=actions_taken,
        metadata={
            "invalid_evolve_reason": "",
            "generation_trace_path": trace_path,
            "assigned_pattern_id": recommended_pattern.pattern_id if recommended_pattern else "",
            "observed_pattern_id": infer_observed_pattern_from_dags(current_dag, new_dag),
            "candidate_index": candidate_index,
            "candidate_style": candidate_style,
        },
    )


async def agenerate_evolve_candidates(
    dag: MASDAG,
    fm_group: str,
    analysis: AnalysisResult,
    negatives: list[ReflectInsight],
    history: list[EvolveResult],
    recommended_pattern: Any | None = None,
    recommended_patterns: list[Any | None] | None = None,
    traces: list[MASTrace] | None = None,
    profiles: list[FMProfile] | None = None,
    num_candidates: int = SKILL_EVOLVE_NUM_CANDIDATES,
    recipe_dir: Path | None = None,
) -> list[EvolveResult]:
    """Generate multiple diff-based mutation candidates for the current DAG."""
    fm_info = GROUP_DEFINITIONS[fm_group]
    current_code = dag_to_python(dag)

    performance_summary = "No performance data available."
    if traces and profiles:
        performance_summary = build_performance_summary(traces, profiles)

    recommended_pattern_text = "No specific recommendation. Use your best judgment."
    if recommended_pattern is not None:
        recommended_pattern_text = (
            f"**{recommended_pattern.name}** (pattern: {recommended_pattern.pattern_id})\n"
            f"{recommended_pattern.description}\n"
            f"This repair direction has been effective for similar failures. "
            f"Prioritize this approach, but adapt based on the diagnosis."
        )

    # Inject repair recipes from offline experience
    recipe_text = _load_recipe_text(fm_group, recipe_dir=recipe_dir)
    if recipe_text:
        recommended_pattern_text += "\n\n" + recipe_text

    history_text = format_history(history)
    negatives_text = format_negatives(negatives)
    metrics_text = build_metrics_text(traces, profiles)
    improvement_areas = build_improvement_areas(analysis, profiles)
    other_context_programs = build_other_context_programs()
    assigned_patterns: list[Any | None]
    if recommended_patterns is not None:
        assigned_patterns = list(recommended_patterns)
    else:
        assigned_patterns = [recommended_pattern] * max(1, num_candidates)
    if not assigned_patterns:
        assigned_patterns = [None]

    async def _generate_one(candidate_index: int, candidate_pattern: Any | None) -> EvolveResult:
        pattern_text = recommended_pattern_text
        if candidate_pattern is None:
            pattern_text = (
                "No specific recommendation for this candidate. Explore a plausible repair "
                "direction that differs meaningfully from the recommended patterns."
            )
        else:
            pattern_text = (
                f"**{candidate_pattern.name}** (pattern: {candidate_pattern.pattern_id})\n"
                f"{candidate_pattern.description}\n"
                f"This repair direction has been effective for similar failures. "
                f"Prioritize this approach, but adapt based on the diagnosis."
            )
        failed_attempts: list[dict[str, Any]] = []
        last_candidate: EvolveResult | None = None

        for attempt_index in range(_GENERATION_RETRY_TIMES):
            user_prompt = _DIRECT_GEN_USER_PROMPT.format(
                current_code=current_code,
                metrics_text=metrics_text,
                improvement_areas=improvement_areas,
                other_context_programs=other_context_programs,
                performance_summary=performance_summary,
                fm_name=fm_info["name"],
                fm_rate=analysis.fm_rate,
                fm_description=fm_info["description"],
                issue_patterns=", ".join(analysis.root_cause_clusters) or "unknown",
                dag_components=", ".join(analysis.metadata.get("dag_components", [])) or "unknown",
                agents=", ".join(analysis.common_agents) or "unknown",
                recommended_pattern_text=pattern_text,
                history_text=history_text,
                negatives_text=negatives_text,
                failed_attempts_text=_format_failed_attempts(failed_attempts),
            )
            messages = [
                {"role": "system", "content": _DIRECT_GEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            response = await acall_llm(
                messages=messages,
                max_tokens=SKILL_EVOLVE_MAX_TOKENS,
                temperature=min(0.75, 0.35 + 0.1 * candidate_index),
            )
            candidate = _response_to_evolve_result(
                response=response,
                current_dag=dag,
                current_code=current_code,
                fm_group=fm_group,
                analysis=analysis,
                negatives=negatives,
                prompt=user_prompt,
                recommended_pattern=candidate_pattern,
                candidate_index=candidate_index,
                candidate_style="",
            )
            last_candidate = candidate

            invalid_reason = str(candidate.metadata.get("invalid_evolve_reason", "")).strip()
            no_material_change = (
                not invalid_reason and not has_material_change(dag, candidate.dag, candidate)
            )
            if not invalid_reason and not no_material_change:
                candidate.metadata["attempts_used"] = attempt_index + 1
                if failed_attempts:
                    candidate.metadata["failed_attempts"] = list(failed_attempts)
                return candidate

            error = invalid_reason or "No concrete DAG change was applied."
            failed_attempts.append({
                "attempt_number": attempt_index + 1,
                "error": error,
                "response_preview": re.sub(r"\s+", " ", response).strip()[:300],
            })

        assert last_candidate is not None
        if not last_candidate.metadata.get("invalid_evolve_reason", ""):
            last_candidate.metadata["invalid_evolve_reason"] = "No concrete DAG change was applied."
        last_candidate.metadata["attempts_used"] = _GENERATION_RETRY_TIMES
        last_candidate.metadata["failed_attempts"] = list(failed_attempts)
        return last_candidate

    raw_candidates = await asyncio.gather(
        *[
            _generate_one(i, assigned_patterns[i])
            for i in range(len(assigned_patterns))
        ]
    )

    deduped: list[EvolveResult] = []
    seen_keys: set[str] = set()
    for candidate in raw_candidates:
        invalid_reason = str(candidate.metadata.get("invalid_evolve_reason", ""))
        if not invalid_reason and hasattr(candidate.dag, "canonical_dict"):
            key = json.dumps(candidate.dag.canonical_dict(), sort_keys=True, ensure_ascii=False)
        elif candidate.modified_source:
            key = candidate.modified_source
        else:
            key = invalid_reason or candidate.change_description
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(candidate)
    return deduped


async def aevolve(
    dag: MASDAG,
    fm_group: str,
    analysis: AnalysisResult,
    negatives: list[ReflectInsight],
    history: list[EvolveResult],
    recommended_pattern: Any | None = None,
    traces: list[MASTrace] | None = None,
    profiles: list[FMProfile] | None = None,
    recipe_dir: Path | None = None,
) -> EvolveResult:
    """Backward-compatible wrapper that returns the first generated candidate."""
    candidates = await agenerate_evolve_candidates(
        dag,
        fm_group,
        analysis,
        negatives,
        history,
        recommended_pattern=recommended_pattern,
        traces=traces,
        profiles=profiles,
        recipe_dir=recipe_dir,
    )
    if candidates:
        return candidates[0]
    return EvolveResult(
        dag=dag,
        analysis_text="",
        modified_source="",
        change_description="Candidate generation returned no results.",
        metadata={"invalid_evolve_reason": "Candidate generation returned no results."},
    )


def reflect(
    fm_group: str,
    original_dag: MASDAG,
    evolve_result: EvolveResult,
    before_fm: float,
    after_fm: float,
    before_pass: float,
    after_pass: float,
) -> ReflectInsight:
    """Post-validation failure analysis.  Returns a lesson for future rounds."""
    fm_info = GROUP_DEFINITIONS[fm_group]
    original_python = dag_to_python(original_dag)
    final_python = evolve_result.modified_source if evolve_result.modified_source else original_python
    changes = [evolve_result.change_description] if evolve_result.change_description else []

    prompt = _REFLECT_PROMPT.format(
        fm_name=fm_info["name"],
        original_python=original_python[:3000],
        final_python=final_python[:3000],
        changes_text="\n".join(f"- {c}" for c in changes) or "No changes recorded.",
        before_fm=before_fm,
        after_fm=after_fm,
        before_pass=before_pass,
        after_pass=after_pass,
    )
    result = call_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)
    return ReflectInsight(
        round_index=0,
        fm_id=fm_group,
        changes_attempted=changes,
        before_fm_rate=before_fm,
        after_fm_rate=after_fm,
        before_pass_rate=before_pass,
        after_pass_rate=after_pass,
        failure_reason=result.get("failure_reason", "Unknown"),
        lesson=result.get("lesson", "Unknown"),
        timestamp=datetime.now().isoformat(),
        metadata={},
    )


# -------------------------------------------------------------------- #
#  Trace persistence                                                    #
# -------------------------------------------------------------------- #

def _persist_generation_trace(
    fm_group: str,
    analysis: AnalysisResult,
    negatives: list[ReflectInsight],
    prompt: str,
    response: str,
    code: str,
) -> str:
    """Save the direct-generation trace to disk."""
    trace_dir = _SKILL_AGENT_TRACE_ROOT / fm_group
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = trace_dir / f"gen_trace_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "fm_group": fm_group,
        "analysis": {
            "fm_id": analysis.fm_id,
            "fm_rate": analysis.fm_rate,
            "common_agents": analysis.common_agents,
            "common_steps": analysis.common_steps,
            "root_cause_clusters": analysis.root_cause_clusters,
            "evidence_snippets": analysis.evidence_snippets,
        },
        "prior_negatives_count": len(negatives),
        "prompt_length": len(prompt),
        "response_length": len(response),
        "generated_code": code,
    }
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(trace_path)
