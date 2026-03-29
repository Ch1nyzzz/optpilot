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
from statistics import mean
from typing import TYPE_CHECKING, Any

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
## Current Code
```python
{current_code}
```

## Current Performance
{performance_summary}

## Diagnosis: Top Failure — {fm_name} ({fm_rate:.0%} occurrence)
{fm_description}

Root causes: {root_causes}
Affected DAG components: {dag_components}
Affected agents: {agents}

## Recommended Repair Direction (from Jacobian experience matrix)
{recommended_pattern_text}

## Prior Modifications In This Round
{history_text}

## Failed Approaches — DO NOT repeat these
{negatives_text}

## Candidate Style
{candidate_style}

Fix the diagnosed problem with a local mutation first. Prefer SEARCH/REPLACE
blocks over full rewrites."""

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


def format_negatives(negatives: list[ReflectInsight]) -> str:
    """Format accumulated negative examples for prompt injection."""
    if not negatives:
        return "None yet."
    lines: list[str] = []
    for i, neg in enumerate(negatives, 1):
        lines.append(
            f"Round {i}: tried [{', '.join(neg.changes_attempted[:3])}] "
            f"→ FM {neg.before_fm_rate:.2f}→{neg.after_fm_rate:.2f}, "
            f"pass {neg.before_pass_rate:.3f}→{neg.after_pass_rate:.3f}. "
            f"Failure: {neg.failure_reason}. Lesson: {neg.lesson}"
        )
    return "\n".join(lines)


def format_history(history: list[EvolveResult]) -> str:
    """Format evolve history for prompt injection."""
    if not history:
        return "No prior modifications in this round."
    lines: list[str] = []
    for i, er in enumerate(history, 1):
        lines.append(f"Iter {i}: {er.change_description[:200]}")
    return "\n".join(lines)


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
    """Extract exact SEARCH/REPLACE blocks from a model response."""
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    return [(search, replace) for search, replace in pattern.findall(response)]


def apply_search_replace_blocks(source: str, blocks: list[tuple[str, str]]) -> str:
    """Apply exact SEARCH/REPLACE blocks sequentially."""
    updated = source
    for search, replace in blocks:
        matches = updated.count(search)
        if matches != 1:
            raise ValueError(
                f"Expected exactly one match for SEARCH block, found {matches}."
            )
        updated = updated.replace(search, replace, 1)
    return updated


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
        try:
            code = apply_search_replace_blocks(current_code, blocks)
            actions_taken = [f"Applied SEARCH/REPLACE block {i + 1}" for i in range(len(blocks))]
        except ValueError as e:
            return EvolveResult(
                dag=current_dag,
                analysis_text="",
                modified_source="",
                change_description=f"Diff mutation failed to apply: {e}",
                actions_taken=[],
                metadata={
                    "invalid_evolve_reason": f"Diff apply error: {e}",
                    "assigned_pattern_id": recommended_pattern.pattern_id if recommended_pattern else "",
                    "candidate_index": candidate_index,
                    "candidate_style": candidate_style,
                },
            )
    else:
        code = extract_python_code(response)
        if not code:
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

    candidate_styles = [
        "Mutate the routing topology with the smallest possible edge-level edit.",
        "Mutate prompts and control conditions conservatively; avoid adding new nodes unless necessary.",
        "Target context preservation explicitly; prefer edits that retain the original task prompt downstream.",
        "Target loop-breaking or verification hardening with one compact structural change.",
    ]
    history_text = format_history(history)
    assigned_patterns: list[Any | None]
    if recommended_patterns is not None:
        assigned_patterns = list(recommended_patterns)
    else:
        assigned_patterns = [recommended_pattern] * max(1, num_candidates)
    if not assigned_patterns:
        assigned_patterns = [None]

    async def _generate_one(candidate_index: int, candidate_pattern: Any | None) -> EvolveResult:
        candidate_style = candidate_styles[candidate_index % len(candidate_styles)]
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
        user_prompt = _DIRECT_GEN_USER_PROMPT.format(
            current_code=current_code,
            performance_summary=performance_summary,
            fm_name=fm_info["name"],
            fm_rate=analysis.fm_rate,
            fm_description=fm_info["description"],
            root_causes=", ".join(analysis.root_cause_clusters) or "unknown",
            dag_components=", ".join(analysis.metadata.get("dag_components", [])) or "unknown",
            agents=", ".join(analysis.common_agents) or "unknown",
            recommended_pattern_text=pattern_text,
            history_text=history_text,
            negatives_text=format_negatives(negatives),
            candidate_style=f"Candidate {candidate_index + 1}: {candidate_style}",
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
        return _response_to_evolve_result(
            response=response,
            current_dag=dag,
            current_code=current_code,
            fm_group=fm_group,
            analysis=analysis,
            negatives=negatives,
            prompt=user_prompt,
            recommended_pattern=candidate_pattern,
            candidate_index=candidate_index,
            candidate_style=candidate_style,
        )

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
