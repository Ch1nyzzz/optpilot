"""Shared helpers for repair workflows and LLM response parsing."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optpilot.models import FMProfile, MASTrace


def summarize_faults(
    fm_id: str,
    profiles: list[FMProfile],
    traces: list[MASTrace],
    limit: int = 3,
) -> str:
    """Aggregate localized failure evidence into a compact text summary."""
    agents: Counter[str] = Counter()
    steps: Counter[str] = Counter()
    causes: Counter[str] = Counter()
    examples: list[str] = []

    for trace, profile in zip(traces, profiles, strict=False):
        loc = profile.localization.get(fm_id)
        if loc is None:
            continue

        agent = (loc.agent or "").strip()
        step = (loc.step or "").strip()
        cause = (loc.root_cause or "").strip()
        context = (loc.context or "").strip().replace("\n", " ")

        if agent:
            agents[agent] += 1
        if step:
            steps[step] += 1
        if cause:
            causes[cause] += 1
        if len(examples) < limit:
            if len(context) > 180:
                context = context[:177] + "..."
            examples.append(
                f"  - trace {trace.trace_id}: agent={agent}, step={step}, "
                f"cause={cause}, context={context}"
            )

    n_affected = sum(1 for profile in profiles if fm_id in profile.active_fm_ids())
    lines = [f"- Observed in {n_affected}/{len(profiles)} traces"]
    if agents:
        lines.append(
            "- Common agents: "
            + ", ".join(f"{agent} ({count})" for agent, count in agents.most_common(limit))
        )
    if steps:
        lines.append(
            "- Common steps: "
            + ", ".join(f"{step} ({count})" for step, count in steps.most_common(limit))
        )
    if causes:
        lines.append(
            "- Common root causes: "
            + " | ".join(cause for cause, _ in causes.most_common(limit))
        )
    if examples:
        lines.append("- Representative evidence:")
        lines.extend(examples)
    return "\n".join(lines)


def extract_fenced_block(response: str, language: str) -> str:
    """Extract the last fenced code block for the given language."""
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    raise ValueError(f"LLM response does not contain a ```{language}``` block")


def extract_preface(response: str, language: str, max_chars: int = 500) -> str:
    """Extract the text before the first fenced block for the given language."""
    match = re.search(rf"```{re.escape(language)}", response)
    if match:
        return response[: match.start()].strip()
    return response[:max_chars].strip()
