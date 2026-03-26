"""Coarse 3-label FM classifier prompt.

The classify() function takes a MAS trace and returns:
    {"C1": bool, "C2": bool, "C3": bool}
"""

import json
import re


# EVOLVE-BLOCK-START
CLASSIFICATION_PROMPT = """\
You are an MAS failure‑analysis expert. Given the execution trace below, decide for each category whether the failure is present.

C1: Agent Compliance – ignores constraints/role, repeats completed work, loses history, or continues after completion.
C2: Inter‑Agent Communication – conversation reset, no clarification, derailment, ignores important info, or action‑reasoning mismatch.
C3: Verification – premature termination, missing/incorrect verification, or superficial checks.

Return ONLY a JSON object with keys C1, C2, C3 and boolean values. No extra text.

Trace:
{trace_content}
"""
# EVOLVE-BLOCK-END


ALL_CATEGORY_IDS = ["C1", "C2", "C3"]


def _prepare_trace(trajectory: str, threshold: int = 8000) -> str:
    """Trim long traces to stay within prompt limits.

    Keeps the first and last 3 KB and inserts an omission notice.
    """
    if len(trajectory) <= threshold:
        return trajectory
    head = trajectory[:3000]
    tail = trajectory[-3000:]
    omitted = len(trajectory) - 6000
    return f"{head}\n\n[... {omitted} chars omitted ...]\n\n{tail}"


def classify(trajectory: str, llm_call) -> dict[str, bool]:
    """Classify a trace into the 3 coarse FM categories."""
    trace_content = _prepare_trace(trajectory)
    prompt = CLASSIFICATION_PROMPT.format(trace_content=trace_content)
    response = llm_call(prompt)

    try:
        match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        data = {}

    return {cat_id: bool(data.get(cat_id, False)) for cat_id in ALL_CATEGORY_IDS}
