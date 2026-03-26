"""Coarse 3-label FM classifier prompt.

The classify() function takes a MAS trace and returns:
    {"C1": bool, "C2": bool, "C3": bool}
"""

import json
import re


# EVOLVE-BLOCK-START
CLASSIFICATION_PROMPT = """\
You are a multi-agent system (MAS) failure analysis expert.
Analyze the execution trace and decide whether each of these coarse failure categories is present.

Categories:
- C1: Agent Compliance problems.
  Includes disobeying task constraints, disobeying role, repeating completed work,
  losing conversation history, or failing to stop after task completion.
- C2: Inter-Agent Communication problems.
  Includes conversation reset, failure to ask clarification, derailment,
  withholding important information, ignoring another agent's input,
  or action-reasoning mismatch.
- C3: Verification problems.
  Includes premature termination, missing or incorrect verification,
  or weak/superficial verification.

Instructions:
- Mark a category true if there is meaningful evidence that at least one problem in that category occurs.
- Mark a category false if the trace does not show convincing evidence for that category.
- Categories are independent. More than one category may be true.

Trace:
{trace_content}

Respond with ONLY a JSON object using exactly these keys:
{{"C1": false, "C2": false, "C3": false}}
"""
# EVOLVE-BLOCK-END


ALL_CATEGORY_IDS = ["C1", "C2", "C3"]


def _prepare_trace(trajectory: str, threshold: int = 15000) -> str:
    if len(trajectory) <= threshold:
        return trajectory
    head = trajectory[:5000]
    tail = trajectory[-5000:]
    return f"{head}\n\n[... {len(trajectory) - 10000} chars omitted ...]\n\n{tail}"


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
