"""FM Classifier prompt — evolved by AdaEvolve.

The classify() function takes a MAS trace and returns a dict of FM labels.
The EVOLVE-BLOCK contains the prompt template that will be optimized.
"""

import json
import re


# EVOLVE-BLOCK-START
CLASSIFICATION_PROMPT = """\
You are an expert analyzing a multi‑agent system (MAS) execution trace.  
Your goal is to decide for each of the 14 failure modes whether it occurs (**true**) or not (**false**).  

**Important:** If any part of the trace contains even a subtle hint or a single piece of evidence for a failure, mark it **true**. Prioritize recall over precision.

### Failure mode definitions (with concrete examples)

**1.1 Disobey Task Specification** – the agent violates the task goal or constraints (e.g., skips a required step, uses wrong data).  

**1.2 Disobey Role Specification** – the agent acts outside its defined role (e.g., a planner provides answers, a verifier tries to act).  

**1.3 Step Repetition** – the same work is done again after it was already completed: re‑deriving a proven result, re‑running an analysis, or restating the same reasoning without new information.  

**1.4 Loss of Conversation History** – the agent forgets earlier discussion and asks about or repeats resolved issues.  

**1.5 Unaware of Termination** – the agent continues acting after the system has declared the task finished (e.g., asks for next step after a “Task completed” message).  

**2.1 Conversation Reset** – dialogue restarts abruptly, losing prior progress.  

**2.2 Fail to Ask Clarification** – the agent proceeds despite ambiguous or missing information.  

**2.3 Task Derailment** – conversation drifts away from the original objective.  

**2.4 Information Withholding** – the agent does not share critical information with other agents.  

**2.5 Ignored Other Agent's Input** – the agent disregards another agent’s contribution.  

**2.6 Action‑Reasoning Mismatch** – the stated reasoning does not match the action or output (e.g., says it will compute X but outputs Y, or reasons about checking data but skips the check).  

**3.1 Premature Termination** – the task ends before objectives are satisfied.  

**3.2 No or Incorrect Verification** – verification step missing or conclusions wrong.  

**3.3 Weak Verification** – verification present but superficial, missing errors.

## Trace
{trace_content}

Respond with ONLY a JSON object containing exactly the following keys (order does not matter), with boolean values **true** or **false** (lower‑case, no quotes around the words):

{{
    "1.1": <true/false>, "1.2": <true/false>, "1.3": <true/false>,
    "1.4": <true/false>, "1.5": <true/false>,
    "2.1": <true/false>, "2.2": <true/false>, "2.3": <true/false>,
    "2.4": <true/false>, "2.5": <true/false>, "2.6": <true/false>,
    "3.1": <true/false>, "3.2": <true/false>, "3.3": <true/false>
}}"""
# EVOLVE-BLOCK-END


ALL_FM_IDS = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]


def _prepare_trace(trajectory: str, threshold: int = 15000) -> str:
    if len(trajectory) <= threshold:
        return trajectory
    head = trajectory[:5000]
    tail = trajectory[-5000:]
    return f"{head}\n\n[... {len(trajectory) - 10000} chars omitted ...]\n\n{tail}"


def classify(trajectory: str, llm_call) -> dict[str, bool]:
    """Classify all 14 FMs for a trace. Returns {fm_id: bool}.

    Args:
        trajectory: Raw MAS execution trace text.
        llm_call: Callable(prompt) -> str, calls the LLM and returns response text.
    """
    trace_content = _prepare_trace(trajectory)
    prompt = CLASSIFICATION_PROMPT.format(trace_content=trace_content)
    response = llm_call(prompt)

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        data = {}

    return {fm_id: bool(data.get(fm_id, False)) for fm_id in ALL_FM_IDS}
