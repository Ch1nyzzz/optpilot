"""FM Classifier prompt — evolved by AdaEvolve.

The classify() function takes a MAS trace and returns a dict of FM labels.
The EVOLVE-BLOCK contains the prompt template that will be optimized.
"""

import json
import re


# EVOLVE-BLOCK-START
CLASSIFICATION_PROMPT = """\
You are a multi-agent system (MAS) fault diagnosis expert. Analyze this MAS execution trace for the following 14 failure modes. For each, answer yes or no.

## Failure Mode Definitions

**Category 1 - Agent Compliance:**
1.1 Disobey Task Specification: Agent fails to follow task constraints or requirements.
1.2 Disobey Role Specification: Agent neglects its defined role, performs actions outside its scope.
1.3 Step Repetition: Unnecessary duplication of already-completed work — re-deriving the same result, re-running the same analysis, or repeating the same reasoning without new information.
1.4 Loss of Conversation History: Agent loses track of prior context, reintroduces resolved issues.
1.5 Unaware of Termination: Agent doesn't know when to stop, continues past task completion.

**Category 2 - Inter-Agent Communication:**
2.1 Conversation Reset: Dialogue unexpectedly restarts, losing prior progress.
2.2 Fail to Ask Clarification: Agent proceeds despite ambiguous or incomplete information.
2.3 Task Derailment: Agent deviates from the original task objective.
2.4 Information Withholding: Agent fails to share critical information with other agents.
2.5 Ignored Other Agent's Input: Agent disregards input from other agents.
2.6 Action-Reasoning Mismatch: Agent's stated reasoning doesn't match its actual actions/output.

**Category 3 - Verification:**
3.1 Premature Termination: Task ends before objectives are fully met.
3.2 No or Incorrect Verification: Verification absent or reaches wrong conclusions.
3.3 Weak Verification: Verification exists but is superficial, misses errors.

## Trace
{trace_content}

Respond with ONLY a JSON object. For each FM, true if present, false if not:

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
