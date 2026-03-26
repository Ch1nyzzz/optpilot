"""FM Classifier prompt — evolved by AdaEvolve.

The classify() function takes a MAS trace and returns a dict of FM labels.
The EVOLVE-BLOCK contains the prompt template that will be optimized.
"""

import json
import re


# EVOLVE-BLOCK-START
CLASSIFICATION_PROMPT = """\
You are an expert analyzing a multi‑agent system (MAS) execution trace.
Your task is to decide for each of the 14 failure modes whether it occurs (**true**) or not (**false**).

**Recall‑first rule:** If you see *any* hint, however subtle, that could indicate a failure, answer **true** for that mode. When in doubt, prefer **true** over **false**.

### Failure mode cues (short definitions + typical textual hints)

1.1 **Disobey Task Specification** – agent breaks the task goal or constraints  
 e.g., “skip this step”, “use the wrong file”, “ignore the requirement”.

1.2 **Disobey Role Specification** – agent acts outside its role  
 e.g., planner gives an answer, verifier tries to plan, role‑mixing language.

1.3 **Step Repetition** – work is performed again after it was already completed  
 look for repeated calculations, re‑deriving a proven result, or restating the same reasoning with no new data.

1.4 **Loss of Conversation History** – agent forgets earlier discussion  
 asks about already‑answered questions or repeats resolved issues.

1.5 **Unaware of Termination** – agent keeps acting after a “Task completed” / “Finished” message.

2.1 **Conversation Reset** – dialogue restarts abruptly, losing prior context.

2.2 **Fail to Ask Clarification** – proceeds despite ambiguous or missing information, never asks “clarify”.

2.3 **Task Derailment** – conversation drifts away from the original objective.

2.4 **Information Withholding** – omits critical information that should be shared.

2.5 **Ignored Other Agent's Input** – disregards another agent’s contribution or answer.

2.6 **Action‑Reasoning Mismatch** – stated reasoning does not match the subsequent action or output  
 e.g., “I will compute X” but output Y, or reasoning about a check that is never performed.

3.1 **Premature Termination** – the task ends before the goals are satisfied.

3.2 **No or Incorrect Verification** – verification step missing or conclusions are wrong.

3.3 **Weak Verification** – verification exists but is superficial or skips key checks.

## Trace
{trace_content}

**Output format:** Respond with ONLY a single JSON object on one line, containing exactly the following keys (order does not matter). Use lower‑case booleans **true** or **false** *without* quotes.

{{"1.1":true,"1.2":true,"1.3":true,"1.4":true,"1.5":true,"2.1":true,"2.2":true,"2.3":true,"2.4":true,"2.5":true,"2.6":true,"3.1":true,"3.2":true,"3.3":true}}
"""
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
