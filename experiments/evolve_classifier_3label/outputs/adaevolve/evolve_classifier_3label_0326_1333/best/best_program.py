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


def _generate_chunks(
    trajectory: str,
    chunk_size: int = 2000,
    overlap: int = 500,
    max_chunks: int = 5,
) -> list[str]:
    """
    Split ``trajectory`` into overlapping chunks.

    * ``chunk_size`` – size of each chunk in characters (default 2 KB).
    * ``overlap`` – number of characters each chunk overlaps with the previous
      one (default 500 bytes) to avoid cutting sentences in half.
    * ``max_chunks`` – upper bound on the number of chunks to keep API usage
      reasonable.

    Returns a list of chunk strings. If the trajectory is short enough,
    a single‑element list containing the whole text is returned.
    """
    if len(trajectory) <= chunk_size:
        return [trajectory]

    chunks = []
    start = 0
    while start < len(trajectory) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunk = trajectory[start:end]
        chunks.append(chunk)
        # Move start forward by chunk_size - overlap to create overlap
        start += chunk_size - overlap
    return chunks


def classify(trajectory: str, llm_call) -> dict[str, bool]:
    """
    Classify a trace into the three coarse FM categories using chunked LLM
    calls.

    The trajectory is split into a few overlapping chunks (see
    ``_generate_chunks``). Each chunk is fed to the same ``CLASSIFICATION_PROMPT``.
    The JSON responses from all chunks are parsed and the final label for each
    category is ``True`` if **any** chunk reports it as true; otherwise ``False``.
    This “any‑true” aggregation helps capture failures that appear only in the
    middle of long traces.
    """
    chunks = _generate_chunks(trajectory)

    # Accumulate votes per category across chunks
    votes = {cat_id: False for cat_id in ALL_CATEGORY_IDS}

    for chunk in chunks:
        prompt = CLASSIFICATION_PROMPT.format(trace_content=chunk)
        response = llm_call(prompt)

        try:
            match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            data = {}

        # Update votes: if any chunk says True, keep it True
        for cat_id in ALL_CATEGORY_IDS:
            if bool(data.get(cat_id, False)):
                votes[cat_id] = True

    return votes
