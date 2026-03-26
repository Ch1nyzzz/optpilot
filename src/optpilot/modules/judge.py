"""Judge - LLM-as-Judge for offline counterfactual evaluation.

Evaluates whether a repair candidate would fix a given FM
without actually re-running the MAS.
"""

from __future__ import annotations

from optpilot.config import JUDGE_MODEL
from optpilot.data.fm_taxonomy import FM_DEFINITIONS
from optpilot.llm import acall_llm_json, call_llm_json
from optpilot.models import (
    FMProfile, JudgeVerdict, MASTrace, RepairCandidate,
)

JUDGE_PROMPT = """\
You are a multi-agent system (MAS) repair evaluator performing counterfactual analysis.

Question: If the repair below had been applied BEFORE execution, would the fault have been prevented?

## Fault
- FM-{fm_id}: {fm_name} -- {fm_description}
- Agent: {agent}, Step: {step}
- Root Cause: {root_cause}
- Context: {failure_context}

## Proposed Repair
- Description: {repair_description}
- Actions:
{repair_actions_text}

## Evaluation Criteria
Think through each step before concluding:
1. Does the repair directly address the root cause?
2. Would the causal chain leading to the fault be broken?
3. Could the repair introduce new failure modes?

Respond with ONLY a JSON object:

{{
    "reasoning": "<step-by-step counterfactual analysis, max 4 sentences>",
    "would_fix": <true or false>,
    "confidence": <float 0.0-1.0>,
    "potential_side_effects": ["<side effect 1>", "..."]
}}"""


class Judge:
    """Offline counterfactual evaluation of repair candidates."""

    def evaluate(
        self,
        trace: MASTrace,
        fm_id: str,
        candidate: RepairCandidate,
        profile: FMProfile,
    ) -> JudgeVerdict:
        """Evaluate whether the repair would fix the FM."""
        fm_info = FM_DEFINITIONS[fm_id]
        loc = profile.localization.get(fm_id)

        actions_text = "\n".join(
            f"  {i+1}. [{a.repair_type.value}] {a.description}"
            for i, a in enumerate(candidate.actions)
        )

        prompt = JUDGE_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            agent=loc.agent if loc else "unknown",
            step=loc.step if loc else "unknown",
            root_cause=loc.root_cause if loc else "unknown",
            failure_context=loc.context if loc else "unknown",
            repair_description=candidate.description,
            repair_actions_text=actions_text or "  (no actions)",
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                model=JUDGE_MODEL,
                max_tokens=4096,
            )
            return JudgeVerdict(
                trace_id=trace.trace_id,
                fm_id=fm_id,
                repair_id=candidate.description[:50],
                would_fix=result.get("would_fix", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            print(f"  Warning: Judge failed for trace {trace.trace_id}, FM-{fm_id}: {e}")
            return JudgeVerdict(
                trace_id=trace.trace_id,
                fm_id=fm_id,
                repair_id=candidate.description[:50],
                would_fix=False,
                confidence=0.0,
                reasoning=f"Judge evaluation failed: {e}",
            )

    async def aevaluate(
        self,
        trace: MASTrace,
        fm_id: str,
        candidate: RepairCandidate,
        profile: FMProfile,
    ) -> JudgeVerdict:
        """Async evaluation for high-concurrency offline pipelines."""
        fm_info = FM_DEFINITIONS[fm_id]
        loc = profile.localization.get(fm_id)

        actions_text = "\n".join(
            f"  {i+1}. [{a.repair_type.value}] {a.description}"
            for i, a in enumerate(candidate.actions)
        )

        prompt = JUDGE_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            agent=loc.agent if loc else "unknown",
            step=loc.step if loc else "unknown",
            root_cause=loc.root_cause if loc else "unknown",
            failure_context=loc.context if loc else "unknown",
            repair_description=candidate.description,
            repair_actions_text=actions_text or "  (no actions)",
        )

        try:
            result = await acall_llm_json(
                [{"role": "user", "content": prompt}],
                model=JUDGE_MODEL,
                max_tokens=4096,
            )
            return JudgeVerdict(
                trace_id=trace.trace_id,
                fm_id=fm_id,
                repair_id=candidate.description[:50],
                would_fix=result.get("would_fix", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            print(f"  Warning: Judge failed for trace {trace.trace_id}, FM-{fm_id}: {e}")
            return JudgeVerdict(
                trace_id=trace.trace_id,
                fm_id=fm_id,
                repair_id=candidate.description[:50],
                would_fix=False,
                confidence=0.0,
                reasoning=f"Judge evaluation failed: {e}",
            )
