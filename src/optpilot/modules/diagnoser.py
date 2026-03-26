"""Diagnoser - FM diagnosis with agent/step level localization.

Takes MAST-annotated traces and uses LLM to localize each active FM
to specific agent + step + root cause. Supports concurrent LLM calls.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from optpilot.config import DIAGNOSER_MAX_WORKERS
from optpilot.data.fm_taxonomy import FM_DEFINITIONS
import re

from optpilot.llm import acall_llm, acall_llm_json, call_llm, call_llm_json
from optpilot.models import FMLabel, FMLocalization, FMProfile, MASTrace

# If trace is shorter than this, send it whole; otherwise summarize
FULL_TRACE_THRESHOLD = 15000  # ~4K tokens


SHORT_CLASSIFICATION_PROMPT = """\
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


CLASSIFICATION_PROMPT = """\
Below I will provide a multiagent system trace. Provide me an analysis of the failure modes and inefficiencies as I will say below.
In the traces, analyze the system behaviour. There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no.
Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end.
Also tell me whether the task is successfully completed or not, as a binary yes or no.
At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better.
Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no.
Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):
*** begin of things you should answer *** @@
A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>
B. Whether the task is successfully completed or not: <yes or no>
C. Whether you encounter any of the failure modes or inefficiencies:
1.1 Disobey Task Specification: <yes or no>
1.2 Disobey Role Specification: <yes or no>
1.3 Step Repetition: <yes or no>
1.4 Loss of Conversation History: <yes or no>
1.5 Unaware of Termination Conditions: <yes or no>
2.1 Conversation Reset: <yes or no>
2.2 Fail to Ask for Clarification: <yes or no>
2.3 Task Derailment: <yes or no>
2.4 Information Withholding: <yes or no>
2.5 Ignored Other Agent's Input: <yes or no>
2.6 Action-Reasoning Mismatch: <yes or no>
3.1 Premature Termination: <yes or no>
3.2 No or Incorrect Verification: <yes or no>
3.3 Weak Verification: <yes or no>
@@*** end of your answer ***

An example answer is:
A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.
B. no
C.
1.1 no
1.2 yes
1.3 no
1.4 no
1.5 no
2.1 no
2.2 no
2.3 yes
2.4 no
2.5 no
2.6 no
3.1 no
3.2 no
3.3 yes

Here is the trace:
{trace_content}

Also, here are the explanations (definitions) of the failure modes and inefficiencies:

1.1 Disobey Task Specification: Agents fail to follow task constraints due to unclear, incomplete, or ambiguous instructions or inadequate constraint interpretation, resulting in incorrect outputs and reduced performance.

1.2 Disobey Role Specification: Agents neglect their defined responsibilities, potentially assuming inappropriate roles.

1.3 Step Repetition: Unnecessary duplication of completed phases from inadequate state or context tracking, inefficient workflow management or failure to recognize task completion.

1.4 Loss of Conversation History: Unexpected context truncation, disregarding recent interaction history and reverting to an antecedent conversational state.

1.5 Unaware of Termination Conditions: Systems fail to recognize stopping criteria, causing unnecessary conversation turns, inefficient use of resources, or potential harm to the correctness of the system.

2.1 Conversation Reset: Unexpected or unwarranted restarting of a dialogue, potentially losing context and progress.

2.2 Fail to Ask for Clarification: Agents cannot request additional information when encountering ambiguous data.

2.3 Task Derailment: Deviation from intended objectives producing irrelevant or unproductive actions.

2.4 Information Withholding: Agents possess critical information but fail to share it, leading to reduced operational efficiency and incorrect or suboptimal decision-making.

2.5 Ignored Other Agent's Input: Systems disregard peer recommendations, causing bad decisions, stalled progress, or missed opportunities.

2.6 Action-Reasoning Mismatch: Inconsistency between stated reasoning and actual outputs, producing unexpected, unintended, or counterproductive behaviors.

3.1 Premature Termination: Tasks end before all necessary information has been exchanged or objectives have been met.

3.2 No or Incorrect Verification: Either verification steps are absent or verifiers fail to execute their specified functions, permitting errors to propagate undetected.

3.3 Weak Verification: Verification mechanisms exist but fail to comprehensively cover all essential aspects, allowing subtle errors, inconsistencies, or vulnerabilities to remain undetected.

Here are some examples of the failure modes and inefficiencies:

Step Repetition: Identical thoughts and actions appeared consecutively. A navigator gave the same response about the _separable function three times verbatim across multiple timestamps.

No or Incorrect Verification: The agent generated incorrect code without running tests or checking results. A math problem about chalk usage produced answer 0 instead of the correct answer 2 because the logic for calculating remaining usable chalk was flawed.

Weak Verification: Code reviewers performed superficial checks without examining logic. A Sudoku solver review only confirmed Finished despite missing validation logic.

Ignored Other Agent's Input: An assistant repeatedly ignored another agent's requests for clarification. When asked for missing problem information about ribbon length, the supervisor responded Continue multiple times rather than providing needed data.

Information Withholding: A navigator discovered a solution internally but did not communicate it to the planner. The navigator found that qdp.py contains QDP file readers and identified case-sensitivity issues but withheld this specific solution information.

Disobey Task Specification: Developers used placeholders like pass statements despite requirements for fully functional code. A Checkers game included unimplemented methods like def update(self): pass.

Action-Reasoning Mismatch: Agents stated they would check multiple components but only examined one. An agent claimed to carefully review implementation and check known issues but only investigated the _separable function.

Unaware of Termination Conditions: An agent repeatedly said Continue despite being told the problem was unsolvable with current information. This continued across 4+ exchanges ignoring explicit statements that data was insufficient.

Loss of Conversation History: An agent forgot previous context, reintroducing issues already resolved. It mentioned reinstalling scikit-learn as already done, then discussed installing lightgbm without referencing earlier progress.

Disobey Role Specification: Agents exceeded their designated roles. A navigator claimed inability to provide code then offered bash commands outside navigation scope.

Conversation Reset: Systems reinitialize, losing accumulated context and repeating earlier analysis identically across multiple Initialized messages.

Task Derailment: Agents created simplified implementations instead of examining actual code. Rather than finding the real _separable function, an agent proposed fictional implementations then modified those.

Premature Termination: Tasks abandoned when encountering obstacles rather than resolving them. A playlist access token error caused immediate failure instead of troubleshooting."""


LOCALIZATION_PROMPT = """\
You are a multi-agent system (MAS) fault diagnosis expert.

Given an MAS execution trace with a known failure mode:
- FM-{fm_id}: {fm_name}
- Definition: {fm_description}

## Trace
{trace_content}

Analyze the trace to localize the fault. Respond with ONLY a JSON object:

{{
    "agent": "<name of the faulty agent>",
    "step": "<phase or step where the fault occurs>",
    "context": "<key context around the fault, max 2 sentences>",
    "root_cause": "<root cause analysis, max 2 sentences>"
}}"""


_ALL_FM_IDS = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]


def _parse_fm_from_mast_response(response: str, fm_id: str) -> bool:
    """Parse a single FM yes/no from MAST-style response text."""
    patterns = [
        rf"{re.escape(fm_id)}\s+.*?(yes|no)",
        rf"{re.escape(fm_id)}\s*[:]?\s*(yes|no)",
        rf"{re.escape(fm_id)}\s+(yes|no)",
        rf"{re.escape(fm_id)}\s*\n\s*(yes|no)",
    ]
    text_lower = response.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1) == "yes"
    return False


def _parse_all_fms_from_mast_response(response: str) -> dict[str, bool]:
    """Parse all FM yes/no labels from MAST-style response text."""
    return {fm_id: _parse_fm_from_mast_response(response, fm_id) for fm_id in _ALL_FM_IDS}


def _prepare_trace_content(trajectory: str) -> str:
    if len(trajectory) <= FULL_TRACE_THRESHOLD:
        return trajectory
    head = trajectory[:5000]
    tail = trajectory[-5000:]
    return f"{head}\n\n[... {len(trajectory) - 10000} chars omitted ...]\n\n{tail}"


class Diagnoser:
    """Diagnose MAS traces: localize FM to agent + step level."""

    def __init__(self, max_workers: int = DIAGNOSER_MAX_WORKERS):
        self.max_workers = max_workers

    def diagnose(self, trace: MASTrace, target_fm: str | None = None) -> FMProfile:
        """Diagnose a trace, localizing active FMs concurrently."""
        profile, fms_to_localize = self._initialize_profile(trace, target_fm)
        if not fms_to_localize:
            return profile

        trace_content = _prepare_trace_content(trace.trajectory)

        # 并行诊断所有 FM
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(fms_to_localize))) as pool:
            futures = {
                pool.submit(self._localize_fm, trace, fm_id, trace_content): fm_id
                for fm_id in fms_to_localize
            }
            for fut in as_completed(futures):
                fm_id = futures[fut]
                profile.localization[fm_id] = fut.result()

        return profile

    async def adiagnose(
        self,
        trace: MASTrace,
        target_fm: str | None = None,
    ) -> FMProfile:
        """Async diagnosis using true coroutine-based LLM concurrency."""
        profile, fms_to_localize = self._initialize_profile(trace, target_fm)
        if not fms_to_localize:
            return profile

        trace_content = _prepare_trace_content(trace.trajectory)
        semaphore = asyncio.Semaphore(min(self.max_workers, len(fms_to_localize)))

        async def localize(fm_id: str) -> tuple[str, FMLocalization]:
            async with semaphore:
                return fm_id, await self._alocalize_fm(trace, fm_id, trace_content)

        results = await asyncio.gather(*(localize(fm_id) for fm_id in fms_to_localize))
        for fm_id, localization in results:
            profile.localization[fm_id] = localization
        return profile

    def diagnose_batch(
        self, traces: list[MASTrace], target_fm: str | None = None,
    ) -> list[FMProfile]:
        """Diagnose multiple traces concurrently."""
        if not traces:
            return []

        # 并行诊断所有 trace
        results: dict[int, FMProfile] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(traces))) as pool:
            futures = {
                pool.submit(self.diagnose, t, target_fm): i
                for i, t in enumerate(traces)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"  Warning: diagnose_batch failed for trace index {idx}: {e}")
                    results[idx] = FMProfile(trace_id=traces[idx].trace_id)

        return [results[i] for i in range(len(traces))]

    async def adiagnose_batch(
        self,
        traces: list[MASTrace],
        target_fm: str | None = None,
    ) -> list[FMProfile]:
        """Async batch diagnosis with bounded coroutine fan-out."""
        if not traces:
            return []

        semaphore = asyncio.Semaphore(min(self.max_workers, len(traces)))
        results: dict[int, FMProfile] = {}

        async def diagnose_one(idx: int, trace: MASTrace) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.adiagnose(trace, target_fm)
                except Exception as e:
                    print(f"  Warning: diagnose_batch failed for trace index {idx}: {e}")
                    results[idx] = FMProfile(trace_id=trace.trace_id)

        await asyncio.gather(
            *(diagnose_one(i, trace) for i, trace in enumerate(traces))
        )
        return [results[i] for i in range(len(traces))]

    def classify_fm(
        self, trace: MASTrace, fm_id: str, model: str | None = None,
        prompt_style: str = "mast",
    ) -> bool:
        """Classify whether a trace exhibits a specific FM. Returns True/False."""
        result = self.classify_all_fms(trace, model=model, prompt_style=prompt_style)
        return result.get(fm_id, False)

    def classify_all_fms(
        self, trace: MASTrace, model: str | None = None,
        prompt_style: str = "mast",
    ) -> dict[str, bool]:
        """Classify all 14 FMs for a trace in a single LLM call.

        Args:
            prompt_style: "mast" for MAST original prompt, "short" for optimized short prompt.
        """
        trace_content = _prepare_trace_content(trace.trajectory)
        kwargs = {}
        if model:
            kwargs["model"] = model

        try:
            if prompt_style == "short":
                prompt = SHORT_CLASSIFICATION_PROMPT.format(trace_content=trace_content)
                result = call_llm_json(
                    [{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    **kwargs,
                )
                return {
                    fm_id: bool(result.get(fm_id, False))
                    for fm_id in _ALL_FM_IDS
                }
            else:
                prompt = CLASSIFICATION_PROMPT.format(trace_content=trace_content)
                response = call_llm(
                    [{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    **kwargs,
                )
                return _parse_all_fms_from_mast_response(response)
        except Exception as e:
            print(f"  Warning: classify_all_fms failed for trace {trace.trace_id}: {e}")
            return {}

    def classify_batch(
        self, traces: list[MASTrace], fm_id: str, model: str | None = None,
        prompt_style: str = "mast",
    ) -> None:
        """Classify FM for multiple traces, mutating trace.mast_annotation in-place."""
        if not traces:
            return

        def _classify_one(trace: MASTrace) -> dict[str, bool]:
            return self.classify_all_fms(trace, model=model, prompt_style=prompt_style)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(traces))) as pool:
            futures = {pool.submit(_classify_one, t): i for i, t in enumerate(traces)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    all_fms = fut.result()
                    # Write all classified FMs, but ensure the requested fm_id is set
                    for fid, present in all_fms.items():
                        traces[idx].mast_annotation[fid] = 1 if present else 0
                    if fm_id not in traces[idx].mast_annotation:
                        traces[idx].mast_annotation[fm_id] = 0
                except Exception as e:
                    print(f"  Warning: classify_batch failed for trace index {idx}: {e}")
                    traces[idx].mast_annotation[fm_id] = 0

    def _localize_fm(
        self, trace: MASTrace, fm_id: str, trace_content: str,
    ) -> FMLocalization:
        fm_info = FM_DEFINITIONS[fm_id]
        prompt = LOCALIZATION_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            trace_content=trace_content,
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
            )
            return FMLocalization(
                agent=result.get("agent", "unknown"),
                step=result.get("step", "unknown"),
                context=result.get("context", ""),
                root_cause=result.get("root_cause", ""),
            )
        except Exception as e:
            print(f"  Warning: Failed to localize FM-{fm_id} for trace {trace.trace_id}: {e}")
            return FMLocalization(
                agent="unknown", step="unknown",
                context="", root_cause=f"Localization failed: {e}",
            )

    async def _alocalize_fm(
        self, trace: MASTrace, fm_id: str, trace_content: str,
    ) -> FMLocalization:
        fm_info = FM_DEFINITIONS[fm_id]
        prompt = LOCALIZATION_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            trace_content=trace_content,
        )

        try:
            result = await acall_llm_json(
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
            )
            return FMLocalization(
                agent=result.get("agent", "unknown"),
                step=result.get("step", "unknown"),
                context=result.get("context", ""),
                root_cause=result.get("root_cause", ""),
            )
        except Exception as e:
            print(f"  Warning: Failed to localize FM-{fm_id} for trace {trace.trace_id}: {e}")
            return FMLocalization(
                agent="unknown",
                step="unknown",
                context="",
                root_cause=f"Localization failed: {e}",
            )

    def _initialize_profile(
        self,
        trace: MASTrace,
        target_fm: str | None,
    ) -> tuple[FMProfile, list[str]]:
        profile = FMProfile(trace_id=trace.trace_id)

        for fm_id, fm_info in FM_DEFINITIONS.items():
            present = trace.mast_annotation.get(fm_id, 0) == 1
            profile.labels[fm_id] = FMLabel(
                fm_id=fm_id,
                fm_name=fm_info["name"],
                category=fm_info["category"],
                present=present,
            )

        fms_to_localize = (
            [target_fm] if target_fm and target_fm in profile.active_fm_ids()
            else profile.active_fm_ids()
        )
        return profile, fms_to_localize
