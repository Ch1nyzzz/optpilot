"""Diagnoser - multi-label issue tagging for the 6-group taxonomy.

Uses the validated 6-group taxonomy (A-F) with MiniMax M2.5.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from optpilot.config import DIAGNOSER_MAX_WORKERS
from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS, GROUP_IDS
from optpilot.llm import acall_llm_json, call_llm_json
from optpilot.models import FMLabel, FMProfile, MASTrace

FULL_TRACE_THRESHOLD = 15000


def _build_classification_prompt() -> str:
    """Build classification prompt from GROUP_DEFINITIONS with full context."""
    lines = [
        "You are a careful evaluator for multi-agent system execution traces.",
        "",
        "Your job is to inspect a trace and mark which problems are present using "
        "the following 6-group taxonomy.",
        "",
        "## Taxonomy Definitions",
        "",
    ]
    for gid in GROUP_IDS:
        gdef = GROUP_DEFINITIONS[gid]
        lines.append(f"### {gid} = {gdef['name']}")
        lines.append(f"**Definition:** {gdef['description']}")
        if gdef.get("analyze_hint"):
            lines.append(f"**What to look for:** {gdef['analyze_hint']}")
        if gdef.get("failure_examples"):
            lines.append("**Concrete examples of this failure:**")
            for ex_line in gdef["failure_examples"].split("\n"):
                ex_line = ex_line.strip()
                if ex_line:
                    lines.append(f"  - {ex_line}")
        lines.append("")

    lines.extend([
        "## Important rules",
        "1. A trace may have multiple true labels.",
        "2. A trace may also have NO failure labels at all.",
        "3. If none of A-F clearly applies, output all false.",
        "4. Do not assume a trace is faulty just because it comes from a benchmark dataset.",
        "5. Label only what is clearly supported by the trace.",
        "6. Prefer precision over recall when evidence is weak.",
        "7. Do NOT force a single root cause. Mark every issue that is clearly present.",
        "",
        "Return ONLY valid JSON with this exact schema:",
        "{",
    ])
    for gid in GROUP_IDS:
        lines.append(f'  "{gid}": false,')
    # Remove trailing comma from last line
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")

    return "\n".join(lines)


CLASSIFICATION_PROMPT = _build_classification_prompt()


CLASSIFICATION_USER_TEMPLATE = """\
Inspect the following multi-agent system trace using the 6-group taxonomy.

Instructions:
- Read the full trace carefully.
- Decide whether each label A-F is true or false.
- If the trace completes normally and no clear failure appears, return all false.
- This is a multi-label judgment task, not a root-cause attribution task.
- Mark every issue that is clearly supported by the trace.

Trace:
{trace_content}"""


def _prepare_trace_content(trajectory: str) -> str:
    if len(trajectory) <= FULL_TRACE_THRESHOLD:
        return trajectory
    head = trajectory[:5000]
    tail = trajectory[-5000:]
    return f"{head}\n\n[... {len(trajectory) - 10000} chars omitted ...]\n\n{tail}"


class Diagnoser:
    """Diagnose MAS traces using 6-group taxonomy (A-F)."""

    def __init__(self, max_workers: int = DIAGNOSER_MAX_WORKERS):
        self.max_workers = max_workers

    def classify(self, trace: MASTrace, model: str | None = None) -> dict[str, bool]:
        """Classify a trace into 6-group labels. Returns {"A": bool, ..., "F": bool}."""
        result = self._classify_raw(trace, model=model)
        return {gid: bool(result.get(gid, False)) for gid in GROUP_IDS}

    def diagnose(self, trace: MASTrace, target_group: str | None = None) -> FMProfile:
        """Diagnose a trace by returning its active issue labels."""
        return self._build_profile(trace)

    async def adiagnose(self, trace: MASTrace, target_group: str | None = None) -> FMProfile:
        """Async diagnosis."""
        return await self._abuild_profile(trace)

    def diagnose_batch(self, traces: list[MASTrace], target_group: str | None = None) -> list[FMProfile]:
        """Diagnose multiple traces concurrently."""
        if not traces:
            return []
        results: dict[int, FMProfile] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(traces))) as pool:
            futures = {
                pool.submit(self.diagnose, t, target_group): i
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

    def classify_batch(self, traces: list[MASTrace]) -> list[FMProfile]:
        """Classify multiple traces without localization."""
        if not traces:
            return []
        results: dict[int, FMProfile] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(traces))) as pool:
            futures = {
                pool.submit(self._build_profile, t): i
                for i, t in enumerate(traces)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"  Warning: classify_batch failed for trace index {idx}: {e}")
                    results[idx] = FMProfile(trace_id=traces[idx].trace_id)
        return [results[i] for i in range(len(traces))]

    async def adiagnose_batch(self, traces: list[MASTrace], target_group: str | None = None) -> list[FMProfile]:
        """Async batch diagnosis."""
        if not traces:
            return []
        semaphore = asyncio.Semaphore(min(self.max_workers, len(traces)))
        results: dict[int, FMProfile] = {}

        async def diagnose_one(idx: int, trace: MASTrace) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.adiagnose(trace, target_group)
                except Exception as e:
                    print(f"  Warning: adiagnose_batch failed for trace index {idx}: {e}")
                    results[idx] = FMProfile(trace_id=trace.trace_id)

        await asyncio.gather(*(diagnose_one(i, t) for i, t in enumerate(traces)))
        return [results[i] for i in range(len(traces))]

    async def aclassify_batch(self, traces: list[MASTrace]) -> list[FMProfile]:
        """Async batch classification without localization."""
        if not traces:
            return []
        semaphore = asyncio.Semaphore(min(self.max_workers, len(traces)))
        results: dict[int, FMProfile] = {}

        async def classify_one(idx: int, trace: MASTrace) -> None:
            async with semaphore:
                try:
                    results[idx] = await self._abuild_profile(trace)
                except Exception as e:
                    print(f"  Warning: aclassify_batch failed for trace index {idx}: {e}")
                    results[idx] = FMProfile(trace_id=trace.trace_id)

        await asyncio.gather(*(classify_one(i, t) for i, t in enumerate(traces)))
        return [results[i] for i in range(len(traces))]

    # --- internal ---

    def _classify_raw(self, trace: MASTrace, model: str | None = None) -> dict[str, object]:
        trace_content = _prepare_trace_content(trace.trajectory)
        kwargs = {}
        if model:
            kwargs["model"] = model
        try:
            result = call_llm_json(
                [
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": CLASSIFICATION_USER_TEMPLATE.format(trace_content=trace_content)},
                ],
                max_tokens=4096,
                **kwargs,
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"  Warning: classify failed for trace {trace.trace_id}: {e}")
            return {}

    async def _aclassify_raw(self, trace: MASTrace, model: str | None = None) -> dict[str, object]:
        trace_content = _prepare_trace_content(trace.trajectory)
        kwargs = {}
        if model:
            kwargs["model"] = model
        try:
            result = await acall_llm_json(
                [
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": CLASSIFICATION_USER_TEMPLATE.format(trace_content=trace_content)},
                ],
                max_tokens=4096,
                **kwargs,
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"  Warning: async classify failed for trace {trace.trace_id}: {e}")
            return {}

    def _build_profile(self, trace: MASTrace) -> FMProfile:
        """Classify and build FMProfile."""
        result = self._classify_raw(trace)
        labels_dict = {gid: bool(result.get(gid, False)) for gid in GROUP_IDS}
        profile = FMProfile(trace_id=trace.trace_id)
        for gid in GROUP_IDS:
            gdef = GROUP_DEFINITIONS[gid]
            profile.labels[gid] = FMLabel(
                fm_id=gid,
                fm_name=gdef["name"],
                category=gid,
                present=labels_dict.get(gid, False),
            )
        return profile

    async def _abuild_profile(self, trace: MASTrace) -> FMProfile:
        """Async classify and build FMProfile."""
        result = await self._aclassify_raw(trace)
        labels_dict = {gid: bool(result.get(gid, False)) for gid in GROUP_IDS}
        profile = FMProfile(trace_id=trace.trace_id)
        for gid in GROUP_IDS:
            gdef = GROUP_DEFINITIONS[gid]
            profile.labels[gid] = FMLabel(
                fm_id=gid,
                fm_name=gdef["name"],
                category=gid,
                present=labels_dict.get(gid, False),
            )
        return profile

    async def _aclassify(self, trace: MASTrace, model: str | None = None) -> dict[str, bool]:
        """Async 6-group classification."""
        result = await self._aclassify_raw(trace, model=model)
        return {gid: bool(result.get(gid, False)) for gid in GROUP_IDS}
