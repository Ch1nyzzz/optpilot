"""Diagnoser - 6-group FM diagnosis with agent/step level localization.

Uses the validated 6-group taxonomy (A-F) with MiniMax M2.5.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from optpilot.config import DIAGNOSER_MAX_WORKERS
from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS, GROUP_IDS
from optpilot.llm import acall_llm_json, call_llm_json
from optpilot.models import FMLabel, FMLocalization, FMProfile, MASTrace

FULL_TRACE_THRESHOLD = 15000


CLASSIFICATION_PROMPT = """\
You are a careful evaluator for multi-agent system execution traces.

Your job is to label a trace using the following 6-group taxonomy.

Definitions:

A = Instruction Non-Compliance
The agent violates an explicit task requirement, constraint, or role boundary.

B = Execution Loop / Stuck
The system repeats the same step or keeps going after it should stop, without meaningful progress.

C = Context Loss
The agent loses previously established context, forgets prior progress, or the conversation effectively resets.

D = Communication Failure
Critical information is not shared, ignored, or clarification is not requested when needed between agents.

E = Task Drift / Reasoning Error
The execution path deviates from the intended task, or the agent's action does not match its stated reasoning.

F = Verification Failure
The system fails to verify properly, verifies too weakly, stops too early, or concludes success incorrectly.

Important rules:
1. A trace may have multiple true labels.
2. A trace may also have NO failure labels at all.
3. If none of A-F clearly applies, output all false.
4. Do not assume a trace is faulty just because it comes from a benchmark dataset.
5. Label only what is clearly supported by the trace.
6. Prefer precision over recall when evidence is weak.
7. Focus on root cause, not just surface symptoms.
8. Keep rationale brief and evidence-based.

Return ONLY valid JSON with this exact schema:
{
  "A": false,
  "B": false,
  "C": false,
  "D": false,
  "E": false,
  "F": false,
  "primary_failure_group": "none",
  "primary_turning_point": "",
  "evidence_snippet": "",
  "rationale": ""
}"""


CLASSIFICATION_USER_TEMPLATE = """\
Label the following multi-agent system trace using the 6-group taxonomy.

Instructions:
- Read the full trace carefully.
- Decide whether each label A-F is true or false.
- If the trace completes normally and no clear failure appears, return all false.
- If one or more labels are true, you MUST choose exactly one `primary_failure_group`
  from A-F as the single dominant root cause that best explains why this task failed.
- If no failure exists, set `primary_failure_group` to "none".
- `primary_turning_point` should be the first step where failure becomes clearly visible. If no failure exists, write "none".
- `evidence_snippet` should be a short quote or short summary of the key evidence.
- `rationale` should be 1-3 sentences only.

Trace:
{trace_content}"""


LOCALIZATION_PROMPT = """\
You are a multi-agent system (MAS) fault diagnosis expert.

Given an MAS execution trace with a known failure group:
- Group {group_id}: {group_name}
- Definition: {group_description}
- Typical repair strategy: {repair_strategy}

## How the DAG executor works
- The MAS is defined as a YAML DAG with nodes (agents, literals, passthrough, loop_counter) and edges.
- Each edge has: trigger (activates the target node), carry_data (passes output to target's input), condition (keyword matching).
- **In loops**: when Agent_Verifier triggers Agent_Code_Executor, the Code_Executor ONLY receives the Verifier's output via carry_data. It does NOT automatically receive the original USER input again.
- If an agent says "I need the problem statement" or produces empty/confused output, it usually means the edge feeding it did not carry the necessary context.
- Loop counters track iterations; when max_iterations is reached, the exit edge fires.

## Trace
{trace_content}

Analyze the trace to localize the fault. Consider both agent-level AND DAG-structure-level causes.

Respond with ONLY a JSON object:

{{
    "agent": "<name of the faulty agent, or 'DAG_structure' if the fault is in edges/routing>",
    "step": "<phase or step where the fault occurs>",
    "context": "<key context around the fault, max 2 sentences>",
    "root_cause": "<root cause analysis — specify if the issue is in an agent prompt, edge carry_data, loop config, or missing edges, max 3 sentences>",
    "dag_component": "<'agent_prompt' | 'edge_carry_data' | 'edge_condition' | 'edge_missing' | 'loop_config' | 'node_config' | 'other'>"
}}"""


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
        """Diagnose a trace: classify 6-group labels + localize active groups."""
        profile = self._build_profile(trace)
        groups_to_localize = (
            [target_group] if target_group and target_group in profile.active_fm_ids()
            else ([profile.primary_failure_id()] if profile.primary_failure_id() else [])
        )
        if not groups_to_localize:
            return profile

        trace_content = _prepare_trace_content(trace.trajectory)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(groups_to_localize))) as pool:
            futures = {
                pool.submit(self._localize_group, trace, gid, trace_content): gid
                for gid in groups_to_localize
            }
            for fut in as_completed(futures):
                gid = futures[fut]
                profile.localization[gid] = fut.result()
                if gid == profile.primary_failure_id():
                    profile.primary_localization = profile.localization[gid]
        return profile

    async def adiagnose(self, trace: MASTrace, target_group: str | None = None) -> FMProfile:
        """Async diagnosis."""
        profile = await self._abuild_profile(trace)
        groups_to_localize = (
            [target_group] if target_group and target_group in profile.active_fm_ids()
            else ([profile.primary_failure_id()] if profile.primary_failure_id() else [])
        )
        if not groups_to_localize:
            return profile

        trace_content = _prepare_trace_content(trace.trajectory)
        semaphore = asyncio.Semaphore(min(self.max_workers, len(groups_to_localize)))

        async def localize(gid: str) -> tuple[str, FMLocalization]:
            async with semaphore:
                return gid, await self._alocalize_group(trace, gid, trace_content)

        results = await asyncio.gather(*(localize(gid) for gid in groups_to_localize))
        for gid, loc in results:
            profile.localization[gid] = loc
            if gid == profile.primary_failure_id():
                profile.primary_localization = loc
        return profile

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

    def _resolve_primary_group(self, result: dict[str, object]) -> str:
        active = [gid for gid in GROUP_IDS if bool(result.get(gid, False))]
        raw_primary = str(result.get("primary_failure_group", "")).strip().upper()
        if raw_primary in active:
            return raw_primary
        return active[0] if active else ""

    def _build_profile(self, trace: MASTrace) -> FMProfile:
        """Classify and build FMProfile."""
        result = self._classify_raw(trace)
        labels_dict = {gid: bool(result.get(gid, False)) for gid in GROUP_IDS}
        profile = FMProfile(trace_id=trace.trace_id, primary_fm_id=self._resolve_primary_group(result))
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
        profile = FMProfile(trace_id=trace.trace_id, primary_fm_id=self._resolve_primary_group(result))
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

    def _localize_group(self, trace: MASTrace, group_id: str, trace_content: str) -> FMLocalization:
        gdef = GROUP_DEFINITIONS[group_id]
        prompt = LOCALIZATION_PROMPT.format(
            group_id=group_id,
            group_name=gdef["name"],
            group_description=gdef["description"],
            repair_strategy=gdef["repair_strategy"],
            trace_content=trace_content,
        )
        try:
            result = call_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)
            return FMLocalization(
                agent=result.get("agent", "unknown"),
                step=result.get("step", "unknown"),
                context=result.get("context", ""),
                root_cause=result.get("root_cause", ""),
                dag_component=result.get("dag_component", "other"),
            )
        except Exception as e:
            print(f"  Warning: localize Group-{group_id} failed for trace {trace.trace_id}: {e}")
            return FMLocalization(agent="unknown", step="unknown", context="", root_cause=f"Localization failed: {e}")

    async def _alocalize_group(self, trace: MASTrace, group_id: str, trace_content: str) -> FMLocalization:
        gdef = GROUP_DEFINITIONS[group_id]
        prompt = LOCALIZATION_PROMPT.format(
            group_id=group_id,
            group_name=gdef["name"],
            group_description=gdef["description"],
            repair_strategy=gdef["repair_strategy"],
            trace_content=trace_content,
        )
        try:
            result = await acall_llm_json([{"role": "user", "content": prompt}], max_tokens=4096)
            return FMLocalization(
                agent=result.get("agent", "unknown"),
                step=result.get("step", "unknown"),
                context=result.get("context", ""),
                root_cause=result.get("root_cause", ""),
                dag_component=result.get("dag_component", "other"),
            )
        except Exception as e:
            print(f"  Warning: async localize Group-{group_id} failed for trace {trace.trace_id}: {e}")
            return FMLocalization(agent="unknown", step="unknown", context="", root_cause=f"Localization failed: {e}")
