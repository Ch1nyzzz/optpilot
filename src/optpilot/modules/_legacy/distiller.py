"""Distiller - validates repairs and distills into Repair Library as skills."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean

from optpilot.data.fm_taxonomy_6group import GROUP_NAMES
from optpilot.library.repair_library import RepairLibrary
from optpilot.llm import call_llm
from optpilot.models import (
    FMProfile, JudgeVerdict, MASTrace, RepairCandidate, RepairEntry,
)

DISTILL_PROMPT = """\
Distill this repair into a reusable one-line skill summary.

Fault: FM-{fm_id} ({fm_name})
Repair: {repair_description}
Actions: {repair_actions}
Rationale: {rationale}

Summarize the root cause pattern (i.e., "when should this skill be applied?") in one sentence, max 50 words.
Output ONLY the summary text, no JSON."""

ONLINE_DISTILL_PROMPT = """\
Distill this validated MAS repair into a reusable one-line skill trigger.

Fault: FM-{fm_id} ({fm_name})

Observed before repair on proposal/validation evidence:
{before_summary}

Observed after repair on holdout validation evidence:
{after_summary}

Validation outcome:
- FM count: {before_count} -> {after_count}
- Pass rate: {before_pass_rate:.3f} -> {after_pass_rate:.3f}
- Avg runtime (s): {before_runtime:.2f} -> {after_runtime:.2f}

Repair:
- Description: {repair_description}
- Actions: {repair_actions}
- Rationale: {rationale}

Summarize when this skill should be applied in one sentence, max 50 words.
Output ONLY the summary text, no JSON."""


def _distill_pattern(fm_id: str, candidate: RepairCandidate) -> str:
    """Use LLM to distill a repair into a root cause pattern summary."""
    actions_text = "; ".join(a.description for a in candidate.actions)
    rationale_text = "; ".join(a.rationale for a in candidate.actions if a.rationale)

    prompt = DISTILL_PROMPT.format(
        fm_id=fm_id,
        fm_name=GROUP_NAMES.get(fm_id, ""),
        repair_description=candidate.description,
        repair_actions=actions_text,
        rationale=rationale_text,
    )

    try:
        return call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).strip().strip('"')
    except Exception:
        # Fallback: use first action's description
        return candidate.actions[0].description[:80] if candidate.actions else candidate.description[:80]


def _mean_non_null(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return mean(present) if present else 0.0


def _pass_rate(traces: list[MASTrace]) -> float:
    values = [1.0 if trace.task_success else 0.0 for trace in traces if trace.task_success is not None]
    return mean(values) if values else 0.0


def _summarize_fault_evidence(
    fm_id: str,
    traces: list[MASTrace],
    profiles: list[FMProfile],
    limit: int = 3,
) -> str:
    matching = [
        (trace, profile)
        for trace, profile in zip(traces, profiles, strict=False)
        if fm_id in profile.active_fm_ids()
    ]
    if not matching:
        return "- No active FM evidence in this split."

    agents = Counter()
    steps = Counter()
    causes = Counter()
    examples: list[str] = []

    for trace, profile in matching:
        loc = profile.localization.get(fm_id)
        if not loc:
            continue
        if loc.agent:
            agents[loc.agent] += 1
        if loc.step:
            steps[loc.step] += 1
        if loc.root_cause:
            causes[loc.root_cause.strip()] += 1
        if len(examples) < limit:
            context = loc.context.strip().replace("\n", " ")
            if len(context) > 160:
                context = context[:157] + "..."
            examples.append(
                f"- Trace {trace.trace_id}: agent={loc.agent}, step={loc.step}, "
                f"cause={loc.root_cause.strip()}, context={context}"
            )

    lines = [f"- Affected traces: {len(matching)}/{len(traces)}"]
    if agents:
        lines.append("- Common agents: " + ", ".join(f"{agent} ({count})" for agent, count in agents.most_common(limit)))
    if steps:
        lines.append("- Common steps: " + ", ".join(f"{step} ({count})" for step, count in steps.most_common(limit)))
    if causes:
        lines.append("- Common root causes: " + " | ".join(cause for cause, _ in causes.most_common(limit)))
    lines.extend(examples)
    return "\n".join(lines)


def _distill_online_pattern(
    fm_id: str,
    candidate: RepairCandidate,
    before_traces: list[MASTrace],
    before_profiles: list[FMProfile],
    after_traces: list[MASTrace],
    after_profiles: list[FMProfile],
) -> str:
    actions_text = "; ".join(a.description for a in candidate.actions)
    rationale_text = "; ".join(a.rationale for a in candidate.actions if a.rationale)
    before_count = sum(1 for p in before_profiles if fm_id in p.active_fm_ids())
    after_count = sum(1 for p in after_profiles if fm_id in p.active_fm_ids())
    before_pass_rate = _pass_rate(before_traces)
    after_pass_rate = _pass_rate(after_traces)
    before_runtime = _mean_non_null([t.latency_s for t in before_traces])
    after_runtime = _mean_non_null([t.latency_s for t in after_traces])

    prompt = ONLINE_DISTILL_PROMPT.format(
        fm_id=fm_id,
        fm_name=GROUP_NAMES.get(fm_id, ""),
        before_summary=_summarize_fault_evidence(fm_id, before_traces, before_profiles),
        after_summary=_summarize_fault_evidence(fm_id, after_traces, after_profiles),
        before_count=before_count,
        after_count=after_count,
        before_pass_rate=before_pass_rate,
        after_pass_rate=after_pass_rate,
        before_runtime=before_runtime,
        after_runtime=after_runtime,
        repair_description=candidate.description,
        repair_actions=actions_text,
        rationale=rationale_text,
    )

    try:
        return call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).strip().strip('"')
    except Exception:
        return candidate.actions[0].description[:80] if candidate.actions else candidate.description[:80]


class Distiller:
    """Distill repair results into Repair Library as reusable skills."""

    def __init__(self, library: RepairLibrary):
        self.library = library

    def distill_offline(
        self,
        fm_id: str,
        candidate: RepairCandidate,
        verdict: JudgeVerdict,
        source_mas: str = "AG2",
    ) -> RepairEntry:
        """Offline distillation: use Judge verdict, mark as 'unvalidated'."""
        pattern = _distill_pattern(fm_id, candidate)
        entry = RepairEntry(
            fm_id=fm_id,
            fm_name=GROUP_NAMES.get(fm_id, ""),
            source_mas=source_mas,
            root_cause_pattern=pattern,
            candidate=candidate,
            status="unvalidated",
            success_rate=verdict.confidence if verdict.would_fix else 0.0,
            n_applied=1,
            n_success=1 if verdict.would_fix else 0,
            created_at=datetime.now().isoformat(),
        )
        self.library.add(entry)
        return entry

    def distill_online(
        self,
        fm_id: str,
        candidate: RepairCandidate,
        before_traces: list[MASTrace],
        before_profiles: list[FMProfile],
        after_traces: list[MASTrace],
        after_profiles: list[FMProfile],
        source_mas: str = "AG2",
    ) -> RepairEntry:
        """Online distillation on holdout tasks.

        A repair only validates when it both reduces the target FM and improves
        pass rate on the holdout validation split.
        """
        before_count = sum(1 for p in before_profiles if fm_id in p.active_fm_ids())
        after_count = sum(1 for p in after_profiles if fm_id in p.active_fm_ids())
        fm_fixed = after_count < before_count

        before_pass_rate = _pass_rate(before_traces)
        after_pass_rate = _pass_rate(after_traces)
        pass_improved = after_pass_rate > before_pass_rate

        before_runtime = _mean_non_null([t.latency_s for t in before_traces])
        after_runtime = _mean_non_null([t.latency_s for t in after_traces])

        before_all_fms = set()
        after_all_fms = set()
        for p in before_profiles:
            before_all_fms.update(p.active_fm_ids())
        for p in after_profiles:
            after_all_fms.update(p.active_fm_ids())
        new_fms = after_all_fms - before_all_fms

        pattern = _distill_online_pattern(
            fm_id,
            candidate,
            before_traces,
            before_profiles,
            after_traces,
            after_profiles,
        )
        status = "validated" if fm_fixed and pass_improved else "failed"
        fm_improvement = 1.0 - (after_count / before_count) if before_count > 0 else 0.0
        pass_delta = after_pass_rate - before_pass_rate
        success_rate = (max(0.0, fm_improvement) + max(0.0, pass_delta)) / 2.0

        entry = RepairEntry(
            fm_id=fm_id,
            fm_name=GROUP_NAMES.get(fm_id, ""),
            source_mas=source_mas,
            root_cause_pattern=pattern,
            candidate=candidate,
            status=status,
            success_rate=max(0.0, success_rate),
            n_applied=before_count,
            n_success=before_count - after_count,
            side_effects=[f"Introduced FM-{f}" for f in new_fms],
            validation_metrics={
                "before_fm_count": before_count,
                "after_fm_count": after_count,
                "fm_fixed": fm_fixed,
                "before_pass_count": sum(1 for trace in before_traces if trace.task_success),
                "after_pass_count": sum(1 for trace in after_traces if trace.task_success),
                "before_pass_rate": before_pass_rate,
                "after_pass_rate": after_pass_rate,
                "pass_improved": pass_improved,
                "before_avg_runtime_s": before_runtime,
                "after_avg_runtime_s": after_runtime,
                "runtime_delta_s": after_runtime - before_runtime,
                "validation_tasks": len(before_traces),
            },
            created_at=datetime.now().isoformat(),
        )
        self.library.add(entry)
        return entry
