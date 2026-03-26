"""Distiller - validates repairs and distills into Repair Library as skills."""

from __future__ import annotations

from datetime import datetime

from optpilot.data.fm_taxonomy import FM_NAMES
from optpilot.library.repair_library import RepairLibrary
from optpilot.llm import call_llm
from optpilot.models import (
    FMProfile, JudgeVerdict, RepairCandidate, RepairEntry,
)

DISTILL_PROMPT = """将以下修复方案蒸馏为一条可复用的 skill 摘要。

故障: FM-{fm_id} ({fm_name})
修复方案: {repair_description}
修复操作: {repair_actions}
原因分析: {rationale}

请用一句话总结这个 skill 的 root cause pattern（即"什么情况下应该用这个 skill"），不超过50字。
直接输出摘要文本，不要 JSON。"""


def _distill_pattern(fm_id: str, candidate: RepairCandidate) -> str:
    """Use LLM to distill a repair into a root cause pattern summary."""
    actions_text = "; ".join(a.description for a in candidate.actions)
    rationale_text = "; ".join(a.rationale for a in candidate.actions if a.rationale)

    prompt = DISTILL_PROMPT.format(
        fm_id=fm_id,
        fm_name=FM_NAMES.get(fm_id, ""),
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
            fm_name=FM_NAMES.get(fm_id, ""),
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
        before_profiles: list[FMProfile],
        after_profiles: list[FMProfile],
        source_mas: str = "AG2",
    ) -> RepairEntry:
        """Online distillation: compare actual FM changes, mark as 'validated' or 'failed'."""
        before_count = sum(1 for p in before_profiles if fm_id in p.active_fm_ids())
        after_count = sum(1 for p in after_profiles if fm_id in p.active_fm_ids())
        fm_fixed = after_count < before_count

        before_all_fms = set()
        after_all_fms = set()
        for p in before_profiles:
            before_all_fms.update(p.active_fm_ids())
        for p in after_profiles:
            after_all_fms.update(p.active_fm_ids())
        new_fms = after_all_fms - before_all_fms

        pattern = _distill_pattern(fm_id, candidate)
        status = "validated" if fm_fixed else "failed"
        success_rate = 1.0 - (after_count / before_count) if before_count > 0 else 0.0

        entry = RepairEntry(
            fm_id=fm_id,
            fm_name=FM_NAMES.get(fm_id, ""),
            source_mas=source_mas,
            root_cause_pattern=pattern,
            candidate=candidate,
            status=status,
            success_rate=max(0.0, success_rate),
            n_applied=before_count,
            n_success=before_count - after_count,
            side_effects=[f"Introduced FM-{f}" for f in new_fms],
            created_at=datetime.now().isoformat(),
        )
        self.library.add(entry)
        return entry
