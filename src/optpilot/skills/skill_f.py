"""Skill F — Verification Failure (FM group F).

Handles: FM-3.1 (Premature Termination), FM-3.2 (Missing Verification),
         FM-3.3 (Incorrect Verification).
"""

from optpilot.skills.base import GenericSkill
from optpilot.skills.registry import register_skill


@register_skill
class SkillF(GenericSkill):
    """Repair workflow for Group F: Verification Failure."""

    FM_GROUP = "F"
    ANALYZE_HINT = (
        "Focus on where output verification is absent, premature, or incorrect. "
        "Look for agents that terminate without checking results, skip verification "
        "steps, or verify solutions but reach wrong conclusions."
    )
