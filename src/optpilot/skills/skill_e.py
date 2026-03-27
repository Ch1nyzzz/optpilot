"""Skill E — Task Drift / Reasoning Error (FM group E).

Handles: FM-2.3 (Task Derailment), FM-2.6 (Reasoning-Action Mismatch).
"""

from optpilot.skills.base import GenericSkill
from optpilot.skills.registry import register_skill


@register_skill
class SkillE(GenericSkill):
    """Repair workflow for Group E: Task Drift / Reasoning Error."""

    FM_GROUP = "E"
    ANALYZE_HINT = (
        "Focus on where agents deviate from the correct execution trajectory. "
        "Look for reasoning chains that diverge from the task objective, "
        "and mismatches between an agent's stated reasoning and its actual actions."
    )
