"""Skill C — Context Loss (FM group C).

Handles: FM-1.4 (Conversation Reset), FM-2.1 (Context Truncation).
"""

from optpilot.skills.base import GenericSkill
from optpilot.skills.registry import register_skill


@register_skill
class SkillC(GenericSkill):
    """Repair workflow for Group C: Context Loss."""

    FM_GROUP = "C"
    ANALYZE_HINT = (
        "Focus on how context is lost between agents or across conversation turns. "
        "Look for truncated information, missing carry_data edges, and state that "
        "fails to propagate through the workflow."
    )
