"""Skill B — Execution Loop / Stuck (FM group B).

Handles: FM-1.3 (Step Repetition), FM-1.5 (Inability to Stop).
"""

from optpilot.skills.base import GenericSkill
from optpilot.skills.registry import register_skill


@register_skill
class SkillB(GenericSkill):
    """Repair workflow for Group B: Execution Loop / Stuck."""

    FM_GROUP = "B"
    ANALYZE_HINT = (
        "Focus on where the system enters repetitive loops or fails to terminate. "
        "Look for agents that repeat the same steps unnecessarily, fail to recognize "
        "stopping criteria, or get stuck in circular conversation patterns."
    )
