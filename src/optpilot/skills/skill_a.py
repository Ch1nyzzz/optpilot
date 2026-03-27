"""Skill A — Instruction Non-Compliance (FM group A).

Handles: FM-1.1 (Disobey Task Specification), FM-1.2 (Agent Overstepping Scope).
"""

from optpilot.skills.base import GenericSkill
from optpilot.skills.registry import register_skill


@register_skill
class SkillA(GenericSkill):
    """Repair workflow for Group A: Instruction Non-Compliance."""

    FM_GROUP = "A"
    ANALYZE_HINT = (
        "Focus on how agents fail to follow task constraints or overstep their "
        "assigned roles. Look for agents that disobey task requirements, produce "
        "outputs outside their role scope, or ignore explicit instructions."
    )
