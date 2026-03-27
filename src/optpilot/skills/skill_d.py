"""Skill D — Communication Failure (FM group D).

Handles: FM-2.2 (Lack of Clarification), FM-2.4 (Information Withholding),
         FM-2.5 (Ignoring Other Agents).
"""

from optpilot.skills.base import GenericSkill
from optpilot.skills.registry import register_skill


@register_skill
class SkillD(GenericSkill):
    """Repair workflow for Group D: Communication Failure."""

    FM_GROUP = "D"
    ANALYZE_HINT = (
        "Focus on breakdowns in information flow between agents. Look for agents "
        "that fail to ask for clarification, withhold critical information, or "
        "ignore input from other agents in the group chat."
    )
