"""Skill Workflows — one per FM group (A-F)."""

from optpilot.skills.registry import get_skill, register_skill  # noqa: F401

# Import all skills to trigger @register_skill decorators.
from optpilot.skills import (  # noqa: F401
    skill_a,
    skill_b,
    skill_c,
    skill_d,
    skill_e,
    skill_f,
)
