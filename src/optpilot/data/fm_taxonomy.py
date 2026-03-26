"""MAST Failure Mode Taxonomy - 14 FMs in 3 categories.

Reference: MAST (Cemri et al., 2025) - Table 1
"""

from optpilot.models import FMCategory

# FM定义: (id, name, category, description)
FM_DEFINITIONS: dict[str, dict] = {
    "1.1": {
        "name": "Disobey Task Specification",
        "category": FMCategory.FC1,
        "description": "Agent fails to follow the task requirements or constraints specified in the user's request.",
    },
    "1.2": {
        "name": "Disobey Role Specification",
        "category": FMCategory.FC1,
        "description": "Agent does not adhere to its assigned role definition, performing actions outside its scope.",
    },
    "1.3": {
        "name": "Step Repetition",
        "category": FMCategory.FC1,
        "description": "Agent repeatedly executes the same step or enters a loop without making progress.",
    },
    "1.4": {
        "name": "Loss of Conversation History",
        "category": FMCategory.FC1,
        "description": "Agent loses track of prior context or conversation, leading to redundant or contradictory actions.",
    },
    "1.5": {
        "name": "Unaware Termination",
        "category": FMCategory.FC1,
        "description": "Agent does not know when to stop, continuing execution past the point of task completion.",
    },
    "2.1": {
        "name": "Conversation Reset",
        "category": FMCategory.FC2,
        "description": "Conversation between agents unexpectedly resets, losing prior progress.",
    },
    "2.2": {
        "name": "Fail Ask Clarification",
        "category": FMCategory.FC2,
        "description": "Agent fails to request clarification when facing ambiguous or incomplete information.",
    },
    "2.3": {
        "name": "Task Derailment",
        "category": FMCategory.FC2,
        "description": "Agent deviates from the original task objective during inter-agent communication.",
    },
    "2.4": {
        "name": "Information Withholding",
        "category": FMCategory.FC2,
        "description": "Agent fails to share critical information with other agents when needed.",
    },
    "2.5": {
        "name": "Ignored Other Agent's Input",
        "category": FMCategory.FC2,
        "description": "Agent disregards or fails to incorporate input from other agents.",
    },
    "2.6": {
        "name": "Reasoning-Action Mismatch",
        "category": FMCategory.FC2,
        "description": "Agent's reasoning does not align with its actual actions or outputs.",
    },
    "3.1": {
        "name": "Premature Termination",
        "category": FMCategory.FC3,
        "description": "Agent terminates execution before the task is fully completed or verified.",
    },
    "3.2": {
        "name": "No or Incomplete Verification",
        "category": FMCategory.FC3,
        "description": "Agent does not verify its output or performs only partial verification.",
    },
    "3.3": {
        "name": "Incorrect Verification",
        "category": FMCategory.FC3,
        "description": "Agent performs verification but reaches incorrect conclusions about output correctness.",
    },
}

# 便捷访问
FM_IDS = list(FM_DEFINITIONS.keys())
FM_NAMES = {k: v["name"] for k, v in FM_DEFINITIONS.items()}
FM_CATEGORIES = {k: v["category"] for k, v in FM_DEFINITIONS.items()}


def get_fm_info(fm_id: str) -> dict:
    """Get full FM info by id."""
    if fm_id not in FM_DEFINITIONS:
        raise ValueError(f"Unknown FM id: {fm_id}")
    return {"fm_id": fm_id, **FM_DEFINITIONS[fm_id]}


def get_category_fms(category: FMCategory) -> list[str]:
    """Get all FM ids in a category."""
    return [k for k, v in FM_DEFINITIONS.items() if v["category"] == category]
