"""Simplified 6-group failure taxonomy with dataset-source label mappings.

Each group maps directly to one repair intervention strategy.

Source-label mapping:
  A (指令遵循):       FM-1.1, FM-1.2
  B (执行循环/卡死):   FM-1.3, FM-1.5
  C (上下文丢失):      FM-1.4, FM-2.1
  D (通信失败):        FM-2.2, FM-2.4, FM-2.5
  E (任务漂移/推理错误): FM-2.3, FM-2.6
  F (验证失败):        FM-3.1, FM-3.2, FM-3.3
"""

GROUP_DEFINITIONS: dict[str, dict] = {
    "A": {
        "name": "Instruction Non-Compliance",
        "name_zh": "指令遵循",
        "original_fms": ["1.1", "1.2"],
        "description": (
            "Agent fails to follow task constraints or role specifications. "
            "Includes disobeying task requirements and acting outside assigned role scope."
        ),
        "repair_strategy": "Optimize prompt design, clarify role definitions and constraints.",
    },
    "B": {
        "name": "Execution Loop / Stuck",
        "name_zh": "执行循环/卡死",
        "original_fms": ["1.3", "1.5"],
        "description": (
            "System enters repetitive loops or fails to terminate. "
            "Includes unnecessary step repetition and inability to recognize stopping criteria."
        ),
        "repair_strategy": "Add loop detection, max iteration limits, explicit termination conditions.",
    },
    "C": {
        "name": "Context Loss",
        "name_zh": "上下文丢失",
        "original_fms": ["1.4", "2.1"],
        "description": (
            "Prior conversation context is lost or reset. "
            "Includes passive context truncation and active conversation reinitialization."
        ),
        "repair_strategy": "Improve state management, context window management, checkpoint mechanisms.",
    },
    "D": {
        "name": "Communication Failure",
        "name_zh": "通信失败",
        "original_fms": ["2.2", "2.4", "2.5"],
        "description": (
            "Information flow between agents breaks down. "
            "Includes failing to ask clarification, withholding critical information, "
            "and ignoring other agents' input."
        ),
        "repair_strategy": "Improve communication protocols, enforce information sharing mechanisms.",
    },
    "E": {
        "name": "Task Drift / Reasoning Error",
        "name_zh": "任务漂移/推理错误",
        "original_fms": ["2.3", "2.6"],
        "description": (
            "Agent deviates from correct execution trajectory. "
            "Includes task derailment from objectives and mismatch between reasoning and actions."
        ),
        "repair_strategy": "Add goal tracking, CoT verification, action-reasoning consistency checks.",
    },
    "F": {
        "name": "Verification Failure",
        "name_zh": "验证失败",
        "original_fms": ["3.1", "3.2", "3.3"],
        "description": (
            "Output verification is absent, premature, or incorrect. "
            "Includes premature termination, missing verification steps, "
            "and verification that reaches wrong conclusions."
        ),
        "repair_strategy": "Multi-level verification: low-level code compilation + high-level goal checking.",
    },
}

GROUP_IDS = list(GROUP_DEFINITIONS.keys())
GROUP_NAMES = {k: v["name"] for k, v in GROUP_DEFINITIONS.items()}

# Reverse mapping: original FM id -> group id
FM_TO_GROUP: dict[str, str] = {}
for gid, gdef in GROUP_DEFINITIONS.items():
    for fm_id in gdef["original_fms"]:
        FM_TO_GROUP[fm_id] = gid


def mast_annotation_to_groups(annotation: dict[str, int]) -> dict[str, int]:
    """Convert raw dataset source annotations to 6-group labels (OR logic)."""
    groups = {}
    for gid, gdef in GROUP_DEFINITIONS.items():
        groups[gid] = int(any(annotation.get(fm, 0) == 1 for fm in gdef["original_fms"]))
    return groups
