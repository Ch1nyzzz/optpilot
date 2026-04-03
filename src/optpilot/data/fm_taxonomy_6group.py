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
        "analyze_hint": (
            "Focus on how agents fail to follow task constraints or overstep their "
            "assigned roles. Look for agents that disobey task requirements, produce "
            "outputs outside their role scope, or ignore explicit instructions."
        ),
        "failure_examples": (
            "Example 1: The task asks 'find all integers n satisfying ...', but "
            "Agent_Problem_Solver assumes n must be positive without justification, "
            "violating the original constraint.\n"
            "Example 2: Agent_Code_Executor is told to write Python code, but outputs "
            "a purely verbal explanation instead, ignoring its role specification.\n"
            "Example 3: Agent_Verifier is instructed to wait for both solutions before "
            "judging, but declares SOLUTION_FOUND after seeing only one agent's answer."
        ),
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
        "analyze_hint": (
            "Focus on where the system enters repetitive loops or fails to terminate. "
            "Look for agents that repeat the same steps unnecessarily, fail to recognize "
            "stopping criteria, or get stuck in circular conversation patterns."
        ),
        "failure_examples": (
            "Example 1: Agent_Verifier keeps saying 'SUGGESTED NEXT SPEAKER: "
            "Agent_Code_Executor' indefinitely, never outputting SOLUTION_FOUND, "
            "causing the loop counter to exhaust.\n"
            "Example 2: Agent_Problem_Solver and Agent_Code_Executor take turns "
            "restating the same approach with minor wording changes, making no progress.\n"
            "Example 3: The system reaches max_iterations=5 on every problem because "
            "no agent produces the SOLUTION_FOUND keyword to trigger the exit edge."
        ),
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
        "analyze_hint": (
            "Focus on how context is lost between agents or across conversation turns. "
            "Look for truncated information, missing carry_data edges, and state that "
            "fails to propagate through the workflow."
        ),
        "failure_examples": (
            "Example 1: Agent_Verifier receives output from Agent_Code_Executor but "
            "has lost the original problem statement, so it verifies against a wrong "
            "interpretation of the task.\n"
            "Example 2: In a later loop iteration, Agent_Problem_Solver restarts from "
            "scratch, ignoring the partial solution established in earlier rounds.\n"
            "Example 3: The carry_data=false edge causes Agent_Verifier to receive an "
            "empty input, losing all prior context from the conversation."
        ),
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
        "analyze_hint": (
            "Focus on breakdowns in information flow between agents. Look for agents "
            "that fail to ask for clarification, withhold critical information, or "
            "ignore input from other agents in the group chat."
        ),
        "failure_examples": (
            "Example 1: Agent_Problem_Solver finds the answer is 42 but buries it in "
            "a long explanation. Agent_Verifier cannot locate the answer and asks "
            "Agent_Code_Executor instead, ignoring Agent_Problem_Solver's work.\n"
            "Example 2: Agent_Code_Executor's code produces a numerical result, but "
            "the agent does not state the result explicitly in text — Agent_Verifier "
            "cannot read code output and misses the answer.\n"
            "Example 3: Agent_Verifier spots a discrepancy between two solutions but "
            "does not ask either agent for clarification, instead guessing arbitrarily."
        ),
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
        "analyze_hint": (
            "Focus on where agents deviate from the correct execution trajectory. "
            "Look for reasoning chains that diverge from the task objective, "
            "and mismatches between an agent's stated reasoning and its actual actions."
        ),
        "failure_examples": (
            "Example 1: The problem asks for 'the number of solutions', but "
            "Agent_Problem_Solver solves for 'the value of x' and outputs a single "
            "number instead of a count.\n"
            "Example 2: Agent_Code_Executor correctly computes the answer as 7 in code, "
            "but then writes 'Therefore the answer is 8' in its explanation, "
            "contradicting its own computation.\n"
            "Example 3: Agent_Verifier's reasoning correctly identifies answer A as "
            "flawed, but then concludes 'SOLUTION_FOUND \\boxed{A}' anyway — a "
            "reasoning-action mismatch."
        ),
    },
    "F": {
        "name": "Verification Failure",
        "name_zh": "验证失败",
        "original_fms": ["3.1", "3.2", "3.3"],
        "description": (
            "A verification/checking role or step exists in the system but "
            "fails to work properly. Includes premature acceptance of wrong "
            "answers, weak or superficial checks, and verification that "
            "reaches wrong conclusions. Does NOT apply when the system has "
            "no verifier/checker role — absence of verification by design "
            "is not a verification failure."
        ),
        "repair_strategy": "Multi-level verification: low-level code compilation + high-level goal checking.",
        "analyze_hint": (
            "Focus on where a verifier/checker role EXISTS but fails to do its job. "
            "Look for verifiers that accept wrong answers, check superficially, or "
            "reach wrong conclusions. If the system has no verifier role at all, "
            "do not flag this group."
        ),
        "failure_examples": (
            "Example 1: Agent_Verifier immediately accepts Agent_Problem_Solver's "
            "answer without checking Agent_Code_Executor's result, even though the "
            "two answers differ.\n"
            "Example 2: Agent_Verifier declares SOLUTION_FOUND after seeing only one "
            "agent's response, skipping verification entirely.\n"
            "Example 3: Agent_Verifier compares both solutions, identifies they disagree, "
            "but picks the wrong one because it fails to verify which computation "
            "is actually correct."
        ),
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
