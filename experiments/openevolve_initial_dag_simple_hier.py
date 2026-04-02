# EVOLVE-BLOCK-START
def build_dag():
    """Build a simplified 3-agent hierarchical DAG for SWE-bench Lite.

    Hierarchical: Planner (top) decomposes and navigates, Coder (middle)
    implements the fix, Verifier (bottom) runs tests. Bounded 3-iteration
    fix-test loop via explicit LoopCounter.
    """

    planner_prompt = (
        "You are the Planner, top of the hierarchy.\n\n"
        "Round 1: Read the bug report. Use search_code and list_files to locate\n"
        "the relevant source files. Use read_file to examine the code around the\n"
        "reported bug. Identify the root cause.\n"
        "Provide the Coder with: (a) file paths to edit, (b) what to change,\n"
        "(c) the root cause explanation.\n\n"
        "Later rounds: You receive test results from the Verifier. If tests\n"
        "failed, re-analyze the issue and provide updated fix instructions.\n\n"
        "When the Verifier confirms tests pass, output exactly:\n"
        "SOLUTION_FOUND \\\\boxed{fixed}"
    )

    coder_prompt = (
        "You are the Coder, middle of the hierarchy.\n\n"
        "You receive the Planner's analysis of where the bug is and what to fix.\n"
        "1. Use read_file to see the exact code that needs changing.\n"
        "2. Use edit_file to apply the fix. Be precise: match the exact text.\n"
        "3. You may need multiple edits across files.\n\n"
        "After editing, describe what you changed and why.\n"
        "Do NOT produce a final answer — that is the Planner's job."
    )

    verifier_prompt = (
        "You are the Verifier, bottom of the hierarchy.\n\n"
        "You receive the Coder's changes and the context of the bug.\n"
        "1. Use run_command to run relevant tests.\n"
        "2. Use read_file to inspect the changes if needed.\n\n"
        "Report clearly: TESTS_PASS if all tests pass, or TESTS_FAIL with\n"
        "details about which tests failed and any error messages.\n"
        "Do NOT produce a final answer — that is the Planner's job."
    )

    introduction_content = (
        "Fix the reported bug using a hierarchical workflow:\n"
        "1. Planner analyzes the bug report and locates relevant code.\n"
        "2. Coder implements the fix based on Planner's analysis.\n"
        "3. Verifier runs tests to confirm the fix works.\n"
        "If tests fail, the cycle repeats with updated context."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Planner", "type": "agent",
         "role": planner_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096},
                    "tools": ["search_code", "list_files", "read_file"]}},
        {"id": "Agent_Coder", "type": "agent",
         "role": coder_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096},
                    "tools": ["read_file", "edit_file", "search_code"]}},
        {"id": "Agent_Verifier", "type": "agent",
         "role": verifier_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096},
                    "tools": ["run_command", "read_file"]}},
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 3,
            "message": "Fix iteration limit reached.",
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context only to Planner (top of hierarchy)
        {"from": "USER", "to": "Agent_Planner",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Planner
        {"from": "Introduction", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True},

        # Hierarchical chain: Planner → Coder → Verifier
        {"from": "Agent_Planner", "to": "Agent_Coder",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Coder", "to": "Agent_Verifier",
         "trigger": True, "condition": "true", "carry_data": True},

        # Verifier → LoopCounter (backflow)
        {"from": "Agent_Verifier", "to": "LoopCounter",
         "trigger": True, "condition": "true", "carry_data": True},

        # Loop continue: re-trigger Planner with Verifier feedback
        {"from": "LoopCounter", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "continue"},

        # Loop exit: force Planner to produce final answer
        {"from": "LoopCounter", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "exit"},

        # Planner → FINAL (keyword termination)
        {"from": "Agent_Planner", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "SimpleHierarchicalSWE",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Simplified 3-agent hierarchical topology for SWE-bench. "
                "Planner decomposes, Coder implements, Verifier tests."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
