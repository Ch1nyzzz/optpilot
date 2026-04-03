# EVOLVE-BLOCK-START
def build_dag():
    """Build the HyperAgent-style centralized dispatch (star) DAG.

    Aligned with the original HyperAgent paper:
    - Planner is the hub that sees the full bug report.
    - Planner fans out sub-tasks to Navigator, Editor, and Executor simultaneously.
    - Sub-agents only receive focused sub-tasks, not the full context.
    - Spoke outputs converge via LoopCounter back to Planner for synthesis.
    - Bounded 3-iteration loop.
    """

    planner_prompt = (
        "You are the Planner, the top-level orchestrator of a hierarchical code debugging workflow.\n\n"
        "You have access to the full bug report. Your responsibilities:\n"
        "1. Analyze the bug report and decompose it into investigation steps.\n"
        "2. Delegate sub-tasks to your interns one at a time:\n"
        "   - Navigator: to explore the codebase and locate relevant files.\n"
        "   - Editor: to implement the fix in specific files.\n"
        "   - Executor: to run tests and verify the fix.\n"
        "3. Provide each intern with a focused sub-task description containing\n"
        "   only the information they need. Do NOT forward the entire bug report.\n"
        "4. Receive their concise results and synthesize the final answer.\n\n"
        "When done, describe the fix that was applied.\n"
        "End with: SOLUTION_FOUND \\\\boxed{fixed}"
    )

    navigator_prompt = (
        "You are the Navigator, an intern reporting to the Planner.\n\n"
        "You receive a focused sub-task from the Planner — only focus on what\n"
        "the Planner asks. The Planner sees the whole picture; you do not.\n\n"
        "Your job:\n"
        "1. Use list_files and search_code to find relevant source files.\n"
        "2. Use read_file to examine the code around the area the Planner specified.\n"
        "3. Identify the root cause within the scope of your sub-task.\n\n"
        "Return a concise summary (under 500 words) of:\n"
        "- Which files and lines are relevant\n"
        "- What the root cause is\n"
        "Do not produce a final answer — that is the Planner's job."
    )

    editor_prompt = (
        "You are the Editor, an intern reporting to the Planner.\n\n"
        "You receive a focused sub-task from the Planner with specific files\n"
        "and changes to make. Only focus on the Planner's instructions.\n\n"
        "Your job:\n"
        "1. Use read_file to see the exact code that needs changing.\n"
        "2. Use edit_file to apply the fix (find old_str, replace with new_str).\n"
        "3. Be precise — match the exact text in the file.\n\n"
        "Return a concise summary (under 300 words) of what you changed and why.\n"
        "Do not produce a final answer — that is the Planner's job."
    )

    executor_prompt = (
        "You are the Executor, an intern reporting to the Planner.\n\n"
        "You receive a focused sub-task from the Planner — typically to run\n"
        "specific tests or verify changes. Only focus on the Planner's request.\n\n"
        "Your job:\n"
        "1. Use run_command to run the tests the Planner specified.\n"
        "2. Use read_file to verify changes if needed.\n\n"
        "Return a concise summary (under 300 words):\n"
        "- TESTS_PASS if all tests pass, or TESTS_FAIL with error details.\n"
        "Do not produce a final answer — that is the Planner's job."
    )

    introduction_content = (
        "Fix the reported bug using a hierarchical delegation workflow:\n"
        "1. Planner reads the bug report and decomposes it into sub-tasks.\n"
        "2. Planner delegates one sub-task at a time to Navigator, Editor, or Executor.\n"
        "3. Navigator locates relevant code. Editor implements the fix. Executor runs tests.\n"
        "4. Each sub-agent returns a concise summary of its results.\n"
        "5. Planner synthesizes all results into the final answer."
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
                    "tools": ["list_files", "search_code", "read_file"]}},
        {"id": "Agent_Navigator", "type": "agent",
         "role": navigator_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096},
                    "tools": ["list_files", "search_code", "read_file"]}},
        {"id": "Agent_Editor", "type": "agent",
         "role": editor_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096},
                    "tools": ["read_file", "edit_file", "search_code"]}},
        {"id": "Agent_Executor", "type": "agent",
         "role": executor_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096},
                    "tools": ["run_command", "read_file"]}},
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 3,
            "message": "Hierarchical delegation iteration limit reached.",
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context ONLY to Planner (the sole orchestrator)
        {"from": "USER", "to": "Agent_Planner",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Planner
        {"from": "Introduction", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True},

        # Star dispatch: Planner fans out to all sub-agents
        {"from": "Agent_Planner", "to": "Agent_Navigator",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Planner", "to": "Agent_Editor",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Planner", "to": "Agent_Executor",
         "trigger": True, "condition": "true", "carry_data": True},

        # Data flow between spokes (Navigator informs Editor/Executor)
        {"from": "Agent_Navigator", "to": "Agent_Editor",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_Navigator", "to": "Agent_Executor",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_Editor", "to": "Agent_Executor",
         "trigger": False, "condition": "true", "carry_data": True},

        # Spoke convergence → LoopCounter
        {"from": "Agent_Executor", "to": "LoopCounter",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Navigator", "to": "LoopCounter",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_Editor", "to": "LoopCounter",
         "trigger": False, "condition": "true", "carry_data": True},

        # Loop continue: re-trigger Planner with all sub-agent results
        {"from": "LoopCounter", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "continue"},

        # Loop exit: force Planner to produce final answer
        {"from": "LoopCounter", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "exit"},

        # Planner final → FINAL
        {"from": "Agent_Planner", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "HyperAgent_Hierarchical",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "HyperAgent-style centralized dispatch (star) topology. Planner is the hub "
                "that fans out sub-tasks to Navigator/Editor/Executor simultaneously."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
