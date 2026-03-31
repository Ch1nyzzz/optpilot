# EVOLVE-BLOCK-START
def build_dag():
    """Build the HyperAgent-style hierarchical workflow DAG.

    Hierarchical: Planner decomposes → Navigator analyzes → Editor solves → Executor verifies.
    Results flow back up to Planner for final synthesis.
    """

    planner_prompt = (
        "You are the Planner, the top-level orchestrator of a hierarchical workflow.\n\n"
        "Your responsibilities:\n"
        "1. Read the user's task and decompose it into clear sub-problems.\n"
        "2. The Navigator will analyze the problem structure.\n"
        "3. The Editor will formulate solution approaches.\n"
        "4. The Executor will verify and compute results.\n"
        "5. You receive all their outputs and synthesize the final answer.\n\n"
        "When you have enough information, produce the final answer exactly as:\n"
        "SOLUTION_FOUND \\\\boxed{answer}\n\n"
        "If the problem is unsolvable, return:\n"
        "SOLUTION_FOUND \\\\boxed{None}"
    )

    navigator_prompt = (
        "You are the Navigator, responsible for analyzing problem structure.\n\n"
        "When the Planner delegates a task to you:\n"
        "1. Identify the mathematical domain and key concepts.\n"
        "2. Break down the problem into its structural components.\n"
        "3. Identify relevant theorems, formulas, or approaches.\n"
        "4. Map out dependencies between sub-problems.\n\n"
        "Provide a clear structural analysis to guide the Editor.\n"
        "Do not produce a final boxed answer — that is the Planner's job."
    )

    editor_prompt = (
        "You are the Editor, responsible for formulating solution approaches.\n\n"
        "You receive the Navigator's structural analysis and the original problem.\n"
        "Your job:\n"
        "1. For each identified sub-problem, formulate a concrete solution approach.\n"
        "2. Work through the mathematical reasoning step by step.\n"
        "3. Produce intermediate results that the Executor can verify.\n"
        "4. Be explicit about each computation step.\n\n"
        "Provide your detailed solution work to the Executor for verification.\n"
        "Do not produce a final boxed answer — that is the Planner's job."
    )

    executor_prompt = (
        "You are the Executor, responsible for verification and final computation.\n\n"
        "You receive the Editor's solution work and the original problem.\n"
        "Your job:\n"
        "1. Verify each computation step for correctness.\n"
        "2. Use Python code to independently verify key calculations:\n"
        "   ```python\n"
        "   # your code here\n"
        "   print(result)\n"
        "   ```\n"
        "3. Check boundary conditions and edge cases.\n"
        "4. Confirm or correct the Editor's results.\n\n"
        "Report your verification results back to the Planner.\n"
        "Do not produce a final boxed answer — that is the Planner's job."
    )

    introduction_content = (
        "We solve the user's task using a hierarchical decomposition workflow:\n\n"
        "1. Planner reads the task and decomposes it into sub-problems.\n"
        "2. Navigator analyzes the problem structure and identifies key components.\n"
        "3. Editor formulates the solution approach for each sub-problem.\n"
        "4. Executor verifies and computes the final answer.\n"
        "5. Planner synthesizes all results into the final answer.\n\n"
        "The Planner orchestrates. Sub-agents work on delegated sub-tasks."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Planner", "type": "agent",
         "role": planner_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384},
                    "tools": ["list_files", "search_code", "read_file"]}},
        {"id": "Agent_Navigator", "type": "agent",
         "role": navigator_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384},
                    "tools": ["list_files", "search_code", "read_file"]}},
        {"id": "Agent_Editor", "type": "agent",
         "role": editor_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384},
                    "tools": ["read_file", "edit_file", "search_code"]}},
        {"id": "Agent_Executor", "type": "agent",
         "role": executor_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384},
                    "tools": ["run_command", "read_file"]}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # Context: USER task to all agents (non-trigger)
        {"from": "USER", "to": "Agent_Planner",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Navigator",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Editor",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Executor",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Planner
        {"from": "Introduction", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True},

        # Hierarchical chain: Planner → Navigator → Editor → Executor → Planner
        {"from": "Agent_Planner", "to": "Agent_Navigator",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Navigator", "to": "Agent_Editor",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Editor", "to": "Agent_Executor",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Executor", "to": "Agent_Planner",
         "trigger": True, "condition": "true", "carry_data": True},

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
                "HyperAgent-style hierarchical workflow. Planner decomposes, "
                "Navigator analyzes, Editor solves, Executor verifies."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
