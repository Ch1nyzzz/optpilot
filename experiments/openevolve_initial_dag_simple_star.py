# EVOLVE-BLOCK-START
def build_dag():
    """Build a simplified 3-agent star topology DAG for GAIA.

    Star topology with Orchestrator (hub) dispatching to Researcher and Solver
    (spokes). Bounded 2-iteration feedback loop via explicit LoopCounter.
    """

    orchestrator_prompt = (
        "You are the Orchestrator, the central hub of a star topology.\n\n"
        "Round 1: Read the task carefully. Plan what information is needed and what\n"
        "computation is required. Your output is sent to the Researcher and Solver.\n\n"
        "Round 2 (if triggered): You receive both specialists' outputs. Synthesize\n"
        "them into the final answer. If they disagree, resolve the conflict.\n\n"
        "When you are confident in the answer, produce exactly:\n"
        "SOLUTION_FOUND \\\\boxed{answer}\n"
        "If the problem is unsolvable, return:\n"
        "SOLUTION_FOUND \\\\boxed{None}"
    )

    researcher_prompt = (
        "You are the Researcher spoke. Your job is to gather information.\n\n"
        "Use web_search to find relevant facts, formulas, or data.\n"
        "Use read_document to fetch and read web pages or provided documents.\n\n"
        "Be concise: report only the findings relevant to the task.\n"
        "Do NOT produce a final boxed answer — that is the Orchestrator's job."
    )

    solver_prompt = (
        "You are the Solver spoke. Your job is to compute and verify answers.\n\n"
        "Use calculator for mathematical expressions.\n"
        "Use python_exec for complex computations or multi-step calculations.\n\n"
        "Show your work step by step, then state the result clearly.\n"
        "Do NOT produce a final boxed answer — that is the Orchestrator's job."
    )

    introduction_content = (
        "Solve the user's task using a hub-spoke workflow:\n"
        "1. Orchestrator plans and delegates sub-tasks.\n"
        "2. Researcher gathers relevant information via search.\n"
        "3. Solver computes and verifies the answer.\n"
        "4. Orchestrator synthesizes specialist outputs into the final answer.\n"
        "The Orchestrator may request another round if the first attempt is insufficient."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Orchestrator", "type": "agent",
         "role": orchestrator_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096}}},
        {"id": "Agent_Researcher", "type": "agent",
         "role": researcher_prompt,
         "config": {"params": {"temperature": 0.3, "max_tokens": 4096},
                    "tools": ["web_search", "read_document"]}},
        {"id": "Agent_Solver", "type": "agent",
         "role": solver_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096},
                    "tools": ["calculator", "python_exec"]}},
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 2,
            "message": "Star dispatch iteration limit reached.",
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context only to Orchestrator
        {"from": "USER", "to": "Agent_Orchestrator",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Orchestrator
        {"from": "Introduction", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True},

        # Star dispatch: Orchestrator → both spokes
        {"from": "Agent_Orchestrator", "to": "Agent_Researcher",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Orchestrator", "to": "Agent_Solver",
         "trigger": True, "condition": "true", "carry_data": True},

        # Researcher feeds Solver (BFS ensures Researcher runs first)
        {"from": "Agent_Researcher", "to": "Agent_Solver",
         "trigger": False, "condition": "true", "carry_data": True},

        # Spoke convergence → LoopCounter
        {"from": "Agent_Solver", "to": "LoopCounter",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Researcher", "to": "LoopCounter",
         "trigger": False, "condition": "true", "carry_data": True},

        # Loop continue: re-trigger Orchestrator
        {"from": "LoopCounter", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "continue"},

        # Loop exit: force Orchestrator to produce final answer
        {"from": "LoopCounter", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "exit"},

        # Orchestrator → FINAL (keyword termination)
        {"from": "Agent_Orchestrator", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "SimpleStarGAIA",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Simplified 3-agent star topology for GAIA. Orchestrator hub "
                "dispatches to Researcher and Solver spokes."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
