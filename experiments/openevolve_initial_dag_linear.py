# EVOLVE-BLOCK-START
def build_dag():
    """Minimal linear topology (hub=0, loop=0).

    Two agents in sequence: Solver generates an answer, Checker verifies
    and refines it. No hub, no feedback loop — single-pass pipeline.
    """

    solver_prompt = (
        "You are the Solver agent. Your task is to solve the given problem "
        "step by step.\n\n"
        "RULES:\n"
        "1. Read the problem carefully and identify what is being asked.\n"
        "2. Think through your approach before answering.\n"
        "3. Show your reasoning clearly.\n"
        "4. State your final answer explicitly.\n\n"
        "Output your solution with clear reasoning, then state the final answer."
    )

    checker_prompt = (
        "You are the Checker agent. You receive the Solver's work and must "
        "verify it.\n\n"
        "RULES:\n"
        "1. Review the Solver's reasoning for logical errors.\n"
        "2. Verify the answer is correct and complete.\n"
        "3. If you find an error, provide the corrected answer.\n"
        "4. If the answer is correct, confirm it.\n\n"
        "Output the verified final answer. Format: SOLUTION_FOUND followed by "
        "the answer."
    )

    introduction_content = (
        "Solve the given problem using a two-agent pipeline:\n"
        "1. Solver works through the problem step by step.\n"
        "2. Checker verifies the solution and produces the final answer."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Solver", "type": "agent",
         "role": solver_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096}}},
        {"id": "Agent_Checker", "type": "agent",
         "role": checker_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096}}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context to Solver
        {"from": "USER", "to": "Agent_Solver",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Solver
        {"from": "Introduction", "to": "Agent_Solver",
         "trigger": True, "condition": "true", "carry_data": True},

        # Solver → Checker
        {"from": "Agent_Solver", "to": "Agent_Checker",
         "trigger": True, "condition": "true", "carry_data": True},

        # Checker also sees original problem for verification
        {"from": "USER", "to": "Agent_Checker",
         "trigger": False, "condition": "true", "carry_data": True},

        # Checker → FINAL
        {"from": "Agent_Checker", "to": "FINAL",
         "trigger": True, "condition": "true", "carry_data": True},
    ]

    return {
        "dag_id": "Linear_Topology",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Minimal 2-agent linear topology. "
                "Solver → Checker, single pass, no loop."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
