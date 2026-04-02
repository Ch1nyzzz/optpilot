# EVOLVE-BLOCK-START
def build_dag():
    """Minimal linear topology with feedback loop (hub=0, loop=1).

    Two agents in sequence with a verify-and-fix cycle: Solver generates
    an answer, Checker verifies it. If incorrect, Checker's feedback
    loops back to Solver for refinement. No hub — pure pipeline with loop.
    """

    solver_prompt = (
        "You are the Solver agent. Your task is to solve the given problem "
        "step by step.\n\n"
        "RULES:\n"
        "1. Read the problem carefully and identify what is being asked.\n"
        "2. Think through your approach before answering.\n"
        "3. Show your reasoning clearly.\n"
        "4. If you receive feedback from the Checker about errors in your "
        "previous attempt, carefully address each issue.\n\n"
        "Output your solution with clear reasoning, then state the final answer."
    )

    checker_prompt = (
        "You are the Checker agent. You receive the Solver's work and must "
        "verify it.\n\n"
        "RULES:\n"
        "1. Review the Solver's reasoning for logical errors.\n"
        "2. Check the answer against the problem requirements.\n"
        "3. If the answer is correct and complete, output:\n"
        "   SOLUTION_FOUND\n"
        "   followed by the verified answer.\n"
        "4. If you find errors, explain specifically what is wrong and what "
        "needs to be fixed. Do NOT include SOLUTION_FOUND in this case."
    )

    introduction_content = (
        "Solve the given problem using an iterative two-agent pipeline:\n"
        "1. Solver works through the problem step by step.\n"
        "2. Checker verifies the solution.\n"
        "3. If the Checker finds errors, feedback goes back to the Solver.\n"
        "The loop continues until the solution is verified or the limit is reached."
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
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 3,
            "message": "Verify-and-fix iteration limit reached.",
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context to Solver and Checker
        {"from": "USER", "to": "Agent_Solver",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Checker",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Solver
        {"from": "Introduction", "to": "Agent_Solver",
         "trigger": True, "condition": "true", "carry_data": True},

        # Solver → Checker
        {"from": "Agent_Solver", "to": "Agent_Checker",
         "trigger": True, "condition": "true", "carry_data": True},

        # Checker → FINAL (keyword termination on success)
        {"from": "Agent_Checker", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},

        # Checker → LoopCounter (when verification fails)
        {"from": "Agent_Checker", "to": "LoopCounter",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": [], "none": ["SOLUTION_FOUND"],
             "regex": [], "case_sensitive": True,
         }}},

        # Loop continue: Checker feedback → Solver for another attempt
        {"from": "LoopCounter", "to": "Agent_Solver",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "continue"},

        # Loop exit: force termination → FINAL
        {"from": "LoopCounter", "to": "FINAL",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "exit"},
    ]

    return {
        "dag_id": "LinearLoop_Topology",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Minimal 2-agent linear topology with verify-and-fix loop. "
                "Solver → Checker → feedback → Solver."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
