# EVOLVE-BLOCK-START
def build_dag():
    """Minimal star topology with feedback loop (hub=1, loop=1).

    Three agents in a hub-spoke pattern with iterative refinement:
    Orchestrator (hub) plans and delegates to two parallel Workers (spokes),
    synthesizes their outputs, and may request another round if the result
    is insufficient.
    """

    orchestrator_prompt = (
        "You are the Orchestrator, the central hub of a star topology.\n\n"
        "Your job:\n"
        "1. Read the task and break it into sub-tasks for two Workers.\n"
        "   Worker_A handles analysis and reasoning.\n"
        "   Worker_B handles computation and verification.\n"
        "2. After receiving their outputs, synthesize them.\n"
        "3. If the combined result is sufficient, output:\n"
        "   SOLUTION_FOUND followed by the final answer.\n"
        "4. If more work is needed, provide refined instructions for the "
        "next round. Do NOT include SOLUTION_FOUND in this case.\n\n"
        "Each round you receive all previous context, so build on prior work."
    )

    worker_a_prompt = (
        "You are Worker_A, a specialist in analysis and reasoning.\n\n"
        "You receive instructions from the Orchestrator. Follow them precisely.\n"
        "Focus on:\n"
        "- Understanding the problem structure\n"
        "- Identifying key information and constraints\n"
        "- Providing analytical reasoning\n\n"
        "Be concise and report only your findings. Do NOT produce a final "
        "answer — that is the Orchestrator's job."
    )

    worker_b_prompt = (
        "You are Worker_B, a specialist in computation and verification.\n\n"
        "You receive instructions from the Orchestrator. Follow them precisely.\n"
        "Focus on:\n"
        "- Performing calculations and computations\n"
        "- Verifying facts and checking work\n"
        "- Providing concrete results\n\n"
        "Be concise and report only your results. Do NOT produce a final "
        "answer — that is the Orchestrator's job."
    )

    introduction_content = (
        "Solve the given task using an iterative hub-spoke workflow:\n"
        "1. Orchestrator plans and delegates sub-tasks.\n"
        "2. Worker_A handles analysis, Worker_B handles computation.\n"
        "3. Orchestrator synthesizes results.\n"
        "4. If needed, Orchestrator requests another round with refined tasks.\n"
        "The loop continues until a solution is found or the limit is reached."
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
        {"id": "Agent_WorkerA", "type": "agent",
         "role": worker_a_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096}}},
        {"id": "Agent_WorkerB", "type": "agent",
         "role": worker_b_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096}}},
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 2,
            "message": "Star dispatch iteration limit reached.",
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context to Orchestrator and Workers
        {"from": "USER", "to": "Agent_Orchestrator",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_WorkerA",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_WorkerB",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Orchestrator
        {"from": "Introduction", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True},

        # Star dispatch: Orchestrator → both Workers
        {"from": "Agent_Orchestrator", "to": "Agent_WorkerA",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Orchestrator", "to": "Agent_WorkerB",
         "trigger": True, "condition": "true", "carry_data": True},

        # Spoke convergence → LoopCounter
        {"from": "Agent_WorkerB", "to": "LoopCounter",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_WorkerA", "to": "LoopCounter",
         "trigger": False, "condition": "true", "carry_data": True},

        # Loop continue: re-trigger Orchestrator with Worker outputs
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
        "dag_id": "StarLoop_Topology",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Minimal 3-agent star topology with iterative refinement. "
                "Orchestrator hub dispatches to Worker spokes with feedback loop."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
