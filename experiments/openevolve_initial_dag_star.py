# EVOLVE-BLOCK-START
def build_dag():
    """Minimal star topology (hub=1, loop=0).

    Three agents in a hub-spoke pattern: Orchestrator (hub) plans and
    delegates to two parallel Workers (spokes), then synthesizes their
    outputs into a final answer. No feedback loop — single dispatch.
    """

    orchestrator_prompt = (
        "You are the Orchestrator, the central hub of a star topology.\n\n"
        "Your job:\n"
        "1. Read the task carefully and break it into sub-tasks.\n"
        "2. Your output is sent to two specialist Workers simultaneously.\n"
        "   Worker_A handles analysis and reasoning.\n"
        "   Worker_B handles computation and verification.\n"
        "3. You will then receive both Workers' outputs and must synthesize "
        "them into the final answer.\n\n"
        "On your FIRST call: decompose the task and provide clear sub-task "
        "instructions for each Worker.\n"
        "On your SECOND call (after receiving Worker outputs): synthesize "
        "their results and output:\n"
        "SOLUTION_FOUND followed by the final answer."
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
        "Solve the given task using a hub-spoke workflow:\n"
        "1. Orchestrator plans and delegates sub-tasks.\n"
        "2. Worker_A handles analysis and reasoning.\n"
        "3. Worker_B handles computation and verification.\n"
        "4. Orchestrator synthesizes both outputs into the final answer."
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
        {"id": "Agent_Synthesizer", "type": "agent",
         "role": (
             "You are the Synthesizer. You receive outputs from Worker_A and "
             "Worker_B. Combine their findings into a single coherent answer.\n\n"
             "Output: SOLUTION_FOUND followed by the final answer."
         ),
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096}}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context to Orchestrator
        {"from": "USER", "to": "Agent_Orchestrator",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Orchestrator
        {"from": "Introduction", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True},

        # Star dispatch: Orchestrator → both Workers
        {"from": "Agent_Orchestrator", "to": "Agent_WorkerA",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Orchestrator", "to": "Agent_WorkerB",
         "trigger": True, "condition": "true", "carry_data": True},

        # Workers also see original problem
        {"from": "USER", "to": "Agent_WorkerA",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_WorkerB",
         "trigger": False, "condition": "true", "carry_data": True},

        # Workers → Synthesizer (WorkerB triggers, WorkerA carries data)
        {"from": "Agent_WorkerB", "to": "Agent_Synthesizer",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_WorkerA", "to": "Agent_Synthesizer",
         "trigger": False, "condition": "true", "carry_data": True},

        # Synthesizer also sees original problem
        {"from": "USER", "to": "Agent_Synthesizer",
         "trigger": False, "condition": "true", "carry_data": True},

        # Synthesizer → FINAL
        {"from": "Agent_Synthesizer", "to": "FINAL",
         "trigger": True, "condition": "true", "carry_data": True},
    ]

    return {
        "dag_id": "Star_Topology",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Minimal 3-agent star topology. Orchestrator hub dispatches "
                "to Worker_A and Worker_B spokes, Synthesizer combines results."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
