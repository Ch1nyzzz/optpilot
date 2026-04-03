# === FROZEN: Answer extraction and termination logic ===
# DO NOT MODIFY anything outside the EVOLVE-BLOCK markers.
# The frozen wrapper ensures correct answer extraction regardless of
# what the evolution does to the core DAG.

ANSWER_INSTRUCTION = (
    "\n\nCRITICAL OUTPUT FORMAT — you MUST follow this exactly:\n"
    "When you have the final answer, output exactly:\n"
    "SOLUTION_FOUND \\\\boxed{your_answer}\n"
    "Do NOT deviate from this format. The system uses keyword detection\n"
    "to extract your answer."
)

TERMINATION_CONDITION = {
    "type": "keyword",
    "config": {
        "any": ["SOLUTION_FOUND"],
        "none": [],
        "regex": [],
        "case_sensitive": True,
    },
}


# EVOLVE-BLOCK-START
def _build_core():
    """Core star topology — evolution can modify this freely.

    Returns dict with keys:
      - agents: list of agent node dicts (id, role, config)
      - extra_nodes: optional list of non-agent nodes (e.g. LoopCounter)
      - edges: list of edge dicts (connections between agents)
      - introduction: str (introduction content for the literal node)
      - terminal_agent: str (id of the agent that produces the final answer)
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
        "their results."
    )

    worker_a_prompt = (
        "You are Worker_A, a specialist in analysis and reasoning.\n\n"
        "You receive instructions from the Orchestrator. Follow them precisely.\n"
        "Focus on:\n"
        "- Understanding the problem structure\n"
        "- Identifying key information and constraints\n"
        "- Providing analytical reasoning\n\n"
        "Be concise and report only your findings. Do NOT produce a final "
        "answer — that is the Synthesizer's job."
    )

    worker_b_prompt = (
        "You are Worker_B, a specialist in computation and verification.\n\n"
        "You receive instructions from the Orchestrator. Follow them precisely.\n"
        "Focus on:\n"
        "- Performing calculations and computations\n"
        "- Verifying facts and checking work\n"
        "- Providing concrete results\n\n"
        "Be concise and report only your results. Do NOT produce a final "
        "answer — that is the Synthesizer's job."
    )

    synthesizer_prompt = (
        "You are the Synthesizer. You receive outputs from Worker_A and "
        "Worker_B. Combine their findings into a single coherent answer."
    )

    introduction = (
        "Solve the given task using a hub-spoke workflow:\n"
        "1. Orchestrator plans and delegates sub-tasks.\n"
        "2. Worker_A handles analysis and reasoning.\n"
        "3. Worker_B handles computation and verification.\n"
        "4. Synthesizer combines both outputs into the final answer."
    )

    agents = [
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
         "role": synthesizer_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096}}},
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
    ]

    return {
        "agents": agents,
        "extra_nodes": [],
        "edges": edges,
        "introduction": introduction,
        "terminal_agent": "Agent_Synthesizer",
    }
# EVOLVE-BLOCK-END


# === FROZEN: DAG assembly with guaranteed termination ===

def build_dag():
    core = _build_core()

    # Frozen scaffolding nodes
    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": core["introduction"],
            "role": "user",
        }},
    ]

    # Add evolved agent nodes, injecting ANSWER_INSTRUCTION into terminal agent
    terminal_id = core["terminal_agent"]
    for agent in core["agents"]:
        if agent["id"] == terminal_id:
            agent = {**agent, "role": agent.get("role", "") + ANSWER_INSTRUCTION}
        nodes.append(agent)

    # Add extra nodes (e.g. LoopCounter)
    for node in core.get("extra_nodes", []):
        nodes.append(node)

    # Frozen FINAL node
    nodes.append({"id": "FINAL", "type": "passthrough", "config": {}})

    # Evolved edges + frozen termination edge
    edges = list(core["edges"])
    edges.append({
        "from": terminal_id, "to": "FINAL",
        "trigger": True, "carry_data": True,
        "condition": TERMINATION_CONDITION,
    })

    return {
        "dag_id": "Star_Topology",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Minimal 4-agent star topology. Orchestrator hub dispatches "
                "to Worker_A and Worker_B spokes, Synthesizer combines results."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
