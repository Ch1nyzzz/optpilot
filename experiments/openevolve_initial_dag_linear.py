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
    """Core linear topology — evolution can modify this freely.

    Returns dict with keys:
      - agents: list of agent node dicts (id, role, config)
      - edges: list of edge dicts (connections between agents)
      - introduction: str (introduction content for the literal node)
      - terminal_agent: str (id of the last agent that produces the answer)
    """

    solver_prompt = (
        "You are the Solver agent. Your task is to solve the given problem "
        "step by step.\n\n"
        "RULES:\n"
        "1. Read the problem carefully and identify what is being asked.\n"
        "2. Think through your approach before answering.\n"
        "3. Show your reasoning clearly.\n"
        "4. State your final answer explicitly."
    )

    checker_prompt = (
        "You are the Checker agent. You receive the Solver's work and must "
        "verify it.\n\n"
        "RULES:\n"
        "1. Review the Solver's reasoning for logical errors.\n"
        "2. Verify the answer is correct and complete.\n"
        "3. If you find an error, provide the corrected answer.\n"
        "4. If the answer is correct, confirm it."
    )

    introduction = (
        "Solve the given problem using a two-agent pipeline:\n"
        "1. Solver works through the problem step by step.\n"
        "2. Checker verifies the solution and produces the final answer."
    )

    agents = [
        {"id": "Agent_Solver", "type": "agent",
         "role": solver_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096}}},
        {"id": "Agent_Checker", "type": "agent",
         "role": checker_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096}}},
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
    ]

    return {
        "agents": agents,
        "edges": edges,
        "introduction": introduction,
        "terminal_agent": "Agent_Checker",
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
