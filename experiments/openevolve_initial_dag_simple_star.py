# === FROZEN: Answer extraction and termination logic ===
# DO NOT MODIFY anything outside the EVOLVE-BLOCK markers.
# The frozen wrapper ensures correct answer extraction regardless of
# what the evolution does to the core DAG.

ANSWER_INSTRUCTION = (
    "\n\nCRITICAL OUTPUT FORMAT — you MUST follow this exactly:\n"
    "When you are confident in the answer, produce exactly:\n"
    "SOLUTION_FOUND \\\\boxed{answer}\n"
    "If the problem is unsolvable, return:\n"
    "SOLUTION_FOUND \\\\boxed{None}\n"
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
      - extra_nodes: list of non-agent nodes to add (e.g. LoopCounter)
      - edges: list of edge dicts (connections between agents)
      - introduction: str (introduction content for the literal node)
      - terminal_agent: str (id of the agent that produces the final answer)
    """

    orchestrator_prompt = (
        "You are the Orchestrator, the central hub of a star topology.\n\n"
        "Round 1: Read the task carefully. Plan what information is needed and what\n"
        "computation is required. Your output is sent to the Researcher and Solver.\n\n"
        "Round 2 (if triggered): You receive both specialists' outputs. Synthesize\n"
        "them into the final answer. If they disagree, resolve the conflict."
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

    introduction = (
        "Solve the user's task using a hub-spoke workflow:\n"
        "1. Orchestrator plans and delegates sub-tasks.\n"
        "2. Researcher gathers relevant information via search.\n"
        "3. Solver computes and verifies the answer.\n"
        "4. Orchestrator synthesizes specialist outputs into the final answer.\n"
        "The Orchestrator may request another round if the first attempt is insufficient."
    )

    agents = [
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
    ]

    extra_nodes = [
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 2,
            "message": "Star dispatch iteration limit reached.",
        }},
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
    ]

    return {
        "agents": agents,
        "extra_nodes": extra_nodes,
        "edges": edges,
        "introduction": introduction,
        "terminal_agent": "Agent_Orchestrator",
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
