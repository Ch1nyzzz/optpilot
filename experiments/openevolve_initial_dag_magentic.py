# EVOLVE-BLOCK-START
def build_dag():
    """Build the Magentic-One-style general-purpose star topology DAG.

    Star topology with Orchestrator coordinating 4 specialists:
    WebSurfer (info retrieval), Coder (code/derivation), Analyst (deep analysis),
    Executor (verification). More complex than AppWorld star.
    """

    orchestrator_prompt = (
        "You are the Orchestrator, the central controller of a general-purpose "
        "multi-agent system.\n\n"
        "Your responsibilities:\n"
        "1. Read the task and create a high-level execution plan.\n"
        "2. Delegate sub-tasks to the right specialist:\n"
        "   - WebSurfer: for retrieving information, formulas, theorems\n"
        "   - Coder: for writing solution code and formal derivations\n"
        "   - Analyst: for deep analysis, edge cases, alternative approaches\n"
        "   - Executor: for verification, testing, final computation\n"
        "3. Coordinate multiple rounds if needed.\n"
        "4. Synthesize all specialist outputs into the final answer.\n\n"
        "When confident, produce the final answer exactly as:\n"
        "SOLUTION_FOUND \\\\boxed{answer}\n\n"
        "If the problem is unsolvable, return:\n"
        "SOLUTION_FOUND \\\\boxed{None}"
    )

    websurfer_prompt = (
        "You are the WebSurfer, responsible for information retrieval and context.\n\n"
        "When the Orchestrator delegates to you:\n"
        "1. Identify what information, formulas, or theorems are needed.\n"
        "2. Recall relevant mathematical concepts from your training.\n"
        "3. Provide precise definitions, formulas, and known results.\n"
        "4. Cite specific theorems or identities that apply.\n\n"
        "Be thorough but concise. Report back to the Orchestrator.\n"
        "Do not produce a final boxed answer — that is the Orchestrator's job."
    )

    coder_prompt = (
        "You are the Coder, responsible for writing solution code and derivations.\n\n"
        "When the Orchestrator delegates to you:\n"
        "1. Translate the mathematical problem into formal steps.\n"
        "2. Write Python code to compute or verify results:\n"
        "   ```python\n"
        "   # your code here\n"
        "   print(result)\n"
        "   ```\n"
        "3. Show your work explicitly — each step of the derivation.\n"
        "4. Handle numerical precision carefully.\n\n"
        "Report your code and results back to the Orchestrator.\n"
        "Do not produce a final boxed answer — that is the Orchestrator's job."
    )

    analyst_prompt = (
        "You are the Analyst, responsible for deep analysis and edge cases.\n\n"
        "When the Orchestrator delegates to you:\n"
        "1. Examine the problem from multiple angles.\n"
        "2. Identify edge cases, boundary conditions, and special cases.\n"
        "3. Look for potential errors in other agents' work.\n"
        "4. Suggest alternative solution approaches if the primary one seems fragile.\n\n"
        "Provide your analysis back to the Orchestrator.\n"
        "Do not produce a final boxed answer — that is the Orchestrator's job."
    )

    executor_prompt = (
        "You are the Executor, the final verification agent.\n\n"
        "You receive the combined work of all other agents and the original problem.\n"
        "Your job:\n"
        "1. Independently verify the proposed answer.\n"
        "2. Re-derive key steps from scratch.\n"
        "3. Check for arithmetic errors, sign errors, off-by-one mistakes.\n"
        "4. Use Python to double-check:\n"
        "   ```python\n"
        "   # verification code\n"
        "   print(result)\n"
        "   ```\n"
        "5. State clearly whether the answer is CORRECT or NEEDS REVISION.\n\n"
        "Report your verification back to the Orchestrator.\n"
        "Do not produce a final boxed answer — that is the Orchestrator's job."
    )

    introduction_content = (
        "We solve the user's task using a general-purpose multi-agent workflow:\n\n"
        "1. Orchestrator reads the task and creates an execution plan.\n"
        "2. WebSurfer retrieves relevant information, formulas, and context.\n"
        "3. Coder writes solution code and mathematical derivations.\n"
        "4. Analyst performs deep analysis and identifies edge cases.\n"
        "5. Executor verifies the solution and runs final checks.\n"
        "6. Orchestrator synthesizes everything into the final answer.\n\n"
        "The Orchestrator coordinates iteratively until confident in the answer."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Orchestrator", "type": "agent",
         "role": orchestrator_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384},
                    "tools": ["web_search", "calculator"]}},
        {"id": "Agent_WebSurfer", "type": "agent",
         "role": websurfer_prompt,
         "config": {"params": {"temperature": 0.3, "max_tokens": 16384},
                    "tools": ["web_search", "read_document"]}},
        {"id": "Agent_Coder", "type": "agent",
         "role": coder_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384},
                    "tools": ["python_exec", "calculator"]}},
        {"id": "Agent_Analyst", "type": "agent",
         "role": analyst_prompt,
         "config": {"params": {"temperature": 0.3, "max_tokens": 16384},
                    "tools": ["web_search", "calculator", "read_document"]}},
        {"id": "Agent_Executor", "type": "agent",
         "role": executor_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384},
                    "tools": ["python_exec", "calculator"]}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # Context: USER task to all agents (non-trigger)
        {"from": "USER", "to": "Agent_Orchestrator",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_WebSurfer",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Coder",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Analyst",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Executor",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Orchestrator
        {"from": "Introduction", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True},

        # Star: Orchestrator → Specialists
        {"from": "Agent_Orchestrator", "to": "Agent_WebSurfer",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Orchestrator", "to": "Agent_Coder",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Orchestrator", "to": "Agent_Analyst",
         "trigger": False, "condition": "true", "carry_data": True},

        # Cross-specialist data flow
        {"from": "Agent_WebSurfer", "to": "Agent_Coder",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_WebSurfer", "to": "Agent_Analyst",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_Coder", "to": "Agent_Analyst",
         "trigger": True, "condition": "true", "carry_data": True},

        # Convergence → Executor
        {"from": "Agent_Analyst", "to": "Agent_Executor",
         "trigger": True, "condition": "true", "carry_data": True},

        # Executor → Orchestrator (results back)
        {"from": "Agent_Executor", "to": "Agent_Orchestrator",
         "trigger": True, "condition": "true", "carry_data": True},

        # Orchestrator final → FINAL
        {"from": "Agent_Orchestrator", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "MagenticOne_Star",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "Magentic-One-style general-purpose star topology. Orchestrator "
                "coordinates WebSurfer, Coder, Analyst, and Executor."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
