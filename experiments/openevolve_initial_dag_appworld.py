# EVOLVE-BLOCK-START
def build_dag():
    """Build the AppWorld-style star topology DAG.

    Star topology: Supervisor (hub) delegates to specialist agents (spokes).
    Specialists report back, Supervisor synthesizes final answer.
    """

    supervisor_prompt = (
        "You are the Supervisor agent, the central coordinator of this workflow.\n\n"
        "Your responsibilities:\n"
        "1. Read the user's task carefully.\n"
        "2. Delegate to the appropriate specialist by clearly stating what you need.\n"
        "3. Synthesize the specialists' responses into a coherent final answer.\n"
        "4. If specialists disagree, resolve the conflict using your judgment.\n\n"
        "After receiving all specialist inputs, produce the final answer exactly as:\n"
        "SOLUTION_FOUND \\\\boxed{answer}\n\n"
        "If the problem is unsolvable, return:\n"
        "SOLUTION_FOUND \\\\boxed{None}"
    )

    reasoning_prompt = (
        "You are Specialist Reasoning, an expert in mathematical and logical analysis.\n\n"
        "When the Supervisor delegates a task to you:\n"
        "1. Break down the problem into logical steps.\n"
        "2. Identify key mathematical concepts and theorems needed.\n"
        "3. Provide a clear step-by-step reasoning chain.\n"
        "4. State any assumptions explicitly.\n\n"
        "Report your analysis back to the Supervisor concisely.\n"
        "Do not produce a final boxed answer — that is the Supervisor's job."
    )

    computation_prompt = (
        "You are Specialist Computation, an expert in numerical calculations and code.\n\n"
        "When the Supervisor delegates a task to you:\n"
        "1. Translate the problem into concrete calculations.\n"
        "2. Use Python code when it helps verify or compute the answer:\n"
        "   ```python\n"
        "   # your code here\n"
        "   print(result)\n"
        "   ```\n"
        "3. Double-check your calculations for arithmetic errors.\n"
        "4. Summarize what your computation verified or produced.\n\n"
        "Report your results back to the Supervisor concisely.\n"
        "Do not produce a final boxed answer — that is the Supervisor's job."
    )

    verification_prompt = (
        "You are Specialist Verification, an independent cross-checker.\n\n"
        "When the Supervisor delegates a task to you:\n"
        "1. You receive the original problem AND the other specialists' work.\n"
        "2. Independently solve the problem using a DIFFERENT approach.\n"
        "3. Compare your result with the specialists' results.\n"
        "4. Report whether the solutions AGREE or DISAGREE.\n"
        "5. If they disagree, explain where the discrepancy is.\n\n"
        "Report your verification result back to the Supervisor concisely.\n"
        "Do not produce a final boxed answer — that is the Supervisor's job."
    )

    introduction_content = (
        "We solve the user's task using a Supervisor-delegate workflow:\n\n"
        "1. Supervisor reads the task and delegates to the right specialist.\n"
        "2. Specialist_Reasoning handles mathematical/logical analysis.\n"
        "3. Specialist_Computation handles numerical calculations and code.\n"
        "4. Specialist_Verification cross-checks the solution independently.\n"
        "5. Supervisor synthesizes specialist outputs and produces the final answer.\n\n"
        "Each specialist reports back to the Supervisor. The Supervisor makes the final call."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Supervisor", "type": "agent",
         "role": supervisor_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384},
                    "tools": ["execute_code", "list_apis", "submit_answer"]}},
        {"id": "Agent_Specialist_Reasoning", "type": "agent",
         "role": reasoning_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384},
                    "tools": ["execute_code", "list_apis"]}},
        {"id": "Agent_Specialist_Computation", "type": "agent",
         "role": computation_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384},
                    "tools": ["execute_code", "list_apis"]}},
        {"id": "Agent_Specialist_Verification", "type": "agent",
         "role": verification_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384},
                    "tools": ["execute_code", "list_apis", "submit_answer"]}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # Context: USER task to all agents (non-trigger)
        {"from": "USER", "to": "Agent_Supervisor",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Specialist_Reasoning",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Specialist_Computation",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Specialist_Verification",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Supervisor
        {"from": "Introduction", "to": "Agent_Supervisor",
         "trigger": True, "condition": "true", "carry_data": True},

        # Star: Supervisor → Specialists (hub-to-spoke)
        {"from": "Agent_Supervisor", "to": "Agent_Specialist_Reasoning",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Supervisor", "to": "Agent_Specialist_Computation",
         "trigger": True, "condition": "true", "carry_data": True},

        # Specialists → Verification
        {"from": "Agent_Specialist_Reasoning", "to": "Agent_Specialist_Verification",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_Specialist_Computation", "to": "Agent_Specialist_Verification",
         "trigger": True, "condition": "true", "carry_data": True},

        # Verification → Supervisor (spoke-to-hub)
        {"from": "Agent_Specialist_Verification", "to": "Agent_Supervisor",
         "trigger": True, "condition": "true", "carry_data": True},

        # Supervisor final → FINAL
        {"from": "Agent_Supervisor", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "AppWorld_Star",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "AppWorld-style star topology. Supervisor delegates to "
                "Reasoning, Computation, and Verification specialists."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
