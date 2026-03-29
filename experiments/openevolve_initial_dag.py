# EVOLVE-BLOCK-START
def build_dag():
    """Build the AG2 MathChat multi-agent DAG configuration."""

    problem_solver_prompt = (
        "You are Agent Problem Solver.\n\n"
        "Solve the user's task directly from the original problem statement.\n"
        "Provide a concise step-by-step mathematical solution.\n"
        "State any assumptions explicitly instead of inventing missing facts.\n"
        "Do not ask for another speaker, and do not emit routing directives."
    )

    code_executor_prompt = (
        "You are Agent Code Executor.\n\n"
        "You receive the original task together with earlier reasoning. "
        "Use Python only when it materially helps verify or compute the answer.\n"
        "If you write code, keep it inside:\n"
        "```python\n"
        "# your code here\n"
        "print(result)\n"
        "```\n\n"
        "After the code block, summarize what the computation verified.\n"
        "Do not ask for the problem again, and do not emit speaker-routing directives."
    )

    verifier_prompt = (
        "You are Agent Verifier, the final decision maker.\n\n"
        "Use the original problem statement together with the Problem Solver and "
        "Code Executor outputs.\n"
        "Check for missing assumptions, arithmetic mistakes, and disagreement "
        "between the two analyses.\n"
        "Do not ask for another speaker.\n\n"
        "When you are confident, return the final answer exactly as:\n"
        "SOLUTION_FOUND \\\\boxed{answer}\n\n"
        "If the information is insufficient or the problem is genuinely "
        "unsolvable from the given data, return:\n"
        "SOLUTION_FOUND \\\\boxed{None}\n\n"
        "Only place the answer itself inside \\\\boxed{}."
    )

    introduction_content = (
        "We will solve the user's math task with a fixed workflow:\n\n"
        "1. Agent_Problem_Solver reasons from the original problem statement.\n"
        "2. Agent_Code_Executor checks the reasoning with Python when useful.\n"
        "3. Agent_Verifier synthesizes the evidence and returns the final boxed answer.\n\n"
        "Keep the original task in view throughout the workflow. Do not ask for another speaker."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Problem_Solver", "type": "agent",
         "role": problem_solver_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384}}},
        {"id": "Agent_Code_Executor", "type": "agent",
         "role": code_executor_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384}}},
        {"id": "Agent_Verifier", "type": "agent",
         "role": verifier_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 16384}}},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        {"from": "USER", "to": "Agent_Problem_Solver",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Code_Executor",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Verifier",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Introduction", "to": "Agent_Problem_Solver",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Problem_Solver", "to": "Agent_Code_Executor",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Problem_Solver", "to": "Agent_Verifier",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "Agent_Code_Executor", "to": "Agent_Verifier",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_Verifier", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "AG2_MathChat",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "AG2 MathChat: context-preserving 3-agent math workflow. "
                "Problem Solver reasons first, Code Executor checks with Python, "
                "and Verifier returns the final boxed answer."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
