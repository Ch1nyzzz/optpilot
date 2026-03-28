# EVOLVE-BLOCK-START
def build_dag():
    """Build the AG2 MathChat multi-agent DAG configuration.

    Returns a dictionary that is directly parseable by MASDAG.from_dict().

    Architecture overview:
      USER ──context──► Problem_Solver ──trigger──► Verifier
      USER ──context──► Code_Executor  ──trigger──► Verifier
      USER ──context──► Verifier
      Introduction ──trigger──► Problem_Solver (starts the flow)
      Verifier ──(SOLUTION_FOUND)──► FINAL
      Verifier ──(no solution)──► Code_Executor (retry loop)
      Verifier ──► Turn Counter ──(max reached)──► FINAL

    To add a new agent:
      1. Add a dict to the `nodes` list with id, type="agent", role, config
      2. Add edges connecting it to other nodes (from/to/trigger/condition/carry_data)
      3. Update the Introduction content to mention the new agent

    To modify routing:
      - Change edge trigger/condition/carry_data fields
      - Add or remove edges between agents

    To adjust behavior:
      - Modify agent role (system prompt)
      - Adjust config.params (temperature, max_tokens)
      - Change loop counter max_iterations
    """

    # ── Agent prompts ──────────────────────────────────────────────

    problem_solver_prompt = (
        "You are Agent Problem Solver, and your role is to collaborate "
        "with other agents to address various challenges.\n\n"
        "For each problem, please follow these steps:\n"
        "1. **Document Your Solution**: Write your solution step by step, "
        "ensuring it is independent of the solutions provided by other agents.\n"
        "2. **Engage in Discussion**: Once you have outlined your approach "
        "and findings, discuss your approach and findings with the other agents."
    )

    code_executor_prompt = (
        "You are Agent Code Executor. You can solve problems only writing "
        "commented Python code.\n\n"
        "For each problem, please follow these steps:\n"
        "1. **Develop Your Solution**: Write your solution in Python code, "
        "detailing each step independently from the solutions provided by other agents.\n"
        "2. **Utilize SymPy**: Feel free to use the SymPy package to facilitate "
        "calculations and enhance your code's efficiency.\n"
        "3. **Display Results**: Ensure that you **print the final result at the "
        "end of your Python code** (e.g., 'print(_result_)').\n"
        "4. **Engage in Discussion**: After obtaining the result from your Python "
        "code, discuss your findings with the other agents.\n\n"
        "Always format your Python code within:\n"
        "```python\n"
        "# your code here\n"
        "print(_result_)\n"
        "```\n\n"
        'If you wish to execute your code, please indicate this by stating '
        '"SUGGESTED NEXT SPEAKER: Agent Code Executor" at the end of your message.'
    )

    verifier_prompt = (
        "You are Agent Verifier.\n\n"
        "Your role is to critically evaluate the solutions proposed by other "
        "agents step by step and provide a final solution.\n\n"
        "1. **Solution Requirement**: Before making any decisions, ensure you "
        "have received solutions from both Agent Code Executor and Agent Problem "
        "Solver. If either proposed solution is missing, do not draw any "
        "conclusions; instead, suggest the next speaker by stating: "
        "SUGGESTED NEXT SPEAKER: _suggested_agent_name_.\n"
        "2. **Avoid Assumptions**: Pay attention to the variables provided in "
        "the original problem statement versus those assumed by the agents. "
        "**Assumed values are not valid for the solution** and can lead to "
        "inaccuracies. Never base your solution on assumed values. Always base "
        "your solution on the explicitly given variables to ensure correctness. "
        "If a problem is deemed unsolvable due to missing information, return: "
        "**SOLUTION_FOUND \\\\boxed{None}**.\n"
        "3. **Evaluating Conflicting Solutions**: If different answers are "
        "presented during the discussion, choose the most appropriate solution "
        "based on your evidence or initiate further discussion to clarify.\n"
        "4. **Final Solution Declaration**: When you are confident about the "
        "final solution, return it as follows: "
        "**SOLUTION_FOUND \\\\boxed{_solution_value_here_}**. Ensure that only "
        "numerical values are placed inside the \\\\boxed{}; any accompanying "
        "text should be outside."
    )

    introduction_content = (
        "Hello everyone. We have assembled a great team today to answer "
        "questions and solve tasks. In attendance are:\n\n"
        "Agent_Code_Executor:\n"
        "                I am Agent Code Executor, specializing in solving "
        "problems by writing Python code.\n"
        "                I have the ability to execute Python code, so feel "
        "free to reach out whenever you need assistance with Python programming.\n"
        "Agent_Problem_Solver:\n"
        "                I am Agent Problem Solver, and I work collaboratively "
        "with other agents to tackle various challenges.\n"
        "Agent_Verifier:\n"
        "                I am Agent Verifier. Please call on me when both "
        "Agent Code Executor and Agent Problem Solver have submitted their "
        "solutions, so I can verify their proposals and provide a final synthesis."
    )

    # ── Nodes ──────────────────────────────────────────────────────

    nodes = [
        # Entry points
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},

        # Agents
        {"id": "Agent_Problem_Solver", "type": "agent",
         "role": problem_solver_prompt,
         "config": {"params": {"temperature": 0.3, "max_tokens": 16384}}},

        {"id": "Agent_Code_Executor", "type": "agent",
         "role": code_executor_prompt,
         "config": {"params": {"temperature": 0.3, "max_tokens": 16384}}},

        {"id": "Agent_Verifier", "type": "agent",
         "role": verifier_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 16384}}},

        # Control
        {"id": "Turn Counter", "type": "loop_counter", "config": {
            "max_iterations": 5,
            "reset_on_emit": True,
            "message": ("Maximum discussion iterations reached. "
                        "Agent_Verifier, please provide your best answer "
                        "now with SOLUTION_FOUND."),
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    # ── Edges ──────────────────────────────────────────────────────

    edges = [
        # Context: USER task goes to all agents
        {"from": "USER", "to": "Agent_Problem_Solver",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Code_Executor",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_Verifier",
         "trigger": False, "condition": "true", "carry_data": True},

        # Start: Introduction triggers Problem_Solver first
        {"from": "Introduction", "to": "Agent_Problem_Solver",
         "trigger": True, "condition": "true", "carry_data": True},

        # Problem_Solver → Verifier
        {"from": "Agent_Problem_Solver", "to": "Agent_Verifier",
         "trigger": True, "condition": "true", "carry_data": True},

        # Code_Executor → Verifier
        {"from": "Agent_Code_Executor", "to": "Agent_Verifier",
         "trigger": True, "condition": "true", "carry_data": True},

        # Verifier found solution → FINAL
        {"from": "Agent_Verifier", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},

        # Verifier delegates to Code_Executor (no solution yet)
        {"from": "Agent_Verifier", "to": "Agent_Code_Executor",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": [],
             "none": ["SOLUTION_FOUND"],
             "regex": [], "case_sensitive": True,
         }}},

        # Loop counting
        {"from": "Agent_Verifier", "to": "Turn Counter",
         "trigger": True, "condition": "true", "carry_data": False},

        # Turn Counter max → FINAL (exit only)
        {"from": "Turn Counter", "to": "FINAL",
         "trigger": True, "condition": "true",
         "carry_data": True, "loop": "exit"},
    ]

    return {
        "dag_id": "AG2_MathChat",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "AG2 MathChat: 3-agent GroupChat for math problem solving. "
                "Based on MAST paper (Cemri et al., 2025) Appendix L."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
