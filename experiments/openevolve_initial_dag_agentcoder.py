# EVOLVE-BLOCK-START
def build_dag():
    """Build an AgentCoder-style 3-agent pipeline DAG for HumanEval.

    Pipeline topology: Programmer → TestDesigner → TestExecutor → Programmer (loop).
    Based on the AgentCoder paper's multi-agent coding workflow.
    """

    programmer_prompt = (
        "You are the Programmer agent. Your task is to write a correct Python "
        "function implementation.\n\n"
        "RULES:\n"
        "1. Read the function signature and docstring carefully.\n"
        "2. Write a complete, correct implementation.\n"
        "3. Think step by step about edge cases before coding.\n"
        "4. If you receive feedback from the TestExecutor about failing tests, "
        "analyze the failures and fix your implementation.\n\n"
        "Output ONLY the complete function (signature + docstring + body). "
        "Do NOT include test code or markdown formatting."
    )

    test_designer_prompt = (
        "You are the TestDesigner agent. Given a function specification and "
        "an implementation from the Programmer, generate test cases.\n\n"
        "RULES:\n"
        "1. Write 3-5 diverse test cases that cover:\n"
        "   - Basic/normal cases\n"
        "   - Edge cases (empty inputs, boundary values, etc.)\n"
        "   - The examples from the docstring\n"
        "2. Output test code as a Python script using assert statements.\n"
        "3. Import the function at the top of your test code.\n"
        "4. Do NOT redefine the function — only write test assertions.\n\n"
        "Output format: a Python code block with assert-based tests."
    )

    test_executor_prompt = (
        "You are the TestExecutor agent. You receive a function implementation "
        "and test cases.\n\n"
        "RULES:\n"
        "1. Combine the function implementation and test cases into a single "
        "Python script.\n"
        "2. Use the python_exec tool to run the combined script.\n"
        "3. If all tests pass, output:\n"
        "   SOLUTION_FOUND\n"
        "   Then output the complete, verified function implementation.\n"
        "4. If tests fail, analyze the error and provide specific feedback:\n"
        "   - Which test(s) failed\n"
        "   - What the expected vs actual output was\n"
        "   - A suggestion for fixing the bug\n\n"
        "IMPORTANT: When outputting the final solution after SOLUTION_FOUND, "
        "include the complete function with \\boxed{} around the function name, "
        "e.g., \\boxed{function_name}."
    )

    introduction_content = (
        "Solve the given coding problem using a multi-agent workflow:\n"
        "1. Programmer writes the initial implementation.\n"
        "2. TestDesigner generates test cases.\n"
        "3. TestExecutor runs tests and verifies correctness.\n"
        "4. If tests fail, feedback goes back to Programmer for fixes.\n"
        "The loop continues until tests pass or the iteration limit is reached."
    )

    nodes = [
        {"id": "USER", "type": "passthrough", "config": {}},
        {"id": "Introduction", "type": "literal", "config": {
            "content": introduction_content,
            "role": "user",
        }},
        {"id": "Agent_Programmer", "type": "agent",
         "role": programmer_prompt,
         "config": {"params": {"temperature": 0.2, "max_tokens": 4096}}},
        {"id": "Agent_TestDesigner", "type": "agent",
         "role": test_designer_prompt,
         "config": {"params": {"temperature": 0.3, "max_tokens": 4096}}},
        {"id": "Agent_TestExecutor", "type": "agent",
         "role": test_executor_prompt,
         "config": {"params": {"temperature": 0.1, "max_tokens": 4096},
                    "tools": ["python_exec"]}},
        {"id": "LoopCounter", "type": "loop_counter", "config": {
            "max_iterations": 3,
            "message": "AgentCoder iteration limit reached.",
        }},
        {"id": "FINAL", "type": "passthrough", "config": {}},
    ]

    edges = [
        # USER context to Programmer and TestDesigner (non-triggering)
        {"from": "USER", "to": "Agent_Programmer",
         "trigger": False, "condition": "true", "carry_data": True},
        {"from": "USER", "to": "Agent_TestDesigner",
         "trigger": False, "condition": "true", "carry_data": True},

        # Introduction triggers Programmer
        {"from": "Introduction", "to": "Agent_Programmer",
         "trigger": True, "condition": "true", "carry_data": True},

        # Pipeline: Programmer → TestDesigner → TestExecutor
        {"from": "Agent_Programmer", "to": "Agent_TestDesigner",
         "trigger": True, "condition": "true", "carry_data": True},
        {"from": "Agent_TestDesigner", "to": "Agent_TestExecutor",
         "trigger": True, "condition": "true", "carry_data": True},

        # TestExecutor also sees Programmer's code for combining
        {"from": "Agent_Programmer", "to": "Agent_TestExecutor",
         "trigger": False, "condition": "true", "carry_data": True},

        # TestExecutor → LoopCounter
        {"from": "Agent_TestExecutor", "to": "LoopCounter",
         "trigger": True, "condition": "true", "carry_data": True},

        # Loop continue: TestExecutor feedback → Programmer for fix
        {"from": "LoopCounter", "to": "Agent_Programmer",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "continue"},

        # Loop exit: force termination
        {"from": "LoopCounter", "to": "Agent_TestExecutor",
         "trigger": True, "condition": "true", "carry_data": True,
         "loop": "exit"},

        # TestExecutor → FINAL (keyword termination)
        {"from": "Agent_TestExecutor", "to": "FINAL",
         "trigger": True, "carry_data": True,
         "condition": {"type": "keyword", "config": {
             "any": ["SOLUTION_FOUND"],
             "none": [], "regex": [], "case_sensitive": True,
         }}},
    ]

    return {
        "dag_id": "AgentCoder_HumanEval",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": (
                "AgentCoder-style 3-agent pipeline for HumanEval. "
                "Programmer → TestDesigner → TestExecutor with iterative "
                "feedback loop."
            ),
            "start": ["USER", "Introduction"],
            "success_nodes": ["FINAL"],
        },
    }
# EVOLVE-BLOCK-END
