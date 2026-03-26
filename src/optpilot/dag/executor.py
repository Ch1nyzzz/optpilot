"""Lightweight DAG executor for MASDAG.

Executes a multi-agent workflow defined as a MASDAG:
- agent nodes: LLM calls
- literal nodes: inject fixed text
- loop_counter nodes: control iteration loops
- passthrough nodes: forward input unchanged

Produces a structured execution trace for diagnosis.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from optpilot.dag.core import MASDAG, DAGNode, DAGEdge


# ---- Execution trace ----

@dataclass
class NodeExecution:
    """Record of a single node execution."""
    node_id: str
    node_type: str
    iteration: int
    input_text: str
    output_text: str
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Full execution trace of a DAG run."""
    dag_id: str
    task_prompt: str
    steps: list[NodeExecution] = field(default_factory=list)
    total_duration_s: float = 0.0
    finished: bool = False
    error: str = ""

    def to_trajectory(self) -> str:
        """Convert to text trajectory for diagnosis."""
        lines = [f"=== DAG Execution: {self.dag_id} ===", f"Task: {self.task_prompt}", ""]
        for step in self.steps:
            lines.append(f"--- [{step.node_type}] {step.node_id} (iter {step.iteration}) ---")
            if step.input_text:
                lines.append(f"Input: {step.input_text[:500]}")
            lines.append(f"Output: {step.output_text[:2000]}")
            lines.append("")
        if self.error:
            lines.append(f"ERROR: {self.error}")
        lines.append(f"Total duration: {self.total_duration_s:.1f}s, Finished: {self.finished}")
        return "\n".join(lines)


# ---- Condition evaluation ----

def evaluate_condition(condition: str | dict, text: str) -> bool:
    """Evaluate an edge condition against output text.

    Supports:
      - "true": always passes
      - {"type": "keyword", "config": {"any": [...], "none": [...]}}: keyword matching
    """
    if isinstance(condition, str):
        return condition.strip().lower() == "true"

    if isinstance(condition, dict):
        ctype = condition.get("type", "")
        cfg = condition.get("config", {})

        if ctype == "keyword":
            any_kw = cfg.get("any", [])
            none_kw = cfg.get("none", [])
            case_sensitive = cfg.get("case_sensitive", True)

            check_text = text if case_sensitive else text.lower()

            if any_kw:
                matched = any(
                    (kw if case_sensitive else kw.lower()) in check_text
                    for kw in any_kw
                )
                if not matched:
                    return False

            if none_kw:
                blocked = any(
                    (kw if case_sensitive else kw.lower()) in check_text
                    for kw in none_kw
                )
                if blocked:
                    return False

            return True

    return False


# ---- DAG Executor ----

# Type alias for the LLM call function that agent nodes use.
# Signature: (system_prompt, user_message, model, **config) -> response_text
LLMCallFn = Callable[..., str]


class DAGExecutor:
    """Execute a MASDAG workflow.

    Args:
        dag: The workflow DAG to execute.
        llm_fn: Function to call LLM. Signature: (messages, model, **kwargs) -> str.
            Compatible with ``optpilot.llm.call_llm``.
        model: Default model name for agent nodes.
        max_global_steps: Safety limit on total node executions.
        timeout: Overall execution timeout in seconds.
    """

    def __init__(
        self,
        dag: MASDAG,
        llm_fn: LLMCallFn,
        model: str = "",
        max_global_steps: int = 200,
        timeout: int = 600,
    ):
        self.dag = dag
        self.llm_fn = llm_fn
        self.model = model
        self.max_global_steps = max_global_steps
        self.timeout = timeout

        # Build adjacency: source -> list of edges
        self._outgoing: dict[str, list[DAGEdge]] = defaultdict(list)
        for edge in dag.edges:
            self._outgoing[edge.source].append(edge)

        # Build incoming trigger edges: target -> list of edges with trigger=True
        self._incoming_triggers: dict[str, list[DAGEdge]] = defaultdict(list)
        for edge in dag.edges:
            if edge.trigger:
                self._incoming_triggers[edge.target].append(edge)

    def run(self, task_prompt: str) -> ExecutionTrace:
        """Execute the DAG with the given task prompt."""
        trace = ExecutionTrace(dag_id=self.dag.dag_id, task_prompt=task_prompt)
        start_time = time.time()
        step_count = 0

        # Node state
        node_outputs: dict[str, str] = {}        # latest output per node
        node_inputs: dict[str, list[str]] = defaultdict(list)  # pending inputs
        loop_counts: dict[str, int] = {}          # loop counter state
        node_iterations: dict[str, int] = defaultdict(int)

        # Determine start nodes from metadata, or find nodes with no incoming trigger edges
        start_nodes = self.dag.metadata.get("start", [])
        if not start_nodes:
            triggered_targets = {e.target for e in self.dag.edges if e.trigger}
            all_nodes = set(self.dag.nodes.keys())
            start_nodes = list(all_nodes - triggered_targets)

        # Initialize start nodes with task prompt
        ready_queue: list[str] = []
        for nid in start_nodes:
            node_inputs[nid].append(task_prompt)
            ready_queue.append(nid)

        # Also propagate via non-trigger edges from start nodes (pre-load context)
        for nid in start_nodes:
            for edge in self._outgoing.get(nid, []):
                if not edge.trigger and edge.carry_data:
                    node_inputs[edge.target].append(task_prompt)

        try:
            while ready_queue:
                if time.time() - start_time > self.timeout:
                    trace.error = f"Timeout exceeded ({self.timeout}s)"
                    break
                if step_count >= self.max_global_steps:
                    trace.error = f"Max steps exceeded ({self.max_global_steps})"
                    break

                node_id = ready_queue.pop(0)
                node = self.dag.nodes.get(node_id)
                if node is None:
                    continue

                # Combine inputs
                combined_input = "\n".join(node_inputs.get(node_id, []))
                node_inputs[node_id] = []  # clear consumed inputs
                node_iterations[node_id] += 1
                step_count += 1

                # Execute node
                t0 = time.time()
                output = self._execute_node(node, combined_input)
                duration = time.time() - t0

                node_outputs[node_id] = output

                trace.steps.append(NodeExecution(
                    node_id=node_id,
                    node_type=node.node_type,
                    iteration=node_iterations[node_id],
                    input_text=combined_input,
                    output_text=output,
                    duration_s=duration,
                ))

                # Process outgoing edges
                for edge in self._outgoing.get(node_id, []):
                    # Evaluate condition
                    if not evaluate_condition(edge.condition, output):
                        continue

                    target_id = edge.target
                    target_node = self.dag.nodes.get(target_id)
                    if target_node is None:
                        continue

                    # For loop_counter targets, handle loop logic
                    if target_node.node_type == "loop_counter":
                        max_iter = target_node.config.get("max_iterations", 3)
                        count = loop_counts.get(target_id, 0) + 1
                        loop_counts[target_id] = count

                        if count >= max_iter:
                            # Loop exhausted: propagate to loop_counter's own outgoing edges
                            # (these typically point to the next phase)
                            loop_counts[target_id] = 0  # reset
                            node_outputs[target_id] = output  # pass through
                            # Record loop counter hit
                            trace.steps.append(NodeExecution(
                                node_id=target_id,
                                node_type="loop_counter",
                                iteration=count,
                                input_text=f"loop exhausted ({count}/{max_iter})",
                                output_text=output,
                            ))
                            # Propagate from loop counter
                            for lc_edge in self._outgoing.get(target_id, []):
                                if lc_edge.trigger and evaluate_condition(lc_edge.condition, output):
                                    payload = output if lc_edge.carry_data else ""
                                    node_inputs[lc_edge.target].append(payload)
                                    if lc_edge.target not in ready_queue:
                                        ready_queue.append(lc_edge.target)
                        else:
                            # Loop continues: re-trigger the loop body
                            # Find non-exit edges from this loop counter
                            for lc_edge in self._outgoing.get(target_id, []):
                                if lc_edge.trigger:
                                    payload = output if lc_edge.carry_data else ""
                                    node_inputs[lc_edge.target].append(payload)
                                    if lc_edge.target not in ready_queue:
                                        ready_queue.append(lc_edge.target)
                        continue

                    # Normal edge: pass data and trigger
                    if edge.carry_data:
                        node_inputs[target_id].append(output)
                    else:
                        node_inputs[target_id].append("")

                    if edge.trigger and target_id not in ready_queue:
                        ready_queue.append(target_id)

            trace.finished = not bool(trace.error)

        except Exception as e:
            trace.error = str(e)

        trace.total_duration_s = time.time() - start_time
        return trace

    def _execute_node(self, node: DAGNode, input_text: str) -> str:
        """Execute a single node and return output text."""

        if node.node_type == "agent":
            return self._execute_agent(node, input_text)

        elif node.node_type == "literal":
            # Literal nodes output their fixed content
            return node.prompt

        elif node.node_type == "passthrough":
            # Pass input through unchanged
            return input_text

        elif node.node_type == "loop_counter":
            # Handled in the main loop; should not reach here normally
            return input_text

        else:
            return input_text

    def _execute_agent(self, node: DAGNode, input_text: str) -> str:
        """Execute an agent node via LLM call."""
        messages = []
        if node.prompt:
            messages.append({"role": "system", "content": node.prompt})
        messages.append({"role": "user", "content": input_text})

        model = node.config.get("model") or self.model

        kwargs: dict[str, Any] = {}
        if "temperature" in node.config:
            kwargs["temperature"] = node.config["temperature"]
        if "max_tokens" in node.config:
            kwargs["max_tokens"] = node.config["max_tokens"]

        return self.llm_fn(messages, model=model, **kwargs)
