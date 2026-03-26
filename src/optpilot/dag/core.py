"""MAS-as-DAG abstraction - unified representation for any MAS."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from optpilot.models import RepairAction, RepairType


@dataclass
class DAGNode:
    node_id: str
    node_type: str       # "agent" | "literal" | "loop_counter" | "passthrough"
    role: str = ""       # agent role name
    prompt: str = ""     # system prompt (agent) or content (literal)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_agent(self) -> bool:
        return self.node_type == "agent"


@dataclass
class DAGEdge:
    source: str    # node_id
    target: str    # node_id
    trigger: bool = True
    condition: str | dict = "true"
    carry_data: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MASDAG:
    dag_id: str = ""
    nodes: dict[str, DAGNode] = field(default_factory=dict)
    edges: list[DAGEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_nodes(self) -> dict[str, DAGNode]:
        return {k: v for k, v in self.nodes.items() if v.is_agent}

    # ---- Serialization ----

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        nodes = []
        for node in self.nodes.values():
            d: dict[str, Any] = {
                "id": node.node_id,
                "type": node.node_type,
            }
            if node.role:
                d["role"] = node.role
            if node.prompt:
                d["prompt"] = node.prompt
            if node.config:
                d["config"] = node.config
            nodes.append(d)

        edges = []
        for edge in self.edges:
            d: dict[str, Any] = {
                "from": edge.source,
                "to": edge.target,
            }
            if not edge.trigger:
                d["trigger"] = False
            if edge.condition != "true":
                d["condition"] = edge.condition
            if not edge.carry_data:
                d["carry_data"] = False
            if edge.config:
                d.update(edge.config)
            edges.append(d)

        result: dict[str, Any] = {"dag_id": self.dag_id, "nodes": nodes, "edges": edges}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict) -> MASDAG:
        """Deserialize from plain dict."""
        dag = cls(dag_id=data.get("dag_id", ""), metadata=data.get("metadata", {}))

        for nd in data.get("nodes", []):
            node_id = nd["id"]
            dag.nodes[node_id] = DAGNode(
                node_id=node_id,
                node_type=nd.get("type", "agent"),
                role=nd.get("role", ""),
                prompt=nd.get("prompt", ""),
                config=nd.get("config", {}),
            )

        for ed in data.get("edges", []):
            extra = {k: v for k, v in ed.items()
                     if k not in ("from", "to", "trigger", "condition", "carry_data")}
            dag.edges.append(DAGEdge(
                source=ed["from"],
                target=ed["to"],
                trigger=ed.get("trigger", True),
                condition=ed.get("condition", "true"),
                carry_data=ed.get("carry_data", True),
                config=extra,
            ))

        return dag

    def save(self, path: str | Path) -> None:
        """Save to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> MASDAG:
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    # ---- Repair ----

    def apply_repair(self, action: RepairAction) -> MASDAG:
        """Apply a repair action, return a new DAG (immutable)."""
        new_dag = copy.deepcopy(self)

        if action.repair_type == RepairType.NODE_MUTATION:
            node = new_dag.nodes.get(action.target)
            if node:
                if "prompt" in action.details:
                    node.prompt = action.details["prompt"]
                if "config" in action.details:
                    node.config.update(action.details["config"])

        elif action.repair_type == RepairType.NODE_ADD:
            new_node = DAGNode(
                node_id=action.target,
                node_type=action.details.get("node_type", "agent"),
                role=action.details.get("role", ""),
                prompt=action.details.get("prompt", ""),
                config=action.details.get("config", {}),
            )
            new_dag.nodes[action.target] = new_node
            for edge_spec in action.details.get("edges", []):
                new_dag.edges.append(DAGEdge(**edge_spec))

        elif action.repair_type == RepairType.NODE_DELETE:
            new_dag.nodes.pop(action.target, None)
            new_dag.edges = [e for e in new_dag.edges
                             if e.source != action.target and e.target != action.target]

        elif action.repair_type == RepairType.EDGE_MUTATION:
            for edge in new_dag.edges:
                if edge.source == action.details.get("source") and edge.target == action.details.get("target"):
                    if "condition" in action.details:
                        edge.condition = action.details["condition"]
                    if "config" in action.details.get("updates", {}):
                        edge.config.update(action.details["updates"]["config"])

        elif action.repair_type == RepairType.EDGE_REWIRE:
            old_src = action.details.get("old_source")
            old_tgt = action.details.get("old_target")
            for edge in new_dag.edges:
                if edge.source == old_src and edge.target == old_tgt:
                    if "new_source" in action.details:
                        edge.source = action.details["new_source"]
                    if "new_target" in action.details:
                        edge.target = action.details["new_target"]

        elif action.repair_type == RepairType.CONFIG_CHANGE:
            node = new_dag.nodes.get(action.target)
            if node:
                node.config.update(action.details)

        return new_dag

    # ---- Utility ----

    def summary(self) -> str:
        """Generate a concise text summary of the DAG for LLM prompts."""
        lines = [f"DAG: {self.dag_id}"]
        lines.append(f"Agents ({len(self.agent_nodes)}):")
        for nid, node in self.agent_nodes.items():
            lines.append(f"  - {nid} [{node.node_type}]: {node.role[:80]}")

        lines.append("Loop counters:")
        for nid, node in self.nodes.items():
            if node.node_type == "loop_counter":
                max_iter = node.config.get("max_iterations", "?")
                lines.append(f"  - {nid}: max_iterations={max_iter}")

        lines.append(f"Edges ({len(self.edges)}):")
        for e in self.edges[:20]:
            cond = e.condition if isinstance(e.condition, str) else "keyword"
            lines.append(f"  {e.source} → {e.target} [trigger={e.trigger}, cond={cond}]")
        if len(self.edges) > 20:
            lines.append(f"  ... and {len(self.edges) - 20} more edges")

        return "\n".join(lines)
