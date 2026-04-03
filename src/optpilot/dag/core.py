"""MAS-as-DAG abstraction - unified representation for any MAS."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict, deque
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

    def canonical_dict(self) -> dict[str, Any]:
        """Serialize to a canonical dict for semantic equality checks.

        Canonicalization intentionally omits schema-default fields via ``to_dict()``
        and normalizes ordering so semantically equivalent DAGs compare equal even if
        their YAML text differs.
        """

        def _canon(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _canon(value[k]) for k in sorted(value)}
            if isinstance(value, list):
                return [_canon(v) for v in value]
            return value

        data = self.to_dict()
        nodes = [_canon(node) for node in data.get("nodes", [])]
        edges = [_canon(edge) for edge in data.get("edges", [])]

        nodes.sort(key=lambda node: (
            str(node.get("id", "")),
            str(node.get("type", "")),
            json.dumps(node, ensure_ascii=False, sort_keys=True),
        ))
        edges.sort(key=lambda edge: (
            str(edge.get("from", "")),
            str(edge.get("to", "")),
            json.dumps(edge.get("condition", "true"), ensure_ascii=False, sort_keys=True),
            json.dumps(edge, ensure_ascii=False, sort_keys=True),
        ))

        canonical = {
            "dag_id": data.get("dag_id", ""),
            "nodes": nodes,
            "edges": edges,
        }
        if "metadata" in data:
            canonical["metadata"] = _canon(data["metadata"])
        return canonical

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

    def structural_errors(self) -> list[str]:
        """Return structural validation errors for the DAG.

        The executor tolerates dangling edges by silently skipping missing
        targets. For optimization workflows this is too permissive: an evolved
        DAG can look high-scoring while never reaching any agent node. This
        validation pass catches those cases before evaluation or adoption.
        """
        errors: list[str] = []
        node_ids = set(self.nodes.keys())

        if not node_ids:
            return ["DAG defines no nodes."]

        start_nodes = list(self.metadata.get("start", []) or [])
        if not start_nodes:
            start_nodes = [nid for nid in node_ids if not any(e.target == nid and e.trigger for e in self.edges)]
        if not start_nodes:
            errors.append("No start nodes are defined or inferrable.")
        missing_start = [nid for nid in start_nodes if nid not in node_ids]
        if missing_start:
            errors.append(f"Start nodes reference missing nodes: {', '.join(sorted(missing_start))}")

        success_nodes = list(self.metadata.get("success_nodes", []) or [])
        if not success_nodes:
            errors.append("No success_nodes are defined in metadata.")
        missing_success = [nid for nid in success_nodes if nid not in node_ids]
        if missing_success:
            errors.append(f"success_nodes reference missing nodes: {', '.join(sorted(missing_success))}")

        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge source is missing: {edge.source} -> {edge.target}")
            if edge.target not in node_ids:
                errors.append(f"Edge target is missing: {edge.source} -> {edge.target}")

        executable: set[str] = {nid for nid in start_nodes if nid in node_ids}
        queue: deque[str] = deque(executable)
        while queue:
            node_id = queue.popleft()
            for edge in self.edges:
                if edge.source != node_id or not edge.trigger or edge.target not in node_ids:
                    continue
                if edge.target in executable:
                    continue
                executable.add(edge.target)
                queue.append(edge.target)

        if not any(self.nodes[nid].is_agent for nid in executable if nid in self.nodes):
            errors.append("No agent node is executable from the configured start nodes.")
        if success_nodes and not any(nid in executable for nid in success_nodes if nid in node_ids):
            errors.append("No success node is reachable via trigger edges from the configured start nodes.")

        return errors

    def is_structurally_valid(self) -> bool:
        """Whether the DAG passes basic structural validation."""
        return not self.structural_errors()

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

    @classmethod
    def from_initial_program(cls, path: str | Path) -> MASDAG:
        """Load from a Python initial program file containing build_dag()."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_initial_dag", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return cls.from_dict(mod.build_dag())

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

    # ---- Topology Feature Extraction ----

    def extract_topology_features(self) -> dict[str, bool]:
        """Extract minimal topology features used by the Jacobian signature.

        Returns ``has_hub`` — the single feature embedded into
        ``FailureSignature.signature_key()`` so different topology families
        occupy separate rows in the global Jacobian matrix.

        * ``has_hub``: an agent whose transitive out-degree (through
          intermediary nodes) reaches >= 60% of all other agents.

        Loop presence is not tracked because evolution adds loops
        autonomously regardless of the initial topology.
        """
        agents = {nid for nid, n in self.nodes.items() if n.is_agent}
        n_agents = len(agents)

        if n_agents == 0:
            return {"has_hub": False}

        # --- has_hub ---
        # Build adjacency from trigger edges, then BFS through intermediaries
        # to find agent-to-agent transitive out-degree.
        trigger_adj: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            if e.trigger:
                trigger_adj[e.source].add(e.target)

        def _reachable_agents(start: str) -> int:
            visited: set[str] = set()
            count = 0
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                if cur != start and cur in agents:
                    count += 1
                    continue  # don't traverse through other agents
                for nb in trigger_adj.get(cur, ()):
                    queue.append(nb)
            return count

        has_hub = False
        if n_agents > 1:
            threshold = math.ceil(n_agents * 0.6)
            for a in agents:
                if _reachable_agents(a) >= threshold:
                    has_hub = True
                    break

        return {"has_hub": has_hub}

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
