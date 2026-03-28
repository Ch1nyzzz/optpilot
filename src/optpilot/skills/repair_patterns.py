"""Repair Pattern catalog and FailureSignature extraction.

Provides structured repair directions for the Jacobian-based recommendation
system.  Each RepairPattern is a named, enumerable repair strategy at a
granularity between "modify prompt" and the specific text changes an LLM
would produce.

The PatternCatalog wraps the pattern dict with JSON persistence so that
meta-evolution (CatalogEvolver) can add/modify/disable patterns at runtime.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from optpilot.dag.core import MASDAG

from optpilot.config import LIBRARY_DIR

if TYPE_CHECKING:
    from optpilot.models import FMProfile
    from optpilot.skills.tools import ChangeRecord


# ------------------------------------------------------------------ #
#  FailureSignature                                                    #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class FailureSignature:
    """Structured failure descriptor used by the online recommender.

    The online Jacobian row key is intentionally coarse: only ``fm_group``.
    ``dag_component`` and ``agent`` are retained as optional metadata for
    diagnostics and offline analysis, but they do not affect online lookup.
    """

    fm_group: str = ""      # "A"-"F"
    dag_component: str = ""  # optional metadata only
    agent: str = ""          # optional metadata only

    def signature_key(self) -> str:
        """Key used as the row index in the Jacobian matrix."""
        return self.fm_group


def extract_failure_signatures(
    fm_group: str,
    profiles: list[FMProfile],
) -> list[FailureSignature]:
    """Extract FailureSignatures from profiles for a given FM group.

    The online recommender is FM-group keyed. Localization-derived metadata is
    retained when available, but it does not change the row key.
    """
    signatures: list[FailureSignature] = []
    for profile in profiles:
        if fm_group not in profile.active_fm_ids():
            continue
        loc = profile.localization.get(fm_group)
        if loc is None:
            signatures.append(FailureSignature(
                fm_group=fm_group,
                dag_component="other",
            ))
            continue
        signatures.append(FailureSignature(
            fm_group=fm_group,
            dag_component=loc.dag_component or "other",
            agent=loc.agent or "",
        ))
    return signatures


def dominant_signature(signatures: list[FailureSignature]) -> FailureSignature:
    """Return the most frequent signature, falling back to a generic one."""
    if not signatures:
        return FailureSignature(fm_group="")
    # Count by signature_key (FM group only)
    key_counts: Counter[str] = Counter()
    key_to_sig: dict[str, FailureSignature] = {}
    for sig in signatures:
        key = sig.signature_key()
        key_counts[key] += 1
        key_to_sig[key] = sig
    top_key, _ = key_counts.most_common(1)[0]
    return key_to_sig[top_key]


# ------------------------------------------------------------------ #
#  RepairPattern                                                       #
# ------------------------------------------------------------------ #

@dataclass
class RepairPattern:
    """A named, enumerable repair strategy."""

    pattern_id: str
    name: str
    description: str          # injected into evolve prompt as direction
    target_components: list[str] = field(default_factory=list)
    effective: bool = True    # False = disabled by meta-evolution


# ------------------------------------------------------------------ #
#  Pattern Catalog                                                     #
# ------------------------------------------------------------------ #

_DEFAULT_CATALOG: dict[str, RepairPattern] = {
    "prompt_add_constraint": RepairPattern(
        pattern_id="prompt_add_constraint",
        name="Add explicit constraint to agent prompt",
        description=(
            "Add explicit constraint-checking instructions to the agent's system prompt. "
            "For example, require the agent to re-read task constraints before answering, "
            "or add a checklist of requirements that must be verified."
        ),
        target_components=["agent_prompt"],
    ),
    "prompt_add_step_by_step": RepairPattern(
        pattern_id="prompt_add_step_by_step",
        name="Add step-by-step reasoning instruction",
        description=(
            "Add structured reasoning instructions to the agent's prompt, such as "
            "'Break the problem into steps and solve each one explicitly before giving "
            "a final answer.' This helps with reasoning errors and task drift."
        ),
        target_components=["agent_prompt"],
    ),
    "prompt_strengthen_verification": RepairPattern(
        pattern_id="prompt_strengthen_verification",
        name="Strengthen verification logic in prompt",
        description=(
            "Enhance the verification agent's prompt to be more rigorous: require "
            "explicit re-computation, unit checking, boundary case testing, or "
            "step-by-step solution validation before accepting an answer."
        ),
        target_components=["agent_prompt"],
    ),
    "prompt_narrow_role": RepairPattern(
        pattern_id="prompt_narrow_role",
        name="Narrow agent role scope",
        description=(
            "Reduce the agent's responsibilities to a focused subset. Remove "
            "conflicting or overly broad instructions that cause the agent to "
            "drift from its primary task."
        ),
        target_components=["agent_prompt"],
    ),
    "edge_fix_carry_data": RepairPattern(
        pattern_id="edge_fix_carry_data",
        name="Fix edge carry_data configuration",
        description=(
            "Fix the carry_data settings on edges so that necessary context "
            "(problem statement, prior results, intermediate data) is properly "
            "passed to downstream agents. Agents receiving incomplete context "
            "produce confused or empty responses."
        ),
        target_components=["edge_carry_data"],
    ),
    "edge_add_context_propagation": RepairPattern(
        pattern_id="edge_add_context_propagation",
        name="Add context propagation across loop iterations",
        description=(
            "Ensure context (especially the original problem statement and "
            "accumulated results) propagates across loop iterations. In loops, "
            "agents often only receive the previous agent's output via carry_data "
            "and lose the original task context."
        ),
        target_components=["edge_carry_data", "edge_missing"],
    ),
    "edge_add_condition": RepairPattern(
        pattern_id="edge_add_condition",
        name="Add or modify edge trigger condition",
        description=(
            "Add or refine edge conditions (keyword matching) to control routing. "
            "This prevents messages from flowing to the wrong agent or prevents "
            "unnecessary iterations."
        ),
        target_components=["edge_condition"],
    ),
    "edge_add_missing": RepairPattern(
        pattern_id="edge_add_missing",
        name="Add missing edge connection",
        description=(
            "Add a missing edge between agents that need to communicate. "
            "Information flow gaps cause agents to operate without necessary "
            "context or results from upstream agents."
        ),
        target_components=["edge_missing"],
    ),
    "topo_add_verification_node": RepairPattern(
        pattern_id="topo_add_verification_node",
        name="Add verification agent node",
        description=(
            "Add a new verification/checker agent node to the DAG. This agent "
            "reviews the output of other agents before the system produces a "
            "final answer, catching errors that would otherwise go undetected."
        ),
        target_components=["agent_prompt", "other"],
    ),
    "topo_split_agent": RepairPattern(
        pattern_id="topo_split_agent",
        name="Split agent into specialized sub-agents",
        description=(
            "Split one agent that has too many responsibilities into two or more "
            "specialized agents with narrower roles. This reduces cognitive load "
            "and improves task focus."
        ),
        target_components=["agent_prompt", "node_config"],
    ),
    "loop_fix_config": RepairPattern(
        pattern_id="loop_fix_config",
        name="Fix loop configuration",
        description=(
            "Adjust loop counter settings: max_iterations, exit conditions, or "
            "loop edge annotations. This fixes systems that loop too many or too "
            "few times, or fail to terminate properly."
        ),
        target_components=["loop_config"],
    ),
    "loop_add_retry": RepairPattern(
        pattern_id="loop_add_retry",
        name="Add error retry loop",
        description=(
            "Add a retry loop that re-routes execution back to a solver agent "
            "when verification fails, with carry_data passing the error feedback. "
            "This gives the system a chance to self-correct."
        ),
        target_components=["loop_config", "edge_missing"],
    ),
    "config_adjust_params": RepairPattern(
        pattern_id="config_adjust_params",
        name="Adjust agent parameters",
        description=(
            "Modify agent configuration parameters such as temperature, "
            "max_tokens, or other model settings to improve output quality "
            "or control verbosity."
        ),
        target_components=["node_config"],
    ),
}


def get_pattern(pattern_id: str) -> RepairPattern | None:
    """Look up a pattern by ID from the default catalog."""
    return _DEFAULT_CATALOG.get(pattern_id)


# ------------------------------------------------------------------ #
#  PatternCatalog — dynamic, persistable wrapper                      #
# ------------------------------------------------------------------ #

_CATALOG_STORE_PATH = LIBRARY_DIR / "pattern_catalog.json"


class PatternCatalog:
    """Dynamic pattern catalog with JSON persistence.

    Loads from ``library_store/pattern_catalog.json`` if present, otherwise
    falls back to ``_DEFAULT_CATALOG``.  Meta-evolution (CatalogEvolver)
    mutates this catalog via ``add_pattern`` / ``update_pattern``.
    """

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or _CATALOG_STORE_PATH
        self._patterns: dict[str, RepairPattern] = {
            pid: RepairPattern(**asdict(p)) for pid, p in _DEFAULT_CATALOG.items()
        }
        self._load()

    # --- dict-like access ---

    def items(self) -> list[tuple[str, RepairPattern]]:
        return list(self._patterns.items())

    def effective_items(self) -> list[tuple[str, RepairPattern]]:
        """Only patterns with ``effective=True``."""
        return [(pid, p) for pid, p in self._patterns.items() if p.effective]

    def get(self, pattern_id: str) -> RepairPattern | None:
        return self._patterns.get(pattern_id)

    def __len__(self) -> int:
        return len(self._patterns)

    def __contains__(self, pattern_id: str) -> bool:
        return pattern_id in self._patterns

    def __getitem__(self, pattern_id: str) -> RepairPattern:
        return self._patterns[pattern_id]

    # --- mutation ---

    def add_pattern(self, pattern: RepairPattern) -> None:
        """Add or overwrite a pattern."""
        self._patterns[pattern.pattern_id] = pattern

    def update_pattern(
        self,
        pattern_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        target_components: list[str] | None = None,
        effective: bool | None = None,
    ) -> bool:
        """Update fields on an existing pattern.  Returns True if found."""
        p = self._patterns.get(pattern_id)
        if p is None:
            return False
        if name is not None:
            p.name = name
        if description is not None:
            p.description = description
        if target_components is not None:
            p.target_components = target_components
        if effective is not None:
            p.effective = effective
        return True

    # --- persistence ---

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            pid: asdict(p) for pid, p in self._patterns.items()
        }
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            for pid, pdata in raw.items():
                self._patterns[pid] = RepairPattern(**pdata)
        except Exception as e:
            print(f"  Warning: failed to load pattern catalog: {e}")

    # --- LLM context ---

    def as_llm_context(self) -> str:
        """Format effective patterns for injection into LLM prompts."""
        lines = ["Available repair patterns:"]
        for pid, p in self.effective_items():
            lines.append(f"- **{p.name}** (`{pid}`): {p.description}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Observed pattern inference from change records                      #
# ------------------------------------------------------------------ #

# Keyword sets for each pattern category
_PROMPT_KEYWORDS = re.compile(
    r"\b(prompt|role|instruction|system_prompt|You are|must|should|always|never|"
    r"step.by.step|verify|check|ensure|require)\b",
    re.IGNORECASE,
)
_EDGE_CARRY_KEYWORDS = re.compile(
    r"\b(carry_data|carry|data.*flow|context.*pass|propagat)\b",
    re.IGNORECASE,
)
_EDGE_CONDITION_KEYWORDS = re.compile(
    r"\b(condition|trigger|keyword|match|route|routing)\b",
    re.IGNORECASE,
)
_TOPO_KEYWORDS = re.compile(
    r"\b(add.*node|new.*agent|add.*agent|DAGNode|node_type.*agent|"
    r"split.*agent|remove.*node)\b",
    re.IGNORECASE,
)
_LOOP_KEYWORDS = re.compile(
    r"\b(loop|max_iteration|loop_counter|exit|continue|retry|max_turns)\b",
    re.IGNORECASE,
)
_CONFIG_KEYWORDS = re.compile(
    r"\b(temperature|max_tokens|top_p|params|config\[)\b",
    re.IGNORECASE,
)
_EDGE_ADD_KEYWORDS = re.compile(
    r"\b(DAGEdge|add.*edge|new.*edge|edges\.append|missing.*edge)\b",
    re.IGNORECASE,
)


def infer_observed_pattern(change_records: list[Any]) -> str:
    """Infer the dominant repair pattern from change_records.

    Analyzes the combined old_str/new_str text to classify the type of
    modification.  Returns a pattern_id or "" if mixed/unclassifiable.
    """
    if not change_records:
        return ""

    # Collect all changed text for analysis
    all_new_text = []
    all_old_text = []
    for cr in change_records:
        old_str = cr.old_str if hasattr(cr, "old_str") else cr.get("old_str", "")
        new_str = cr.new_str if hasattr(cr, "new_str") else cr.get("new_str", "")
        all_old_text.append(old_str)
        all_new_text.append(new_str)

    combined_old = "\n".join(all_old_text)
    combined_new = "\n".join(all_new_text)
    # Focus on what was added (the diff)
    combined = combined_new + "\n" + combined_old

    # Score each category
    scores: dict[str, int] = {
        "prompt": len(_PROMPT_KEYWORDS.findall(combined)),
        "edge_carry": len(_EDGE_CARRY_KEYWORDS.findall(combined)),
        "edge_condition": len(_EDGE_CONDITION_KEYWORDS.findall(combined)),
        "topo": len(_TOPO_KEYWORDS.findall(combined)),
        "loop": len(_LOOP_KEYWORDS.findall(combined)),
        "config": len(_CONFIG_KEYWORDS.findall(combined)),
        "edge_add": len(_EDGE_ADD_KEYWORDS.findall(combined)),
    }

    if sum(scores.values()) == 0:
        return ""

    top_category = max(scores, key=lambda k: scores[k])
    top_score = scores[top_category]

    # If no clear winner (top < 2 or close to second), return mixed
    sorted_scores = sorted(scores.values(), reverse=True)
    if top_score < 2:
        return ""
    if len(sorted_scores) > 1 and sorted_scores[1] > 0:
        ratio = top_score / sorted_scores[1]
        if ratio < 1.5:
            return ""  # too close to call

    # Map category to most likely pattern_id
    category_to_pattern: dict[str, str] = {
        "prompt": "prompt_add_constraint",  # generic prompt change
        "edge_carry": "edge_fix_carry_data",
        "edge_condition": "edge_add_condition",
        "topo": "topo_add_verification_node",
        "loop": "loop_fix_config",
        "config": "config_adjust_params",
        "edge_add": "edge_add_missing",
    }
    return category_to_pattern.get(top_category, "")


def infer_observed_pattern_from_dags(
    original_dag: MASDAG,
    candidate_dag: MASDAG,
) -> str:
    """Infer the dominant observed repair pattern from DAG-level changes.

    This is the online attribution path used by the Jacobian. It intentionally
    maps concrete edits into a small, stable pattern vocabulary.

    Returns:
        A pattern_id when the change is dominated by one pattern family, or
        ``""`` when the edit is mixed / unclassifiable.
    """
    categories = _dag_change_categories(original_dag, candidate_dag)
    if len(categories) != 1:
        return ""

    category = next(iter(categories))
    category_to_pattern = {
        "prompt": "prompt_add_constraint",
        "edge_carry": "edge_fix_carry_data",
        "edge_condition": "edge_add_condition",
        "topology": "topo_add_verification_node",
        "loop": "loop_fix_config",
        "config": "config_adjust_params",
        "edge_add": "edge_add_missing",
    }
    return category_to_pattern.get(category, "")


def _dag_change_categories(
    original_dag: MASDAG,
    candidate_dag: MASDAG,
) -> set[str]:
    """Classify DAG-level differences into coarse repair categories."""
    categories: set[str] = set()

    original_nodes = original_dag.nodes
    candidate_nodes = candidate_dag.nodes

    original_agents = {nid for nid, node in original_nodes.items() if node.node_type == "agent"}
    candidate_agents = {nid for nid, node in candidate_nodes.items() if node.node_type == "agent"}
    if original_agents != candidate_agents:
        categories.add("topology")

    original_edges = {_edge_key(edge): edge for edge in original_dag.edges}
    candidate_edges = {_edge_key(edge): edge for edge in candidate_dag.edges}
    if set(original_edges) != set(candidate_edges):
        categories.add("edge_add")

    shared_nodes = set(original_nodes) & set(candidate_nodes)
    for node_id in shared_nodes:
        old_node = original_nodes[node_id]
        new_node = candidate_nodes[node_id]

        if old_node.node_type != new_node.node_type:
            categories.add("topology")
            continue

        if old_node.node_type == "agent":
            if old_node.prompt != new_node.prompt or old_node.role != new_node.role:
                categories.add("prompt")
            if _agent_params(old_node.config) != _agent_params(new_node.config):
                categories.add("config")
        elif old_node.node_type == "loop_counter":
            if old_node.config != new_node.config:
                categories.add("loop")
        elif old_node.config != new_node.config:
            categories.add("config")

    for edge_key in set(original_edges) & set(candidate_edges):
        old_edge = original_edges[edge_key]
        new_edge = candidate_edges[edge_key]
        if old_edge.carry_data != new_edge.carry_data:
            categories.add("edge_carry")
        if old_edge.condition != new_edge.condition or old_edge.trigger != new_edge.trigger:
            categories.add("edge_condition")
        old_loop = old_edge.config.get("loop", "") if old_edge.config else ""
        new_loop = new_edge.config.get("loop", "") if new_edge.config else ""
        if old_loop != new_loop:
            categories.add("loop")

    return categories


def _edge_key(edge: Any) -> tuple[str, str]:
    return (edge.source, edge.target)


def _agent_params(config: dict[str, Any]) -> Any:
    return (config or {}).get("params", {})
