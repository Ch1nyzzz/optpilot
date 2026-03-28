"""Core data models for OptPilot."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from optpilot.dag.core import MASDAG


# === Trace ===

@dataclass
class MASTrace:
    trace_id: int
    mas_name: str          # "ChatDev", "MetaGPT"
    llm_name: str          # "GPT-4o", "gpt-oss-120b"
    benchmark_name: str    # "ProgramDev"
    trajectory: str        # full conversation log
    trace_path: str = ""   # persisted trace artifact path, if saved
    task_key: str = ""     # task identifier
    mast_annotation: dict[str, int] = field(default_factory=dict)  # group_id -> 0/1
    task_success: bool | None = None
    task_score: float | None = None
    latency_s: float | None = None

    def active_fm_ids(self) -> list[str]:
        return [k for k, v in self.mast_annotation.items() if v == 1]


# === FM Profile ===

@dataclass
class FMLabel:
    fm_id: str
    fm_name: str
    category: str  # group ID: "A"-"F"
    present: bool
    confidence: float = 1.0


@dataclass
class FMLocalization:
    agent: str         # which agent, or "DAG_structure"
    step: str          # which phase/step
    context: str       # relevant snippet
    root_cause: str    # analysis
    dag_component: str = "other"  # agent_prompt | edge_carry_data | edge_condition | edge_missing | loop_config | node_config | other


@dataclass
class FMProfile:
    trace_id: int
    labels: dict[str, FMLabel] = field(default_factory=dict)
    localization: dict[str, FMLocalization] = field(default_factory=dict)  # fm_id -> loc

    def active_fms(self) -> list[FMLabel]:
        return [l for l in self.labels.values() if l.present]

    def active_fm_ids(self) -> list[str]:
        return [l.fm_id for l in self.active_fms()]


# === Repair Actions ===

class RepairType(str, Enum):
    NODE_MUTATION = "node_mutation"
    NODE_ADD = "node_add"
    NODE_DELETE = "node_delete"
    EDGE_MUTATION = "edge_mutation"
    EDGE_REWIRE = "edge_rewire"
    CONFIG_CHANGE = "config_change"


@dataclass
class RepairAction:
    repair_type: RepairType
    target: str          # target node/edge id
    description: str     # human-readable
    details: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class RepairCandidate:
    fm_id: str
    actions: list[RepairAction] = field(default_factory=list)
    description: str = ""
    source: str = "generated"  # "generated" | "library"
    confidence: float = 0.0


# === Repair Library Entry ===

@dataclass
class RepairEntry:
    entry_id: str = ""
    entry_kind: str = "hint"  # "hint" | "wrapped"
    fm_id: str = ""
    fm_name: str = ""
    source_mas: str = ""
    root_cause_pattern: str = ""  # 蒸馏后的 root cause 类型摘要 (skill key)
    when_not_to_use: str = ""
    candidate: RepairCandidate | None = None
    status: str = "unvalidated"  # "unvalidated" | "validated" | "failed"
    success_rate: float = 0.0
    n_applied: int = 0
    n_success: int = 0
    side_effects: list[str] = field(default_factory=list)
    avoid_actions: list[str] = field(default_factory=list)
    supporting_entry_ids: list[str] = field(default_factory=list)
    counter_entry_ids: list[str] = field(default_factory=list)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = uuid4().hex[:8]


# === Judge Verdict ===

@dataclass
class JudgeVerdict:
    trace_id: int
    fm_id: str
    repair_id: str = ""
    would_fix: bool = False
    confidence: float = 0.0
    reasoning: str = ""


# === Serialization ===

# === Skill Workflow Models ===


@dataclass
class EvolveResult:
    """Return value of Skill.evolve(): modified DAG + full change evidence."""

    dag: MASDAG
    analysis_text: str          # LLM reasoning
    modified_source: str        # complete modified source (Python build_dag())
    change_description: str     # one-line summary
    actions_taken: list[str] = field(default_factory=list)  # human-readable diffs
    change_records: list[Any] = field(default_factory=list)  # ChangeRecord list for Forger replay
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectInsight:
    """Return value of Skill.reflect(): failure analysis + lesson."""

    round_index: int
    fm_id: str
    changes_attempted: list[str]    # from evolve_history
    before_fm_rate: float
    after_fm_rate: float
    before_pass_rate: float
    after_pass_rate: float
    failure_reason: str
    lesson: str
    timestamp: str = ""


@dataclass
class SkillBudget:
    """Computation budget tracker for a Skill run."""

    max_llm_calls: int = 30
    max_batch_runs: int = 10
    max_wall_time_s: float = 600.0
    used_llm_calls: int = 0
    used_batch_runs: int = 0
    start_time: float = 0.0

    def check(self) -> bool:
        """Return False if budget exhausted."""
        if self.used_llm_calls >= self.max_llm_calls:
            return False
        if self.used_batch_runs >= self.max_batch_runs:
            return False
        if self.start_time > 0 and time.time() - self.start_time > self.max_wall_time_s:
            return False
        return True


@dataclass
class AnalysisResult:
    """Return value of Skill.analyze()."""

    fm_id: str
    fm_rate: float
    common_agents: list[str] = field(default_factory=list)
    common_steps: list[str] = field(default_factory=list)
    root_cause_clusters: list[str] = field(default_factory=list)
    dag_summary: str = ""
    evidence_snippets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Return value of BaseSkill.run()."""

    success: bool
    fm_id: str
    dag: MASDAG | None = None
    inner_iterations: int = 0
    outer_rounds: int = 0
    final_fm_rate: float = 1.0
    final_pass_rate: float = 0.0
    negatives: list[ReflectInsight] = field(default_factory=list)
    budget_used: SkillBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# === Serialization ===

def to_json(obj: Any) -> str:
    """Serialize dataclass to JSON string."""
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)
