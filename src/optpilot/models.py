"""Core data models for OptPilot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
from uuid import uuid4


# === FM Categories ===

class FMCategory(str, Enum):
    FC1 = "FC1"  # System Design
    FC2 = "FC2"  # Inter-Agent Misalignment
    FC3 = "FC3"  # Task Verification


# === Trace ===

@dataclass
class MASTrace:
    trace_id: int
    mas_name: str          # "ChatDev", "MetaGPT"
    llm_name: str          # "GPT-4o", "gpt-oss-120b"
    benchmark_name: str    # "ProgramDev"
    trajectory: str        # full conversation log
    task_key: str = ""     # task identifier
    mast_annotation: dict[str, int] = field(default_factory=dict)  # fm_id -> 0/1
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
    category: FMCategory
    present: bool
    confidence: float = 1.0


@dataclass
class FMLocalization:
    agent: str         # which agent
    step: str          # which phase/step
    context: str       # relevant snippet
    root_cause: str    # analysis


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

def to_json(obj: Any) -> str:
    """Serialize dataclass to JSON string."""
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)
