"""Repair Jacobian — experience-driven repair direction recommendation.

Maintains a matrix mapping (FM group, RepairPattern) → success statistics.
Used to recommend the most promising repair directions for a given failure,
and updated after each repair attempt.

Two tracking modes:
- **assigned_pattern_id**: pre-selected pattern used for online recommendation.
- **observed_pattern_id**: inferred from actual DAG-level edits for attribution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from optpilot.config import LIBRARY_DIR
from optpilot.skills.repair_patterns import (
    FailureSignature,
    PatternCatalog,
    RepairPattern,
)

_JACOBIAN_DIR = LIBRARY_DIR / "jacobian"


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class JacobianEntry:
    """Statistics for one (signature_key, pattern_id) cell."""

    n_applied: int = 0
    n_success: int = 0
    total_fm_delta: float = 0.0
    total_pass_delta: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_applied if self.n_applied > 0 else 0.0

    @property
    def avg_fm_improvement(self) -> float:
        """Average FM rate change (negative = improvement)."""
        return self.total_fm_delta / self.n_applied if self.n_applied > 0 else 0.0

    @property
    def avg_pass_improvement(self) -> float:
        """Average pass rate change (positive = improvement)."""
        return self.total_pass_delta / self.n_applied if self.n_applied > 0 else 0.0


@dataclass
class RepairOutcome:
    """Record of a single repair attempt (inner iteration level)."""

    fm_group: str
    dag_component: str
    agent: str
    assigned_pattern_id: str
    observed_pattern_id: str
    success: bool
    fm_delta: float
    pass_delta: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def signature(self) -> FailureSignature:
        return FailureSignature(
            fm_group=self.fm_group,
            dag_component=self.dag_component,
            agent=self.agent,
        )


# ------------------------------------------------------------------ #
#  RepairJacobian                                                      #
# ------------------------------------------------------------------ #

class RepairJacobian:
    """Experience-driven repair recommendation matrix.

    Matrix layout:
        rows = FM groups        (e.g. "A", "D")
        cols = pattern IDs      (e.g. "prompt_add_constraint", "edge_fix_carry_data")
        cell = JacobianEntry    (n_applied, n_success, deltas)
    """

    def __init__(
        self,
        catalog: PatternCatalog | None = None,
        base_dir: Path | None = None,
    ):
        self.base_dir = base_dir or _JACOBIAN_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog or PatternCatalog()

        # Online recommendation matrix: {sig_key: {pattern_id: JacobianEntry}}
        self.matrix: dict[str, dict[str, JacobianEntry]] = {}

        # Full outcome log (assigned + observed)
        self.outcomes: list[RepairOutcome] = []

        # Try to load persisted state
        self._load()

    # ---------------------------------------------------------------- #
    #  Recommend                                                        #
    # ---------------------------------------------------------------- #

    def recommend(
        self,
        sig: FailureSignature,
        top_k: int = 3,
    ) -> list[tuple[RepairPattern, float]]:
        """Recommend top-k repair patterns for a failure signature.

        Returns list of (RepairPattern, score) sorted by score descending.
        Score = success_rate when data exists, or an FM-group prior for cold
        start.
        """
        sig_key = sig.signature_key()
        entries = self.matrix.get(sig_key, {})

        scored: list[tuple[str, float]] = []

        for pattern_id, pattern in self.catalog.effective_items():
            entry = entries.get(pattern_id)
            if entry and entry.n_applied >= 1:
                # Data-driven score: blend empirical success_rate with cold-start prior.
                # As n_applied grows, trust data more; at n=1 still weight data > prior.
                prior = self._cold_start_score(sig, pattern)
                alpha = entry.n_applied / (entry.n_applied + 2.0)  # at n=1: 0.33, n=3: 0.6, n=5: 0.71
                score = alpha * entry.success_rate + (1 - alpha) * prior
            else:
                # Cold-start: heuristic prior based on FM group.
                score = self._cold_start_score(sig, pattern)

            scored.append((pattern_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[RepairPattern, float]] = []
        for pattern_id, score in scored[:top_k]:
            pattern = self.catalog[pattern_id]
            results.append((pattern, score))
        return results

    @staticmethod
    def _cold_start_score(sig: FailureSignature, pattern: RepairPattern) -> float:
        """Heuristic score when no empirical data exists."""
        priors: dict[str, dict[str, float]] = {
            "A": {
                "prompt_add_constraint": 0.55,
                "prompt_narrow_role": 0.45,
                "prompt_add_step_by_step": 0.30,
            },
            "B": {
                "loop_fix_config": 0.55,
                "edge_add_condition": 0.45,
                "loop_add_retry": 0.35,
            },
            "C": {
                "edge_add_context_propagation": 0.55,
                "edge_fix_carry_data": 0.50,
                "edge_add_missing": 0.35,
            },
            "D": {
                "edge_add_missing": 0.55,
                "edge_fix_carry_data": 0.45,
                "edge_add_condition": 0.30,
            },
            "E": {
                "prompt_add_step_by_step": 0.55,
                "prompt_add_constraint": 0.40,
                "topo_split_agent": 0.25,
            },
            "F": {
                "prompt_strengthen_verification": 0.60,
                "topo_add_verification_node": 0.45,
                "loop_add_retry": 0.20,
            },
        }
        return priors.get(sig.fm_group, {}).get(pattern.pattern_id, 0.10)

    # ---------------------------------------------------------------- #
    #  Update                                                           #
    # ---------------------------------------------------------------- #

    def update(self, outcome: RepairOutcome) -> None:
        """Update the matrix with a repair outcome.

        Uses **observed** pattern when available (what the repair actually
        changed). When the observed pattern is unknown, successful runs do not
        credit the assigned recommendation, to avoid false positive attribution.
        """
        sig_key = outcome.signature.signature_key()

        # Prefer observed (what actually happened) over assigned (what we suggested).
        if outcome.observed_pattern_id:
            self._update_entry(sig_key, outcome.observed_pattern_id, outcome)

        # If observed differs from assigned, also record a "tried but not
        # followed" datapoint on assigned so its score decays naturally.
        if (
            outcome.observed_pattern_id
            and outcome.assigned_pattern_id
            and outcome.observed_pattern_id != outcome.assigned_pattern_id
        ):
            self._update_entry(
                sig_key,
                outcome.assigned_pattern_id,
                outcome,
                override_success=False,  # LLM chose not to follow it
            )
        elif outcome.assigned_pattern_id and not outcome.success:
            # When the repair failed and we cannot confidently identify what
            # happened, it is still safe to record that the recommended pattern
            # did not lead to success.
            self._update_entry(
                sig_key,
                outcome.assigned_pattern_id,
                outcome,
                override_success=False,
            )

        self.outcomes.append(outcome)

    def _update_entry(
        self,
        sig_key: str,
        pattern_id: str,
        outcome: RepairOutcome,
        override_success: bool | None = None,
    ) -> None:
        """Increment a single matrix cell."""
        if sig_key not in self.matrix:
            self.matrix[sig_key] = {}
        if pattern_id not in self.matrix[sig_key]:
            self.matrix[sig_key][pattern_id] = JacobianEntry()

        entry = self.matrix[sig_key][pattern_id]
        entry.n_applied += 1
        success = override_success if override_success is not None else outcome.success
        if success:
            entry.n_success += 1
        entry.total_fm_delta += outcome.fm_delta
        entry.total_pass_delta += outcome.pass_delta

    # ---------------------------------------------------------------- #
    #  Persistence                                                      #
    # ---------------------------------------------------------------- #

    def save(self) -> None:
        """Persist matrix and outcomes to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Save matrix
        matrix_data: dict[str, dict[str, dict[str, Any]]] = {}
        for sig_key, patterns in self.matrix.items():
            matrix_data[sig_key] = {
                pid: asdict(entry) for pid, entry in patterns.items()
            }
        matrix_path = self.base_dir / "matrix.json"
        matrix_path.write_text(
            json.dumps(matrix_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Append new outcomes to JSONL
        outcomes_path = self.base_dir / "outcomes.jsonl"
        with open(outcomes_path, "a", encoding="utf-8") as f:
            for outcome in self.outcomes:
                f.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")

        # Clear in-memory outcomes after flush (they're on disk now)
        self.outcomes.clear()

    def _load(self) -> None:
        """Load persisted state from disk."""
        # Load matrix
        matrix_path = self.base_dir / "matrix.json"
        if matrix_path.exists():
            try:
                raw = json.loads(matrix_path.read_text(encoding="utf-8"))
                for sig_key, patterns in raw.items():
                    self.matrix[sig_key] = {}
                    for pid, entry_data in patterns.items():
                        self.matrix[sig_key][pid] = JacobianEntry(**entry_data)
            except Exception as e:
                print(f"  Warning: failed to load Jacobian matrix: {e}")

        # Outcomes are append-only on disk; don't reload into memory
        # (they're for offline analysis, not online recommendation)

    # ---------------------------------------------------------------- #
    #  Analysis                                                         #
    # ---------------------------------------------------------------- #

    def offline_summary(self) -> dict[str, Any]:
        """Aggregate outcomes by observed_pattern_id for offline analysis.

        Reads from the outcomes.jsonl file to produce a summary view.
        """
        outcomes_path = self.base_dir / "outcomes.jsonl"
        if not outcomes_path.exists():
            return {}

        observed_stats: dict[str, dict[str, Any]] = {}
        try:
            for line in outcomes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                obs_pid = record.get("observed_pattern_id", "")
                if not obs_pid:
                    continue
                if obs_pid not in observed_stats:
                    observed_stats[obs_pid] = {
                        "n_applied": 0, "n_success": 0,
                        "total_fm_delta": 0.0, "total_pass_delta": 0.0,
                    }
                stats = observed_stats[obs_pid]
                stats["n_applied"] += 1
                if record.get("success"):
                    stats["n_success"] += 1
                stats["total_fm_delta"] += record.get("fm_delta", 0.0)
                stats["total_pass_delta"] += record.get("pass_delta", 0.0)
        except Exception as e:
            print(f"  Warning: failed to read outcomes for offline summary: {e}")

        return observed_stats

    def format_matrix_summary(self) -> str:
        """Human-readable matrix summary for logging."""
        if not self.matrix:
            return "Jacobian matrix is empty (cold start)."

        lines: list[str] = ["Jacobian Matrix Summary:"]
        for sig_key in sorted(self.matrix):
            entries = self.matrix[sig_key]
            top = sorted(
                entries.items(),
                key=lambda kv: kv[1].success_rate,
                reverse=True,
            )[:3]
            top_strs = [
                f"{pid}({e.success_rate:.0%}, n={e.n_applied})"
                for pid, e in top
            ]
            lines.append(f"  {sig_key}: {', '.join(top_strs)}")
        return "\n".join(lines)

    def format_evolution_report(self, fm_group: str) -> str:
        """Detailed report for CatalogEvolver — what worked, what didn't,
        and where the LLM diverged from recommendations.

        Covers:
        1. Per-pattern success rates for this FM group (from matrix)
        2. Assigned-vs-observed divergence (from outcomes.jsonl)
        """
        lines: list[str] = [f"# Jacobian Experience Report for FM Group {fm_group}", ""]

        # --- 1. Matrix stats for this FM group ---
        relevant = {
            sig_key: entries
            for sig_key, entries in self.matrix.items()
            if sig_key == fm_group
        }
        if relevant:
            lines.append("## Pattern Success Rates (online matrix)")
            lines.append("")
            for sig_key in sorted(relevant):
                lines.append(f"### {sig_key}")
                entries = sorted(
                    relevant[sig_key].items(),
                    key=lambda kv: kv[1].n_applied,
                    reverse=True,
                )
                for pid, entry in entries:
                    lines.append(
                        f"- `{pid}`: {entry.success_rate:.0%} success "
                        f"(n={entry.n_applied}, "
                        f"avg_fm_delta={entry.avg_fm_improvement:+.3f}, "
                        f"avg_pass_delta={entry.avg_pass_improvement:+.3f})"
                    )
                lines.append("")
        else:
            lines.append("## Pattern Success Rates")
            lines.append("No matrix data for this FM group yet (cold start).")
            lines.append("")

        # --- 2. Assigned-vs-observed divergence ---
        outcomes_path = self.base_dir / "outcomes.jsonl"
        divergences: list[dict[str, str]] = []
        if outcomes_path.exists():
            try:
                for line in outcomes_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("fm_group") != fm_group:
                        continue
                    assigned = rec.get("assigned_pattern_id", "")
                    observed = rec.get("observed_pattern_id", "")
                    if assigned and observed and assigned != observed:
                        divergences.append({
                            "assigned": assigned,
                            "observed": observed,
                            "success": rec.get("success", False),
                        })
            except Exception:
                pass

        if divergences:
            lines.append("## Recommendation Divergence")
            lines.append("Cases where the LLM ignored the recommended pattern:")
            lines.append("")
            # Aggregate
            div_counts: dict[str, dict[str, int]] = {}
            for d in divergences:
                key = f"{d['assigned']} → {d['observed']}"
                if key not in div_counts:
                    div_counts[key] = {"total": 0, "success": 0}
                div_counts[key]["total"] += 1
                if d["success"]:
                    div_counts[key]["success"] += 1

            for key, counts in sorted(div_counts.items(), key=lambda x: x[1]["total"], reverse=True):
                rate = counts["success"] / counts["total"] if counts["total"] else 0
                lines.append(f"- {key}: {counts['total']}x ({rate:.0%} success)")
            lines.append("")
            lines.append(
                "If the LLM consistently diverges from a pattern AND the "
                "observed alternative succeeds more often, consider updating "
                "the catalog: refine the ignored pattern's description or "
                "add the preferred alternative as a first-class pattern."
            )
        else:
            lines.append("## Recommendation Divergence")
            lines.append("No divergence data yet (LLM followed recommendations or data unavailable).")

        lines.append("")
        return "\n".join(lines)
