"""Repair Library - CRUD + retrieval for repair entries."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from optpilot.models import RepairCandidate, RepairAction, RepairEntry, RepairType


class RepairLibrary:
    """Persistent repair library backed by JSON file."""

    def __init__(self, store_path: str | Path):
        self.store_path = Path(store_path)
        self.entries: list[RepairEntry] = self._load()

    def add(self, entry: RepairEntry) -> None:
        self.entries.append(entry)
        self._save()

    def search(
        self,
        fm_id: str,
        mas_name: str | None = None,
        status: str | None = None,
        top_k: int = 3,
    ) -> list[RepairEntry]:
        """Search library by FM id, optionally filter by MAS and status."""
        matches = [e for e in self.entries if e.fm_id == fm_id]
        if status:
            matches = [e for e in matches if e.status == status]

        # Sort: same MAS first, then by success_rate descending
        def sort_key(e: RepairEntry) -> tuple:
            same_mas = 1 if mas_name and e.source_mas == mas_name else 0
            return (same_mas, e.success_rate)

        matches.sort(key=sort_key, reverse=True)
        return matches[:top_k]

    def update_stats(self, entry_id: str, success: bool) -> None:
        """Update an entry's statistics after application."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.n_applied += 1
                if success:
                    entry.n_success += 1
                entry.success_rate = entry.n_success / entry.n_applied if entry.n_applied > 0 else 0.0
                self._save()
                return

    def get_stats(self) -> dict:
        """Get library statistics."""
        return {
            "total_entries": len(self.entries),
            "validated": sum(1 for e in self.entries if e.status == "validated"),
            "unvalidated": sum(1 for e in self.entries if e.status == "unvalidated"),
            "failed": sum(1 for e in self.entries if e.status == "failed"),
            "fm_coverage": list(set(e.fm_id for e in self.entries)),
        }

    def _load(self) -> list[RepairEntry]:
        if not self.store_path.exists():
            return []
        try:
            data = json.loads(self.store_path.read_text())
            entries = []
            for d in data:
                # Reconstruct nested dataclasses
                cand_data = d.pop("candidate", None)
                candidate = None
                if cand_data:
                    actions = [
                        RepairAction(
                            repair_type=RepairType(a["repair_type"]),
                            **{k: v for k, v in a.items() if k != "repair_type"}
                        )
                        for a in cand_data.pop("actions", [])
                    ]
                    candidate = RepairCandidate(actions=actions, **cand_data)
                entries.append(RepairEntry(candidate=candidate, **d))
            return entries
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load library from {self.store_path}: {e}")
            return []

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self.entries]
        self.store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
