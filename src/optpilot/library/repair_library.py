"""Repair Library - CRUD + retrieval for repair entries."""

from __future__ import annotations

import atexit
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from optpilot.models import RepairCandidate, RepairAction, RepairEntry, RepairType


class RepairLibrary:
    """Persistent repair library backed by JSON file."""

    def __init__(self, store_path: str | Path, autosave: bool = False):
        self.store_path = Path(store_path)
        self.autosave = autosave
        self.entries: list[RepairEntry] = self._load()
        self._entries_by_fm: dict[str, list[RepairEntry]] = defaultdict(list)
        self._dirty = False
        self._rebuild_index()
        atexit.register(self.flush)

    def add(self, entry: RepairEntry, save: bool | None = None) -> None:
        self.entries.append(entry)
        self._entries_by_fm[entry.fm_id].append(entry)
        self._dirty = True
        if self._should_save(save):
            self.flush()

    def search(
        self,
        fm_id: str,
        mas_name: str | None = None,
        status: str | None = None,
        top_k: int = 3,
    ) -> list[RepairEntry]:
        """Search library by FM id, optionally filter by MAS and status.

        By default, failed entries are excluded from retrieval.
        Wrapped skills are preferred over raw hints when available.
        """
        matches = list(self._entries_by_fm.get(fm_id, []))
        if status:
            matches = [e for e in matches if e.status == status]
        else:
            matches = [e for e in matches if e.status != "failed"]

        wrapped_matches = [e for e in matches if e.entry_kind == "wrapped"]
        if wrapped_matches:
            matches = wrapped_matches

        # Sort: same MAS first, then prefer wrapped+validated entries, then success_rate.
        def sort_key(e: RepairEntry) -> tuple:
            same_mas = 1 if mas_name and e.source_mas == mas_name else 0
            kind_priority = 1 if e.entry_kind == "wrapped" else 0
            status_priority = {"validated": 2, "unvalidated": 1}.get(e.status, 0)
            return (same_mas, kind_priority, status_priority, e.success_rate)

        matches.sort(key=sort_key, reverse=True)
        return matches[:top_k]

    def get_entries(
        self,
        fm_id: str | None = None,
        *,
        entry_kind: str | None = None,
        include_failed: bool = True,
    ) -> list[RepairEntry]:
        """Return entries for inspection, wrap-up, or export."""
        entries = list(self.entries if fm_id is None else self._entries_by_fm.get(fm_id, []))
        if entry_kind is not None:
            entries = [e for e in entries if e.entry_kind == entry_kind]
        if not include_failed:
            entries = [e for e in entries if e.status != "failed"]
        return entries

    def replace_wrapped_entries(
        self,
        fm_id: str,
        wrapped_entries: list[RepairEntry],
        save: bool | None = None,
    ) -> None:
        """Replace canonical wrapped skills for one FM while keeping raw hints."""
        self.entries = [
            entry for entry in self.entries
            if not (entry.fm_id == fm_id and entry.entry_kind == "wrapped")
        ]
        self.entries.extend(wrapped_entries)
        self._dirty = True
        self._rebuild_index()
        if self._should_save(save):
            self.flush()

    def update_stats(self, entry_id: str, success: bool, save: bool | None = None) -> None:
        """Update an entry's statistics after application."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.n_applied += 1
                if success:
                    entry.n_success += 1
                entry.success_rate = entry.n_success / entry.n_applied if entry.n_applied > 0 else 0.0
                self._dirty = True
                if self._should_save(save):
                    self.flush()
                return

    def get_stats(self) -> dict:
        """Get library statistics."""
        return {
            "total_entries": len(self.entries),
            "wrapped": sum(1 for e in self.entries if e.entry_kind == "wrapped"),
            "hints": sum(1 for e in self.entries if e.entry_kind == "hint"),
            "validated": sum(1 for e in self.entries if e.status == "validated"),
            "unvalidated": sum(1 for e in self.entries if e.status == "unvalidated"),
            "failed": sum(1 for e in self.entries if e.status == "failed"),
            "fm_coverage": list(dict.fromkeys(e.fm_id for e in self.entries)),
        }

    def flush(self) -> None:
        """Persist pending library changes to disk."""
        if not self._dirty:
            return
        self._save()
        self._dirty = False

    def _load(self) -> list[RepairEntry]:
        if not self.store_path.exists():
            return []
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
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
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_index(self) -> None:
        self._entries_by_fm.clear()
        for entry in self.entries:
            self._entries_by_fm[entry.fm_id].append(entry)

    def _should_save(self, save: bool | None) -> bool:
        if save is not None:
            return save
        return self.autosave
