"""NegativesStore — persist ReflectInsight across runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from optpilot.models import ReflectInsight


class NegativesStore:
    """JSON-backed storage for accumulated negative examples per FM group."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, fm_group: str) -> Path:
        return self.base_dir / f"negatives_{fm_group}.json"

    def load(self, fm_group: str) -> list[ReflectInsight]:
        path = self._path(fm_group)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [ReflectInsight(**entry) for entry in data]
        except Exception:
            return []

    def extend(self, fm_group: str, insights: list[ReflectInsight]) -> None:
        if not insights:
            return
        existing = self.load(fm_group)
        existing.extend(insights)
        self._save(fm_group, existing)

    def _save(self, fm_group: str, insights: list[ReflectInsight]) -> None:
        data = [asdict(i) for i in insights]
        self._path(fm_group).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
