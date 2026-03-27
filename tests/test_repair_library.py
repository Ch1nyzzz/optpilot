from __future__ import annotations

import json

from optpilot.library.repair_library import RepairLibrary
from optpilot.models import RepairEntry


def test_repair_library_buffers_writes_until_flush(tmp_path) -> None:
    store_path = tmp_path / "library.json"
    library = RepairLibrary(store_path)

    entry = RepairEntry(fm_id="B", fm_name="Execution Loop / Stuck", source_mas="AG2")
    library.add(entry)

    assert store_path.exists() is False
    assert library.search("B")[0].entry_id == entry.entry_id

    library.flush()

    assert store_path.exists() is True
    data = json.loads(store_path.read_text(encoding="utf-8"))
    assert data[0]["entry_id"] == entry.entry_id

    library.update_stats(entry.entry_id, success=True)
    updated = json.loads(store_path.read_text(encoding="utf-8"))
    assert updated[0]["n_applied"] == 0

    library.flush()
    updated = json.loads(store_path.read_text(encoding="utf-8"))
    assert updated[0]["n_applied"] == 1
    assert updated[0]["n_success"] == 1


def test_repair_library_search_excludes_failed_by_default(tmp_path) -> None:
    store_path = tmp_path / "library.json"
    library = RepairLibrary(store_path)

    library.add(RepairEntry(fm_id="B", status="failed", success_rate=0.9))
    validated = RepairEntry(fm_id="B", status="validated", success_rate=0.4)
    library.add(validated)

    matches = library.search("B")

    assert [entry.status for entry in matches] == ["validated"]
    assert matches[0].entry_id == validated.entry_id
