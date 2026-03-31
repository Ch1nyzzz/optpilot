from __future__ import annotations

import pytest

from optpilot.models import MASTrace
from optpilot.modules.diagnoser import Diagnoser


@pytest.mark.anyio
async def test_diagnoser_adiagnose_returns_labels_only(monkeypatch) -> None:
    async def fake_aclassify_raw(self, trace, model=None):
        return {
            "A": False, "B": True, "C": False, "D": False, "E": False, "F": True,
        }

    monkeypatch.setattr(Diagnoser, "_aclassify_raw", fake_aclassify_raw)

    diagnoser = Diagnoser(max_workers=4)
    trace = MASTrace(
        trace_id=1,
        mas_name="AG2",
        llm_name="MiniMaxAI/MiniMax-M2.5",
        benchmark_name="GSM",
        trajectory="trace body",
    )

    profile = await diagnoser.adiagnose(trace)

    assert profile.active_fm_ids() == ["B", "F"]
    assert profile.localization == {}


@pytest.mark.anyio
async def test_diagnoser_aclassify_batch_skips_localization(monkeypatch) -> None:
    async def fake_aclassify_raw(self, trace, model=None):
        return {
            "A": False, "B": True, "C": False, "D": False, "E": False, "F": False,
        }

    monkeypatch.setattr(Diagnoser, "_aclassify_raw", fake_aclassify_raw)

    diagnoser = Diagnoser(max_workers=4)
    traces = [
        MASTrace(
            trace_id=i,
            mas_name="AG2",
            llm_name="MiniMaxAI/MiniMax-M2.5",
            benchmark_name="GSM",
            trajectory="trace body",
        )
        for i in range(2)
    ]

    profiles = await diagnoser.aclassify_batch(traces)

    assert len(profiles) == 2
    assert all(profile.active_fm_ids() == ["B"] for profile in profiles)
    assert all(profile.localization == {} for profile in profiles)
