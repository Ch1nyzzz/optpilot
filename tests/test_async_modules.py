from __future__ import annotations

import pytest

from optpilot.models import FMLocalization, MASTrace
from optpilot.modules.diagnoser import Diagnoser


@pytest.mark.anyio
async def test_diagnoser_adiagnose_uses_async_localization(monkeypatch) -> None:
    async def fake_aclassify_raw(self, trace, model=None):
        return {
            "A": False, "B": True, "C": False, "D": False, "E": False, "F": True,
            "primary_failure_group": "F",
        }

    async def fake_alocalize_group(self, trace, gid, trace_content):
        return FMLocalization(
            agent="assistant",
            step="review",
            context=f"{gid}-ctx",
            root_cause=f"{gid}-cause",
        )

    monkeypatch.setattr(Diagnoser, "_aclassify_raw", fake_aclassify_raw)
    monkeypatch.setattr(Diagnoser, "_alocalize_group", fake_alocalize_group)

    diagnoser = Diagnoser(max_workers=4)
    trace = MASTrace(
        trace_id=1,
        mas_name="AG2",
        llm_name="MiniMaxAI/MiniMax-M2.5",
        benchmark_name="GSM",
        trajectory="trace body",
    )

    profile = await diagnoser.adiagnose(trace)

    assert profile.primary_fm_id == "F"
    assert sorted(profile.localization) == ["F"]
    assert profile.localization["F"].root_cause == "F-cause"
    assert profile.primary_localization is not None
    assert profile.primary_localization.root_cause == "F-cause"


@pytest.mark.anyio
async def test_diagnoser_aclassify_batch_skips_localization(monkeypatch) -> None:
    async def fake_aclassify_raw(self, trace, model=None):
        return {
            "A": False, "B": True, "C": False, "D": False, "E": False, "F": False,
            "primary_failure_group": "B",
        }

    async def fail_alocalize_group(self, trace, gid, trace_content):
        raise AssertionError("localization should not be called in classify-only mode")

    monkeypatch.setattr(Diagnoser, "_aclassify_raw", fake_aclassify_raw)
    monkeypatch.setattr(Diagnoser, "_alocalize_group", fail_alocalize_group)

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
    assert all(profile.primary_fm_id == "B" for profile in profiles)
    assert all(profile.localization == {} for profile in profiles)
