from __future__ import annotations

import pytest

from optpilot.models import FMLocalization, MASTrace
from optpilot.modules.diagnoser import Diagnoser


@pytest.mark.anyio
async def test_diagnoser_adiagnose_uses_async_localization(monkeypatch) -> None:
    async def fake_aclassify(self, trace, model=None):
        return {"A": False, "B": True, "C": False, "D": False, "E": False, "F": True}

    async def fake_alocalize_group(self, trace, gid, trace_content):
        return FMLocalization(
            agent="assistant",
            step="review",
            context=f"{gid}-ctx",
            root_cause=f"{gid}-cause",
        )

    monkeypatch.setattr(Diagnoser, "_aclassify", fake_aclassify)
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

    assert sorted(profile.localization) == ["B", "F"]
    assert profile.localization["B"].agent == "assistant"
    assert profile.localization["F"].root_cause == "F-cause"
