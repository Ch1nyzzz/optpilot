from __future__ import annotations

import pytest

from optpilot.models import MASTrace
from optpilot.modules.diagnoser import Diagnoser


@pytest.mark.anyio
async def test_diagnoser_adiagnose_uses_async_localization(monkeypatch) -> None:
    async def fake_acall_llm_json(messages, model=None, temperature=0.2, max_tokens=8192):
        return {
            "agent": "assistant",
            "step": "review",
            "context": "ctx",
            "root_cause": "cause",
        }

    monkeypatch.setattr("optpilot.modules.diagnoser.acall_llm_json", fake_acall_llm_json)

    diagnoser = Diagnoser(max_workers=4)
    trace = MASTrace(
        trace_id=1,
        mas_name="AG2",
        llm_name="MiniMaxAI/MiniMax-M2.5",
        benchmark_name="GSM",
        trajectory="trace body",
        mast_annotation={"1.3": 1, "3.3": 1},
    )

    profile = await diagnoser.adiagnose(trace)

    assert sorted(profile.localization) == ["1.3", "3.3"]
    assert profile.localization["1.3"].agent == "assistant"
    assert profile.localization["3.3"].root_cause == "cause"
