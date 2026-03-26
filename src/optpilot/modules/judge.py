"""Judge - LLM-as-Judge for offline counterfactual evaluation.

Evaluates whether a repair candidate would fix a given FM
without actually re-running the MAS.
"""

from __future__ import annotations

from optpilot.config import JUDGE_MODEL
from optpilot.data.fm_taxonomy import FM_DEFINITIONS
from optpilot.llm import call_llm_json
from optpilot.models import (
    FMProfile, JudgeVerdict, MASTrace, RepairCandidate,
)

JUDGE_PROMPT = """你是一个多智能体系统 (MAS) 修复评估专家。

请进行反事实推理：如果在执行前就应用了以下修复方案，该故障是否会被避免？

## 故障信息
- FM-{fm_id}: {fm_name}
- 定义: {fm_description}
- 出错 Agent: {agent}
- 出错 Step: {step}
- Root Cause: {root_cause}
- 故障上下文: {failure_context}

## 修复方案
- 描述: {repair_description}
- 具体操作:
{repair_actions_text}

## 评估要求
请从以下几个方面评估:
1. 修复方案是否直接针对 root cause?
2. 修复后，导致故障的执行路径是否会改变?
3. 修复是否可能引入新的问题?

输出严格 JSON 格式:

{{
    "would_fix": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "详细的推理过程 (不超过300字)",
    "potential_side_effects": ["可能的副作用列表"]
}}"""


class Judge:
    """Offline counterfactual evaluation of repair candidates."""

    def evaluate(
        self,
        trace: MASTrace,
        fm_id: str,
        candidate: RepairCandidate,
        profile: FMProfile,
    ) -> JudgeVerdict:
        """Evaluate whether the repair would fix the FM."""
        fm_info = FM_DEFINITIONS[fm_id]
        loc = profile.localization.get(fm_id)

        actions_text = "\n".join(
            f"  {i+1}. [{a.repair_type.value}] {a.description}"
            for i, a in enumerate(candidate.actions)
        )

        prompt = JUDGE_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            agent=loc.agent if loc else "unknown",
            step=loc.step if loc else "unknown",
            root_cause=loc.root_cause if loc else "unknown",
            failure_context=loc.context if loc else "unknown",
            repair_description=candidate.description,
            repair_actions_text=actions_text or "  (no actions)",
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                model=JUDGE_MODEL,
                max_tokens=4096,
            )
            return JudgeVerdict(
                trace_id=trace.trace_id,
                fm_id=fm_id,
                repair_id=candidate.description[:50],
                would_fix=result.get("would_fix", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            print(f"  Warning: Judge failed for trace {trace.trace_id}, FM-{fm_id}: {e}")
            return JudgeVerdict(
                trace_id=trace.trace_id,
                fm_id=fm_id,
                repair_id=candidate.description[:50],
                would_fix=False,
                confidence=0.0,
                reasoning=f"Judge evaluation failed: {e}",
            )
