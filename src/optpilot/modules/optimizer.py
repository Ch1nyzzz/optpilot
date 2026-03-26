"""Optimizer - generates repair candidates by retrieving skills from library or generating new ones.

Retrieval: fm_id 粗筛 → LLM 看 root_cause 匹配最佳 skill → 适配或生成新方案。
"""

from __future__ import annotations

from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy import FM_DEFINITIONS
from optpilot.library.repair_library import RepairLibrary
from optpilot.llm import call_llm_json
from optpilot.models import (
    FMProfile, MASTrace, RepairAction, RepairCandidate, RepairEntry, RepairType,
)

SKILL_MATCH_PROMPT = """你是一个 MAS 修复专家。根据当前故障的 root cause，从已有的 skill 库中选择最匹配的 skill。

## 当前故障
- FM-{fm_id}: {fm_name}
- Agent: {agent}
- Root Cause: {root_cause}

## 可用 Skills
{skills_text}

请判断哪个 skill 最适用于当前故障。输出 JSON:

{{
    "matched_skill_id": "最匹配的 skill 编号 (如 '1')，如果没有合适的则填 'none'",
    "match_reason": "为什么选择这个 skill (一句话)",
    "confidence": 0.0-1.0
}}"""

REPAIR_GENERATION_PROMPT = """你是一个多智能体系统 (MAS) 架构优化专家。

根据以下诊断结果，生成具体的修复方案。

## 故障信息
- FM-{fm_id}: {fm_name}
- 定义: {fm_description}
- 出错 Agent: {agent}
- 出错 Step: {step}
- Root Cause: {root_cause}

## MAS 架构信息
{mas_context}

## 可用修复操作类型
1. node_mutation: 修改 agent 的 prompt 或配置
2. node_add: 增加新 agent
3. node_delete: 删除冗余 agent
4. edge_mutation: 修改通信协议或终止条件
5. config_change: 修改循环次数、超时等配置

请生成一个针对性的修复方案。输出严格 JSON 格式:

{{
    "description": "修复方案的高层描述 (一句话)",
    "actions": [
        {{
            "repair_type": "操作类型 (上述5种之一)",
            "target": "目标 agent 或配置项",
            "description": "具体修改描述",
            "details": {{}},
            "rationale": "为什么这个修改能解决问题"
        }}
    ],
    "confidence": 0.0-1.0
}}"""

REPAIR_WITH_SKILL_PROMPT = """你是一个多智能体系统 (MAS) 架构优化专家。

根据以下诊断结果，生成具体的修复方案。可以参考已有的 skill（但不必照搬，根据实际情况决定是否采用、部分采用或完全另起方案）。

## 故障信息
- FM-{fm_id}: {fm_name}
- 定义: {fm_description}
- 出错 Agent: {agent}
- 出错 Step: {step}
- Root Cause: {root_cause}

## MAS 架构信息
{mas_context}

## 参考 Skill（仅供参考）
- Pattern: {skill_pattern}
- 方案: {skill_description}
- 操作: {skill_actions}

## 可用修复操作类型
1. node_mutation: 修改 agent 的 prompt 或配置
2. node_add: 增加新 agent
3. node_delete: 删除冗余 agent
4. edge_mutation: 修改通信协议或终止条件
5. config_change: 修改循环次数、超时等配置

请生成针对性的修复方案。输出严格 JSON 格式:

{{
    "description": "修复方案的高层描述 (一句话)",
    "used_skill": true/false,
    "actions": [
        {{
            "repair_type": "操作类型 (上述5种之一)",
            "target": "目标 agent 或配置项",
            "description": "具体修改描述",
            "details": {{}},
            "rationale": "为什么这个修改能解决问题"
        }}
    ],
    "confidence": 0.0-1.0
}}"""


class Optimizer:
    """Generate repair candidates: match skill from library or create new."""

    def __init__(self, library: RepairLibrary):
        self.library = library

    def generate_repair(
        self,
        fm_id: str,
        profile: FMProfile,
        trace: MASTrace,
        dag: MASDAG | None = None,
    ) -> RepairCandidate:
        """Generate a repair candidate for the given FM.

        1. fm_id 粗筛 library 中同 FM 的 skills
        2. LLM 看当前 root_cause 匹配最佳 skill
        3. 匹配到 → 适配该 skill
        4. 没匹配到 → 生成全新方案
        """
        # 1. 粗筛
        candidates = self.library.search(fm_id, mas_name=trace.mas_name)
        if not candidates:
            print(f"  Library 无 FM-{fm_id} 的 skill，生成新方案...")
            return self._generate_new(fm_id, profile, trace, dag)

        # 2. LLM 匹配最相关 skill
        matched = self._match_skill(fm_id, profile, candidates)
        if matched:
            print(f"  参考 skill [{matched.entry_id}]: {matched.root_cause_pattern[:60]}")
            return self._generate_with_skill(matched, fm_id, profile, trace, dag)

        # 3. 无匹配，从头生成
        print(f"  Library 有 {len(candidates)} 条 skill 但无匹配，生成新方案...")
        return self._generate_new(fm_id, profile, trace, dag)

    def _match_skill(
        self, fm_id: str, profile: FMProfile, candidates: list[RepairEntry],
    ) -> RepairEntry | None:
        """Use LLM to match current root cause against library skills."""
        fm_info = FM_DEFINITIONS[fm_id]
        loc = profile.localization.get(fm_id)

        skills_text = "\n".join(
            f"Skill {i+1} (id={e.entry_id}): {e.root_cause_pattern} "
            f"[success_rate={e.success_rate:.0%}, applied={e.n_applied}x]"
            for i, e in enumerate(candidates)
        )

        prompt = SKILL_MATCH_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            agent=loc.agent if loc else "unknown",
            root_cause=loc.root_cause if loc else "unknown",
            skills_text=skills_text,
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            skill_id = result.get("matched_skill_id", "none")
            confidence = result.get("confidence", 0)

            if skill_id == "none" or confidence < 0.5:
                return None

            # Map skill number to entry
            idx = int(skill_id) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except Exception as e:
            print(f"  Skill matching failed: {e}")

        return None

    def _generate_with_skill(
        self, skill: RepairEntry, fm_id: str, profile: FMProfile,
        trace: MASTrace, dag: MASDAG | None,
    ) -> RepairCandidate:
        """Generate repair with a matched skill as reference (not forced)."""
        fm_info = FM_DEFINITIONS[fm_id]
        loc = profile.localization.get(fm_id)

        skill_actions = "; ".join(
            a.description for a in (skill.candidate.actions if skill.candidate else [])
        )

        prompt = REPAIR_WITH_SKILL_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            agent=loc.agent if loc else "unknown",
            step=loc.step if loc else "unknown",
            root_cause=loc.root_cause if loc else "unknown",
            mas_context=self._get_mas_context(trace, dag),
            skill_pattern=skill.root_cause_pattern,
            skill_description=skill.candidate.description if skill.candidate else "",
            skill_actions=skill_actions,
        )

        result = call_llm_json(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        candidate = self._parse_result(result, fm_id)
        candidate.source = "library" if result.get("used_skill") else "generated"
        return candidate

    def _generate_new(
        self, fm_id: str, profile: FMProfile, trace: MASTrace, dag: MASDAG | None,
    ) -> RepairCandidate:
        fm_info = FM_DEFINITIONS[fm_id]
        loc = profile.localization.get(fm_id)

        prompt = REPAIR_GENERATION_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            agent=loc.agent if loc else "unknown",
            step=loc.step if loc else "unknown",
            root_cause=loc.root_cause if loc else "unknown",
            mas_context=self._get_mas_context(trace, dag),
        )

        result = call_llm_json(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        return self._parse_result(result, fm_id)

    def _get_mas_context(self, trace: MASTrace, dag: MASDAG | None) -> str:
        if dag:
            return dag.summary()
        return (
            f"MAS: {trace.mas_name}, LLM: {trace.llm_name}, "
            f"Benchmark: {trace.benchmark_name}\n"
            f"简单 2-agent 对话系统 (Student + Assistant)，多轮对话解题。"
        )

    def _parse_result(self, result: dict, fm_id: str) -> RepairCandidate:
        actions = []
        for a in result.get("actions", []):
            try:
                rt = a.get("repair_type", "node_mutation")
                if rt not in [e.value for e in RepairType]:
                    rt = "node_mutation"
                actions.append(RepairAction(
                    repair_type=RepairType(rt),
                    target=a.get("target", ""),
                    description=a.get("description", ""),
                    details=a.get("details", {}),
                    rationale=a.get("rationale", ""),
                ))
            except (ValueError, KeyError):
                continue

        return RepairCandidate(
            fm_id=fm_id,
            actions=actions,
            description=result.get("description", ""),
            source="generated",
            confidence=result.get("confidence", 0.5),
        )
