"""YAMLOptimizer - 直接读写 YAML 的 MAS 优化器。

不经过 DAG 抽象，Optimizer 直接看到完整 YAML 配置，
根据诊断结果输出修改后的 YAML。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from optpilot.data.fm_taxonomy import FM_DEFINITIONS
from optpilot.llm import call_llm
from optpilot.models import FMProfile, MASTrace


OPTIMIZE_PROMPT = """你是一个多智能体系统 (MAS) 架构优化专家。

你将看到一个 MAS 的完整 YAML 配置和该系统的故障诊断结果。
你的任务是修改 YAML 配置来修复这些故障。

## 当前 YAML 配置
```yaml
{yaml_content}
```

## 故障诊断结果
{diagnosis_text}

## 故障 Trace (供参考)
{trace_excerpt}

## 你可以做的修改
1. **修改 agent prompt** (nodes[].config.role) — 改变 agent 行为
2. **修改 tooling** (nodes[].config.tooling) — 增减 agent 可用工具
3. **修改模型参数** (nodes[].config.params) — temperature, max_tokens 等
4. **增加 agent** — 加新 node + 对应 edges
5. **删除 agent** — 移除 node + 清理 edges
6. **修改消息流** (edges) — 改 trigger, condition, carry_data, clear_context 等
7. **修改终止条件** (edges[].condition) — keyword 匹配规则
8. **修改循环上限** (loop_counter.config.max_iterations)
9. **修改 literal prompt** (literal nodes 的 config.content)

## 输出要求
1. 先输出你的分析和修改方案（不超过 300 字）
2. 然后输出修改后的**完整 YAML**，用 ```yaml ... ``` 包裹
3. YAML 必须是合法的、可以直接被引擎执行的完整配置
4. 只做针对诊断结果的必要修改，不要做无关改动
5. 保持 vars, version 等元信息不变"""


OPTIMIZE_WITH_SKILLS_PROMPT = """你是一个多智能体系统 (MAS) 架构优化专家。

你将看到一个 MAS 的完整 YAML 配置和该系统的故障诊断结果。
你的任务是修改 YAML 配置来修复这些故障。
以下是一些来自经验库的参考 skill（仅供参考，不必照搬）。

## 当前 YAML 配置
```yaml
{yaml_content}
```

## 故障诊断结果
{diagnosis_text}

## 故障 Trace (供参考)
{trace_excerpt}

## 经验库参考 Skills
{skills_text}

## 你可以做的修改
1. **修改 agent prompt** (nodes[].config.role) — 改变 agent 行为
2. **修改 tooling** (nodes[].config.tooling) — 增减 agent 可用工具
3. **修改模型参数** (nodes[].config.params) — temperature, max_tokens 等
4. **增加 agent** — 加新 node + 对应 edges
5. **删除 agent** — 移除 node + 清理 edges
6. **修改消息流** (edges) — 改 trigger, condition, carry_data, clear_context 等
7. **修改终止条件** (edges[].condition) — keyword 匹配规则
8. **修改循环上限** (loop_counter.config.max_iterations)
9. **修改 literal prompt** (literal nodes 的 config.content)

## 输出要求
1. 先输出你的分析和修改方案（不超过 300 字）
2. 然后输出修改后的**完整 YAML**，用 ```yaml ... ``` 包裹
3. YAML 必须是合法的、可以直接被引擎执行的完整配置
4. 只做针对诊断结果的必要修改，不要做无关改动
5. 保持 vars, version 等元信息不变"""


def _build_diagnosis_text(profile: FMProfile) -> str:
    """把 FMProfile 中所有活跃 FM 的诊断信息格式化。"""
    lines = []
    for fm_id in profile.active_fm_ids():
        fm_info = FM_DEFINITIONS.get(fm_id, {})
        loc = profile.localization.get(fm_id)
        lines.append(f"### FM-{fm_id}: {fm_info.get('name', '?')}")
        lines.append(f"- 定义: {fm_info.get('description', '?')}")
        if loc:
            lines.append(f"- Agent: {loc.agent}")
            lines.append(f"- Step: {loc.step}")
            lines.append(f"- Root Cause: {loc.root_cause}")
            lines.append(f"- Context: {loc.context}")
        lines.append("")
    return "\n".join(lines) if lines else "No active FMs diagnosed."


def _extract_yaml(response: str) -> str:
    """从 LLM 回复中提取 ```yaml ... ``` 块。"""
    match = re.search(r'```yaml\s*\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: 找最后一个 yaml 块
    matches = list(re.finditer(r'```yaml\s*\n(.*?)```', response, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    raise ValueError("LLM response does not contain a ```yaml``` block")


def _extract_analysis(response: str) -> str:
    """提取 YAML 块之前的分析文本。"""
    match = re.search(r'```yaml', response)
    if match:
        return response[:match.start()].strip()
    return response[:500].strip()


class YAMLOptimizer:
    """直接操作 YAML 的 MAS 优化器。"""

    def __init__(self, library=None):
        self.library = library

    def optimize(
        self,
        yaml_path: str | Path,
        profile: FMProfile,
        trace: MASTrace,
    ) -> dict:
        """根据诊断结果优化 YAML 配置。

        Args:
            yaml_path: 当前 YAML 配置文件路径
            profile: 诊断结果 (所有活跃 FM 的 localization)
            trace: 故障 trace

        Returns:
            {
                "original_yaml": str,
                "modified_yaml": str,
                "analysis": str,
                "fm_ids": list[str],
                "yaml_valid": bool,
            }
        """
        yaml_path = Path(yaml_path)
        original_yaml = yaml_path.read_text(encoding="utf-8")
        diagnosis_text = _build_diagnosis_text(profile)
        trace_excerpt = trace.trajectory[:3000]

        # 查库
        skills_text = self._get_relevant_skills(profile)

        if skills_text:
            prompt = OPTIMIZE_WITH_SKILLS_PROMPT.format(
                yaml_content=original_yaml,
                diagnosis_text=diagnosis_text,
                trace_excerpt=trace_excerpt,
                skills_text=skills_text,
            )
        else:
            prompt = OPTIMIZE_PROMPT.format(
                yaml_content=original_yaml,
                diagnosis_text=diagnosis_text,
                trace_excerpt=trace_excerpt,
            )

        response = call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=16384,
        )

        # 解析
        analysis = _extract_analysis(response)
        modified_yaml = _extract_yaml(response)

        # 验证 YAML 合法性
        yaml_valid = False
        try:
            parsed = yaml.safe_load(modified_yaml)
            # 基本结构检查
            graph = parsed.get("graph", parsed)
            has_nodes = "nodes" in graph and len(graph["nodes"]) > 0
            has_edges = "edges" in graph
            yaml_valid = has_nodes and has_edges
        except Exception:
            pass

        return {
            "original_yaml": original_yaml,
            "modified_yaml": modified_yaml,
            "analysis": analysis,
            "fm_ids": profile.active_fm_ids(),
            "yaml_valid": yaml_valid,
        }

    def optimize_and_save(
        self,
        yaml_path: str | Path,
        profile: FMProfile,
        trace: MASTrace,
        output_path: str | Path | None = None,
    ) -> dict:
        """优化并保存修改后的 YAML。"""
        result = self.optimize(yaml_path, profile, trace)

        if result["yaml_valid"]:
            out = Path(output_path) if output_path else Path(yaml_path).with_suffix(".optimized.yaml")
            out.write_text(result["modified_yaml"], encoding="utf-8")
            result["output_path"] = str(out)
        else:
            result["output_path"] = None
            print("  WARNING: Generated YAML is invalid, not saving.")

        return result

    def _get_relevant_skills(self, profile: FMProfile) -> str:
        """从 library 中检索与当前 FM 相关的 skills。"""
        if not self.library:
            return ""

        fm_ids = profile.active_fm_ids()
        skills = []
        for fm_id in fm_ids:
            entries = self.library.search(fm_id)
            for e in entries[:2]:  # 每个 FM 最多 2 条
                skills.append(
                    f"- FM-{e.fm_id} [{e.status}, success={e.success_rate:.0%}]: "
                    f"{e.root_cause_pattern}"
                )

        return "\n".join(skills) if skills else ""
