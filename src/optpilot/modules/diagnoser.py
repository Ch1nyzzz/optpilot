"""Diagnoser - FM diagnosis with agent/step level localization.

Takes MAST-annotated traces and uses LLM to localize each active FM
to specific agent + step + root cause. Supports concurrent LLM calls.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from optpilot.data.fm_taxonomy import FM_DEFINITIONS
from optpilot.llm import call_llm_json
from optpilot.models import FMLabel, FMLocalization, FMProfile, MASTrace

# If trace is shorter than this, send it whole; otherwise summarize
FULL_TRACE_THRESHOLD = 15000  # ~4K tokens


LOCALIZATION_PROMPT = """你是一个多智能体系统 (MAS) 故障诊断专家。

以下是一个 MAS 执行轨迹。已知该轨迹存在以下故障：
- FM-{fm_id}: {fm_name}
- 定义: {fm_description}

轨迹内容:
{trace_content}

请分析该轨迹，定位故障发生的位置和原因。输出严格 JSON 格式:

{{
    "agent": "出错的 agent 名称",
    "step": "出错发生在哪个阶段/步骤",
    "context": "故障相关的关键上下文 (简要描述，不超过200字)",
    "root_cause": "故障的根本原因分析 (不超过200字)"
}}"""


def _prepare_trace_content(trajectory: str) -> str:
    if len(trajectory) <= FULL_TRACE_THRESHOLD:
        return trajectory
    head = trajectory[:5000]
    tail = trajectory[-5000:]
    return f"{head}\n\n[... {len(trajectory) - 10000} chars omitted ...]\n\n{tail}"


class Diagnoser:
    """Diagnose MAS traces: localize FM to agent + step level."""

    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers

    def diagnose(self, trace: MASTrace, target_fm: str | None = None) -> FMProfile:
        """Diagnose a trace, localizing active FMs concurrently."""
        profile = FMProfile(trace_id=trace.trace_id)

        for fm_id, fm_info in FM_DEFINITIONS.items():
            present = trace.mast_annotation.get(fm_id, 0) == 1
            profile.labels[fm_id] = FMLabel(
                fm_id=fm_id,
                fm_name=fm_info["name"],
                category=fm_info["category"],
                present=present,
            )

        fms_to_localize = (
            [target_fm] if target_fm and target_fm in profile.active_fm_ids()
            else profile.active_fm_ids()
        )
        if not fms_to_localize:
            return profile

        trace_content = _prepare_trace_content(trace.trajectory)

        # 并行诊断所有 FM
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(fms_to_localize))) as pool:
            futures = {
                pool.submit(self._localize_fm, trace, fm_id, trace_content): fm_id
                for fm_id in fms_to_localize
            }
            for fut in as_completed(futures):
                fm_id = futures[fut]
                profile.localization[fm_id] = fut.result()

        return profile

    def diagnose_batch(
        self, traces: list[MASTrace], target_fm: str | None = None,
    ) -> list[FMProfile]:
        """Diagnose multiple traces concurrently."""
        if not traces:
            return []

        # 并行诊断所有 trace
        results: dict[int, FMProfile] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(traces))) as pool:
            futures = {
                pool.submit(self.diagnose, t, target_fm): i
                for i, t in enumerate(traces)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"  Warning: diagnose_batch failed for trace index {idx}: {e}")
                    results[idx] = FMProfile(trace_id=traces[idx].trace_id)

        return [results[i] for i in range(len(traces))]

    def _localize_fm(
        self, trace: MASTrace, fm_id: str, trace_content: str,
    ) -> FMLocalization:
        fm_info = FM_DEFINITIONS[fm_id]
        prompt = LOCALIZATION_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            trace_content=trace_content,
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
            )
            return FMLocalization(
                agent=result.get("agent", "unknown"),
                step=result.get("step", "unknown"),
                context=result.get("context", ""),
                root_cause=result.get("root_cause", ""),
            )
        except Exception as e:
            print(f"  Warning: Failed to localize FM-{fm_id} for trace {trace.trace_id}: {e}")
            return FMLocalization(
                agent="unknown", step="unknown",
                context="", root_cause=f"Localization failed: {e}",
            )
