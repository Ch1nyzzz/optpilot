"""MAS Runner 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from optpilot.dag.core import MASDAG
from optpilot.models import MASTrace


class MASRunner(ABC):
    """运行 MAS 并收集 trace 的通用接口。"""

    mas_name: str

    @abstractmethod
    def run_task(
        self,
        task_prompt: str,
        dag: MASDAG | None = None,
        output_dir: str | Path | None = None,
        timeout: int | None = None,
    ) -> MASTrace:
        """运行单个任务并收集执行轨迹。"""

    def run_batch(
        self,
        tasks: list[str],
        dag: MASDAG | None = None,
        output_base: str | Path | None = None,
    ) -> list[MASTrace]:
        """顺序运行多个任务。子类可覆盖实现并行。"""
        traces = []
        for i, task in enumerate(tasks):
            out_dir = Path(output_base) / f"task_{i}" if output_base else None
            print(f"  Running task {i + 1}/{len(tasks)}: {task[:60]}...")
            trace = self.run_task(task, dag=dag, output_dir=out_dir)
            trace.trace_id = i
            traces.append(trace)
        return traces
