"""Runner - executes MASDAG workflows and collects traces.

Uses the built-in DAGExecutor instead of external subprocess calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from optpilot.config import TARGET_MODEL
from optpilot.dag.core import MASDAG
from optpilot.dag.executor import DAGExecutor, ExecutionTrace
from optpilot.llm import call_llm
from optpilot.models import MASTrace
from optpilot.modules.base_runner import MASRunner


class OptPilotRunner(MASRunner):
    """Run MASDAG workflows directly and collect execution traces."""

    mas_name = "OptPilot"

    def __init__(
        self,
        dag: MASDAG | None = None,
        dag_path: str | Path | None = None,
        model: str = TARGET_MODEL,
        score_fn: Callable[[str, MASDAG, ExecutionTrace], float | None] | None = None,
        max_steps: int = 200,
        timeout: int = 600,
    ):
        """Initialize runner.

        Args:
            dag: A MASDAG instance to execute. Mutually exclusive with dag_path.
            dag_path: Path to a MASDAG YAML file.
            model: Default LLM model for agent nodes.
            score_fn: Optional benchmark scorer. Should return a higher-is-better score.
            max_steps: Safety limit on total node executions.
            timeout: Overall execution timeout in seconds.
        """
        if dag is not None:
            self._dag = dag
        elif dag_path is not None:
            self._dag = MASDAG.load(dag_path)
        else:
            self._dag = None

        self.model = model
        self.score_fn = score_fn
        self.max_steps = max_steps
        self.timeout = timeout

    @property
    def dag(self) -> MASDAG:
        if self._dag is None:
            raise ValueError("No DAG loaded. Pass dag or dag_path to __init__, or call set_dag().")
        return self._dag

    def set_dag(self, dag: MASDAG) -> None:
        """Replace the current DAG (e.g. after applying a repair)."""
        self._dag = dag

    def run_task(
        self,
        task_prompt: str,
        dag: MASDAG | None = None,
        output_dir: str | Path | None = None,
        timeout: int | None = None,
    ) -> MASTrace:
        """Run a single task on the DAG and return a trace.

        Args:
            task_prompt: The task description.
            dag: Optional DAG override (e.g. a repaired variant).
            output_dir: Directory to save execution artifacts (optional).
            timeout: Override default timeout.
        """
        run_dag = dag or self.dag
        executor = DAGExecutor(
            dag=run_dag,
            llm_fn=call_llm,
            model=self.model,
            max_global_steps=self.max_steps,
            timeout=timeout or self.timeout,
        )

        exec_trace = executor.run(task_prompt)

        # Save trace if output_dir is specified
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "trace.txt").write_text(exec_trace.to_trajectory(), encoding="utf-8")

        task_success = self._infer_task_success(run_dag, exec_trace)
        task_score = self._score_trace(task_prompt, run_dag, exec_trace, task_success)

        return MASTrace(
            trace_id=-1,
            mas_name=run_dag.dag_id or "OptPilot",
            llm_name=self.model,
            benchmark_name="ProgramDev",
            trajectory=exec_trace.to_trajectory(),
            task_key=task_prompt[:50],
            task_success=task_success,
            task_score=task_score,
            latency_s=exec_trace.total_duration_s,
        )

    def run_batch(
        self,
        tasks: list[str],
        dag: MASDAG | None = None,
        output_base: str | Path | None = None,
    ) -> list[MASTrace]:
        """Run multiple tasks sequentially."""
        traces = []
        for i, task in enumerate(tasks):
            out_dir = Path(output_base) / f"task_{i}" if output_base else None
            print(f"  Running task {i + 1}/{len(tasks)}: {task[:60]}...")
            trace = self.run_task(task, dag=dag, output_dir=out_dir)
            trace.trace_id = i
            traces.append(trace)
        return traces

    def _infer_task_success(self, dag: MASDAG, exec_trace: ExecutionTrace) -> bool:
        if not exec_trace.finished or bool(exec_trace.error):
            return False

        executed_nodes = {step.node_id for step in exec_trace.steps}
        success_nodes = dag.metadata.get("success_nodes", [])
        if success_nodes:
            return any(node_id in executed_nodes for node_id in success_nodes)

        if "FINAL" in dag.nodes:
            return "FINAL" in executed_nodes

        return True

    def _score_trace(
        self,
        task_prompt: str,
        dag: MASDAG,
        exec_trace: ExecutionTrace,
        task_success: bool,
    ) -> float:
        if self.score_fn is not None:
            score = self.score_fn(task_prompt, dag, exec_trace)
            if score is not None:
                return float(score)
        return 1.0 if task_success else 0.0
