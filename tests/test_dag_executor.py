from __future__ import annotations

from collections import Counter

from optpilot.dag.core import DAGEdge, DAGNode, MASDAG
from optpilot.dag.executor import DAGExecutor


def test_loop_counter_only_triggers_exit_after_max_iterations() -> None:
    dag = MASDAG(
        dag_id="loop_test",
        nodes={
            "start": DAGNode(node_id="start", node_type="literal", prompt="seed"),
            "worker": DAGNode(node_id="worker", node_type="passthrough"),
            "loop": DAGNode(
                node_id="loop",
                node_type="loop_counter",
                config={"max_iterations": 3},
            ),
            "done": DAGNode(node_id="done", node_type="literal", prompt="done"),
        },
        edges=[
            DAGEdge(source="start", target="worker"),
            DAGEdge(source="worker", target="loop"),
            DAGEdge(source="loop", target="worker"),
            DAGEdge(source="loop", target="done"),
        ],
        metadata={"start": ["start"]},
    )

    executor = DAGExecutor(dag=dag, llm_fn=lambda *args, **kwargs: "")
    trace = executor.run("ignored")

    counts = Counter(step.node_id for step in trace.steps)

    assert trace.finished is True
    assert trace.error == ""
    assert counts["worker"] == 3
    assert counts["done"] == 1
    assert [step.node_id for step in trace.steps][-2:] == ["loop", "done"]
