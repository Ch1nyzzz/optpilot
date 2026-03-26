"""Runner 注册与工厂函数。"""

from __future__ import annotations

from optpilot.dag.core import MASDAG
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.runner import OptPilotRunner


def create_runner(dag: MASDAG | None = None, **kwargs) -> MASRunner:
    """创建 OptPilotRunner 实例。"""
    return OptPilotRunner(dag=dag, **kwargs)
