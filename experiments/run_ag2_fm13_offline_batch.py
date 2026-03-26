"""Run a fixed offline inspection batch for AG2 FM-1.3.

Usage:
    python -m experiments.run_ag2_fm13_offline_batch
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR
from experiments.offline_pipeline import run_offline_pipeline


BATCH_SIZE = 8


if __name__ == "__main__":
    run_offline_pipeline(
        mas_name="AG2",
        fm_id="1.3",
        dag_path=str(DAG_DIR / "chatdev.yaml"),
        benchmark=None,
        max_traces=BATCH_SIZE,
        use_wandb=False,
    )
