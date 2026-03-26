"""Experiment tracking - JSON local + Weights & Biases dual track."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from optpilot.config import RESULTS_DIR


class Tracker:
    """Dual-track experiment tracker: local JSON + W&B."""

    def __init__(self, exp_name: str, use_wandb: bool = False, wandb_project: str = "optpilot"):
        self.exp_name = exp_name
        self.results: list[dict] = []
        self._wandb_run = None

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={"exp_name": exp_name},
                )
            except Exception as e:
                print(f"W&B init failed: {e}. Continuing with local tracking only.")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to both local and W&B."""
        entry = {"timestamp": datetime.now().isoformat(), **metrics}
        if step is not None:
            entry["step"] = step
        self.results.append(entry)

        if self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def log_result(self, result: Any) -> None:
        """Log a single result (dataclass or dict)."""
        if hasattr(result, "__dataclass_fields__"):
            self.results.append(asdict(result))
        else:
            self.results.append(result)

    def save_local(self, filename: str | None = None) -> Path:
        """Save results to local JSON file."""
        if filename is None:
            filename = f"{self.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = RESULTS_DIR / filename
        path.write_text(json.dumps(self.results, ensure_ascii=False, indent=2))
        print(f"Results saved to {path}")
        return path

    def log_artifact(self, path: str, name: str) -> None:
        """Upload artifact to W&B."""
        if self._wandb_run:
            import wandb
            artifact = wandb.Artifact(name, type="result")
            artifact.add_file(path)
            self._wandb_run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish tracking, save local and close W&B."""
        self.save_local()
        if self._wandb_run:
            import wandb
            wandb.finish()
