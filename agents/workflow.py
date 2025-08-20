"""High level workflow interface using the configurable :class:`Orchestrator`."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from core.orchestrator import Orchestrator


class SQLWorkflow:
    """Thin wrapper around :class:`Orchestrator` for backward compatibility."""

    def __init__(self, db_path: str, config_path: str | None = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "core" / "workflow.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Inject the runtime database path into configured steps that require it
        for step_name in ("schema_retriever", "executor"):
            params = config.get("steps", {}).get(step_name, {}).setdefault("params", {})
            params["db_path"] = db_path

        self.orchestrator = Orchestrator(config)

    def run(self, question: str, viz_options: Optional[Dict[str, any]] = None):
        """Execute the orchestrated workflow for ``question``."""
        return self.orchestrator.run(question, viz_options)
