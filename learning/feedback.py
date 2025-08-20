import json
import os
from datetime import datetime
from typing import Any, Optional


class FeedbackCollector:
    """Collects interaction data for training."""

    LOG_FILE = os.path.join("data", "feedback", "training_examples.jsonl")

    @classmethod
    def log_interaction(
        cls,
        query: Optional[str],
        schema: Any,
        generated_sql: str,
        error: Optional[str],
        corrected_sql: Optional[str],
    ) -> None:
        """Append a structured record of an interaction to the log file."""
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "schema": schema,
            "generated_sql": generated_sql,
            "error": error,
            "corrected_sql": corrected_sql,
        }
        with open(cls.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
