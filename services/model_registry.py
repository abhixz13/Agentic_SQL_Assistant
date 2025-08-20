"""Simple on-disk registry for fine‑tuned models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


def _read_registry() -> List[Dict]:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text())
    return []


def _write_registry(entries: List[Dict]) -> None:
    _REGISTRY_PATH.write_text(json.dumps(entries, indent=2))


def register_model(base_model: str, model_path: str, metrics: Optional[Dict] = None) -> Dict:
    """Record a newly fine‑tuned model in the registry."""

    entries = _read_registry()
    entry = {
        "base_model": base_model,
        "model_path": model_path,
        "metrics": metrics or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    entries.append(entry)
    _write_registry(entries)
    return entry


def get_latest_model(base_model: str) -> Optional[str]:
    """Return the most recently registered model for ``base_model``.

    If no fine‑tuned models have been registered, ``None`` is returned.
    """

    entries = [e for e in _read_registry() if e["base_model"] == base_model]
    if not entries:
        return None
    latest = max(entries, key=lambda e: e["timestamp"])
    return latest["model_path"]
