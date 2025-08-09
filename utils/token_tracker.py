#!/usr/bin/env python3
"""
Token usage tracker for OpenAI API calls across the app lifecycle.
Thread-safe, opt-in, no-ops if usage isn't available in responses.
"""
from __future__ import annotations
import threading
from typing import Optional, Any, Dict

class _TokenTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0

    def add(self, prompt: int = 0, completion: int = 0, total: Optional[int] = None) -> None:
        with self._lock:
            self._prompt_tokens += int(prompt or 0)
            self._completion_tokens += int(completion or 0)
            if total is not None:
                self._total_tokens += int(total or 0)
            else:
                self._total_tokens += int(prompt or 0) + int(completion or 0)

    def get_totals(self) -> Dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
            }


token_tracker = _TokenTracker()


def record_openai_usage_from_response(resp: Any) -> None:
    """Best-effort extraction of usage from OpenAI SDK response objects."""
    try:
        usage = getattr(resp, "usage", None)
        if not usage:
            return
        # OpenAI SDK v1 returns an object with attributes
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        # Fallback if dict-like
        if prompt is None and isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            total = usage.get("total_tokens")
        token_tracker.add(prompt or 0, completion or 0, total)
    except Exception:
        # Never break the app due to usage parsing issues
        return 