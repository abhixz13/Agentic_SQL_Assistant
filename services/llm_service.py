import logging
from typing import Any, Dict, List

from openai import OpenAI

from .model_registry import get_latest_model

logger = logging.getLogger(__name__)


class LLMService:
    """Central service for invoking LLM models and logging usage."""

    _client: OpenAI | None = None

    @classmethod
    def _get_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI()
        return cls._client

    @classmethod
    def invoke(cls, model: str, messages: List[Dict[str, str]], **opts: Any):
        """Invoke a chat completion model.

        If a fine‑tuned variant of ``model`` has been registered via the
        :mod:`services.model_registry`, it will be used instead of the base model.
        """

        target_model = get_latest_model(model) or model

        client = cls._get_client()
        response = client.chat.completions.create(
            model=target_model, messages=messages, **opts
        )
        usage = getattr(response, "usage", None)
        if usage:
            logger.info(
                "LLM usage - prompt: %s, completion: %s, total: %s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        return response
