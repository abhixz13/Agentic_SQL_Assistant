import logging
from typing import Any, Dict, List

from openai import OpenAI

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
        client = cls._get_client()
        response = client.chat.completions.create(model=model, messages=messages, **opts)
        usage = getattr(response, "usage", None)
        if usage:
            logger.info(
                "LLM usage - prompt: %s, completion: %s, total: %s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        return response
