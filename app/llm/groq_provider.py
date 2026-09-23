"""Groq cloud provider (free tier) — used for the hosted demo.

Groq exposes an OpenAI-style chat API and is very fast, which makes the
deployed demo feel snappy. The API key is read from the environment only and
is never logged or hard-coded.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base import LLMProvider, Message
from app.logging import get_logger

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider backed by Groq's hosted models.

    Args:
        model: Override the configured ``GROQ_MODEL``.
        api_key: Override the configured ``GROQ_API_KEY`` (env by default).

    Raises:
        ValueError: If no API key is available.
    """

    name = "groq"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from groq import Groq

        settings = get_settings()
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it in the environment to use the Groq provider."
            )
        self.model = model or settings.groq_model
        self._client = Groq(api_key=key)

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Call Groq's chat completions API and return the message text."""
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**request)
        content = resp.choices[0].message.content or ""
        logger.debug("groq_generate", model=self.model, chars=len(content))
        return content.strip()
