"""Local Ollama provider (default, free).

Talks to a local Ollama server over its HTTP chat API. No API key, no network
egress, no cost — the default for development.
"""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.llm.base import LLMProvider, Message
from app.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server.

    Args:
        model: Override the configured ``OLLAMA_MODEL``.
        base_url: Override the configured ``OLLAMA_BASE_URL``.
    """

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Call Ollama's ``/api/chat`` endpoint and return the message text."""
        options: dict[str, float | int] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        logger.debug("ollama_generate", model=self.model, chars=len(content))
        return content.strip()
