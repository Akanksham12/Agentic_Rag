"""The LLM provider interface.

``LLMProvider`` is the single seam through which the whole system talks to a
language model. Concrete providers (Ollama, Groq) implement only :meth:`generate`
(raw text). :meth:`generate_json` — used by the agent's decision nodes to get
structured verdicts — is implemented *once* here on the base class in terms of
``generate``, so the JSON-coaxing/parsing logic isn't duplicated per provider.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    """A single chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


def system(content: str) -> Message:
    """Build a system message."""
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    """Build a user message."""
    return {"role": "user", "content": content}


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    Attributes:
        name: Short provider id (e.g. ``"ollama"``, ``"groq"``).
        model: The model identifier in use.
    """

    name: str = "base"
    model: str = ""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text completion for ``messages``."""
        raise NotImplementedError

    def generate_json(
        self,
        messages: list[Message],
        temperature: float = 0.0,
    ) -> dict:
        """Return a parsed JSON object from the model.

        Appends an instruction to emit JSON only, then robustly extracts the
        first JSON object from the response (tolerating code fences or prose).

        Raises:
            ValueError: If no JSON object can be parsed from the output.
        """
        guided = [
            *messages,
            system("Respond with a single valid JSON object only. No prose, no code fences."),
        ]
        raw = self.generate(guided, temperature=temperature)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Best-effort extraction of a JSON object from model output."""
        text = text.strip()
        # Strip ```json ... ``` fences if present.
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fall back to the first balanced-looking {...} span.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from model output: {text[:200]!r}")
