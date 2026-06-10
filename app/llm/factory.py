"""Provider selection — the one place that maps env config to an implementation.

Callers ask for :func:`get_provider` and receive an :class:`LLMProvider`; they
never import a concrete provider directly. Swapping local<->cloud is therefore
a single environment variable (``LLM_PROVIDER``) with zero code changes.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base import LLMProvider
from app.logging import get_logger

logger = get_logger(__name__)


def get_provider(name: str | None = None) -> LLMProvider:
    """Return an LLM provider instance.

    Args:
        name: ``"ollama"`` or ``"groq"``. Defaults to the configured
            ``LLM_PROVIDER``.

    Raises:
        ValueError: If ``name`` is not a known provider.
    """
    name = (name or get_settings().llm_provider).lower()
    logger.info("selecting_llm_provider", provider=name)

    if name == "ollama":
        from app.llm.ollama_provider import OllamaProvider

        return OllamaProvider()
    if name == "groq":
        from app.llm.groq_provider import GroqProvider

        return GroqProvider()
    raise ValueError(f"Unknown LLM provider {name!r}; expected 'ollama' or 'groq'.")


def get_judge_provider() -> LLMProvider:
    """Return the provider used as the eval LLM-judge.

    Prefers ``JUDGE_PROVIDER`` (default ``groq`` for credible scores). Falls
    back to the main ``LLM_PROVIDER`` if Groq is configured but has no key, so
    ``make eval`` still runs fully locally when no cloud key is present.
    """
    settings = get_settings()
    if settings.judge_provider == "groq" and not settings.groq_api_key:
        logger.warning("judge_falling_back_to_local", reason="no GROQ_API_KEY")
        return get_provider(settings.llm_provider)
    return get_provider(settings.judge_provider)
