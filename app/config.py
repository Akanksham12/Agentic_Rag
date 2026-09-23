"""Centralised, typed application configuration.

Every tunable knob in the system is declared here and loaded from environment
variables / a local ``.env`` file via ``pydantic-settings``. No other module
should read ``os.environ`` directly — they import :func:`get_settings` instead.
This gives us one validated source of truth and a single place to document
configuration (mirrored in ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from the environment / ``.env``.

    Field names map to upper-case environment variables (e.g. ``top_k`` <-
    ``TOP_K``). Defaults are chosen so the system runs fully locally and free
    with zero configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider selection ---------------------------------------------
    llm_provider: Literal["ollama", "groq"] = "ollama"

    # Local Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Groq cloud (free tier). Key is read from the environment only.
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Embeddings (always local) ------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Vector store -------------------------------------------------------
    chroma_dir: str = "./.chroma"
    collection_name: str = "documents"

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval / agent behaviour ----------------------------------------
    top_k: int = 4
    max_attempts: int = 2
    enable_grounding_check: bool = True

    # --- Corpus -------------------------------------------------------------
    corpus_dir: str = "./data/corpus"

    # --- Evaluation ---------------------------------------------------------
    judge_provider: Literal["ollama", "groq"] = "groq"

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"

    # --- Metadata -----------------------------------------------------------
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the ``.env`` file and environment are read once per process.
    """
    return Settings()
