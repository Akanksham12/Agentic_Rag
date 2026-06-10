"""Structured logging setup.

We emit structured (JSON) logs rather than ``print`` statements so that, once
deployed, logs are machine-parseable and each event carries context as
key/value pairs (e.g. ``route_taken``, ``attempts``, ``doc_count``). This is
configured once at process start via :func:`configure_logging`; modules obtain
a bound logger with :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config import get_settings

_configured = False


def configure_logging() -> None:
    """Configure structlog + stdlib logging. Idempotent.

    Renders JSON to stdout. The log level is taken from settings
    (``LOG_LEVEL``). Safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
