"""Core data types shared across the ingestion pipeline.

Keeping these in one tiny module (rather than scattering dicts) gives the rest
of the system typed, self-documenting units to pass around:

- :class:`Document` — a whole loaded file (text + provenance metadata).
- :class:`Chunk` — one retrievable slice of a document, with a stable id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single loaded source file.

    Attributes:
        content: The extracted plain text of the whole file.
        metadata: Provenance, e.g. ``{"source": "attention.pdf",
            "type": "pdf", "path": "data/corpus/attention.pdf"}``.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Human-readable origin of this document (filename or ``"unknown"``)."""
        return str(self.metadata.get("source", "unknown"))


@dataclass
class Chunk:
    """A retrievable slice of a :class:`Document`.

    Attributes:
        content: The chunk text.
        metadata: Inherited document metadata plus chunk-local fields such as
            ``chunk_index``.
        id: Stable, unique identifier used as the vector-store primary key.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    @property
    def source(self) -> str:
        """Human-readable origin of this chunk (filename or ``"unknown"``)."""
        return str(self.metadata.get("source", "unknown"))
