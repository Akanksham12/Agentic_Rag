"""Chunking: split documents into overlapping, retrievable :class:`Chunk` units.

The :class:`Splitter` protocol is the *swap seam* — the ingestion pipeline
depends only on this interface, so the strategy can be changed (e.g. to a
semantic splitter) without touching anything downstream.

:class:`RecursiveCharacterSplitter` is the default. It tries a prioritised list
of separators (paragraph -> line -> sentence -> word -> character), keeping
related text together where possible while bounding chunk size, and adds a
configurable overlap so context isn't lost at boundaries.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.ingestion.models import Chunk, Document

# Separators tried in order: keep paragraphs whole, then lines, then sentences,
# then words, finally fall back to raw characters so size is always bounded.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@runtime_checkable
class Splitter(Protocol):
    """Strategy interface for turning a :class:`Document` into chunks."""

    def split(self, document: Document) -> list[Chunk]:
        """Split ``document`` into an ordered list of :class:`Chunk` objects."""
        ...


class RecursiveCharacterSplitter:
    """Recursively split text on a prioritised separator list, with overlap.

    Args:
        chunk_size: Maximum size (characters) of an emitted chunk.
        chunk_overlap: Characters of trailing context repeated at the start of
            the next chunk. Must be smaller than ``chunk_size``.
        separators: Ordered separators to try; defaults to
            :data:`DEFAULT_SEPARATORS`.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Chunk]:
        """Split a document, attaching inherited metadata and stable ids."""
        text = document.content.strip()
        if not text:
            return []

        pieces = self._split_text(text, self.separators)
        source = document.metadata.get("source", "unknown")
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    content=piece,
                    metadata={**document.metadata, "chunk_index": index},
                    id=f"{source}::{index}",
                )
            )
        return chunks

    # -- internals -----------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split ``text`` using the highest-priority separator that
        appears in it, recursing into any piece still larger than the limit."""
        final: list[str] = []

        # Choose the first separator present in the text; "" means char split.
        separator = separators[-1]
        remaining: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                remaining = separators[i + 1 :]
                break

        splits = list(text) if separator == "" else text.split(separator)

        # Merge small splits up to chunk_size; recurse into oversized ones.
        good: list[str] = []
        for piece in splits:
            if len(piece) < self.chunk_size:
                good.append(piece)
                continue
            if good:
                final.extend(self._merge(good, separator))
                good = []
            if remaining:
                final.extend(self._split_text(piece, remaining))
            else:
                final.append(piece)
        if good:
            final.extend(self._merge(good, separator))
        return final

    def _merge(self, splits: list[str], separator: str) -> list[str]:
        """Greedily merge splits into <= chunk_size chunks, carrying overlap.

        After emitting a chunk, the leading splits are dropped only until the
        retained tail is within ``chunk_overlap`` — that tail becomes the start
        of the next chunk, giving continuity across boundaries.
        """
        sep_len = len(separator)
        chunks: list[str] = []
        current: list[str] = []
        total = 0

        for piece in splits:
            addition = len(piece) + (sep_len if current else 0)
            if current and total + addition > self.chunk_size:
                merged = separator.join(current).strip()
                if merged:
                    chunks.append(merged)
                # Drop from the front until the tail fits the overlap budget.
                while total > self.chunk_overlap and current:
                    removed = current.pop(0)
                    total -= len(removed) + (sep_len if current else 0)
            current.append(piece)
            total += len(piece) + (sep_len if len(current) > 1 else 0)

        merged = separator.join(current).strip()
        if merged:
            chunks.append(merged)
        return chunks
