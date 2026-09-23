"""Persistent vector store backed by Chroma.

Chroma was chosen over FAISS because it ships persistence, metadata storage,
and a document store in one package — FAISS would require bolting those on
ourselves. We supply pre-computed embeddings (from :class:`~app.embeddings.
embedder.Embedder`) rather than using Chroma's built-in embedder, so the
embedding model stays a single owned concern.

The collection is configured for cosine space; reported ``score`` is cosine
similarity in ``[0, 1]`` (higher = more similar).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.ingestion.models import Chunk
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """One retrieved chunk plus its similarity to the query."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))


class ChromaStore:
    """Thin wrapper over a persistent Chroma collection.

    Args:
        persist_dir: On-disk directory for the Chroma database.
        collection_name: Name of the collection to use/create.
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        import chromadb

        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_dir
        self.collection_name = collection_name or settings.collection_name
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        # Cosine space so distances map cleanly to a [0, 1] similarity score.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_ready", persist_dir=self.persist_dir, collection=self.collection_name
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks with their pre-computed embeddings.

        Using ``upsert`` (keyed on the chunk's stable id) makes re-ingestion
        idempotent rather than creating duplicates.

        Returns:
            The number of chunks written.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )
        logger.info("chunks_upserted", count=len(chunks))
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        source: str | None = None,
    ) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to a query embedding."""
        top_k = top_k or get_settings().top_k
        if self.count() == 0:
            return []
        where = {"source": source} if source else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            SearchResult(content=doc, metadata=meta or {}, score=1.0 - float(dist))
            for doc, meta, dist in zip(docs, metas, distances, strict=False)
        ]

    def count(self) -> int:
        """Number of chunks currently stored."""
        return int(self._collection.count())
