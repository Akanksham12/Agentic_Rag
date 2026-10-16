"""Persistent vector store backed by Chroma.

Chroma was chosen over FAISS because it ships persistence, metadata storage,
and a document store in one package — FAISS would require bolting those on
ourselves. We supply pre-computed embeddings (from :class:`~app.embeddings.
embedder.Embedder`) rather than using Chroma's built-in embedder, so the
embedding model stays a single owned concern.

The collection is configured for cosine space; reported ``score`` is one minus
Chroma's cosine distance (higher = more similar).
"""

from __future__ import annotations

import re
from collections import Counter
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
            "chroma_ready",
            persist_dir=self.persist_dir,
            collection=self.collection_name,
            distance_metric=(self._collection.metadata or {}).get("hnsw:space", "l2"),
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
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to a query embedding."""
        top_k = top_k or get_settings().top_k
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        matches = [
            SearchResult(content=doc, metadata=meta or {}, score=1.0 - float(dist))
            for doc, meta, dist in zip(docs, metas, distances, strict=False)
        ]
        logger.info(
            "vector_search_results",
            distance_metric=(self._collection.metadata or {}).get("hnsw:space", "l2"),
            results=[
                {
                    "rank": rank,
                    "source": match.source,
                    "distance": float(distance),
                    "score": match.score,
                }
                for rank, (match, distance) in enumerate(
                    zip(matches, distances, strict=True), start=1
                )
            ],
        )
        for rank, (match, distance) in enumerate(zip(matches, distances, strict=True), start=1):
            logger.info(
                "retrieved_chunk_debug",
                rank=rank,
                source=match.source,
                distance=float(distance),
                score=match.score,
                content=match.content[:800],
            )
        return matches

    def keyword_search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Keyword fallback used when semantic similarity is weak or noisy."""
        top_k = top_k or get_settings().top_k
        if self.count() == 0:
            return []

        chunk_data = self._collection.get(include=["documents", "metadatas"])
        docs = chunk_data.get("documents", [])
        metas = chunk_data.get("metadatas", [])
        tokens = {
            token.lower()
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 2
        }
        if not tokens:
            return []

        scored: list[tuple[float, SearchResult]] = []
        for doc, meta in zip(docs, metas, strict=False):
            if not doc:
                continue
            doc_tokens = Counter(re.findall(r"[a-z0-9]+", doc.lower()))
            overlap = sum(doc_tokens[token] for token in tokens if token in doc_tokens)
            if overlap <= 0:
                continue
            score = overlap / max(1, len(tokens))
            scored.append((score, SearchResult(content=doc, metadata=meta or {}, score=score)))

        scored.sort(key=lambda item: item[0], reverse=True)
        matches = [match for _, match in scored[:top_k]]
        logger.info("keyword_search_results", query=query, hits=len(matches))
        return matches

    def count(self) -> int:
        """Number of chunks currently stored."""
        return int(self._collection.count())
