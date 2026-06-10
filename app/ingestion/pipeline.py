"""The ingestion pipeline: load -> chunk -> embed -> store.

This is the composition root for ingestion. It wires the four stages together
but depends only on their interfaces, so any stage can be swapped (a different
splitter, embedder, or store) by injecting it into the constructor — which is
exactly how tests and the eval harness exercise it in isolation.
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import Embedder
from app.ingestion.chunking import RecursiveCharacterSplitter, Splitter
from app.ingestion.loaders import load_directory, load_file
from app.ingestion.models import Document
from app.logging import get_logger
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


class IngestionPipeline:
    """Coordinates loading, chunking, embedding, and storage.

    All collaborators default to the configured implementations but can be
    injected for testing or to swap strategies.

    Args:
        splitter: Chunking strategy (defaults to a configured
            :class:`RecursiveCharacterSplitter`).
        embedder: Embedding model wrapper (defaults to :class:`Embedder`).
        store: Vector store (defaults to :class:`ChromaStore`).
    """

    def __init__(
        self,
        splitter: Splitter | None = None,
        embedder: Embedder | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        settings = get_settings()
        self.splitter = splitter or RecursiveCharacterSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.embedder = embedder or Embedder()
        self.store = store or ChromaStore()

    def ingest_documents(self, documents: list[Document]) -> dict[str, int]:
        """Chunk, embed, and store a batch of already-loaded documents.

        Returns:
            Stats: ``{"files", "chunks", "doc_count"}`` where ``doc_count`` is
            the total chunks in the store after this run.
        """
        chunks = [chunk for doc in documents for chunk in self.splitter.split(doc)]
        if not chunks:
            logger.warning("no_chunks_produced", files=len(documents))
            return {"files": len(documents), "chunks": 0, "doc_count": self.store.count()}

        embeddings = self.embedder.embed_texts([c.content for c in chunks])
        self.store.add_chunks(chunks, embeddings)
        stats = {"files": len(documents), "chunks": len(chunks), "doc_count": self.store.count()}
        logger.info("ingestion_complete", **stats)
        return stats

    def ingest_paths(self, paths: list[str | Path]) -> dict[str, int]:
        """Load and ingest specific files by path."""
        return self.ingest_documents([load_file(p) for p in paths])

    def ingest_directory(self, directory: str | Path) -> dict[str, int]:
        """Load and ingest every supported file in a directory."""
        return self.ingest_documents(load_directory(directory))
