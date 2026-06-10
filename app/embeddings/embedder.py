"""Local text embeddings via sentence-transformers (all-MiniLM-L6-v2).

Embeddings are intentionally a separate concern from text generation: they are
*always* computed locally and free, regardless of which LLM provider answers
questions. That separation is what keeps even the hosted demo at $0.

The model is loaded lazily on first use so importing this module (e.g. in
tests) stays cheap.
"""

from __future__ import annotations

from app.config import get_settings
from app.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Wraps a sentence-transformers model to embed documents and queries.

    Args:
        model_name: HuggingFace model id. Defaults to the configured
            ``EMBEDDING_MODEL`` (all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embedding_model
        self._model = None  # lazy

    @property
    def model(self):
        """The underlying SentenceTransformer, loaded on first access."""
        if self._model is None:
            # Imported lazily to avoid the heavy torch import at module load.
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality (384 for MiniLM-L6)."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Vectors are L2-normalised for cosine.

        Args:
            texts: Document/chunk strings.

        Returns:
            One embedding (list of floats) per input string.
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string, returning one normalised vector."""
        return self.embed_texts([text])[0]
