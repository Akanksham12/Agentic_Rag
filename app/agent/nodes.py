"""Agent node functions.

Each node is a method on :class:`AgentNodes`, which holds the injectable
dependencies (LLM provider, embedder, vector store, settings). LangGraph calls
a node with the current :class:`AgentState` and merges its returned partial
update. This commit implements the basic-RAG primitives: ``retrieve``,
``grade_documents``, and ``generate``. The adaptive nodes (router, query
transform, grounding) are added when the full graph is assembled.
"""

from __future__ import annotations

from app.agent.prompts import (
    GENERATE_SYSTEM,
    GENERATE_TEMPLATE,
    GRADE_SYSTEM,
    GRADE_TEMPLATE,
)
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder
from app.llm.base import LLMProvider, system, user
from app.llm.factory import get_provider
from app.logging import get_logger
from app.vectorstore.chroma_store import ChromaStore, SearchResult

logger = get_logger(__name__)


def format_context(results: list[SearchResult]) -> str:
    """Render retrieved chunks as numbered, source-tagged passages for citation."""
    if not results:
        return "(no context retrieved)"
    return "\n\n".join(
        f"[{i}] (source: {r.source})\n{r.content}" for i, r in enumerate(results, start=1)
    )


def to_sources(results: list[SearchResult]) -> list[dict]:
    """Serialise retrieved chunks into citable provenance for the API response."""
    return [
        {"source": r.source, "score": round(r.score, 3), "snippet": r.content[:200].strip()}
        for r in results
    ]


class AgentNodes:
    """Container for agent node implementations and their dependencies.

    Args:
        provider: LLM provider (defaults to the configured one).
        embedder: Embedding model wrapper.
        store: Vector store.
        settings: Application settings.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        embedder: Embedder | None = None,
        store: ChromaStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or get_provider()
        self.embedder = embedder or Embedder()
        self.store = store or ChromaStore()

    # -- basic RAG primitives -----------------------------------------------

    def retrieve(self, state: AgentState) -> dict:
        """Embed the (possibly rewritten) query and fetch top-k chunks."""
        query = state.get("rewritten_question") or state["question"]
        results = self.store.similarity_search(
            self.embedder.embed_query(query), top_k=self.settings.top_k
        )
        logger.info("retrieve", query=query, hits=len(results), attempt=state.get("attempts", 0) + 1)
        return {"retrieved": results, "attempts": state.get("attempts", 0) + 1}

    def grade_documents(self, state: AgentState) -> dict:
        """Self-check: do the retrieved chunks actually answer the question?"""
        context = format_context(state.get("retrieved", []))
        try:
            verdict = self.provider.generate_json(
                [
                    system(GRADE_SYSTEM),
                    user(GRADE_TEMPLATE.format(question=state["question"], context=context)),
                ]
            )
            relevant = bool(verdict.get("relevant"))
        except ValueError:
            # If the model returns unparseable output, fail open to "relevant"
            # so we still attempt an answer rather than wrongly abstaining.
            logger.warning("grade_parse_failed", falling_back="relevant")
            relevant = True
        grade = "relevant" if relevant else "insufficient"
        logger.info("grade_documents", grade=grade)
        return {"grade": grade}

    def generate(self, state: AgentState) -> dict:
        """Generate an answer grounded in the retrieved context, with citations."""
        results = state.get("retrieved", [])
        context = format_context(results)
        answer = self.provider.generate(
            [
                system(GENERATE_SYSTEM),
                user(GENERATE_TEMPLATE.format(context=context, question=state["question"])),
            ]
        )
        logger.info("generate", chars=len(answer), sources=len(results))
        return {"answer": answer, "sources": to_sources(results), "abstained": False}
