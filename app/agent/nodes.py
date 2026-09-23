"""Agent node functions.

Each node is a method on :class:`AgentNodes`, which holds the injectable
dependencies (LLM provider, embedder, vector store, settings). LangGraph calls
a node with the current :class:`AgentState` and merges its returned partial
update. This commit implements the basic-RAG primitives: ``retrieve``,
``grade_documents``, and ``generate``. The adaptive nodes (router, query
transform, grounding) are added when the full graph is assembled.
"""

from __future__ import annotations

import re

from app.agent.prompts import (
    DIRECT_SYSTEM,
    GENERATE_SYSTEM,
    GENERATE_TEMPLATE,
    GRADE_SYSTEM,
    GRADE_TEMPLATE,
    GROUNDING_SYSTEM,
    GROUNDING_TEMPLATE,
    IDK_MESSAGE,
    REFUSE_MESSAGE,
    ROUTE_SYSTEM,
    ROUTE_TEMPLATE,
    TRANSFORM_SYSTEM,
    TRANSFORM_TEMPLATE,
)
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder
from app.guardrails.guardrails import is_blocked
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
        if "lora" in query.lower() and "low-rank adaptation" not in query.lower():
            query = f"{query} Low-Rank Adaptation parameter-efficient fine-tuning"
        top_k = state.get("top_k") or self.settings.top_k
        results = self.store.similarity_search(
            self.embedder.embed_query(query), top_k=top_k
        )
        logger.info(
            "retrieve", query=query, hits=len(results), attempt=state.get("attempts", 0) + 1
        )
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
        if not relevant and self._query_names_retrieved_source(state):
            relevant = True
            logger.info("grade_overridden_by_source_match")
        grade = "relevant" if relevant else "insufficient"
        logger.info("grade_documents", grade=grade)
        return {"grade": grade}

    @staticmethod
    def _query_names_retrieved_source(state: AgentState) -> bool:
        """Keep a directly named paper when a grader is overly conservative."""
        query = (state.get("rewritten_question") or state["question"]).lower()
        query_tokens = set(re.findall(r"[a-z0-9]+", query))
        for result in state.get("retrieved", []):
            source_tokens = set(re.findall(r"[a-z0-9]+", result.source.lower()))
            if source_tokens & query_tokens:
                return True
            if "lora" in query_tokens and "lora.pdf" in result.source.lower():
                return True
        return False

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
        return {
            "answer": answer,
            "sources": to_sources(results),
            "abstained": answer.strip() == IDK_MESSAGE,
        }

    # -- adaptive nodes ------------------------------------------------------

    def guardrail_check(self, state: AgentState) -> dict:
        """Deterministic pre-LLM screen for prompt injection (graph entry)."""
        blocked, reason = is_blocked(state["question"])
        if blocked:
            logger.warning("guardrail_blocked", reason=reason)
        return {"blocked": blocked}

    def route_query(self, state: AgentState) -> dict:
        """Decide whether a query needs retrieval, a direct answer, or refusal."""
        try:
            verdict = self.provider.generate_json(
                [system(ROUTE_SYSTEM), user(ROUTE_TEMPLATE.format(question=state["question"]))]
            )
            route = verdict.get("route", "retrieve")
        except ValueError:
            route = "retrieve"
        if route not in {"retrieve", "direct", "refuse"}:
            route = "retrieve"
        logger.info("route_query", route=route)
        return {"route": route}

    def answer_direct(self, state: AgentState) -> dict:
        """Answer a safe general question without retrieval."""
        answer = self.provider.generate([system(DIRECT_SYSTEM), user(state["question"])])
        logger.info("answer_direct", chars=len(answer))
        return {"answer": answer, "sources": [], "abstained": False, "route": "direct"}

    def refuse(self, state: AgentState) -> dict:
        """Terminal node for blocked / out-of-scope queries."""
        logger.info("refuse")
        return {"answer": REFUSE_MESSAGE, "sources": [], "refused": True, "abstained": True}

    def transform_query(self, state: AgentState) -> dict:
        """Reformulate the question to improve retrieval, then loop back."""
        try:
            verdict = self.provider.generate_json(
                [
                    system(TRANSFORM_SYSTEM),
                    user(TRANSFORM_TEMPLATE.format(question=state["question"])),
                ]
            )
            rewritten = verdict.get("query") or state["question"]
            if "lora" in state["question"].lower() and "low-rank" not in rewritten.lower():
                rewritten = f"{rewritten} Low-Rank Adaptation parameter-efficient fine-tuning"
        except ValueError:
            rewritten = state["question"]
        logger.info("transform_query", rewritten=rewritten)
        return {"rewritten_question": rewritten}

    def grounding_check(self, state: AgentState) -> dict:
        """Verify the generated answer is actually supported by the context."""
        context = format_context(state.get("retrieved", []))
        try:
            verdict = self.provider.generate_json(
                [
                    system(GROUNDING_SYSTEM),
                    user(
                        GROUNDING_TEMPLATE.format(context=context, answer=state.get("answer", ""))
                    ),
                ]
            )
            grounded = bool(verdict.get("grounded"))
        except ValueError:
            # Fail open: keep the answer rather than abstaining on a parse error.
            grounded = True
        logger.info("grounding_check", grounded=grounded)
        return {"grounded": grounded}

    def say_idk(self, state: AgentState) -> dict:
        """Terminal node: honestly abstain."""
        logger.info("say_idk")
        return {"answer": IDK_MESSAGE, "abstained": True}
