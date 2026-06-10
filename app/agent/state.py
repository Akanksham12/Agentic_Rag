"""The agent's shared state.

LangGraph threads a single mutable state dict through every node; each node
returns a partial update that is merged in. Keeping the schema in one typed
place makes the data flow explicit and self-documenting.
"""

from __future__ import annotations

from typing import TypedDict

from app.vectorstore.chroma_store import SearchResult


class AgentState(TypedDict, total=False):
    """State passed between agent nodes.

    Fields are optional (``total=False``) because they are filled in as the
    graph progresses.

    Attributes:
        question: The original user question.
        rewritten_question: A reformulated query (set by ``transform_query``).
        retrieved: Chunks returned by the latest retrieval.
        attempts: Number of retrieval attempts made so far.
        blocked: Whether the deterministic guardrail blocked the query.
        route: Routing verdict — ``"retrieve" | "direct" | "refuse"``.
        grade: Document-grading verdict — ``"relevant" | "insufficient"``.
        answer: The final answer text.
        grounded: Whether the grounding check passed.
        abstained: Whether the agent declined to answer ("I don't know").
        refused: Whether a guardrail/scope check blocked the query.
        sources: Citable provenance for the answer (for the API response).
    """

    question: str
    rewritten_question: str
    top_k: int
    retrieved: list[SearchResult]
    attempts: int
    blocked: bool
    route: str
    grade: str
    answer: str
    grounded: bool
    abstained: bool
    refused: bool
    sources: list[dict]
