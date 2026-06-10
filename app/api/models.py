"""Pydantic request/response models for the API.

These typed schemas drive request validation and the auto-generated OpenAPI
docs at ``/docs`` — the contract is the code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health and active configuration."""

    status: str = Field(examples=["ok"])
    version: str
    provider: str = Field(description="Active LLM provider", examples=["ollama", "groq"])
    model: str = Field(description="Active generation model")
    doc_count: int = Field(description="Chunks currently in the vector store")


class QueryRequest(BaseModel):
    """A question for the agent."""

    question: str = Field(min_length=1, examples=["What problem does LoRA address?"])
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Optional retrieval depth override"
    )


class Source(BaseModel):
    """One citable retrieved passage."""

    source: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """The agent's answer plus its decision trace."""

    answer: str
    sources: list[Source]
    route_taken: str = Field(description='"retrieve" | "direct" | "refused"')
    attempts: int = Field(description="Retrieval attempts (incl. reformulations)")
    grounded: bool | None = Field(description="Grounding-gate verdict, if run")
    abstained: bool


class IngestResponse(BaseModel):
    """Result of an ingestion request."""

    ingested_files: int
    chunks_created: int
    doc_count: int
