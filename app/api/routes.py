"""API endpoints: /health, /query, /ingest.

Handlers are thin — they validate input, call the shared agent/pipeline held in
``app.state``, and return typed responses. All heavy objects (agent, embedder,
vector store) are created once at startup, not per request.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import RedirectResponse

from app.api.models import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.logging import get_logger

logger = get_logger("api.routes")
router = APIRouter()


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Report service status and the active provider/model/doc count."""
    settings = request.app.state.settings
    agent = request.app.state.agent
    model = settings.groq_model if settings.llm_provider == "groq" else settings.ollama_model
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        provider=settings.llm_provider,
        model=model,
        doc_count=agent.nodes.store.count(),
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer a question through the agentic RAG graph.

    The response includes the decision trace (route taken, attempts, grounded,
    abstained) so callers can see how the agent reasoned.
    """
    agent = request.app.state.agent
    result = agent.run(body.question, top_k=body.top_k)
    logger.info(
        "query_handled",
        route=result["route_taken"],
        attempts=result["attempts"],
        abstained=result["abstained"],
    )
    return QueryResponse(**result)


@router.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(
    request: Request, files: Annotated[list[UploadFile], File()]
) -> IngestResponse:
    """Ingest uploaded PDF/Markdown files into the vector store.

    Files are written to a temp dir, run through the pipeline, then removed.
    """
    pipeline = request.app.state.pipeline
    tmp = Path(tempfile.mkdtemp(prefix="ingest_"))
    saved: list[Path] = []
    try:
        for upload in files:
            dest = tmp / Path(upload.filename or "upload").name
            with dest.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            saved.append(dest)
        stats = pipeline.ingest_paths(saved)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return IngestResponse(
        ingested_files=stats["files"],
        chunks_created=stats["chunks"],
        doc_count=stats["doc_count"],
    )
