"""FastAPI application entry point.

Builds the app, wires the router, and uses a lifespan handler to construct the
expensive singletons (agent, embedder, vector store, ingestion pipeline) exactly
once at startup. The agent and pipeline deliberately share one embedder and one
store so that ingested documents are immediately queryable.

If the store is empty at startup, the bundled corpus is ingested so the demo
works the moment the server is up. (Downloading the corpus is the job of
scripts/ingest_corpus.py / the Docker build — startup only ingests files that
are already present.)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.agent.graph import RagAgent
from app.api.routes import router
from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.logging import configure_logging, get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared singletons and (optionally) pre-ingest the corpus."""
    configure_logging()
    settings = get_settings()
    logger.info("startup", provider=settings.llm_provider)

    agent = RagAgent(settings=settings)
    # Share the agent's embedder + store so /ingest writes where /query reads.
    pipeline = IngestionPipeline(embedder=agent.nodes.embedder, store=agent.nodes.store)

    if agent.nodes.store.count() == 0:
        corpus = Path(settings.corpus_dir)
        has_files = corpus.exists() and any(
            corpus.glob("*.pdf")
        ) or (corpus.exists() and any(corpus.glob("*.md")))
        if has_files:
            logger.info("startup_preingest", corpus=str(corpus))
            pipeline.ingest_directory(corpus)
        else:
            logger.warning("startup_empty_store", hint="run scripts/ingest_corpus.py")

    app.state.settings = settings
    app.state.agent = agent
    app.state.pipeline = pipeline
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Agentic RAG",
        version=settings.app_version,
        description=(
            "An adaptive + corrective RAG agent over a local document store. "
            "POST a question to /query and inspect the decision trace in the "
            "response."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
