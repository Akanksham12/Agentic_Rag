"""Gradio entry point for the Hugging Face Space."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import spaces

from app.agent.graph import RagAgent
from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.logging import configure_logging, get_logger
from scripts.ingest_corpus import download_corpus

logger = get_logger("gradio_app")


def build_agent() -> RagAgent:
    """Create the agent and ensure the demo corpus is indexed."""
    configure_logging()
    settings = get_settings()
    agent = RagAgent(settings=settings)
    pipeline = IngestionPipeline(embedder=agent.nodes.embedder, store=agent.nodes.store)

    if agent.nodes.store.count() == 0:
        corpus_dir = Path(settings.corpus_dir)
        download_corpus(corpus_dir)
        pipeline.ingest_directory(corpus_dir)
        logger.info("space_corpus_ingested", doc_count=agent.nodes.store.count())

    return agent


agent = build_agent()


@spaces.GPU
def _zerogpu_marker() -> None:
    """Declare ZeroGPU compatibility without requiring GPU for this app."""
    return None


def answer_question(question: str, top_k: int) -> tuple[str, list[dict], dict]:
    """Run one question through the existing agent and format its trace."""
    if not question or not question.strip():
        return "Please enter a question.", [], {}

    result = agent.run(question.strip(), top_k=int(top_k))
    trace = {
        "route": result["route_taken"],
        "attempts": result["attempts"],
        "grounded": result["grounded"],
        "abstained": result["abstained"],
    }
    return result["answer"], result["sources"], trace


with gr.Blocks(title="Agentic RAG") as demo:
    gr.Markdown("# Agentic RAG\nAsk questions about the indexed AI/ML papers.")
    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="What problem does LoRA address?",
            scale=4,
        )
        top_k = gr.Slider(1, 10, value=4, step=1, label="Top-k", scale=1)
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    sources = gr.JSON(label="Sources")
    trace = gr.JSON(label="Decision trace")

    submit.click(answer_question, inputs=[question, top_k], outputs=[answer, sources, trace])
    question.submit(answer_question, inputs=[question, top_k], outputs=[answer, sources, trace])


if __name__ == "__main__":
    demo.launch()
