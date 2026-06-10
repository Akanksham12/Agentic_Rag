# Agentic RAG

An **agentic** Retrieval-Augmented Generation system with a real **evaluation
harness**, served over **FastAPI**. It runs **100% locally for free** during
development (Ollama + local embeddings + Chroma) and deploys cheaply to a free
host by swapping the LLM provider with a single environment variable.

> Status: under construction — see [the design spec](docs/superpowers/specs/2026-06-10-agentic-rag-design.md)
> and the commit history, which builds the system one explainable capability at a time.

## Why this is "agentic", not plain RAG

The retrieval loop is an **adaptive + corrective RAG** agent (LangGraph): it
decides whether a query even needs retrieval, grades whether the retrieved
context actually answers the question, reformulates and retries weak queries,
checks its own answer for grounding, and abstains ("I don't know") rather than
hallucinating. Full architecture, setup, eval results, and an honest list of
limitations land in this README as the build progresses.

## Quick start (local, free)

```bash
# 1. Install Ollama and pull the default model
ollama pull llama3.2:3b

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Copy env and ingest the demo corpus
cp .env.example .env
python scripts/ingest_corpus.py

# 4. Run the API
uvicorn app.main:app --reload   # docs at http://localhost:8000/docs
```

(Each step becomes real as its commit lands.)
