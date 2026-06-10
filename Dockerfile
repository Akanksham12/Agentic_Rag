# Container for Hugging Face Spaces (Docker SDK) — free CPU tier.
# Generation runs remotely on Groq, so no GPU is needed here; only the local
# MiniLM embedder runs on CPU.
FROM python:3.11-slim

WORKDIR /app

# Keep the HF model cache inside the image so weights are baked at build time.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
COPY data ./data
RUN pip install -e .

# Bake the embedding model + pre-ingest the corpus at build time so the first
# request is fast and the demo works the moment the Space is up. Embeddings are
# local and free; no LLM key is needed for this step.
RUN python scripts/ingest_corpus.py

# Hugging Face Spaces routes to port 7860.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
