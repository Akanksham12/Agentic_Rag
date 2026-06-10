# Deployment — Hugging Face Spaces (free)

The hosted demo runs **without your laptop** by switching the LLM provider to
**Groq** (free tier). Local MiniLM embeddings run on the Space's CPU; text
generation is remote, so **no GPU is required** and the cost is **$0**.

## Why Hugging Face Spaces (over Render)

- Genuinely free CPU tier, no credit card.
- 16 GB RAM — comfortable for MiniLM on CPU.
- Docker SDK gives full control over the runtime.
- Signals ML-community fluency to reviewers.

(Render's free web service sleeps on idle and its free tier has tightened; HF
Spaces is the cleaner free option for an ML demo.)

## Prerequisites

1. A free Hugging Face account.
2. A free Groq API key — https://console.groq.com

## Steps

1. **Create a Space**: New Space → SDK = **Docker** → blank.
2. **Push this repo** to the Space's git remote (or connect the GitHub repo).
   The repo root already contains a `Dockerfile`.
3. **Add front matter** (already in `README.md`) so the Space serves the app on
   port 7860:
   ```yaml
   ---
   title: Agentic RAG
   emoji: 🔎
   sdk: docker
   app_port: 7860
   ---
   ```
4. **Set Space secrets** (Settings → Variables and secrets):

   | Name | Value | Kind |
   | --- | --- | --- |
   | `LLM_PROVIDER` | `groq` | variable |
   | `GROQ_API_KEY` | *your key* | **secret** |
   | `GROQ_MODEL` | `llama-3.1-8b-instant` | variable |
   | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | variable |
   | `MAX_ATTEMPTS` | `2` | variable |

5. The Space builds the image (installs deps, bakes MiniLM, pre-ingests the
   corpus) and starts. When it's running, open the Space URL — interactive API
   docs are at **`/docs`**.

## Verify

- `GET /health` → `{"provider": "groq", "doc_count": <n>, ...}`
- `POST /query` with `{"question": "What problem does LoRA solve?"}` → a grounded,
  cited answer with a decision trace.

## Notes

- The build downloads the corpus PDFs and the MiniLM weights, so the first build
  is the slow one; subsequent requests are fast.
- To run the hosted demo on the local model instead (not recommended on a free
  CPU Space — it would be slow), set `LLM_PROVIDER=ollama` and provide an
  reachable `OLLAMA_BASE_URL`.
