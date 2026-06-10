# Agentic RAG — Design Spec

_Date: 2026-06-10 · Status: approved_

## Context & goal

A portfolio project demonstrating AI-engineering competence for an internship
application: an **agentic** (not plain) RAG system with a real **evaluation
harness**, served as a **FastAPI** app, runnable **100% locally for free** and
**deployable for free**. It must land as ~13 logical, interview-explainable git
commits, and every architectural decision must be defensible in an interview.

Hard cost constraint: LLM generation defaults to a local Ollama model behind an
interface, swappable to Groq free-tier via one env var. Embeddings are always
local (sentence-transformers MiniLM), so even the hosted demo costs $0.

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Cloud LLM provider | Groq | OpenAI-compatible, fast demo latency, generous free tier. |
| Vector DB | Chroma | Built-in persistence + metadata filtering + doc store. |
| Local model | `llama3.2:3b` | Fits 4GB VRAM (RTX 3050), snappy across multi-call loops. |
| Eval judge | Groq (default) | A 3B local model is a weak judge; Groq keeps scores credible, free. |
| Embeddings | all-MiniLM-L6-v2, local always | Separate concern from generation; keeps demo free. |
| Demo corpus | 3–5 classic arXiv AI/ML PDFs | On-theme; ships with matching eval Q/A set. |
| Deploy | Hugging Face Spaces (Docker) | Free CPU tier, no card; generation remote so no GPU. |

## Agent design (LangGraph) — adaptive + corrective RAG

State: `question`, `rewritten_question`, `retrieved_docs`, `attempts`,
`answer`, plus route/grade/grounding verdicts.

Nodes:
- `guardrail_check` (entry) — deterministic injection/scope screen before any
  LLM call. Blocked → `refuse` (END).
- `route_query` — needs retrieval → `retrieve`; safe general → `answer_direct`
  (END); out-of-scope → `refuse` (END).
- `retrieve` — top-k from Chroma.
- `grade_documents` — self-check: relevant → `generate`; weak & retries left →
  `transform_query`; weak & no retries → `say_idk` (END).
- `transform_query` — reformulate, `attempts++`, loop to `retrieve`; bounded by
  `MAX_ATTEMPTS`.
- `generate` — answer grounded in context, cited.
- `grounding_check` (toggleable) — faithfulness gate; grounded → `answer`,
  else → `say_idk`. Runtime twin of the eval faithfulness metric.

## Two swap seams (dependency inversion)

- `LLMProvider` Protocol (`generate`, `generate_json`) — `OllamaProvider`
  (default), `GroqProvider`; `factory.get_provider()` reads `LLM_PROVIDER`.
  `GROQ_API_KEY` only from env.
- `Splitter` Protocol (`split`) — `RecursiveCharacterSplitter` default;
  `SemanticSplitter` addable without touching the pipeline.

## Eval harness

`eval_set.yaml` (~12 `{id, question, expected_answer, expected_source, type}`
spanning in-scope / multi-hop / out-of-scope / injection). Metrics: hit-rate@k
(deterministic), faithfulness (LLM-judge), answer correctness (LLM-judge),
refusal accuracy. `make eval` → table + `eval/results.md`. Documents
LLM-judge non-determinism honestly.

## Guardrails

Do: injection-pattern flagging; retrieved text delimited + system-prompted as
data-not-instructions; router refuses out-of-scope. Don't (documented):
obfuscated/encoded/multilingual injection, indirect injection inside ingested
PDFs, model-level jailbreaks, semantically-out-of-scope-but-lexically-similar
queries. No dedicated classifier model.

## API

`GET /health` (status, provider, model, doc_count); `POST /ingest` (multipart
upload); `POST /query` (`answer`, `sources[]`, decision trace: `route_taken`,
`attempts`, `grounded`, `abstained`). Auto `/docs`. Corpus pre-ingested at
startup.

## Commit roadmap

1. Scaffold (config, logging, packaging). 2. Loaders + Chunk. 3. Splitter (TDD).
4. Embedder + Chroma. 5. Pipeline + corpus. 6. LLM providers. 7. Agent core
(basic RAG). 8. Full LangGraph (agentic) + tests. 9. Guardrails. 10. FastAPI.
11. Eval. 12. README. 13. Deployment.

## Deployment

HF Spaces Docker: install deps, bake MiniLM + pre-ingest corpus, uvicorn on
7860. Secrets: `LLM_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_MODEL`,
`EMBEDDING_MODEL`, `MAX_ATTEMPTS`.
