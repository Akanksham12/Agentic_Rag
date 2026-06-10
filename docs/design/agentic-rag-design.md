# Agentic RAG — Design Notes

## Overview

An agentic (not plain) RAG system with an evaluation harness, served as a
FastAPI app. It runs entirely locally at no cost during development and deploys
for free. LLM generation defaults to a local Ollama model behind an interface,
swappable to Groq's free tier via one environment variable; embeddings are
always local (sentence-transformers MiniLM), so even the hosted demo costs
nothing.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Cloud LLM provider | Groq | OpenAI-compatible API, fast, generous free tier. |
| Vector DB | Chroma | Built-in persistence + metadata filtering + doc store. |
| Local model | `llama3.2:3b` | Fits 4 GB VRAM (RTX 3050), responsive across multi-call loops. |
| Eval judge | Groq (default) | A 3B local model is a weak judge; Groq keeps scores credible, free. |
| Embeddings | all-MiniLM-L6-v2, local always | Separate concern from generation; keeps the demo free. |
| Demo corpus | classic arXiv AI/ML PDFs | Coherent theme; ships with a matching eval set. |
| Deploy | Hugging Face Spaces (Docker) | Free CPU tier; generation is remote, so no GPU. |

## Agent design (LangGraph) — adaptive + corrective RAG

State: `question`, `rewritten_question`, `retrieved`, `attempts`, `answer`, plus
route/grade/grounding verdicts.

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

- `LLMProvider` (`generate`, `generate_json`) — `OllamaProvider` (default),
  `GroqProvider`; `factory.get_provider()` reads `LLM_PROVIDER`. `GROQ_API_KEY`
  only from env.
- `Splitter` (`split`) — `RecursiveCharacterSplitter` default; a semantic
  splitter can be added without touching the pipeline.

## Eval harness

`eval_set.yaml` spans in-scope / multi-hop / out-of-scope / injection. Metrics:
hit-rate@k (deterministic), faithfulness (LLM-judge), answer correctness
(LLM-judge), refusal accuracy. `make eval` → table + `eval/results.md`.
LLM-judge non-determinism is documented.

## Guardrails

Catches: injection-pattern flagging; retrieved text delimited and
system-prompted as data-not-instructions; router refuses out-of-scope. Does not
catch: obfuscated/encoded/non-English injection, indirect injection inside
ingested documents, model-level jailbreaks, semantically-out-of-scope queries
that are lexically similar. No dedicated classifier.

## API

`GET /health` (status, provider, model, doc_count); `POST /ingest` (multipart
upload); `POST /query` (`answer`, `sources[]`, decision trace: `route_taken`,
`attempts`, `grounded`, `abstained`). Auto `/docs`. Corpus pre-ingested at
startup.

## Deployment

Hugging Face Spaces (Docker): install deps, bake MiniLM + pre-ingest corpus,
uvicorn on 7860. Secrets: `LLM_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_MODEL`,
`EMBEDDING_MODEL`, `MAX_ATTEMPTS`.
