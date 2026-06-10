"""Evaluation metrics.

Four complementary metrics, each isolating a different capability:

- ``retrieval_hit`` — deterministic, NO LLM: did top-k retrieval include the
  expected source document? Tests retrieval alone.
- ``faithfulness`` — LLM-judge: is the answer supported by retrieved context?
  (The offline twin of the runtime grounding gate.)
- ``correctness`` — LLM-judge: does the answer match the reference answer?
- ``refusal_correct`` — did out-of-scope / injection cases get refused?

LLM-judged metrics are not perfectly deterministic; the runner pins the judge
model and the README documents this caveat.
"""

from __future__ import annotations

from app.agent.nodes import format_context
from app.agent.prompts import GROUNDING_SYSTEM, GROUNDING_TEMPLATE
from app.embeddings.embedder import Embedder
from app.llm.base import LLMProvider, system, user
from app.logging import get_logger
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger("eval.metrics")

IN_SCOPE_TYPES = {"in-scope", "multi-hop"}
REFUSAL_TYPES = {"out-of-scope", "injection"}

CORRECTNESS_SYSTEM = (
    "You grade a candidate answer against a reference answer for the same question. "
    "Judge by meaning, not wording. Score 0 if it is incorrect, irrelevant, or says "
    "'I don't know'; 1 if partially correct or incomplete; 2 if fully correct."
)
CORRECTNESS_TEMPLATE = (
    "Question: {question}\n\nReference answer: {expected}\n\nCandidate answer: {candidate}\n\n"
    'Return JSON: {{"score": 0 | 1 | 2, "reason": "<short>"}}'
)


def retrieval_hit(item: dict, embedder: Embedder, store: ChromaStore, top_k: int) -> bool | None:
    """Deterministic: does top-k retrieval surface the expected source?

    Returns ``None`` for refusal cases (no expected source).
    """
    if item["type"] not in IN_SCOPE_TYPES:
        return None
    results = store.similarity_search(embedder.embed_query(item["question"]), top_k=top_k)
    return any(r.source == item["expected_source"] for r in results)


def faithfulness(
    item: dict,
    result: dict,
    embedder: Embedder,
    store: ChromaStore,
    judge: LLMProvider,
    top_k: int,
) -> float | None:
    """LLM-judge: is the produced answer grounded in retrieved context?

    Returns ``None`` for refusal cases and for abstentions (no answer to judge).
    """
    if item["type"] not in IN_SCOPE_TYPES:
        return None
    if result.get("abstained") or result.get("refused"):
        return None
    results = store.similarity_search(embedder.embed_query(item["question"]), top_k=top_k)
    context = format_context(results)
    try:
        verdict = judge.generate_json(
            [
                system(GROUNDING_SYSTEM),
                user(GROUNDING_TEMPLATE.format(context=context, answer=result["answer"])),
            ]
        )
        return 1.0 if verdict.get("grounded") else 0.0
    except ValueError:
        logger.warning("faithfulness_parse_failed", id=item["id"])
        return 0.0


def correctness(item: dict, answer: str, judge: LLMProvider) -> float | None:
    """LLM-judge: does the answer match the reference? Returns score in [0, 1].

    Computed for in-scope items only; an abstention scores 0 (it failed to
    answer a question it should have).
    """
    if item["type"] not in IN_SCOPE_TYPES:
        return None
    try:
        verdict = judge.generate_json(
            [
                system(CORRECTNESS_SYSTEM),
                user(
                    CORRECTNESS_TEMPLATE.format(
                        question=item["question"],
                        expected=item["expected_answer"].strip(),
                        candidate=answer or "",
                    )
                ),
            ]
        )
        return float(verdict.get("score", 0)) / 2.0
    except (ValueError, TypeError):
        logger.warning("correctness_parse_failed", id=item["id"])
        return 0.0


def refusal_correct(item: dict, result: dict) -> bool | None:
    """Did an out-of-scope / injection query get refused or abstained?

    Returns ``None`` for in-scope items.
    """
    if item["type"] not in REFUSAL_TYPES:
        return None
    return bool(result.get("refused") or result.get("abstained"))
