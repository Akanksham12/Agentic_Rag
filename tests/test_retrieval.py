"""Tests for retrieval and the agent control flow.

These use lightweight fakes for the LLM provider, embedder, and vector store so
the graph's *decision logic* (routing, the bounded reformulation loop, the
grounding gate, abstention, refusal) is verified deterministically and fast,
without a live model or network.
"""

from __future__ import annotations

from app.agent.graph import build_graph
from app.agent.nodes import AgentNodes
from app.agent.prompts import IDK_MESSAGE, REFUSE_MESSAGE
from app.config import Settings
from app.llm.base import LLMProvider, Message
from app.vectorstore.chroma_store import SearchResult


class FakeProvider(LLMProvider):
    """Provider whose structured verdicts are fixed per construction.

    ``generate_json`` inspects which node's system prompt it received (via a
    distinctive keyword) and returns the configured verdict.
    """

    name = "fake"

    def __init__(
        self,
        route: str = "retrieve",
        relevant: bool = True,
        grounded: bool = True,
        answer: str = "FAKE ANSWER [1]",
    ) -> None:
        self.route = route
        self.relevant = relevant
        self.grounded = grounded
        self.answer = answer

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens=None) -> str:
        return self.answer

    def generate_json(self, messages: list[Message], temperature: float = 0.0) -> dict:
        text = " ".join(m["content"] for m in messages).lower()
        if "classify" in text:
            return {"route": self.route}
        if "relevance grader" in text:
            return {"relevant": self.relevant}
        if "rewrite" in text:
            return {"query": "rewritten query"}
        if "supported" in text:
            return {"grounded": self.grounded}
        return {}


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def similarity_search(self, embedding, top_k=None) -> list[SearchResult]:
        return self._results

    def count(self) -> int:
        return len(self._results)


def make_agent(
    route: str = "retrieve",
    relevant: bool = True,
    grounded: bool = True,
    results: list[SearchResult] | None = None,
    settings: Settings | None = None,
):
    if results is None:
        results = [SearchResult("context about transformers", {"source": "a.pdf"}, 0.9)]
    settings = settings or Settings(max_attempts=2)
    nodes = AgentNodes(
        provider=FakeProvider(route=route, relevant=relevant, grounded=grounded),
        embedder=FakeEmbedder(),
        store=FakeStore(results),
        settings=settings,
    )
    return build_graph(nodes, settings)


# -- retrieve node -----------------------------------------------------------


def test_retrieve_increments_attempts_and_returns_hits() -> None:
    nodes = AgentNodes(
        provider=FakeProvider(),
        embedder=FakeEmbedder(),
        store=FakeStore([SearchResult("x", {"source": "a"}, 0.5)]),
        settings=Settings(),
    )
    out = nodes.retrieve({"question": "q", "attempts": 0})
    assert out["attempts"] == 1
    assert len(out["retrieved"]) == 1


# -- routing -----------------------------------------------------------------


def test_out_of_scope_query_is_refused() -> None:
    agent = make_agent(route="refuse")
    final = agent.invoke({"question": "what's the weather?", "attempts": 0})
    assert final.get("refused") is True
    assert final["abstained"] is True
    assert final["answer"] == REFUSE_MESSAGE


def test_direct_route_answers_without_retrieval() -> None:
    agent = make_agent(route="direct")
    final = agent.invoke({"question": "hi", "attempts": 0})
    assert final["route"] == "direct"
    assert "FAKE ANSWER" in final["answer"]
    assert final["sources"] == []


# -- corrective loop + grounding --------------------------------------------


def test_relevant_docs_produce_grounded_cited_answer() -> None:
    agent = make_agent(route="retrieve", relevant=True, grounded=True)
    final = agent.invoke({"question": "what is a transformer?", "attempts": 0})
    assert "FAKE ANSWER" in final["answer"]
    assert final["grounded"] is True
    assert final["abstained"] is False
    assert final["sources"]


def test_persistently_weak_retrieval_exhausts_retries_then_abstains() -> None:
    agent = make_agent(route="retrieve", relevant=False, settings=Settings(max_attempts=2))
    final = agent.invoke({"question": "q", "attempts": 0})
    assert final["abstained"] is True
    assert final["answer"] == IDK_MESSAGE
    assert final["attempts"] == 2  # looped exactly up to the bound


def test_ungrounded_answer_is_replaced_by_abstention() -> None:
    agent = make_agent(route="retrieve", relevant=True, grounded=False)
    final = agent.invoke({"question": "q", "attempts": 0})
    assert final["abstained"] is True
    assert final["answer"] == IDK_MESSAGE


def test_grounding_gate_can_be_disabled() -> None:
    settings = Settings(max_attempts=2, enable_grounding_check=False)
    agent = make_agent(route="retrieve", relevant=True, grounded=False, settings=settings)
    final = agent.invoke({"question": "q", "attempts": 0})
    # With the gate off, the answer stands even though grounded would be False.
    assert "FAKE ANSWER" in final["answer"]
    assert final["abstained"] is False
