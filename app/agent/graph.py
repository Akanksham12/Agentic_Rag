"""Assemble the agent nodes into a LangGraph StateGraph.

This is the adaptive + corrective RAG control flow:

    START -> route_query --(retrieve)--> retrieve -> grade_documents
                         --(direct)----> answer_direct -> END
                         --(refuse)----> refuse -> END

    grade_documents --(relevant)----------> generate
                    --(insufficient,retry)-> transform_query -> retrieve
                    --(insufficient,done)--> say_idk -> END

    generate -> grounding_check --(grounded)----> END
                                --(ungrounded)--> say_idk -> END

The reformulation loop is bounded by ``MAX_ATTEMPTS``; the grounding gate is
toggled by ``ENABLE_GROUNDING_CHECK``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentNodes
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.logging import get_logger

logger = get_logger(__name__)


def _guardrail_decision(state: AgentState) -> str:
    """Branch key after the deterministic guardrail screen."""
    return "blocked" if state.get("blocked") else "ok"


def _route_decision(state: AgentState) -> str:
    """Branch key after routing."""
    return state.get("route", "retrieve")


def _make_grade_decision(settings: Settings):
    """Build the post-grading branch function, bounding retries by max_attempts."""

    def decide(state: AgentState) -> str:
        if state.get("grade") == "relevant":
            return "relevant"
        if state.get("attempts", 0) < settings.max_attempts:
            return "transform"
        return "idk"

    return decide


def _grounding_decision(state: AgentState) -> str:
    """Branch key after the grounding gate."""
    return "ok" if state.get("grounded") else "idk"


def build_graph(nodes: AgentNodes | None = None, settings: Settings | None = None):
    """Build and compile the agent graph.

    Args:
        nodes: Node implementations (deps injectable for tests).
        settings: Application settings (controls retries + grounding toggle).

    Returns:
        A compiled LangGraph runnable exposing ``.invoke(state)``.
    """
    settings = settings or get_settings()
    nodes = nodes or AgentNodes(settings=settings)

    graph = StateGraph(AgentState)
    graph.add_node("guardrail_check", nodes.guardrail_check)
    graph.add_node("route_query", nodes.route_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("transform_query", nodes.transform_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("answer_direct", nodes.answer_direct)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("say_idk", nodes.say_idk)

    graph.add_edge(START, "guardrail_check")
    graph.add_conditional_edges(
        "guardrail_check",
        _guardrail_decision,
        {"blocked": "refuse", "ok": "route_query"},
    )
    graph.add_conditional_edges(
        "route_query",
        _route_decision,
        {"retrieve": "retrieve", "direct": "answer_direct", "refuse": "refuse"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _make_grade_decision(settings),
        {"relevant": "generate", "transform": "transform_query", "idk": "say_idk"},
    )
    graph.add_edge("transform_query", "retrieve")

    if settings.enable_grounding_check:
        graph.add_node("grounding_check", nodes.grounding_check)
        graph.add_edge("generate", "grounding_check")
        graph.add_conditional_edges(
            "grounding_check", _grounding_decision, {"ok": END, "idk": "say_idk"}
        )
    else:
        graph.add_edge("generate", END)

    graph.add_edge("answer_direct", END)
    graph.add_edge("refuse", END)
    graph.add_edge("say_idk", END)

    logger.info("agent_graph_built", grounding=settings.enable_grounding_check)
    return graph.compile()


class RagAgent:
    """High-level wrapper around the compiled graph.

    Exposes a single :meth:`run` that returns the answer plus the decision
    trace (route taken, attempts, grounded, abstained) used by the API.
    """

    def __init__(self, nodes: AgentNodes | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.nodes = nodes or AgentNodes(settings=self.settings)
        self.graph = build_graph(self.nodes, self.settings)

    def run(self, question: str) -> dict:
        """Answer ``question`` and return the answer with its decision trace."""
        final = self.graph.invoke({"question": question, "attempts": 0})
        route = "refused" if final.get("refused") else final.get("route", "retrieve")
        return {
            "answer": final.get("answer", ""),
            "sources": final.get("sources", []),
            "route_taken": route,
            "attempts": final.get("attempts", 0),
            "grounded": final.get("grounded"),
            "abstained": bool(final.get("abstained", False)),
        }
