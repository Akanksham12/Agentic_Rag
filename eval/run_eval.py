"""Single-command evaluation runner.

Run from the repo root::

    python -m eval.run_eval        # or: make eval

Runs every item in ``eval/eval_set.yaml`` through the agent, computes the four
metrics, prints a results table, and writes ``eval/results.md`` (the artifact
you can screenshot for the README). The answering provider is whatever
``LLM_PROVIDER`` is set to; the judge defaults to Groq (see
``get_judge_provider``) for credible scores, falling back to local if no key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from rich import box
from rich.console import Console
from rich.table import Table

# The legacy Windows console encodes as cp1252 and chokes on Unicode box-drawing
# glyphs. Prefer UTF-8 stdout where possible; the console table also uses an
# ASCII box as a belt-and-suspenders fallback.
try:  # pragma: no cover - platform dependent
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.agent.graph import RagAgent
from app.config import get_settings
from app.llm.factory import get_judge_provider
from app.logging import get_logger
from eval import metrics

logger = get_logger("eval.run")
console = Console()

EVAL_SET_PATH = Path("eval/eval_set.yaml")
RESULTS_PATH = Path("eval/results.md")


def _mean(values: list[float | bool | None]) -> float:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else 0.0


def _fmt(value: float | bool | None) -> str:
    """ASCII formatting for the console (cp1252-safe)."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    return f"{value:.2f}"


def _fmt_md(value: float | bool | None) -> str:
    """Unicode formatting for the markdown artifact (written as UTF-8)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    return f"{value:.2f}"


def run() -> dict:
    """Execute the evaluation and return aggregate scores."""
    settings = get_settings()
    agent = RagAgent(settings=settings)
    judge = get_judge_provider()
    embedder, store = agent.nodes.embedder, agent.nodes.store
    items = yaml.safe_load(EVAL_SET_PATH.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for item in items:
        logger.info("evaluating", id=item["id"], type=item["type"])
        result = agent.run(item["question"])
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "hit": metrics.retrieval_hit(item, embedder, store, settings.top_k),
                "faithful": metrics.faithfulness(
                    item, result, embedder, store, judge, settings.top_k
                ),
                "correct": metrics.correctness(item, result["answer"], judge),
                "refusal": metrics.refusal_correct(item, result),
                "route": result["route_taken"],
            }
        )

    aggregates = {
        "hit_rate": _mean([r["hit"] for r in rows]),
        "faithfulness": _mean([r["faithful"] for r in rows]),
        "correctness": _mean([r["correct"] for r in rows]),
        "refusal_accuracy": _mean([r["refusal"] for r in rows]),
    }

    _print_table(rows, aggregates, settings, judge)
    _write_markdown(rows, aggregates, settings, judge)
    return aggregates


def _print_table(rows, aggregates, settings, judge) -> None:
    table = Table(title="Agentic RAG - Evaluation", box=box.ASCII)
    for col in ("id", "type", "hit", "faithful", "correct", "refusal", "route"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["id"], r["type"], _fmt(r["hit"]), _fmt(r["faithful"]),
            _fmt(r["correct"]), _fmt(r["refusal"]), r["route"],
        )
    console.print(table)
    console.print(
        f"[bold]Hit-rate@{settings.top_k}[/]: {aggregates['hit_rate']:.2f}   "
        f"[bold]Faithfulness[/]: {aggregates['faithfulness']:.2f}   "
        f"[bold]Correctness[/]: {aggregates['correctness']:.2f}   "
        f"[bold]Refusal acc[/]: {aggregates['refusal_accuracy']:.2f}"
    )
    console.print(
        f"[dim]answering={settings.llm_provider} · judge={judge.name}:{judge.model}[/]"
    )


def _write_markdown(rows, aggregates, settings, judge) -> None:
    lines = [
        "# Evaluation Results",
        "",
        f"_Answering provider: `{settings.llm_provider}` · "
        f"judge: `{judge.name}:{judge.model}` · top_k={settings.top_k}_",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Retrieval hit-rate @{settings.top_k} | {aggregates['hit_rate']:.2f} |",
        f"| Faithfulness (grounded) | {aggregates['faithfulness']:.2f} |",
        f"| Answer correctness (LLM-judge) | {aggregates['correctness']:.2f} |",
        f"| Refusal accuracy | {aggregates['refusal_accuracy']:.2f} |",
        "",
        "| id | type | hit | faithful | correct | refusal | route |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['type']} | {_fmt_md(r['hit'])} | {_fmt_md(r['faithful'])} "
            f"| {_fmt_md(r['correct'])} | {_fmt_md(r['refusal'])} | {r['route']} |"
        )
    lines.append("")
    lines.append(
        "_LLM-judged metrics (faithfulness, correctness) are not perfectly "
        "deterministic; the judge model is pinned for reproducibility._"
    )
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote {RESULTS_PATH}[/]")


if __name__ == "__main__":
    run()
