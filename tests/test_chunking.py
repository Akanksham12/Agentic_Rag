"""Tests for the recursive character splitter (written before the impl — TDD).

These encode the contract we expect of *any* splitter:
- never emit a chunk larger than ``chunk_size``
- short text stays a single chunk
- consecutive chunks overlap (context continuity for retrieval)
- chunks carry unique ids and inherited + chunk-local metadata
- empty/whitespace input yields nothing
- the concrete splitter satisfies the swappable ``Splitter`` protocol
"""

from __future__ import annotations

from app.ingestion.chunking import RecursiveCharacterSplitter, Splitter
from app.ingestion.models import Document


def make_splitter(size: int = 100, overlap: int = 20) -> RecursiveCharacterSplitter:
    return RecursiveCharacterSplitter(chunk_size=size, chunk_overlap=overlap)


def test_short_text_is_a_single_chunk() -> None:
    splitter = make_splitter()
    chunks = splitter.split(Document(content="hello world", metadata={"source": "a.md"}))
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"


def test_no_chunk_exceeds_chunk_size() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    splitter = make_splitter(size=100, overlap=20)
    chunks = splitter.split(Document(content=text, metadata={"source": "a.md"}))
    assert len(chunks) > 1
    assert all(len(c.content) <= 100 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(200))
    splitter = make_splitter(size=80, overlap=30)
    chunks = splitter.split(Document(content=text, metadata={"source": "a.md"}))
    assert len(chunks) >= 2
    shared = any(
        set(a.content.split()) & set(b.content.split())
        for a, b in zip(chunks, chunks[1:], strict=False)
    )
    assert shared, "expected adjacent chunks to share overlapping tokens"


def test_chunks_have_unique_ids_and_inherited_metadata() -> None:
    text = " ".join(f"word{i}" for i in range(300))
    splitter = make_splitter(size=100, overlap=20)
    chunks = splitter.split(
        Document(content=text, metadata={"source": "paper.pdf", "type": "pdf"})
    )
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert all(c.metadata["source"] == "paper.pdf" for c in chunks)
    assert all(c.metadata["type"] == "pdf" for c in chunks)
    assert all("chunk_index" in c.metadata for c in chunks)


def test_empty_document_yields_no_chunks() -> None:
    splitter = make_splitter()
    assert splitter.split(Document(content="   ", metadata={"source": "a.md"})) == []


def test_concrete_splitter_satisfies_protocol() -> None:
    assert isinstance(make_splitter(), Splitter)
