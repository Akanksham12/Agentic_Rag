"""Document loaders: turn files on disk into :class:`Document` objects.

Responsibility is deliberately narrow — extract clean text and attach
provenance metadata. Loaders know nothing about chunking, embedding, or
storage, so new formats can be added without touching the rest of the
pipeline.

Supported formats: PDF (``.pdf``) and Markdown (``.md`` / ``.markdown``).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import Document
from app.logging import get_logger

logger = get_logger(__name__)

# File extensions this module knows how to load.
PDF_SUFFIXES = {".pdf"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | MARKDOWN_SUFFIXES


def load_pdf(path: str | Path) -> Document:
    """Load a PDF, concatenating the text of all pages.

    Args:
        path: Path to a ``.pdf`` file.

    Returns:
        A :class:`Document` whose ``content`` is the joined page text and whose
        metadata records the source, type, and page count.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    content = "\n\n".join(p for p in pages if p)
    logger.info("loaded_pdf", source=path.name, pages=len(reader.pages), chars=len(content))
    return Document(
        content=content,
        metadata={"source": path.name, "type": "pdf", "path": str(path), "pages": len(reader.pages)},
    )


def load_markdown(path: str | Path) -> Document:
    """Load a Markdown (or plain-text) file as-is.

    Args:
        path: Path to a ``.md`` / ``.markdown`` file.

    Returns:
        A :class:`Document` with the raw file text and source metadata.
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    logger.info("loaded_markdown", source=path.name, chars=len(content))
    return Document(
        content=content,
        metadata={"source": path.name, "type": "markdown", "path": str(path)},
    )


def load_file(path: str | Path) -> Document:
    """Load a single file, dispatching on its extension.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return load_pdf(path)
    if suffix in MARKDOWN_SUFFIXES:
        return load_markdown(path)
    raise ValueError(
        f"Unsupported file type {suffix!r} for {path.name}; "
        f"supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def load_directory(directory: str | Path) -> list[Document]:
    """Load every supported file in a directory (non-recursive).

    Unsupported files are skipped with a debug log rather than raising, so a
    mixed corpus folder loads cleanly.

    Args:
        directory: Folder to scan.

    Returns:
        A list of loaded :class:`Document` objects (possibly empty).
    """
    directory = Path(directory)
    documents: list[Document] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append(load_file(entry))
        elif entry.is_file():
            logger.debug("skipped_unsupported_file", source=entry.name)
    logger.info("loaded_directory", directory=str(directory), documents=len(documents))
    return documents
