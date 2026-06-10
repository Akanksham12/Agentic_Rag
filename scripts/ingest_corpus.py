"""Download the demo corpus (classic AI/ML papers) and ingest it into Chroma.

Run from the repo root::

    python scripts/ingest_corpus.py

PDFs are downloaded on demand (not committed to the repo, to avoid
redistributing third-party papers). Re-running is safe: downloads are skipped
if the file already exists, and ingestion upserts on stable chunk ids.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.logging import get_logger

logger = get_logger("ingest_corpus")

# On-theme corpus for an AI-Engineer portfolio: foundational papers.
CORPUS: dict[str, str] = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "rag.pdf": "https://arxiv.org/pdf/2005.11401",
    "chain_of_thought.pdf": "https://arxiv.org/pdf/2201.11903",
    "lora.pdf": "https://arxiv.org/pdf/2106.09685",
}


def download_corpus(corpus_dir: Path) -> None:
    """Download any missing corpus PDFs into ``corpus_dir``."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in CORPUS.items():
        target = corpus_dir / filename
        if target.exists():
            logger.info("corpus_file_present", file=filename)
            continue
        logger.info("downloading", file=filename, url=url)
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        logger.info("downloaded", file=filename, bytes=len(resp.content))


def main() -> None:
    settings = get_settings()
    corpus_dir = Path(settings.corpus_dir)
    download_corpus(corpus_dir)

    pipeline = IngestionPipeline()
    stats = pipeline.ingest_directory(corpus_dir)
    print(
        f"\nIngested {stats['files']} files -> {stats['chunks']} chunks. "
        f"Vector store now holds {stats['doc_count']} chunks."
    )


if __name__ == "__main__":
    main()
