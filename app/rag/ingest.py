import os
import json
from pathlib import Path
from typing import List

from app.db.dao import init_db, clear_support_docs, add_support_doc_chunk
from app.logs.logger import logger
from .embeddings import get_embedding

KNOWLEDGE_BASE_DIR = Path("knowledge_base")


def _read_text_files(base_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not base_dir.exists():
        return files
    for path in base_dir.rglob("*"):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            files.append(path)
    return files


def _chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Simple character-based chunking by paragraphs / lines."""
    lines = text.splitlines()
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def build_support_doc_index() -> None:
    """Ingest all .txt/.md files in knowledge_base/ and build embeddings index."""
    logger.info("Starting support docs ingestion")
    init_db()

    files = _read_text_files(KNOWLEDGE_BASE_DIR)
    if not files:
        logger.warning("No support docs found in knowledge_base/")
        return

    logger.info(f"Found {len(files)} support files. Clearing old index...")
    clear_support_docs()

    for fpath in files:
        logger.info(f"Ingesting {fpath}")
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(text)
        doc_id = str(fpath.relative_to(KNOWLEDGE_BASE_DIR))

        for idx, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            emb_json = json.dumps(emb)
            title = fpath.stem
            add_support_doc_chunk(
                doc_id=doc_id,
                chunk_index=idx,
                title=title,
                content=chunk,
                embedding_json=emb_json,
            )

    logger.info("Support docs ingestion completed.")


if __name__ == "__main__":
    build_support_doc_index()
