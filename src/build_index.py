#!/usr/bin/env python3
"""Offline ingestion for search_business_docs: chunk the Markdown docs in config/rag_docs/,
embed each chunk (CPU, fastembed), and write the chunk rows + embedding matrix that src/rag.py
loads at server startup.

Chunking is heading-based (splits on ## / ### Markdown headings) rather than fixed token windows
— the docs are authored with clear section headings, so this is both simpler (no tokenizer
dependency) and more precise than a generic windowed split. A char-count fallback only kicks in
for a single section that's unusually long.

Safe to re-run any time the docs change: always clears and rebuilds from scratch (no incremental
indexing — not needed at this corpus size, and simpler to reason about).

Run: python3 src/build_index.py
"""
import os
from datetime import datetime

import numpy as np

import db
import rag

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "config", "rag_docs")

CHUNK_CHAR_CEILING = 1800     # ~450 tokens at ~4 chars/token — a section longer than this gets
                              # windowed rather than embedded as one oversized chunk
WINDOW_OVERLAP_RATIO = 0.15


def parse_markdown_text(raw_text: str) -> list[tuple[str | None, str]]:
    """Split raw Markdown into (section, text) pairs, one per ## or ### heading. `section` is the
    heading path, e.g. "Fees > Overdraft" for text under an H3 nested under an H2. Text before any
    H2/H3 heading (e.g. an intro paragraph under the H1 title) gets section=None."""
    h2 = h3 = None
    buffer: list[str] = []
    chunks: list[tuple[str | None, str]] = []

    def flush():
        content = "\n".join(buffer).strip()
        buffer.clear()
        if content:
            section = " > ".join(s for s in (h2, h3) if s) or None
            chunks.append((section, content))

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            flush()
            h3 = stripped[4:].strip()
        elif stripped.startswith("## "):
            flush()
            h2 = stripped[3:].strip()
            h3 = None
        elif stripped.startswith("# "):
            flush()   # H1 title line — just a boundary, not tracked as a section itself
        else:
            buffer.append(line)
    flush()
    return chunks


def _window_split(text: str, ceiling: int = CHUNK_CHAR_CEILING,
                  overlap_ratio: float = WINDOW_OVERLAP_RATIO) -> list[str]:
    """Char-based sliding-window fallback for a single section that exceeds `ceiling`."""
    if len(text) <= ceiling:
        return [text]
    step = max(1, int(ceiling * (1 - overlap_ratio)))
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + ceiling, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start += step
    return pieces


def chunk_document(doc_name: str, raw_text: str) -> list[dict]:
    """One Markdown file -> a list of {"doc_name", "section", "text"} chunk dicts."""
    result = []
    for section, text in parse_markdown_text(raw_text):
        pieces = _window_split(text)
        if len(pieces) == 1:
            result.append({"doc_name": doc_name, "section": section, "text": pieces[0]})
        else:
            for i, piece in enumerate(pieces, start=1):
                labeled = f"{section} (part {i})" if section else f"(part {i})"
                result.append({"doc_name": doc_name, "section": labeled, "text": piece})
    return result


def build(conn, docs_dir: str = DOCS_DIR) -> dict:
    """Chunk every *.md in docs_dir, embed in one batch, write chunk rows + the .npy matrix.
    Returns a summary dict. Raises if the resulting row/matrix counts don't line up."""
    db.init_schema(conn)   # safe to call every run — CREATE TABLE IF NOT EXISTS, no-op if present
    db.clear_rag_chunks(conn)

    doc_files = sorted(f for f in os.listdir(docs_dir) if f.endswith(".md")) if os.path.isdir(docs_dir) else []
    all_chunks: list[dict] = []
    for fname in doc_files:
        with open(os.path.join(docs_dir, fname), "r") as f:
            raw = f.read()
        all_chunks.extend(chunk_document(fname, raw))

    if not all_chunks:
        print(f"  [build_index] No chunks found in {docs_dir} — nothing to index.")
        return {"doc_count": 0, "chunk_count": 0}

    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=rag.EMBED_MODEL_NAME)
    # Passage mode (embed), NOT query mode (query_embed) — bge is trained asymmetrically and
    # mixing the two modes silently degrades retrieval quality. rag.search_docs() uses query_embed().
    vectors = np.array(list(model.embed([c["text"] for c in all_chunks])), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    updated_at = datetime.now().astimezone().isoformat()
    for chunk in all_chunks:
        db.insert_rag_chunk(conn, chunk["doc_name"], chunk["section"], chunk["text"], updated_at)

    os.makedirs(os.path.dirname(rag.INDEX_MATRIX_PATH), exist_ok=True)
    np.save(rag.INDEX_MATRIX_PATH, vectors)

    stored = db.fetch_all_rag_chunks(conn)
    if len(stored) != vectors.shape[0]:
        raise RuntimeError(
            f"rag_chunks row count ({len(stored)}) != embedding matrix row count ({vectors.shape[0]}) "
            "— insertion order must match embed order, something went wrong."
        )

    return {"doc_count": len(doc_files), "chunk_count": len(all_chunks), "matrix_shape": vectors.shape}


if __name__ == "__main__":
    conn = db.connect()
    summary = build(conn)
    print(f"Indexed {summary['chunk_count']} chunks from {summary['doc_count']} docs in {DOCS_DIR}")
    if summary["chunk_count"]:
        print(f"  Embedding matrix: {summary['matrix_shape']} -> {rag.INDEX_MATRIX_PATH}")
        for row in db.fetch_all_rag_chunks(conn)[:10]:
            preview = row["text"][:60].replace("\n", " ")
            print(f"  [{row['id']}] {row['doc_name']} :: {row['section']} :: {preview}...")
        if summary["chunk_count"] > 10:
            print(f"  ... and {summary['chunk_count'] - 10} more")
    conn.close()
