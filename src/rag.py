#!/usr/bin/env python3
"""Retrieval for the search_business_docs tool: CPU-only embeddings (fastembed, ONNX INT8) +
flat NumPy cosine similarity over a small corpus of HBL policy/product documents.

Purely informational — no identity/verification gating like banking.py, since business info is
non-sensitive. Must never touch the GPU the STT/LLM/TTS models share; everything here is CPU-only
by construction (fastembed ships pre-quantized ONNX models, no torch/CUDA dependency at all).

Retrieval quality depends on using the right encode mode for bge models, which are trained
asymmetrically: query_embed() (with a query-specific prefix) for the caller's question here at
search time, embed() (passage mode, no prefix) for the document chunks at build time in
build_index.py. Mixing them up silently degrades results without raising any error.
"""
import os

import numpy as np

import db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_MATRIX_PATH = os.path.join(PROJECT_ROOT, "data", "rag_index.npy")

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Calibrated against the real corpus + live model: on-topic HBL questions scored 0.68-0.85
# cosine similarity against their best-matching chunk; off-topic questions (weather, jokes, "what
# time is it") scored 0.48-0.57 even against their nearest (irrelevant) chunk — bge-small doesn't
# produce near-zero scores for unrelated text, so the threshold has to sit in that gap rather than
# near zero. 0.60 cleanly separates the two groups in that sample; revisit if real caller queries
# turn out to sit closer to the boundary than this test set did.
SIMILARITY_THRESHOLD = 0.60
TOP_K = 3

# Lazy module-level singletons, populated by init(). Left empty (not loaded) until init() runs,
# so importing this module never pays model-load cost — only actually starting the server does.
_embed_model = None
_doc_matrix = None            # np.ndarray, shape (N, 384), float32, L2-normalized, or None
_chunk_meta: list[dict] = []  # row i here corresponds to row i of _doc_matrix


def init(conn) -> None:
    """Load the embedding model and the prebuilt index (matrix + metadata cache) once, at server
    startup. If build_index.py hasn't been run yet, leaves the index empty so search_docs()
    returns no_match instead of crashing — the server can still start without a corpus."""
    global _embed_model, _doc_matrix, _chunk_meta
    from fastembed import TextEmbedding
    _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)

    _chunk_meta = db.fetch_all_rag_chunks(conn)
    if os.path.exists(INDEX_MATRIX_PATH):
        matrix = np.load(INDEX_MATRIX_PATH)
        if matrix.shape[0] != len(_chunk_meta):
            print(f"  [rag] WARNING: index matrix has {matrix.shape[0]} rows but rag_chunks table "
                  f"has {len(_chunk_meta)} — stale index, re-run src/build_index.py. Disabling RAG "
                  f"until rebuilt.")
            _doc_matrix = None
            _chunk_meta = []
        else:
            _doc_matrix = matrix
            print(f"  [rag] Loaded {matrix.shape[0]} chunks, dim={matrix.shape[1]}")
    else:
        _doc_matrix = None
        print("  [rag] No index found at data/rag_index.npy — run src/build_index.py to enable "
              "search_business_docs. Until then it will report no_match.")


def warmup() -> None:
    """Pay ONNX session cold-start cost once at startup, not on the first real caller query."""
    if _embed_model is not None:
        list(_embed_model.query_embed(["warmup"]))


def search_docs(query: str) -> dict:
    """Embed the caller's question and return the top matching chunks above the similarity
    threshold. Returns {"status": "found", "results": [{"text","source","section"}, ...]} or
    {"status": "no_match"} — never raw exceptions, never more than the chunk text itself."""
    query = (query or "").strip()
    if not query or _doc_matrix is None or _embed_model is None:
        return {"status": "no_match"}

    query_vec = next(iter(_embed_model.query_embed([query])))
    query_vec = np.asarray(query_vec, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return {"status": "no_match"}
    query_vec = query_vec / norm

    scores = _doc_matrix @ query_vec   # cosine similarity, single matmul (both sides normalized)
    order = np.argsort(scores)[::-1][:TOP_K]

    results = []
    for idx in order:
        score = float(scores[idx])
        if score < SIMILARITY_THRESHOLD:
            continue
        chunk = _chunk_meta[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["doc_name"],
            "section": chunk.get("section"),
        })

    if not results:
        return {"status": "no_match"}
    return {"status": "found", "results": results}
