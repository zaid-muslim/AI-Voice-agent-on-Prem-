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
# Inside data/ (already a persisted volume mount for the worker container, see docker-compose.yml)
# rather than fastembed's own default (/tmp/fastembed_cache under Docker) — otherwise every fresh
# container re-downloads the ~130MB ONNX model from scratch. That download alone can exceed
# LiveKit's default 10s initialize_process_timeout, repeatedly killing and restarting prewarm
# before the download ever completes — confirmed live, not hypothetical.
EMBED_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "fastembed_cache")

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
    _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=EMBED_CACHE_DIR)

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


def build_retrieval_query(history: list[dict], window: int = 3, max_prev_chars: int = 150) -> str:
    """Build the text to embed for retrieval from recent conversation, so a pronoun-y follow-up
    ("what's the age eligibility for it?") still retrieves the right document. A bare utterance
    embeds toward generic terms ("eligibility") and pulls the wrong account; prepending the last
    exchange carries the topic (whether it was named in the caller's question or the agent's prior
    answer). Measured effect on the hard follow-up case: correct chunk rank 2 @ 0.69 -> rank 1 @ 0.96.

    This is only ever one of two hybrid queries (see search_docs_multi) — the raw utterance is always
    searched too — so the prepended context can only help recall, never suppress the plain question.

    `history` ends with the current user turn. Leading assistant messages (the opening greeting,
    which precedes any real question) are dropped so boilerplate doesn't pollute the query. Uses the
    last `window` user/assistant messages, capping prior ones to `max_prev_chars` so the current
    utterance stays dominant."""
    msgs = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    while msgs and msgs[0]["role"] == "assistant":   # drop the greeting (assistant, no prior user)
        msgs.pop(0)
    recent = msgs[-window:]
    parts = []
    for i, m in enumerate(recent):
        text = m["content"]
        if i < len(recent) - 1:      # cap prior context; keep the current (last) utterance in full
            text = text[:max_prev_chars]
        parts.append(text)
    return " ".join(parts).strip()


def _scores_for(query: str):
    """Cosine similarity of one query against every chunk, or None if retrieval isn't available."""
    query = (query or "").strip()
    if not query or _doc_matrix is None or _embed_model is None:
        return None
    query_vec = next(iter(_embed_model.query_embed([query])))
    query_vec = np.asarray(query_vec, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return None
    query_vec = query_vec / norm
    return _doc_matrix @ query_vec   # single matmul (both sides L2-normalized)


def _top_results(scores, k: int) -> list[dict]:
    results = []
    for idx in np.argsort(scores)[::-1][:k]:
        score = float(scores[idx])
        if score < SIMILARITY_THRESHOLD:
            continue
        chunk = _chunk_meta[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["doc_name"],
            "section": chunk.get("section"),
            "score": round(score, 4),   # cosine similarity, for logging/diagnostics
        })
    return results


def search_docs(query: str) -> dict:
    """Embed the caller's question and return the top matching chunks above the similarity
    threshold. Returns {"status": "found", "results": [{"text","source","section","score"}, ...]}
    or {"status": "no_match"} — never raw exceptions, never more than the chunk text itself."""
    scores = _scores_for(query)
    if scores is None:
        return {"status": "no_match"}
    results = _top_results(scores, TOP_K)
    return {"status": "found", "results": results} if results else {"status": "no_match"}


def search_docs_multi(queries: list[str], k: int | None = None) -> dict:
    """Retrieve for several query phrasings and interleave each query's top hits (round-robin,
    de-duplicated) into a combined set of up to `k` chunks. Used for context-aware retrieval: pass
    both the raw current utterance and a context-expanded query.

    Interleaving (rather than a global score merge) is deliberate: when one query is *confidently
    wrong* — e.g. a context-expanded query dominated by the previous answer scores unrelated chunks
    at 0.78 while the raw utterance's correct chunk sits at 0.66 — a score merge would still bury the
    correct chunk. Reserving slots per query guarantees the raw utterance's best matches are present
    regardless of the other query's scores, so a fresh topical question and a pronoun follow-up are
    both covered."""
    k = k or TOP_K
    per_query = []
    for q in queries:
        s = _scores_for(q)
        if s is not None:
            per_query.append(_top_results(s, k))
    if not per_query:
        return {"status": "no_match"}

    merged, seen = [], set()
    for rank in range(k):
        for results in per_query:
            if rank < len(results):
                r = results[rank]
                key = (r["source"], r["section"])
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
                    if len(merged) >= k:
                        break
        if len(merged) >= k:
            break
    return {"status": "found", "results": merged} if merged else {"status": "no_match"}
