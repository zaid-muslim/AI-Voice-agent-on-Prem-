"""
RAG retrieval over hospital_kb.py's static hospital info (hours,
departments, doctors, insurance, billing, prescriptions, visiting policy,
lab results, parking).

APPROACH: semantic search via sentence-transformers embeddings, so a query
like "who can I see for my heart" matches the cardiology entries even
without the literal word "cardiology" anywhere in it. That's the actual
point of RAG being "smart" - keyword matching alone can't do this.

GRACEFUL DEGRADATION: if sentence-transformers isn't installed yet, this
falls back automatically to a keyword-overlap scorer instead of crashing -
noticeably worse matching, but the agent stays functional. Install the real
thing on your machine:
    pip install sentence-transformers --break-system-packages

HONEST CAVEAT: the environment this was written in has no network access to
install sentence-transformers, so the semantic path could not be executed
there. The self-test below runs whichever backend is actually available -
it will report which one, and on your machine (with the real dependency
installed) test #3 should go from a likely FAIL to a PASS. That specific
before/after is the proof the semantic path is worth installing.

NOT for appointment availability or booking - check_availability and
book_appointment remain the only tools for those; this is for the static
informational questions those tools were never meant to answer.

LIVEKIT COMPAT NOTE (this revision): the original version of this file was
written for Pipecat - it took a `params` object as its first argument and
spoke results through `params.result_callback(string)` instead of
returning anything, and it was wrapped in `@with_adaptive_filler` from a
`tool_filler.py` that doesn't exist in a LiveKit-style agent. Both are
gone. `search_hospital_info(query)` now just returns a plain dict, which
is exactly what compat.py's `_call()` expects to await and hand back -
LiveKit's own function-calling loop wants the same shape natively, so this
also removes the need for compat.py as a translation layer for this tool.
"""

import asyncio
from typing import List, Tuple

try:
    from .hospital_kb import HOSPITAL_KB
except ImportError:
    from hospital_kb import HOSPITAL_KB

TOP_K = 3

# --- semantic backend (preferred) ------------------------------------------

_embedder = None
_kb_embeddings = None
_kb_texts = None
_kb_entries = None  # snapshot of entries the vectors were built from


def _try_load_embedder() -> bool:
    """Lazy-load once. Returns True if the real semantic backend is usable.
    Also the hook compat.py's warm_rag() calls at startup, before the first
    real call is accepted, so the model doesn't lazy-load mid-conversation
    and blow through a function-call timeout."""
    global _embedder, _kb_embeddings, _kb_texts, _kb_entries
    if _embedder is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # Snapshot the entries we embed ALONGSIDE their vectors, so a later
        # reload_kb() that reorders/resizes HOSPITAL_KB can never make the
        # position indices in _semantic_search misalign with the vectors.
        _kb_entries = list(HOSPITAL_KB)
        _kb_texts = [f"{e['title']}. {e['text']}" for e in _kb_entries]
        _kb_embeddings = _embedder.encode(_kb_texts, normalize_embeddings=True)
        return True
    except Exception:
        _embedder = None
        _kb_embeddings = None
        _kb_entries = None
        return False


def reindex() -> bool:
    """Re-embed the CURRENT HOSPITAL_KB from scratch. Call this AFTER
    hospital_kb.save_kb() writes an admin edit, so semantic search reflects
    the new/changed/removed entries immediately (for calls that start after
    this returns). Returns True if the semantic backend re-embedded, False
    if running on the keyword fallback (in which case there's nothing to
    re-embed - the fallback always reads HOSPITAL_KB live). Safe to call
    even if the embedder was never loaded."""
    global _kb_embeddings, _kb_texts, _kb_entries
    if _embedder is None:
        # Either never loaded, or on keyword fallback. Try a fresh load so
        # an admin save can be the thing that first warms it.
        return _try_load_embedder()
    _kb_entries = list(HOSPITAL_KB)
    _kb_texts = [f"{e['title']}. {e['text']}" for e in _kb_entries]
    _kb_embeddings = _embedder.encode(_kb_texts, normalize_embeddings=True)
    return True


def _semantic_search(query: str, k: int) -> List[Tuple[dict, float]]:
    import numpy as np

    q_emb = _embedder.encode([query], normalize_embeddings=True)[0]
    scores = _kb_embeddings @ q_emb
    top_idx = np.argsort(-scores)[:k]
    # Index into _kb_entries (the snapshot the vectors were built from), NOT
    # the live HOSPITAL_KB, so results always line up with their vectors
    # even if HOSPITAL_KB was reloaded since the last reindex().
    return [(_kb_entries[i], float(scores[i])) for i in top_idx]


# --- fallback: keyword overlap (zero extra dependency) ---------------------


def _keyword_search(query: str, k: int) -> List[Tuple[dict, float]]:
    q_words = set(query.lower().split())
    scored = []
    for entry in HOSPITAL_KB:
        entry_words = set((entry["title"] + " " + entry["text"]).lower().split())
        overlap = len(q_words & entry_words)
        scored.append((entry, float(overlap)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def _search(query: str, k: int = TOP_K) -> List[Tuple[dict, float]]:
    if _try_load_embedder():
        return _semantic_search(query, k)
    return _keyword_search(query, k)


async def search_hospital_info(query: str) -> dict:
    """Search Riverside General's general information: hours, departments,
    doctors, insurance, billing, prescriptions, visiting policy, lab
    results, or parking. Do NOT use this for appointment availability or
    booking - use check_availability / book_appointment for those.

    Returns {"status": "ok", "answer": "..."} on a match, or
    {"status": "not_found", "message": "..."} if nothing scored above zero.
    """
    results = await asyncio.to_thread(_search, query, TOP_K)
    relevant = [entry for entry, score in results if score > 0]
    if not relevant:
        return {
            "status": "not_found",
            "message": "No information found for that. Tell the caller you "
            "don't have that information rather than guessing.",
        }
    return {"status": "ok", "answer": " ".join(entry["text"] for entry in relevant)}


# --- self-test --------------------------------------------------------

if __name__ == "__main__":

    async def _run():
        backend_is_semantic = _try_load_embedder()
        backend = (
            "semantic (sentence-transformers)"
            if backend_is_semantic
            else "keyword fallback"
        )
        print(f"Backend in use: {backend}\n")

        results = []

        r1 = await search_hospital_info("what are your hours")
        ok1 = r1["status"] == "ok" and any(
            w in r1["answer"] for w in ("8 AM", "8:00", "24 hours")
        )
        results.append(("direct keyword query: hours", ok1, r1))

        r2 = await search_hospital_info("who works in cardiology")
        ok2 = r2["status"] == "ok" and any(
            w in r2["answer"] for w in ("Malik", "Siddiqui", "Cardiology")
        )
        results.append(("direct keyword query: cardiology", ok2, r2))

        r3 = await search_hospital_info("who can I see for my heart")
        ok3 = r3["status"] == "ok" and any(
            w in r3["answer"] for w in ("Malik", "Siddiqui", "Cardiology", "cardiology")
        )
        note = (
            "(zero shared words with 'cardiology' - keyword fallback is "
            "EXPECTED to fail this; sentence-transformers should pass it)"
        )
        results.append((f"semantic query: heart -> cardiology {note}", ok3, r3))

        r4 = await search_hospital_info("asdkjqwoe nonsense gibberish query")
        ok4 = r4["status"] in ("not_found",) or (
            r4["status"] == "ok"  # semantic backend may still weakly match; that's fine
        )
        results.append(("nonsense query doesn't crash", ok4, r4))

        for desc, ok, detail in results:
            status = (
                "PASS"
                if ok
                else (
                    "FAIL (expected under fallback)"
                    if not backend_is_semantic
                    else "FAIL"
                )
            )
            print(f"[{status}] {desc}\n       -> {detail}")

        if not backend_is_semantic:
            print(
                "\nRunning on the keyword fallback here - install "
                "sentence-transformers on your machine and re-run this file "
                "to check whether test #3 flips to a real PASS."
            )

    asyncio.run(_run())
