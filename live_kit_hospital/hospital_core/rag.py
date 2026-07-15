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
"""

import asyncio
from typing import List, Tuple

try:
    from .hospital_kb import HOSPITAL_KB
except ImportError:
    from hospital_kb import HOSPITAL_KB

try:
    from .tool_filler import with_adaptive_filler
except ImportError:
    from tool_filler import with_adaptive_filler

TOP_K = 3
MIN_KEYWORD_OVERLAP = 1  # fallback-only: need at least this many shared words

# --- semantic backend (preferred) ------------------------------------------

_embedder = None
_kb_embeddings = None
_kb_texts = None


def _try_load_embedder() -> bool:
    """Lazy-load once. Returns True if the real semantic backend is usable."""
    global _embedder, _kb_embeddings, _kb_texts
    if _embedder is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _kb_texts = [f"{e['title']}. {e['text']}" for e in HOSPITAL_KB]
        _kb_embeddings = _embedder.encode(_kb_texts, normalize_embeddings=True)
        return True
    except Exception:
        _embedder = None
        _kb_embeddings = None
        return False


def _semantic_search(query: str, k: int) -> List[Tuple[dict, float]]:
    import numpy as np

    q_emb = _embedder.encode([query], normalize_embeddings=True)[0]
    scores = _kb_embeddings @ q_emb
    top_idx = np.argsort(-scores)[:k]
    return [(HOSPITAL_KB[i], float(scores[i])) for i in top_idx]


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


def _format_results(results: List[Tuple[dict, float]]) -> str:
    relevant = [entry for entry, score in results if score > 0]
    if not relevant:
        return "no matching information found"
    return " ".join(entry["text"] for entry in relevant)


@with_adaptive_filler(threshold_secs=0.6, filler_text="Let me look that up.")
async def search_hospital_info(params, query: str):
    """Search Riverside General's general information: hours, departments,
    doctors, insurance, billing, prescriptions, visiting policy, lab
    results, or parking. Do NOT use this for appointment availability or
    booking - use check_availability / book_appointment for those.
    """
    results = await asyncio.to_thread(_search, query, TOP_K)
    await params.result_callback(_format_results(results))


# --- self-test --------------------------------------------------------

if __name__ == "__main__":

    class FakeParams:
        def __init__(self):
            self.result = None

        async def result_callback(self, result):
            self.result = result

    async def _run():
        backend_is_semantic = _try_load_embedder()
        backend = (
            "semantic (sentence-transformers)"
            if backend_is_semantic
            else "keyword fallback"
        )
        print(f"Backend in use: {backend}\n")

        results = []

        p1 = FakeParams()
        await search_hospital_info(p1, "what are your hours")
        ok1 = any(w in p1.result for w in ("8 AM", "8:00", "24 hours"))
        results.append(("direct keyword query: hours", ok1, p1.result))

        p2 = FakeParams()
        await search_hospital_info(p2, "who works in cardiology")
        ok2 = any(w in p2.result for w in ("Patel", "Osei", "Cardiology"))
        results.append(("direct keyword query: cardiology", ok2, p2.result))

        p3 = FakeParams()
        await search_hospital_info(p3, "who can I see for my heart")
        ok3 = any(w in p3.result for w in ("Patel", "Osei", "Cardiology", "cardiology"))
        note = (
            "(zero shared words with 'cardiology' - keyword fallback is "
            "EXPECTED to fail this; sentence-transformers should pass it)"
        )
        results.append((f"semantic query: heart -> cardiology {note}", ok3, p3.result))

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
