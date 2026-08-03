"""Generic semantic-search RAG engine, generalized from
``app/hospital_core/rag.py``.

APPROACH (unchanged): sentence-transformers embeddings so a query like
"who can I see for my heart" matches cardiology entries without the literal
word "cardiology." Graceful degradation to a keyword-overlap scorer if
sentence-transformers isn't importable, so the agent stays functional
either way.

WHAT GENERALIZED: one ``RagEngine`` instance per domain (the "Silo"
multi-tenant pattern - one knowledge base per domain, not a shared vector
store with tenant partitioning, since this platform only needs
domain-level isolation, not per-customer-at-scale isolation). The
``HOSPITAL_KB``-style module-level list becomes a plain ``list[dict]``
passed into the constructor by ``kb_store.py``.
"""

from __future__ import annotations

import asyncio

TOP_K = 3


class RagEngine:
    """One domain's semantic-search index over its own knowledge base.

    Attributes:
        entries: The KB entries this engine searches, each a dict with
            at least ``title`` and ``text`` keys.
    """

    def __init__(self, entries: list[dict]) -> None:
        """Initialize with a domain's KB entries (no embedding yet - lazy).

        Args:
            entries: KB entries, each ``{"id", "category", "title",
                "text"}`` (matching ``hospital_kb.py``'s entry shape).
        """
        self.entries = entries
        self._embedder = None
        self._kb_embeddings = None
        self._kb_texts: list[str] | None = None

    def try_load_embedder(self) -> bool:
        """Lazy-load the embedder once. Called by ``compat``-style
        warm-up hooks at process start, before any real call, so the
        model doesn't lazy-load mid-conversation and blow a tool-call
        timeout.

        Returns:
            True if the real semantic backend is usable.
        """
        if self._embedder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._kb_texts = [f"{e['title']}. {e['text']}" for e in self.entries]
            self._kb_embeddings = self._embedder.encode(
                self._kb_texts, normalize_embeddings=True
            )
            return True
        except Exception:  # noqa: BLE001 - any import/load failure means "no semantic backend"
            self._embedder = None
            self._kb_embeddings = None
            self._kb_texts = None
            return False

    def reindex(self, entries: list[dict]) -> bool:
        """Re-embed from a fresh entry list (e.g. after a KB edit).

        Args:
            entries: The new/current KB entries.

        Returns:
            True if the semantic backend re-embedded, False if running
            on the keyword fallback (nothing to re-embed there - it
            reads ``entries`` live on every call).
        """
        self.entries = entries
        if self._embedder is None:
            return self.try_load_embedder()
        self._kb_texts = [f"{e['title']}. {e['text']}" for e in self.entries]
        self._kb_embeddings = self._embedder.encode(
            self._kb_texts, normalize_embeddings=True
        )
        return True

    def _semantic_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        import numpy as np

        q_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
        scores = self._kb_embeddings @ q_emb
        top_idx = np.argsort(-scores)[:k]
        return [(self.entries[i], float(scores[i])) for i in top_idx]

    def _keyword_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        q_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            entry_words = set((entry["title"] + " " + entry["text"]).lower().split())
            overlap = len(q_words & entry_words)
            scored.append((entry, float(overlap)))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def _search(self, query: str, k: int = TOP_K) -> list[tuple[dict, float]]:
        if self.try_load_embedder():
            return self._semantic_search(query, k)
        return self._keyword_search(query, k)

    async def search(self, query: str) -> dict:
        """Search this domain's knowledge base for a caller's question.

        Args:
            query: The caller's question, rephrased as a short search
                query.

        Returns:
            ``{"status": "ok", "answer": "..."}`` on a match, or
            ``{"status": "not_found", "message": "..."}`` if nothing
            scored above zero.
        """
        results = await asyncio.to_thread(self._search, query, TOP_K)
        relevant = [entry for entry, score in results if score > 0]
        if not relevant:
            return {
                "status": "not_found",
                "message": "No information found for that. Tell the caller "
                "you don't have that information rather than guessing.",
            }
        return {
            "status": "ok",
            "answer": " ".join(entry["text"] for entry in relevant),
        }
