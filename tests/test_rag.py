"""Tests for the RAG retrieval logic (src/rag.py, src/build_index.py, and the rag_chunks
accessors in src/db.py). No live embedding model or network — the embed calls are monkeypatched
with small fixed vectors, matching the "no live model" philosophy of test_banking.py.

Run: pytest tests/ -q
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import build_index  # noqa: E402
import db            # noqa: E402
import rag           # noqa: E402


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


# ── Heading-based chunking (build_index.parse_markdown_text) ────────────────────

FIXTURE_MD = """# Test Doc

Intro paragraph before any heading.

## Fees

General fee info here.

### Overdraft

Overdraft fee is high.

## Eligibility

Age 18+.
"""


def test_parse_markdown_splits_on_headings():
    chunks = build_index.parse_markdown_text(FIXTURE_MD)
    assert chunks == [
        (None, "Intro paragraph before any heading."),
        ("Fees", "General fee info here."),
        ("Fees > Overdraft", "Overdraft fee is high."),
        ("Eligibility", "Age 18+."),
    ]


def test_parse_markdown_skips_empty_sections():
    md = "## A\n## B\ntext under B\n"
    chunks = build_index.parse_markdown_text(md)
    assert chunks == [("B", "text under B")]


# ── Context-aware retrieval query (rag.build_retrieval_query) ─────────────────────

def test_build_retrieval_query_first_turn_is_just_current():
    history = [{"role": "user", "content": "tell me about the rutba account"}]
    assert rag.build_retrieval_query(history) == "tell me about the rutba account"


def test_build_retrieval_query_prepends_last_exchange():
    history = [
        {"role": "user", "content": "something for old citizens"},
        {"role": "assistant", "content": "The HBL Rutba account is for senior citizens aged 55 and above."},
        {"role": "user", "content": "what is the age eligibility for it"},
    ]
    q = rag.build_retrieval_query(history)
    # current utterance present in full, and prior exchange folded in to carry the topic
    assert q.endswith("what is the age eligibility for it")
    assert "Rutba" in q and "old citizens" in q


def test_build_retrieval_query_caps_prior_context_but_not_current():
    long_prev = "x" * 500
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": long_prev},
        {"role": "user", "content": "current question in full"},
    ]
    q = rag.build_retrieval_query(history, max_prev_chars=200)
    assert q.endswith("current question in full")
    assert q.count("x") == 200   # prior assistant message capped


def test_build_retrieval_query_drops_leading_greeting():
    # The opening greeting is an assistant message with no preceding user turn; it's boilerplate
    # and must not pollute the retrieval query on the caller's first real question.
    history = [
        {"role": "assistant", "content": "Good afternoon, thank you for calling HBL, how may I help?"},
        {"role": "user", "content": "what accounts do you offer"},
    ]
    assert rag.build_retrieval_query(history) == "what accounts do you offer"


def test_build_retrieval_query_ignores_tool_and_system_messages():
    history = [
        {"role": "user", "content": "about rutba"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": '{"status":"found"}', "tool_call_id": "1"},
        {"role": "user", "content": "eligibility?"},
    ]
    q = rag.build_retrieval_query(history)
    assert "found" not in q and q.endswith("eligibility?")


def test_chunk_document_labels_with_doc_name():
    chunks = build_index.chunk_document("fees.md", FIXTURE_MD)
    assert all(c["doc_name"] == "fees.md" for c in chunks)
    assert chunks[1] == {"doc_name": "fees.md", "section": "Fees", "text": "General fee info here."}


# ── Windowing fallback for oversized sections ────────────────────────────────────

def test_window_split_short_text_is_one_piece():
    assert build_index._window_split("short text", ceiling=100) == ["short text"]


def test_window_split_long_text_produces_overlapping_pieces():
    text = "0123456789" * 12   # 120 distinct-position chars, so slices are easy to verify
    pieces = build_index._window_split(text, ceiling=50, overlap_ratio=0.2)
    assert len(pieces) > 1
    assert all(len(p) <= 50 for p in pieces)
    assert pieces[-1] == text[-len(pieces[-1]):]   # last piece reaches the end of the text
    # consecutive pieces overlap: the tail of piece i reappears at the head of piece i+1
    overlap_len = int(50 * 0.2)
    assert pieces[0][-overlap_len:] == pieces[1][:overlap_len]


def test_chunk_document_falls_back_to_windowing_for_oversized_section():
    long_body = "word " * 500   # 2500 chars, well past the default 1800-char ceiling
    md = f"## Big Section\n\n{long_body}\n"
    chunks = build_index.chunk_document("big.md", md)
    assert len(chunks) > 1
    assert chunks[0]["section"] == "Big Section (part 1)"
    assert chunks[1]["section"] == "Big Section (part 2)"


# ── db.py rag_chunks accessors ────────────────────────────────────────────────────

def test_insert_and_fetch_all_rag_chunks_in_order(conn):
    id1 = db.insert_rag_chunk(conn, "a.md", "Sec A", "text a", "2026-01-01T00:00:00")
    id2 = db.insert_rag_chunk(conn, "b.md", None, "text b", "2026-01-01T00:00:00")
    assert id2 == id1 + 1
    rows = db.fetch_all_rag_chunks(conn)
    assert [r["id"] for r in rows] == [id1, id2]
    assert rows[0]["doc_name"] == "a.md" and rows[0]["section"] == "Sec A"
    assert rows[1]["doc_name"] == "b.md" and rows[1]["section"] is None


def test_clear_rag_chunks_resets_autoincrement(conn):
    db.insert_rag_chunk(conn, "a.md", None, "text", "2026-01-01T00:00:00")
    db.insert_rag_chunk(conn, "b.md", None, "text", "2026-01-01T00:00:00")
    db.clear_rag_chunks(conn)
    assert db.fetch_all_rag_chunks(conn) == []
    new_id = db.insert_rag_chunk(conn, "c.md", None, "text", "2026-01-01T00:00:00")
    assert new_id == 1   # autoincrement restarted


# ── rag.search_docs retrieval math (fixed fake vectors, no real model) ──────────────

class FakeEmbedModel:
    """Stands in for fastembed.TextEmbedding — query_embed() returns a fixed vector regardless
    of the query text, so tests are deterministic and don't need the real model/network."""
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def query_embed(self, texts):
        return [self.vector for _ in texts]


# 4-dim orthonormal fixture: three "chunks" pointing along separate axes.
FIXTURE_MATRIX = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
], dtype=np.float32)
FIXTURE_META = [
    {"doc_name": "a.md", "section": "A", "text": "chunk a"},
    {"doc_name": "b.md", "section": "B", "text": "chunk b"},
    {"doc_name": "c.md", "section": "C", "text": "chunk c"},
]


def _install_fixture_index(monkeypatch, query_vector):
    monkeypatch.setattr(rag, "_doc_matrix", FIXTURE_MATRIX)
    monkeypatch.setattr(rag, "_chunk_meta", FIXTURE_META)
    monkeypatch.setattr(rag, "_embed_model", FakeEmbedModel(query_vector))


def test_search_docs_returns_closest_match_above_threshold(monkeypatch):
    _install_fixture_index(monkeypatch, [0.0, 0.95, 0.05, 0.0])   # close to chunk b
    out = rag.search_docs("what about b?")
    assert out["status"] == "found"
    assert out["results"][0]["source"] == "b.md"
    assert out["results"][0]["text"] == "chunk b"


def test_search_docs_no_match_below_threshold(monkeypatch):
    _install_fixture_index(monkeypatch, [0.0, 0.0, 0.0, 1.0])   # orthogonal to all fixture chunks
    out = rag.search_docs("something unrelated")
    assert out == {"status": "no_match"}


def test_search_docs_empty_query_is_no_match(monkeypatch):
    _install_fixture_index(monkeypatch, [1.0, 0.0, 0.0, 0.0])
    assert rag.search_docs("") == {"status": "no_match"}
    assert rag.search_docs("   ") == {"status": "no_match"}


def test_search_docs_no_index_loaded_is_no_match(monkeypatch):
    monkeypatch.setattr(rag, "_doc_matrix", None)
    monkeypatch.setattr(rag, "_embed_model", FakeEmbedModel([1.0, 0.0, 0.0, 0.0]))
    assert rag.search_docs("anything") == {"status": "no_match"}


class MultiFakeEmbedModel:
    """Returns a different fixed vector per query string, so multi-query merge can be exercised."""
    def __init__(self, mapping):
        self.mapping = mapping

    def query_embed(self, texts):
        return [np.asarray(self.mapping[t], dtype=np.float32) for t in texts]


def test_search_docs_multi_merges_best_score_per_chunk(monkeypatch):
    # query "q_b" points at chunk b; query "q_c" points at chunk c. The merge should surface both.
    monkeypatch.setattr(rag, "_doc_matrix", FIXTURE_MATRIX)
    monkeypatch.setattr(rag, "_chunk_meta", FIXTURE_META)
    monkeypatch.setattr(rag, "_embed_model", MultiFakeEmbedModel({
        "q_b": [0.0, 1.0, 0.0, 0.0],
        "q_c": [0.0, 0.0, 1.0, 0.0],
    }))
    out = rag.search_docs_multi(["q_b", "q_c"])
    assert out["status"] == "found"
    sources = {r["source"] for r in out["results"]}
    assert "b.md" in sources and "c.md" in sources   # both query topics represented


def test_search_docs_limits_to_top_k_sorted_by_similarity(monkeypatch):
    # A query with positive similarity to all three fixture chunks, strongest to weakest a>b>c.
    # Threshold is zeroed out here so this test isolates top-k limiting from threshold filtering
    # (covered separately above) — it shouldn't need updating whenever the real threshold is retuned.
    monkeypatch.setattr(rag, "TOP_K", 2)
    monkeypatch.setattr(rag, "SIMILARITY_THRESHOLD", 0.0)
    _install_fixture_index(monkeypatch, [0.8, 0.5, 0.3, 0.0])
    out = rag.search_docs("broad query")
    assert out["status"] == "found"
    assert len(out["results"]) == 2   # capped at TOP_K even though all 3 are above threshold
    assert out["results"][0]["source"] == "a.md"   # highest similarity first
    assert out["results"][1]["source"] == "b.md"


# ── build_index.build() end-to-end, against a bare (schema-less) connection ─────────────────
# Regression test: build() must create the schema itself (db.init_schema), not assume the caller
# already did — a live run against an existing bank.db that predated the rag_chunks table hit
# exactly this ("no such table: rag_chunks") before build() called init_schema itself.

class FakePassageEmbedModel:
    """Stands in for fastembed.TextEmbedding at build time — embed() (passage mode) returns a
    fixed-dim vector per input text, deterministic and network-free."""
    def __init__(self, model_name=None):
        pass

    def embed(self, texts):
        return [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in texts]


def test_build_creates_schema_on_bare_connection(tmp_path, monkeypatch):
    docs_dir = tmp_path / "rag_docs"
    docs_dir.mkdir()
    (docs_dir / "one.md").write_text("## Section One\n\nSome content here.\n")

    bare_conn = db.connect(":memory:")   # deliberately NOT calling db.init_schema first

    # Stub sys.modules["fastembed"] so build()'s `from fastembed import TextEmbedding` succeeds
    # without the real (heavy, network-fetched) package installed — works whether or not the real
    # fastembed is present, keeping this test network-free like the rest of the suite.
    fake_module = types.ModuleType("fastembed")
    fake_module.TextEmbedding = FakePassageEmbedModel
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    monkeypatch.setattr(rag, "INDEX_MATRIX_PATH", str(tmp_path / "rag_index.npy"))

    summary = build_index.build(bare_conn, docs_dir=str(docs_dir))

    assert summary["chunk_count"] == 1
    rows = db.fetch_all_rag_chunks(bare_conn)
    assert len(rows) == 1
    assert rows[0]["doc_name"] == "one.md"
    assert os.path.exists(str(tmp_path / "rag_index.npy"))
    bare_conn.close()
