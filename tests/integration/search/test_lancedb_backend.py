"""Tests for the LanceDB two-tier backend (RFC-090 §3.7, #855).

Uses a real embedded LanceDB in a temp dir with tiny 4-dim vectors (fast).
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import (
    InsightDocument,
    SearchBackend,
    SearchQuery,
    SegmentDocument,
)

pytestmark = pytest.mark.integration

lancedb = pytest.importorskip("lancedb")

from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402


def _seg(i, text, show, emb):
    return SegmentDocument(
        id=i, text=text, show_id=show, episode_id="ep1", start_time=0.0, end_time=1.0, embedding=emb
    )


@pytest.fixture
def backend(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.upsert_segment(_seg("s1", "Sam Altman talks about OpenAI", "A", [0.1, 0.2, 0.3, 0.4]))
    b.upsert_segment(_seg("s2", "Tim Cook and Apple earnings", "B", [0.9, 0.1, 0.0, 0.1]))
    b.upsert_insight(
        InsightDocument(
            id="insight:1",
            text="Altman argues AI scaling continues",
            show_id="A",
            episode_id="ep1",
            entity_type="claim",
            confidence=0.9,
            derived=False,
            embedding=[0.1, 0.2, 0.3, 0.45],
        )
    )
    b.create_indices()
    return b


def test_satisfies_protocol(backend):
    assert isinstance(backend, SearchBackend)


def test_bm25_named_entity(backend):
    # BM25 retrieves the proper noun across both tiers.
    res = backend.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="all"))
    ids = {r.doc_id for r in res}
    assert "s1" in ids and "insight:1" in ids
    assert all(r.signal == "bm25" for r in res)
    assert res[0].rank == 1


def test_vector_search_orders_by_similarity(backend):
    res = backend.search_vector(
        SearchQuery(text="", embedding=[0.1, 0.2, 0.3, 0.4], tier="segment")
    )
    assert res and res[0].doc_id == "s1"  # closest segment
    assert all(r.signal == "vector" and r.source_tier == "segment" for r in res)
    # similarity score is higher-is-better.
    assert res[0].score >= res[-1].score


def test_tier_filter_segment_only(backend):
    res = backend.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))
    assert {r.source_tier for r in res} == {"segment"}


def test_payload_excludes_embedding(backend):
    res = backend.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="all"))
    assert "embedding" not in res[0].payload
    assert res[0].payload.get("show_id") == "A"


def test_filters_where_show(backend):
    res = backend.search_bm25(
        SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment", filters={"show_id": "B"})
    )
    assert res == []  # s1 is show A; filtered out


def test_health_and_delete(backend):
    h = backend.health()
    assert h["status"] == "ok" and h["segments"] == 2 and h["insights"] == 1
    backend.delete("s1", "segment")
    assert backend.health()["segments"] == 1


@pytest.mark.critical_path
def test_create_indices_skips_ann_on_small_tables(backend):
    # 2 segment / 1 insight rows are far below _MIN_VECTOR_INDEX_ROWS, so the IVF
    # ANN build (which can SIGSEGV on too-few rows) is skipped — FTS still builds
    # and the tables stay searchable via brute force. Must not raise.
    assert backend._MIN_VECTOR_INDEX_ROWS > 2
    backend.create_indices()
    res = backend.search_bm25(SearchQuery(text="OpenAI", embedding=[0, 0, 0, 0], tier="segment"))
    assert "s1" in {r.doc_id for r in res}  # brute-force FTS still finds the row


@pytest.mark.critical_path
def test_create_indices_noop_when_no_tables(tmp_path):
    # A fresh backend has no tier tables on disk: every tier short-circuits on the
    # ``table is None`` branch — create_indices() is a clean no-op.
    empty = LanceDBBackend(str(tmp_path / "empty"), embed_dim=4)
    empty.create_indices()  # must not raise


@pytest.mark.critical_path
def test_sql_str_escapes_and_to_sql_builds_clause(backend):
    assert LanceDBBackend._sql_str("O'Brien") == "O''Brien"
    assert backend._to_sql({}) is None
    assert backend._to_sql({"show_id": "A"}) == "show_id = 'A'"
    # Embedded quote is escaped so it cannot break out of the literal.
    assert backend._to_sql({"id": "x'y"}) == "id = 'x''y'"


# --- stale-index reopen (#stack-test search flake) ---------------------------------------------
# The API holds a long-lived LanceDBBackend while the pipeline container runs compact() on the
# shared corpus volume. A cached table handle then references pruned data fragments and reads throw
# "Not found: .../segments.lance/data/<frag>". The backend must reopen the table and retry.

from podcast_scraper.search.backends.lancedb_backend import (  # noqa: E402
    _is_stale_index_error,
)


def test_is_stale_index_error_classification() -> None:
    assert _is_stale_index_error(
        RuntimeError("Not found: /app/output/search/lance_index/segments.lance/data/ab12")
    )
    assert _is_stale_index_error(
        RuntimeError("LanceError(IO): Object at location /app/output/x not found")
    )
    # Not a stale-fragment error -> must NOT reopen (would mask real failures).
    assert not _is_stale_index_error(RuntimeError("connection refused"))
    assert not _is_stale_index_error(ValueError("some other error"))
    assert not _is_stale_index_error(
        RuntimeError("segments.lance/data/ab12 exists")
    )  # no "not found"


def test_read_tier_reopens_once_on_stale_fragment(backend, monkeypatch) -> None:
    """A read that hits a compacted-away fragment reopens the table exactly once and returns the
    fresh result — the search does not fail."""
    stale = RuntimeError("Not found: /app/output/search/lance_index/segments.lance/data/ab12")
    stale_table, fresh_table = object(), object()
    calls = {"reopen": 0}
    monkeypatch.setattr(backend, "_open_if_exists", lambda tier: stale_table)

    def _reopen(tier):
        calls["reopen"] += 1
        return fresh_table

    monkeypatch.setattr(backend, "_reopen_table", _reopen)

    def run(table):
        if table is stale_table:
            raise stale
        return [{"id": "recovered"}]

    assert backend._read_tier("segment", run) == [{"id": "recovered"}]
    assert calls["reopen"] == 1


def test_read_tier_propagates_non_stale_error(backend, monkeypatch) -> None:
    """A non-stale error must propagate (reopening would hide a real bug)."""
    monkeypatch.setattr(backend, "_open_if_exists", lambda tier: object())
    monkeypatch.setattr(
        backend, "_reopen_table", lambda tier: pytest.fail("must not reopen on a non-stale error")
    )

    def run(table):
        raise ValueError("genuine failure")

    with pytest.raises(ValueError, match="genuine failure"):
        backend._read_tier("segment", run)


def test_search_survives_cross_backend_compaction(tmp_path) -> None:
    """Real scenario: the API's cached handle survives a compaction done by a SECOND backend on the
    same path (the pipeline). Search must still return the row after the compaction."""
    path = str(tmp_path / "lance")
    api = LanceDBBackend(path, embed_dim=4)
    api.upsert_segment(_seg("s1", "Sam Altman OpenAI", "A", [0.1, 0.2, 0.3, 0.4]))
    api.create_indices()
    # API reads once -> caches the segment table handle at the current version.
    assert api.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))

    # "pipeline": a separate backend writes more rows + compacts (prunes superseded fragments).
    pipeline = LanceDBBackend(path, embed_dim=4)
    for i in range(6):
        pipeline.upsert_segment(_seg(f"x{i}", f"filler text {i}", "A", [0.1, 0.1, 0.1, 0.1]))
    pipeline.create_indices()
    pipeline.compact()

    # API reads again with its (now stale) cached handle -> reopens + still finds s1.
    res = api.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))
    assert any(r.doc_id == "s1" for r in res)
