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


# --- concurrency + cross-process compaction (documented LanceDB server pattern) ----------------
# The api holds a long-lived backend and serves search from many threads while the pipeline — a
# SEPARATE process/backend on the same path — writes and compact()s. The backend opens every table
# FRESH per read (never caches a handle), so a compaction can neither strand a read with a
# pruned-fragment "not found" nor corrupt a sibling read (the native use-after-free that segfaulted
# the api when a cached handle was swapped under concurrent readers). These exercise that at the
# integration tier so a regression surfaces here, not only in the full stack test.

import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402


def test_fresh_open_sees_a_second_backends_writes(tmp_path) -> None:
    """A read opens the table fresh, so rows another backend committed after this one was built are
    visible on the next read — no reopen/reconnect needed."""
    path = str(tmp_path / "lance")
    api = LanceDBBackend(path, embed_dim=4)
    api.upsert_segment(_seg("s1", "Sam Altman OpenAI", "A", [0.1, 0.2, 0.3, 0.4]))
    api.create_indices()
    assert api.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))

    # A second backend (the pipeline) appends a new row after `api` already served a query.
    pipeline = LanceDBBackend(path, embed_dim=4)
    pipeline.upsert_segment(_seg("s2", "Sundar Pichai Google", "A", [0.4, 0.3, 0.2, 0.1]))
    pipeline.create_indices()

    # `api` sees s2 on its next read — no cached-handle staleness.
    res = api.search_bm25(SearchQuery(text="Pichai", embedding=[0, 0, 0, 0], tier="segment"))
    assert any(r.doc_id == "s2" for r in res)


def test_search_survives_cross_process_compaction(tmp_path) -> None:
    """A compaction (optimize + prune) run by a SECOND backend must not strand this backend's
    search — the fresh handle points at the current version, never the pruned fragments."""
    path = str(tmp_path / "lance")
    api = LanceDBBackend(path, embed_dim=4)
    api.upsert_segment(_seg("s1", "Sam Altman OpenAI", "A", [0.1, 0.2, 0.3, 0.4]))
    api.create_indices()
    assert api.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))

    pipeline = LanceDBBackend(path, embed_dim=4)
    for i in range(6):
        pipeline.upsert_segment(_seg(f"x{i}", f"filler text {i}", "A", [0.1, 0.1, 0.1, 0.1]))
    pipeline.create_indices()
    pipeline.compact()  # prunes the fragments/versions the earlier writes created

    res = api.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))
    assert any(r.doc_id == "s1" for r in res)


def test_concurrent_hybrid_reads_during_compaction(tmp_path) -> None:
    """Regression guard for the api SIGSEGV: many threads run hybrid search on one backend while a
    second backend compacts the same path in a loop. With fresh-per-read handles nothing shared is
    mutated, so every read returns cleanly (no raise) and finds the always-present target row.

    (A native segfault is timing/platform dependent and may not reproduce here; this pins the
    correct behaviour — no stale-fragment errors, no shared-handle corruption — at the integration
    tier so a regression to cached handles is caught long before the stack test.)
    """
    path = str(tmp_path / "lance")
    api = LanceDBBackend(path, embed_dim=4)
    api.upsert_segment(_seg("s1", "Sam Altman OpenAI keynote", "A", [0.1, 0.2, 0.3, 0.4]))
    for i in range(12):
        api.upsert_segment(_seg(f"f{i}", f"filler passage number {i}", "A", [0.2, 0.2, 0.2, 0.2]))
    api.create_indices()

    pipeline = LanceDBBackend(path, embed_dim=4)
    stop = threading.Event()
    errors: list[BaseException] = []

    def _compactor() -> None:
        while not stop.is_set():
            try:
                pipeline.upsert_segment(
                    _seg("s1", "Sam Altman OpenAI keynote", "A", [0.1, 0.2, 0.3, 0.4])
                )
                pipeline.create_indices()
                pipeline.compact()
            except BaseException as exc:  # noqa: BLE001 - record; the reads are what we assert on
                errors.append(exc)
                return

    def _reader(_i: int) -> bool:
        q = SearchQuery(text="Altman keynote", embedding=[0.1, 0.2, 0.3, 0.4], tier="segment")
        return any(r.doc_id == "s1" for r in api.search_hybrid(q))

    compactor = threading.Thread(target=_compactor, daemon=True)
    compactor.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_reader, range(48)))
    finally:
        stop.set()
        compactor.join(timeout=10)

    # Every concurrent read completed without raising; s1 is always present so every read finds it.
    assert all(results)
