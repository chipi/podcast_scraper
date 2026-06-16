"""Integration test for the from-corpus two-tier indexer (RFC-090 Phase 2 / B).

Stubs the (already-tested) FAISS-indexer corpus extraction and exercises this
module's own logic — embed → map → upsert → both tiers queryable — against a real
LanceDB index with real MiniLM embeddings.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search import hybrid_search as hs, two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402


def _stub_extraction(monkeypatch, tmp_path, rows):
    meta_path = tmp_path / "corpus" / "metadata" / "ep1.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta_path])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: tmp_path / "corpus")
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)


def test_builds_both_tiers_and_is_queryable(tmp_path, monkeypatch):
    rows = [
        (
            "insight:1",
            "The central bank is shifting monetary policy",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "show1", "grounded": True},
        ),
        (
            "chunk:1",
            "Markets moved sharply as the central bank signaled a policy shift",
            {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "feed_id": "show1",
                "timestamp_start_ms": 2000,
                "timestamp_end_ms": 5000,
            },
        ),
        # A non-two-tier row (quote) must be ignored.
        ("quote:1", "we are shifting", {"doc_type": "quote", "episode_id": "ep1"}),
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)

    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"
    stats = tti.build_two_tier_index(corpus, lance)

    assert stats.episodes == 1
    assert stats.insights == 1 and stats.segments == 1  # quote ignored

    backend = LanceDBBackend(str(lance))
    health = backend.health()
    assert health["insights"] == 1 and health["segments"] == 1

    # Index meta records the model + the model's real dim (derived, not assumed 384).
    meta = backend.read_index_meta()
    assert meta is not None and meta["embedding_model"]
    assert meta["embed_dim"] == 384  # MiniLM

    rows_out = hs.hybrid_candidates(corpus, "central bank policy shift", top_k=5)
    assert rows_out is not None and len(rows_out) >= 1
    by_id = {r.doc_id: r for r in rows_out}
    assert by_id["chunk:1"].metadata["timestamp_start_ms"] == 2000  # ms preserved via seconds


def test_writes_metadata_json_sidecar_with_char_offsets(tmp_path, monkeypatch):
    """#1010 regression: GIL offset verify reads ``search/metadata.json`` (doc_id -> meta with
    char_start/char_end). FAISS wrote it; the LanceDB build must re-emit it from chunk meta.
    """
    rows = [
        (
            "chunk:0",
            "Markets moved sharply as the central bank signaled a policy shift",
            {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "feed_id": "show1",
                "char_start": 0,
                "char_end": 64,
                "timestamp_start_ms": 0,
                "timestamp_end_ms": 4000,
            },
        ),
        (
            "insight:1",
            "The central bank is shifting monetary policy",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "show1", "grounded": True},
        ),
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    tti.build_two_tier_index(corpus, corpus / "search" / "lance_index")

    # Sidecar lands next to the lance index, where the verifier resolves <root>/search.
    assert (corpus / "search" / "metadata.json").is_file(), "metadata.json sidecar not written"

    # The verifier's OWN reader must parse it and recover the transcript chunk char spans.
    from podcast_scraper.search.gil_chunk_offset_verify import (
        load_index_metadata_map,
        transcript_chunk_spans_by_episode,
    )

    meta_map = load_index_metadata_map(corpus / "search")
    assert meta_map["chunk:0"]["char_start"] == 0 and meta_map["chunk:0"]["char_end"] == 64
    assert transcript_chunk_spans_by_episode(meta_map) == {"ep1": [(0, 64)]}
    # Bloat guard: the sidecar must NOT re-duplicate the embedded chunk text — the verifier
    # only needs the char offsets (see two_tier_indexer text-strip).
    assert "text" not in meta_map["chunk:0"], "sidecar should not carry full chunk text"


def test_linking_populates_compounds(tmp_path, monkeypatch):
    """Native index links insight↔segment so dedup actually produces a CompoundResult."""
    import json

    from podcast_scraper.search.backend import SearchQuery
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend
    from podcast_scraper.search.dedup import deduplicate

    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    gi = corpus / "ep1.gi.json"
    gi.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "insight:n1", "type": "Insight", "properties": {"text": "x"}},
                    {
                        "id": "quote:q1",
                        "type": "Quote",
                        "properties": {"timestamp_start_ms": 1000, "timestamp_end_ms": 4000},
                    },
                ],
                "edges": [{"type": "SUPPORTED_BY", "from": "insight:n1", "to": "quote:q1"}],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        (
            "insight:s:insight:n1",
            "central bank policy shift",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s", "source_id": "insight:n1"},
        ),
        (
            "chunk:s:1",
            "the central bank signaled a policy shift in markets",
            {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "feed_id": "s",
                "timestamp_start_ms": 0,
                "timestamp_end_ms": 10000,
            },
        ),  # span covers the quote (1–4s)
    ]
    monkeypatch.setattr(
        tti, "discover_metadata_files", lambda root: [corpus / "metadata" / "ep1.json"]
    )
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    monkeypatch.setattr(tti, "_gi_path", lambda root, mp, doc: gi)

    lance = corpus / "search" / "lance_index"
    stats = tti.build_two_tier_index(corpus, lance)
    assert stats.linked == 1  # the insight linked to the segment

    backend = LanceDBBackend(str(lance))
    hits = backend.search_bm25(
        SearchQuery(text="central bank policy shift", embedding=[], tier="all")
    )
    compounds = [r for r in deduplicate(hits) if r.source_tier == "compound"]
    assert compounds, "linked insight+segment must dedup into a CompoundResult"


def test_faiss_metadata_parity_publish_date_and_source_id(tmp_path, monkeypatch):
    """Regression: hybrid hits must carry FAISS-parity metadata fields.

    - ``publish_date``: the shared ``since`` filter drops any hit lacking it, which
      silently zeroed out the digest topic-bands (always pass a ``since`` bound) when
      a corpus was served via LanceDB instead of FAISS.
    - ``source_id``: the viewer's "Show on graph" affordance reads it (focusable
      tiers) to resolve the graph node; without it the handoff never renders.
    """
    rows = [
        (
            "insight:1",
            "The central bank is shifting monetary policy",
            {
                "doc_type": "insight",
                "episode_id": "ep1",
                "feed_id": "show1",
                "grounded": True,
                "publish_date": "2026-06-07",
                "source_id": "insight:n1",
            },
        ),
        (
            "kg_topic:1",
            "monetary policy",
            {
                "doc_type": "kg_topic",
                "episode_id": "ep1",
                "feed_id": "show1",
                "publish_date": "2026-06-07",
                "source_id": "topic:monetary-policy",
            },
        ),
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"
    tti.build_two_tier_index(corpus, lance)

    rows_out = hs.hybrid_candidates(corpus, "monetary policy", top_k=5)
    assert rows_out is not None and rows_out
    # Every hit (insight + aux tier) carries the episode publish date.
    assert all(r.metadata.get("publish_date") == "2026-06-07" for r in rows_out)
    # The focusable kg_topic hit carries its canonical graph node id.
    topic_hit = next(r for r in rows_out if r.metadata.get("doc_type") == "kg_topic")
    assert topic_hit.metadata.get("source_id") == "topic:monetary-policy"


def test_stale_schema_is_detected_and_read_falls_back(tmp_path, monkeypatch):
    """A pre-schema-bump index is flagged stale, so the read path skips it (FAISS)."""
    from podcast_scraper.search.backends import lancedb_backend as lb

    rows = [("insight:1", "x", {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s"})]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"
    tti.build_two_tier_index(corpus, lance)

    # Fresh build is current-schema → not stale → served.
    assert lb.stored_schema_version(lance) == lb.LANCE_SCHEMA_VERSION
    assert lb.lance_index_is_stale(lance) is False
    assert hs.hybrid_candidates(corpus, "x", top_k=5) is not None

    # Simulate a pre-versioning index (meta without schema_version → version 1).
    import json

    meta = json.loads((lance / "index_meta.json").read_text())
    meta.pop("schema_version", None)
    (lance / "index_meta.json").write_text(json.dumps(meta))
    assert lb.stored_schema_version(lance) == 1
    assert lb.lance_index_is_stale(lance) is True
    # Read path must skip a stale index and defer to FAISS (returns None).
    assert hs.hybrid_candidates(corpus, "x", top_k=5) is None


def test_stale_schema_build_clears_index_and_sidecar_then_rebuilds(tmp_path, monkeypatch):
    """A present-but-stale index (no drop_existing) is wiped via _clear_index — including the
    sibling metadata.json sidecar — then rebuilt fresh (covers the stale-schema reindex path)."""
    import json

    from podcast_scraper.search.backends import lancedb_backend as lb

    rows = [
        (
            "chunk:0",
            "central bank policy text",
            {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "feed_id": "s",
                "char_start": 0,
                "char_end": 24,
            },
        )
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"

    tti.build_two_tier_index(corpus, lance)  # fresh build → index + metadata.json sidecar
    # Make it pre-versioning stale, and plant a stale entry in the existing sidecar.
    meta = json.loads((lance / "index_meta.json").read_text())
    meta.pop("schema_version", None)
    (lance / "index_meta.json").write_text(json.dumps(meta))
    assert lb.lance_index_is_stale(lance) is True
    (corpus / "search" / "metadata.json").write_text('{"stale:doc": {"doc_type": "transcript"}}')

    # Rebuild WITHOUT drop_existing → the stale branch clears index + sidecar, then rebuilds.
    tti.build_two_tier_index(corpus, lance)
    assert lb.lance_index_is_stale(lance) is False  # rebuilt fresh
    sidecar = json.loads((corpus / "search" / "metadata.json").read_text())
    assert "stale:doc" not in sidecar and "chunk:0" in sidecar  # old sidecar wiped, fresh written


def test_limit_episodes_caps_walk(tmp_path, monkeypatch):
    rows = [("insight:1", "x", {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s"})]
    # Two metadata files, but limit_episodes=0 stops before any work.
    meta = tmp_path / "corpus" / "metadata" / "ep1.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta, meta])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: tmp_path / "corpus")
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    (tmp_path / "corpus" / "metadata").mkdir(parents=True)

    stats = tti.build_two_tier_index(
        tmp_path / "corpus", tmp_path / "corpus" / "search" / "li", limit_episodes=0
    )
    assert stats.episodes == 0 and stats.insights == 0


def test_repeated_reindex_stays_bounded_via_compaction(tmp_path, monkeypatch):
    """Repeated (incremental) reindex must not grow the index unboundedly.

    LanceDB is MVCC and the indexer upserts per-document, so without compaction every
    build piles up fragments + versions (observed in prod: 8k fragments / 2.7G on one
    table). Each build now compacts + prunes old versions, so the retained-version
    count stays bounded no matter how many times we reindex.
    """
    rows = [
        ("insight:1", "central bank policy", {"doc_type": "insight", "episode_id": "ep1"}),
        (
            "chunk:1",
            "markets moved as the bank signaled a shift",
            {"doc_type": "transcript", "episode_id": "ep1"},
        ),
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"

    def _max_retained_versions() -> int:
        return max(
            (len(list(p.glob("*"))) for p in lance.glob("*.lance/_versions")),
            default=0,
        )

    # Five incremental reindexes — the path the pipeline runs after each batch.
    for _ in range(5):
        tti.build_two_tier_index(corpus, lance)

    # Compaction keeps retained versions tiny (a handful, not one-per-build-per-doc).
    assert _max_retained_versions() <= 4, f"versions unbounded: {_max_retained_versions()}"
    # Data is still correct + queryable after compaction.
    backend = LanceDBBackend(str(lance))
    health = backend.health()
    assert health["insights"] == 1 and health["segments"] == 1


def test_configurable_batch_size_flushes_in_chunks(tmp_path, monkeypatch):
    """A small ``upsert_batch_size`` flushes mid-build, but every row still lands."""
    rows = [
        (
            f"insight:{i}",
            f"insight {i} on monetary policy",
            {"doc_type": "insight", "episode_id": "ep1"},
        )
        for i in range(5)
    ] + [
        (
            f"chunk:{i}",
            f"transcript chunk {i}: markets moved on the policy signal",
            {"doc_type": "transcript", "episode_id": "ep1"},
        )
        for i in range(5)
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"

    # batch_size=2 over 5 rows/tier → multiple flushes + a final partial flush.
    stats = tti.build_two_tier_index(corpus, lance, upsert_batch_size=2)
    assert stats.insights == 5 and stats.segments == 5

    backend = LanceDBBackend(str(lance))
    health = backend.health()
    assert health["insights"] == 5 and health["segments"] == 5  # nothing dropped across flushes


def test_aux_buffer_flushes_mid_build_at_batch_size(tmp_path, monkeypatch):
    """A build with more aux rows than upsert_batch_size flushes the aux tier mid-walk
    (covers the len(aux_buf) >= batch aux-flush branch) and lands every row."""
    rows = [
        (
            f"kg_topic:{i}",
            f"topic {i} on monetary policy",
            {"doc_type": "kg_topic", "episode_id": "ep1", "feed_id": "s"},
        )
        for i in range(5)
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"

    # batch_size=2 over 5 aux rows -> at least two mid-build aux flushes + a final partial.
    stats = tti.build_two_tier_index(corpus, lance, upsert_batch_size=2)
    assert stats.aux == 5 and stats.segments == 0 and stats.insights == 0
    assert LanceDBBackend(str(lance)).health()["aux"] == 5  # nothing dropped across flushes


def test_grounding_quote_texts_links_insight_to_segment(tmp_path, monkeypatch):
    """Text-containment grounding (verbatim quote text) links an insight to the segment that
    contains it — the primary linking path that needs no segment timestamps."""
    import json

    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    gi = corpus / "ep1.gi.json"
    gi.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "insight:n1", "type": "Insight", "properties": {"text": "x"}},
                    # A Quote with verbatim text (drives _insight_grounding_quote_texts).
                    {
                        "id": "quote:q1",
                        "type": "Quote",
                        "properties": {"text": "central bank signaled a policy shift"},
                    },
                    # A Quote with blank text must be ignored (the strip()/isinstance guard).
                    {"id": "quote:q2", "type": "Quote", "properties": {"text": "   "}},
                ],
                "edges": [{"type": "SUPPORTED_BY", "from": "insight:n1", "to": "quote:q1"}],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        (
            "insight:s:insight:n1",
            "the central bank policy outlook",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s", "source_id": "insight:n1"},
        ),
        (
            "chunk:s:1",
            "markets reacted after the central bank signaled a policy shift today",
            {"doc_type": "transcript", "episode_id": "ep1", "feed_id": "s"},
        ),  # contains the verbatim quote text -> text-containment link
    ]
    monkeypatch.setattr(
        tti, "discover_metadata_files", lambda root: [corpus / "metadata" / "ep1.json"]
    )
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    monkeypatch.setattr(tti, "_gi_path", lambda root, mp, doc: gi)

    stats = tti.build_two_tier_index(corpus, corpus / "search" / "lance_index")
    assert stats.linked == 1  # insight linked to the segment via verbatim quote text


def test_grounding_quote_texts_handles_unreadable_gi(tmp_path):
    """_insight_grounding_quote_texts returns {} for a missing or malformed gi.json
    (covers the is_file()=False and the OSError/ValueError guards)."""
    # Missing file -> {}.
    assert tti._insight_grounding_quote_texts(tmp_path / "nope.gi.json") == {}
    # Malformed JSON -> the try/except returns {}.
    bad = tmp_path / "bad.gi.json"
    bad.write_text("{not json", encoding="utf-8")
    assert tti._insight_grounding_quote_texts(bad) == {}


def test_grounding_quotes_handles_unreadable_gi(tmp_path):
    """_insight_grounding_quotes (timestamp variant) returns {} for missing/malformed gi.json."""
    assert tti._insight_grounding_quotes(tmp_path / "nope.gi.json") == {}
    bad = tmp_path / "bad.gi.json"
    bad.write_text("{not json", encoding="utf-8")
    assert tti._insight_grounding_quotes(bad) == {}


def test_limit_episodes_breaks_after_first_episode(tmp_path, monkeypatch):
    """limit_episodes=1 over two metadata files processes one then breaks on the second
    (covers the mid-walk break after a prior episode was counted)."""
    rows = [("insight:1", "x", {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s"})]
    meta = tmp_path / "corpus" / "metadata" / "ep1.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta, meta])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: tmp_path / "corpus")
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    (tmp_path / "corpus" / "metadata").mkdir(parents=True)

    stats = tti.build_two_tier_index(
        tmp_path / "corpus", tmp_path / "corpus" / "search" / "li", limit_episodes=1
    )
    assert stats.episodes == 1 and stats.insights == 1  # second file skipped by the break


def test_skips_episode_with_empty_metadata(tmp_path, monkeypatch):
    """An episode whose metadata fails to load is skipped (the `if not doc: continue` path)
    without bumping the episode count or producing rows."""
    rows = [("insight:1", "x", {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s"})]
    meta_a = tmp_path / "corpus" / "metadata" / "ep_empty.json"
    meta_b = tmp_path / "corpus" / "metadata" / "ep_ok.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta_a, meta_b])
    # First file loads empty (skipped), second loads fine.
    monkeypatch.setattr(
        tti,
        "_load_metadata_file",
        lambda p: {} if p == meta_a else {"episode": {"episode_id": "ep1"}},
    )
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: tmp_path / "corpus")
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    (tmp_path / "corpus" / "metadata").mkdir(parents=True)

    stats = tti.build_two_tier_index(tmp_path / "corpus", tmp_path / "corpus" / "search" / "li")
    assert stats.episodes == 1 and stats.insights == 1  # only the loadable episode counted


def test_unknown_doc_type_rows_are_ignored(tmp_path, monkeypatch):
    """A row whose doc_type is none of insight/transcript/aux falls through the dispatch
    and contributes to no tier (covers the if/elif chain's implicit else)."""
    rows = [
        ("insight:1", "central bank policy", {"doc_type": "insight", "episode_id": "ep1"}),
        # doc_type not recognized -> dropped silently.
        ("weird:1", "mystery row", {"doc_type": "totally_unknown", "episode_id": "ep1"}),
        ("nodt:1", "no doc_type at all", {"episode_id": "ep1"}),
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    stats = tti.build_two_tier_index(corpus, corpus / "search" / "li")
    assert stats.insights == 1 and stats.segments == 0 and stats.aux == 0  # unknowns ignored


def test_grounding_quote_text_edge_without_matching_quote(tmp_path):
    """A SUPPORTED_BY edge pointing at a quote id with no captured text yields no mapping
    (covers the edge-loop branch where quote_id is absent from quote_text)."""
    import json

    gi = tmp_path / "ep.gi.json"
    gi.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "q1", "type": "Quote", "properties": {"text": "captured"}},
                ],
                "edges": [
                    # A non-SUPPORTED_BY edge is skipped by the type guard.
                    {"type": "MENTIONS", "from": "ins1", "to": "q1"},
                    # Edge references q_missing, which has no text entry -> skipped.
                    {"type": "SUPPORTED_BY", "from": "ins1", "to": "q_missing"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert tti._insight_grounding_quote_texts(gi) == {}


def test_drop_existing_clears_before_full_reindex(tmp_path, monkeypatch):
    """``drop_existing=True`` wipes the prior index so a full reindex starts clean."""
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"

    _stub_extraction(
        monkeypatch,
        tmp_path,
        [("insight:old", "first build content", {"doc_type": "insight", "episode_id": "ep1"})],
    )
    tti.build_two_tier_index(corpus, lance)
    assert LanceDBBackend(str(lance)).health()["insights"] == 1

    # Re-stub with different content and rebuild full (clean slate).
    _stub_extraction(
        monkeypatch,
        tmp_path,
        [("insight:new", "second build content", {"doc_type": "insight", "episode_id": "ep1"})],
    )
    tti.build_two_tier_index(corpus, lance, drop_existing=True)

    # Old row must be gone (cleared), only the new one present.
    rows_out = hs.hybrid_candidates(corpus, "build content", top_k=10) or []
    ids = {r.doc_id for r in rows_out}
    assert "insight:new" in ids
    assert "insight:old" not in ids
