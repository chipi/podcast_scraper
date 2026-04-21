"""Unit tests for corpus similarity helpers."""

from __future__ import annotations

from podcast_scraper.search.corpus_similar import (
    build_similarity_query,
    merge_similar_episode_hits,
)


def test_build_similarity_query_prefers_summary_then_title() -> None:
    q = build_similarity_query("T", ["alpha bravo"], "ignored", max_chars=500)
    assert "T" in q and "alpha" in q and "ignored" not in q

    q2 = build_similarity_query(None, [], "only title here")
    assert q2 == "only title here"


def test_build_similarity_query_truncates_at_word_boundary() -> None:
    long_bullet = "word " * 2000
    q = build_similarity_query(None, [long_bullet], "x", max_chars=40)
    assert len(q) <= 40
    assert not q.endswith(" ")


def test_merge_similar_episode_hits_dedupes_and_drops_source() -> None:
    rows = [
        {
            "score": 0.9,
            "metadata": {"feed_id": "a", "episode_id": "1", "doc_type": "summary"},
            "text": "one",
        },
        {
            "score": 0.95,
            "metadata": {"feed_id": "a", "episode_id": "1", "doc_type": "quote"},
            "text": "better",
        },
        {
            "score": 0.5,
            "metadata": {"feed_id": "a", "episode_id": "2", "doc_type": "summary"},
            "text": "two",
        },
        {
            "score": 0.8,
            "metadata": {"feed_id": "a", "episode_id": "src", "doc_type": "summary"},
            "text": "src chunk",
        },
    ]
    out = merge_similar_episode_hits(
        rows,
        source_feed_id="a",
        source_episode_id="src",
        top_k=10,
    )
    keys = {(x["metadata"]["feed_id"], x["metadata"]["episode_id"]) for x in out}
    assert ("a", "src") not in keys
    assert len(out) == 2
    ep1 = next(x for x in out if x["metadata"]["episode_id"] == "1")
    assert ep1["score"] == 0.95
    assert ep1["text"] == "better"
