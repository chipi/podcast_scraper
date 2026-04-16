"""Integration: Library similar-episodes helpers (RFC-067) + ``run_similar_episodes``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.search.corpus_similar import (
    build_similarity_query,
    episode_scope_key,
    merge_similar_episode_hits,
    run_similar_episodes,
)

pytestmark = pytest.mark.integration


def test_build_similarity_query_concat_and_truncation() -> None:
    q = build_similarity_query("Title", ["b1", "b2"], "Ep")
    assert "Title" in q and "b1" in q
    long_b = "word " * 2000
    q2 = build_similarity_query(None, [long_b], "fallback title here")
    assert len(q2) <= 6000


def test_build_similarity_query_falls_back_to_episode_title() -> None:
    assert build_similarity_query(None, [], "  Only title  ").endswith("Only title")


def test_episode_scope_key_and_merge_similar() -> None:
    assert episode_scope_key({"episode_id": "e", "feed_id": "f"}) == ("f", "e")
    assert episode_scope_key({}) is None

    rows = [
        {
            "score": 0.5,
            "metadata": {"episode_id": "b", "feed_id": "f1"},
            "text": "t",
        },
        {
            "score": 0.9,
            "metadata": {"episode_id": "b", "feed_id": "f1"},
            "text": "better",
        },
        {
            "score": 0.99,
            "metadata": {"episode_id": "src", "feed_id": "f1"},
            "text": "self",
        },
    ]
    merged = merge_similar_episode_hits(
        rows,
        source_feed_id="f1",
        source_episode_id="src",
        top_k=5,
    )
    assert len(merged) == 1
    assert merged[0]["metadata"]["episode_id"] == "b"
    assert merged[0]["score"] == pytest.approx(0.9)


def test_run_similar_episodes_insufficient_text() -> None:
    out = run_similar_episodes(
        Path("/tmp"),
        summary_title=None,
        summary_bullets=[],
        episode_title="short",
        source_feed_id="f",
        source_episode_id="e",
    )
    assert out.error == "insufficient_text"


def test_run_similar_episodes_delegates_to_corpus_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_search(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        assert "renewable" in query.lower()
        assert kwargs.get("dedupe_kg_surfaces") is False
        return CorpusSearchOutcome(
            results=[
                {
                    "score": 0.8,
                    "text": "peer",
                    "metadata": {"episode_id": "other", "feed_id": "fx", "doc_type": "summary"},
                },
            ],
        )

    monkeypatch.setattr(
        "podcast_scraper.search.corpus_similar.run_corpus_search",
        fake_search,
    )
    out = run_similar_episodes(
        tmp_path,
        summary_title="Renewable energy trends",
        summary_bullets=["Solar grows fast.", "Wind capacity doubles."],
        episode_title="Episode",
        source_feed_id="fx",
        source_episode_id="self",
        top_k=3,
    )
    assert out.error is None
    assert out.query_used
    assert len(out.items) >= 1
