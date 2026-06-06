"""Unit tests for the shared search-capability core (RFC-095 §3)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.capability import doc_types_for_tier, structured_corpus_search
from podcast_scraper.search.corpus_search import CorpusSearchOutcome

pytestmark = pytest.mark.unit


def test_doc_types_for_tier() -> None:
    assert doc_types_for_tier("insight") == ["insight"]
    assert doc_types_for_tier("segment") == ["transcript"]
    assert doc_types_for_tier("both") is None
    assert doc_types_for_tier(None) is None
    assert doc_types_for_tier("bogus") is None


def test_structured_search_stamps_tier_and_intent(tmp_path, monkeypatch) -> None:
    def fake_run(root, query, **kwargs):
        return CorpusSearchOutcome(
            results=[
                {"doc_id": "i1", "score": 0.9, "metadata": {"doc_type": "insight"}, "text": "x"},
                {"doc_id": "t1", "score": 0.8, "metadata": {"doc_type": "transcript"}, "text": "y"},
                {"doc_id": "k1", "score": 0.5, "metadata": {"doc_type": "kg_entity"}, "text": "z"},
            ],
            lift_stats={"transcript_hits_returned": 1, "lift_applied": 0},
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    # "Jane Doe" trips the name regex → entity_lookup intent.
    out = structured_corpus_search(tmp_path, "Jane Doe")
    assert out["query_type"] == "entity_lookup"
    assert [r["source_tier"] for r in out["results"]] == ["insight", "segment", "aux"]
    assert out["error"] is None
    assert out["lift_stats"] == {"transcript_hits_returned": 1, "lift_applied": 0}


def test_structured_search_error_passthrough(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        lambda *a, **k: CorpusSearchOutcome(error="no_index", detail="/x"),
    )
    out = structured_corpus_search(tmp_path, "anything")
    assert out["error"] == "no_index"
    assert out["detail"] == "/x"
    assert out["results"] == []
    assert out["lift_stats"] is None


def test_structured_search_preserves_lifted_and_quotes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        lambda *a, **k: CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "t1",
                    "score": 0.7,
                    "metadata": {"doc_type": "transcript"},
                    "text": "chunk",
                    "lifted": {"insight": {"id": "i1"}},
                },
                {
                    "doc_id": "i1",
                    "score": 0.6,
                    "metadata": {"doc_type": "insight"},
                    "text": "ins",
                    "supporting_quotes": [{"text": "q"}],
                },
            ]
        ),
    )
    out = structured_corpus_search(tmp_path, "q")
    assert out["results"][0]["lifted"] == {"insight": {"id": "i1"}}
    assert out["results"][1]["supporting_quotes"] == [{"text": "q"}]
