"""Scoped searches narrow the QUERY, not the results (operator 2026-10-10).

Same defect as search within one episode (``test_episode_scoped_search``), on the other scopes:
a show (``feed``), a date window (``since``) and a set of episodes (Search's "Mine"). Each used to
rank the whole corpus, keep the top few hundred, and filter afterwards — so on a large corpus the
scope's own matches never made the cut. Each is now a LanceDB prefilter, so rows outside the scope
are never ranked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backend import SegmentDocument  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402

DIM = 4
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]


def _seg(
    n: int, episode: str, *, show: str, date: str, near: bool, text: str = "risk drawdown"
) -> SegmentDocument:
    return SegmentDocument(
        id=f"{episode}_chunk_{n}",
        text=text if near else "one passing mention of risk",
        show_id=show,
        episode_id=episode,
        start_time=float(n),
        end_time=float(n) + 5.0,
        embedding=[1.0, 0.01 * n, 0.0, 0.0] if near else [0.0, 0.0, 1.0, 0.0],
        publish_date=date,
    )


def _corpus(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fifty rows in show ``crowd`` from 2020 sit right on the query; one far-off row in show
    ``quiet`` from 2026 is the only one inside every scope under test."""
    from podcast_scraper.search import hybrid_search

    b = LanceDBBackend(str(root / "search" / "lance_index"), embed_dim=DIM)
    b.upsert_segments(
        [
            *(
                _seg(i, f"ep-crowd-{i}", show="crowd", date="2020-03-01T00:00:00", near=True)
                for i in range(50)
            ),
            _seg(0, "ep-quiet", show="quiet", date="2026-05-01T00:00:00", near=False),
        ]
    )
    b.create_indices()
    b.write_index_meta("test-model")
    # The query embedding is the only ML in the path; a fixed vector keeps these tests model-free.
    monkeypatch.setattr(hybrid_search.embedding_loader, "encode", lambda *a, **k: list(QUERY_VEC))


def _episodes(out: dict) -> list:
    assert not out.get("error"), out
    return [r["metadata"].get("episode_id") for r in out["results"]]


def test_a_show_scope_finds_the_show_on_a_crowded_corpus(tmp_path, monkeypatch):
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    assert _episodes(structured_corpus_search(tmp_path, "risk", feed="quiet", top_k=1)) == [
        "ep-quiet"
    ]


def test_a_date_window_finds_recent_episodes_on_a_crowded_corpus(tmp_path, monkeypatch):
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    assert _episodes(structured_corpus_search(tmp_path, "risk", since="2026-01-01", top_k=1)) == [
        "ep-quiet"
    ]


def test_a_set_of_episodes_finds_them_on_a_crowded_corpus(tmp_path, monkeypatch):
    """Search's "Mine": the listener's episodes are part of the query."""
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    out = structured_corpus_search(
        tmp_path, "risk", episode_ids=["ep-quiet", "ep-unknown"], top_k=1
    )
    assert _episodes(out) == ["ep-quiet"]


def test_the_scopes_never_let_anything_else_through(tmp_path, monkeypatch):
    from podcast_scraper.search.capability import structured_corpus_search

    _corpus(tmp_path, monkeypatch)
    for kwargs in ({"feed": "quiet"}, {"since": "2026-01-01"}, {"episode_ids": ["ep-quiet"]}):
        assert set(_episodes(structured_corpus_search(tmp_path, "risk", top_k=20, **kwargs))) == {
            "ep-quiet"
        }, kwargs


def test_the_filter_syntax_quotes_values_and_rejects_unknown_keys(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=DIM)
    assert b._to_sql({"episode_id": "e'1"}) == "episode_id = 'e''1'"
    assert b._to_sql({"episode_id__in": ["a", "b"]}) == "episode_id IN ('a', 'b')"
    assert b._to_sql({"episode_id__in": []}) == "1 = 0"  # an empty world matches nothing
    assert b._to_sql({"publish_date__gte": "2026-01-01"}) == "publish_date >= '2026-01-01'"
    assert b._to_sql({"show_id__contains": "Quiet"}) == "lower(show_id) LIKE '%quiet%'"
    with pytest.raises(ValueError):
        b._to_sql({"show_id; drop": "x"})
    with pytest.raises(ValueError):
        b._to_sql({"show_id__regex": "x"})


def test_the_query_scope_is_never_stricter_than_the_result_filters():
    from podcast_scraper.search.corpus_search import _query_scope

    scope = _query_scope(feed=" quiet ", since="2026-01-01", episode_id=None, episode_ids=None)
    assert scope == {"show_id__contains": "quiet", "publish_date__gte": "2025-12-31"}
    # A type filter goes in only when every type lives in the aux table (the one with doc_type).
    assert _query_scope(
        doc_types=["storyline"], feed=None, since=None, episode_id=None, episode_ids=None
    ) == {"doc_type__in": ["storyline"]}
    assert (
        _query_scope(
            doc_types=["insight", "storyline"],
            feed=None,
            since=None,
            episode_id=None,
            episode_ids=None,
        )
        is None
    )
