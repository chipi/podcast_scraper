"""Unit tests for the catalog MCP tools (RFC-095 slice 3) — corpus_catalog mocked."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import catalog as cat

pytestmark = pytest.mark.unit


class _Row:
    """Minimal CatalogEpisodeRow stand-in (only the fields the tools read)."""

    def __init__(self, **kw):
        defaults = {
            "metadata_relative_path": "m/e.json",
            "feed_id": "f1",
            "feed_title": "Show One",
            "episode_id": "e1",
            "episode_title": "Ep One",
            "publish_date": "2026-06-01",
            "summary_title": "Sum",
            "summary_text": "Body",
            "gi_relative_path": "m/e.gi.json",
            "kg_relative_path": "m/e.kg.json",
            "has_gi": True,
            "has_kg": True,
            "duration_seconds": 1200,
            "episode_number": 3,
            "feed_rss_url": "http://x/rss",
        }
        defaults.update(kw)
        for key, value in defaults.items():
            setattr(self, key, value)


def test_list_feeds(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.server.corpus_catalog.build_catalog_rows_cumulative",
        lambda root: [_Row()],
    )
    monkeypatch.setattr(
        "podcast_scraper.server.corpus_catalog.aggregate_feeds",
        lambda rows: [{"feed_id": "f1", "display_title": "Show One", "episode_count": 1}],
    )
    out = cat.list_feeds(ctx)
    assert out["feeds"][0]["feed_id"] == "f1"


def test_list_episodes_filters_and_limits(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    rows = [
        _Row(feed_id="f1", publish_date="2026-06-05", metadata_relative_path="m/a.json"),
        _Row(feed_id="f2", publish_date="2026-06-04", metadata_relative_path="m/b.json"),
        _Row(feed_id="f1", publish_date="2026-05-01", metadata_relative_path="m/c.json"),
    ]
    monkeypatch.setattr(
        "podcast_scraper.server.corpus_catalog.build_catalog_rows_cumulative",
        lambda root: rows,
    )
    # feed=f1 + since 2026-06-01 → only m/a.json.
    out = cat.list_episodes(ctx, feed="f1", since="2026-06-01")
    assert out["count"] == 1
    assert out["episodes"][0]["metadata_path"] == "m/a.json"
    # limit caps the result.
    assert cat.list_episodes(ctx, limit=2)["count"] == 2


def test_episode_detail_found(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.server.corpus_catalog.catalog_row_for_metadata_path",
        lambda root, relpath: _Row(metadata_relative_path=relpath),
    )
    out = cat.episode_detail(ctx, "m/e.json")
    assert out["episode"]["episode_id"] == "e1"
    assert out["episode"]["summary_text"] == "Body"
    assert out["episode"]["gi_relative_path"] == "m/e.gi.json"


def test_episode_detail_not_found(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.server.corpus_catalog.catalog_row_for_metadata_path",
        lambda root, relpath: None,
    )
    out = cat.episode_detail(ctx, "m/missing.json")
    assert out["episode"] is None
    assert out["error"] == "not_found"


def test_top_people(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    captured = {}

    def fake_top(root, limit):
        captured["root"] = root
        captured["limit"] = limit
        return {"persons": [{"person_id": "person:p"}], "total_persons": 1}

    monkeypatch.setattr("podcast_scraper.server.routes.corpus_persons.top_persons", fake_top)
    out = cat.top_people(ctx, limit=999)
    assert out["persons"][0]["person_id"] == "person:p"
    assert captured["root"] == tmp_path.resolve()
    assert captured["limit"] == 50  # clamped to [1, 50]
