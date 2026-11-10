"""A digest topic band waits for its search: a slow search still gives the band its hits.

The band used to give up after 0.8 s, but its executor's ``with`` exit waited for the search anyway,
so the timeout saved no time and only dropped the band. On a prod-sized index a band's search takes
about 0.4-1 s warm and seconds cold, so the digest lost topics at random, and the cached digest
stayed incomplete until the next reindex (found by the release-1 route comparison, 2026-11-09).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.routes import corpus_digest as route


def _row() -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path="feeds/f/e1.metadata.json",
        feed_id="f",
        feed_title="F",
        episode_id="e1",
        episode_title="Episode one",
        publish_date="2026-11-01",
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="",
        kg_relative_path="",
        bridge_relative_path="",
        has_gi=False,
        has_kg=False,
        has_bridge=False,
    )


def test_a_slow_search_still_gives_the_band(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_search(*_a: object, **_k: object) -> SimpleNamespace:
        time.sleep(1.0)  # longer than the 0.8 s the band used to allow
        hit = {"score": 0.9, "metadata": {"episode_id": "e1", "feed_id": "f"}}
        return SimpleNamespace(error=None, results=[hit])

    monkeypatch.setattr(route, "run_corpus_search", slow_search)
    row = _row()
    band = route._topic_band_for_query(
        Path("/c"),
        "tech",
        "technology",
        "technology",
        start=datetime(2026, 10, 1, tzinfo=timezone.utc),
        end=datetime(2026, 11, 9, tzinfo=timezone.utc),
        since="",
        scope={("f", "e1"): row},
        feed_titles_by_feed_id={},
        feed_rss_urls_by_feed_id={},
        feed_descriptions_by_feed_id={},
    )
    assert band is not None, "the band was dropped for a slow search"
    assert [h.metadata_relative_path for h in band.hits] == [row.metadata_relative_path]
