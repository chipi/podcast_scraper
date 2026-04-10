"""Unit tests for corpus digest selection (RFC-068)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.corpus_digest import (
    digest_row_dict,
    diversify_digest_rows,
    episode_in_utc_window,
    filter_rows_in_window,
    utc_bounds_for_window,
)


def _row(
    *,
    path: str,
    feed: str,
    title: str,
    pub: str | None,
) -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path=path,
        feed_id=feed,
        feed_title=None,
        episode_id=None,
        episode_title=title,
        publish_date=pub,
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="x.gi.json",
        kg_relative_path="x.kg.json",
        has_gi=False,
        has_kg=False,
    )


def test_digest_row_dict_includes_visual_fields() -> None:
    row = CatalogEpisodeRow(
        metadata_relative_path="m.metadata.json",
        feed_id="f",
        feed_title="S",
        episode_id="e",
        episode_title="T",
        publish_date="2024-06-01",
        summary_title=None,
        summary_bullets=("b1",),
        summary_text=None,
        gi_relative_path="m.gi.json",
        kg_relative_path="m.kg.json",
        has_gi=True,
        has_kg=False,
        feed_image_url="https://f.example/a.png",
        episode_image_url="https://e.example/b.png",
        duration_seconds=3600,
        episode_number=12,
    )
    d = digest_row_dict(row)
    assert d.get("feed_rss_url") is None
    assert d.get("feed_description") is None
    assert d["feed_image_url"] == "https://f.example/a.png"
    assert d["episode_image_url"] == "https://e.example/b.png"
    assert d["duration_seconds"] == 3600
    assert d["episode_number"] == 12
    assert d["feed_display_title"] == "S"
    assert d["summary_preview"] == "b1"


def test_digest_row_dict_includes_feed_rss_url_and_description() -> None:
    row = CatalogEpisodeRow(
        metadata_relative_path="m.metadata.json",
        feed_id="f",
        feed_title="S",
        episode_id="e",
        episode_title="T",
        publish_date="2024-06-01",
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="m.gi.json",
        kg_relative_path="m.kg.json",
        has_gi=False,
        has_kg=False,
        feed_rss_url="https://example.com/rss",
        feed_description="About the show",
    )
    d = digest_row_dict(row)
    assert d["feed_rss_url"] == "https://example.com/rss"
    assert d["feed_description"] == "About the show"


def test_digest_row_dict_resolves_feed_title_from_feed_index() -> None:
    row = CatalogEpisodeRow(
        metadata_relative_path="m.metadata.json",
        feed_id="f",
        feed_title=None,
        episode_id="e",
        episode_title="T",
        publish_date="2024-06-01",
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="m.gi.json",
        kg_relative_path="m.kg.json",
        has_gi=False,
        has_kg=False,
    )
    d = digest_row_dict(row, feed_titles_by_feed_id={"f": "From sibling"})
    assert d["feed_display_title"] == "From sibling"


def test_digest_row_dict_resolves_feed_rss_from_sibling_index() -> None:
    row = CatalogEpisodeRow(
        metadata_relative_path="m.metadata.json",
        feed_id="f",
        feed_title="T",
        episode_id="e",
        episode_title="Ep",
        publish_date="2024-06-01",
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="m.gi.json",
        kg_relative_path="m.kg.json",
        has_gi=False,
        has_kg=False,
        feed_rss_url=None,
    )
    d = digest_row_dict(
        row,
        feed_rss_urls_by_feed_id={"f": "https://sibling/rss"},
        feed_descriptions_by_feed_id={"f": "Sibling desc"},
    )
    assert d["feed_rss_url"] == "https://sibling/rss"
    assert d["feed_description"] == "Sibling desc"


def test_utc_bounds_24h_and_7d() -> None:
    now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    s, e = utc_bounds_for_window("24h", since=None, now_utc=now)
    assert e == now
    assert (e - s).total_seconds() == 86400

    s7, e7 = utc_bounds_for_window("7d", since=None, now_utc=now)
    assert e7 == now
    assert (e7 - s7).days == 7


def test_utc_bounds_since_requires_param() -> None:
    now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="since_required"):
        utc_bounds_for_window("since", since=None, now_utc=now)


def test_utc_bounds_since() -> None:
    now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    s, e = utc_bounds_for_window("since", since="2024-06-01", now_utc=now)
    assert s == datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
    assert e == now


def test_episode_in_window_midnight_compare() -> None:
    start = datetime(2024, 6, 10, 15, 30, tzinfo=timezone.utc)
    end = datetime(2024, 6, 20, 15, 30, tzinfo=timezone.utc)
    row = _row(path="a.json", feed="f", title="t", pub="2024-06-15")
    assert episode_in_utc_window(row, start, end) is True
    assert (
        episode_in_utc_window(
            _row(path="b.json", feed="f", title="t", pub="2024-06-09"), start, end
        )
        is False
    )


def test_filter_rows_preserves_order() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    rows = [
        _row(path="a.json", feed="A", title="old", pub="2024-03-01"),
        _row(path="b.json", feed="B", title="new", pub="2024-06-01"),
    ]
    out = filter_rows_in_window(rows, start, end)
    assert [r.metadata_relative_path for r in out] == ["a.json", "b.json"]


def test_diversify_round_robin_caps_per_feed() -> None:
    # Newest-first global order (as catalog sort_key): A2, A1, A0, B1, B0, C0
    rows = [
        _row(path="a2.json", feed="A", title="2", pub="2024-06-12"),
        _row(path="a1.json", feed="A", title="1", pub="2024-06-11"),
        _row(path="a0.json", feed="A", title="0", pub="2024-06-10"),
        _row(path="b1.json", feed="B", title="1", pub="2024-06-06"),
        _row(path="b0.json", feed="B", title="0", pub="2024-06-05"),
        _row(path="c.json", feed="C", title="c", pub="2024-06-01"),
    ]

    out = diversify_digest_rows(rows, max_rows=6, per_feed_cap=2)
    feeds = [r.feed_id for r in out]
    assert feeds.count("A") <= 2
    assert feeds.count("B") <= 2
    assert feeds.count("C") <= 2
    # C has only one episode; caps stop A/B after two each → five rows total.
    assert len(out) == 5
    assert out[0].metadata_relative_path == "a2.json"
    assert out[1].metadata_relative_path == "b1.json"
    assert out[2].metadata_relative_path == "c.json"
