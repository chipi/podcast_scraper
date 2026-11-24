"""What a news listing reports: its backlog over the cap, how far its cursor is behind, and items
that fell out of the feed before they were taken (next-cycle item 24, 2026-11-24)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import pytest

from podcast_scraper.news import hn, listing, rss
from podcast_scraper.obs import events
from podcast_scraper.rss import downloader
from podcast_scraper.rss.feeds_spec import NewsSourceSpec

pytestmark = pytest.mark.unit

_FEED = """<?xml version="1.0"?><rss version="2.0"><channel><title>F</title>
<item><title>A</title><link>https://news.test/a</link>
<pubDate>Tue, 17 Nov 2026 06:00:00 +0000</pubDate></item>
<item><title>B</title><link>https://news.test/b</link>
<pubDate>Tue, 17 Nov 2026 07:00:00 +0000</pubDate></item>
<item><title>C</title><link>https://news.test/c</link>
<pubDate>Tue, 17 Nov 2026 08:00:00 +0000</pubDate></item>
<item><title>D</title><link>https://news.test/d</link>
<pubDate>Tue, 17 Nov 2026 09:00:00 +0000</pubDate></item>
</channel></rss>"""


class _Source:
    news = NewsSourceSpec(max_items=2)
    language_override = None
    url = "https://news.test/feed.xml"


@pytest.fixture
def emitted(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(events, "emit_event", lambda kind, **f: seen.append({"kind": kind, **f}))
    monkeypatch.setattr(
        downloader,
        "fetch_rss_feed_url",
        lambda url, *a, **k: httpx.Response(
            200, content=_FEED.encode(), request=httpx.Request("GET", url)
        ),
    )
    return seen


def test_a_listing_over_its_cap_reports_the_backlog(emitted: List[Dict[str, Any]]) -> None:
    items, cursor = rss.list_items(_Source(), "2026-11-17T06:00:00Z")  # A is the cursor's item
    assert [i.metadata["title"] for i in items] == ["B", "C"]
    (event,) = emitted
    assert event["kind"] == "news_source_listing" and event["source"] == _Source.url
    assert (event["waiting"], event["taken"], event["backlog"]) == (3, 2, 1)
    assert event["cursor_after"] == cursor == "2026-11-17T08:00:00Z"
    assert event["lost_between"] is None
    rss.list_items(_Source(), cursor)
    assert emitted[1]["backlog"] == 0  # caught up on the next run


def test_a_feed_that_starts_after_the_cursor_reports_what_was_lost(
    emitted: List[Dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        rss.list_items(_Source(), "2026-11-16T12:00:00Z")
    assert emitted[0]["lost_between"] == ["2026-11-16T12:00:00Z", "2026-11-17T06:00:00Z"]
    assert "never listed" in caplog.text and _Source.url in caplog.text


def test_the_cursor_lag_is_hours_behind_now(emitted: List[Dict[str, Any]]) -> None:
    now = datetime(2026, 11, 17, 19, 0, tzinfo=timezone.utc)
    listing.report(
        "rss",
        _Source(),
        cursor_before=None,
        cursor_after="2026-11-17T07:00:00Z",
        waiting=0,
        taken=0,
        now=now,
    )
    assert emitted[0]["cursor_lag_hours"] == 12.0


def test_hacker_news_reports_its_backlog_and_no_lost_window(
    monkeypatch: pytest.MonkeyPatch, emitted: List[Dict[str, Any]]
) -> None:
    hits = {
        "hits": [
            {
                "objectID": str(n),
                "created_at_i": n * 100,
                "title": str(n),
                "url": f"https://x.test/{n}",
            }
            for n in (1, 2, 3)
        ]
    }
    monkeypatch.setattr(
        downloader,
        "fetch_url",
        lambda url, *a, **k: httpx.Response(200, json=hits, request=httpx.Request("GET", url)),
    )
    hn.list_items(_Source(), "50")
    (event,) = emitted
    assert (event["waiting"], event["taken"], event["backlog"]) == (3, 2, 1)
    assert event["lost_between"] is None
