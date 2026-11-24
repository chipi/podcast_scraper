"""The news writer's API transports (E05 slice 3): Hacker News (Algolia) and the Guardian
Content API, on fake responses."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx
import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.news import guardian, hn
from podcast_scraper.rss import downloader
from podcast_scraper.rss.feeds_spec import NewsSourceSpec
from podcast_scraper.utils.log_redaction import redact_for_log

pytestmark = pytest.mark.unit


class _Source:
    news = NewsSourceSpec(max_items=2)
    language_override = None

    def __init__(self, url: str) -> None:
        self.url = url


def _serve(monkeypatch: pytest.MonkeyPatch, body: Dict[str, Any], seen: List[str]) -> None:
    def fake(url: str, *a: Any, **k: Any) -> httpx.Response:
        seen.append(url)
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(downloader, "fetch_url", fake)


# --- Hacker News ---

HITS = {
    "hits": [
        {"objectID": "3", "created_at_i": 300, "title": "C", "url": "https://c.test/a"},
        {"objectID": "1", "created_at_i": 100, "title": "A", "url": "https://a.test/a"},
        {"objectID": "4", "created_at_i": 400, "title": "Ask HN: anyone?", "url": None},
        {"objectID": "2", "created_at_i": 200, "title": "B", "url": "https://b.test/a"},
    ]
}


def test_hn_lists_linked_stories_newest_first_up_to_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: List[str] = []
    _serve(monkeypatch, HITS, seen)
    items, cursor = hn.list_items(_Source("https://hn.test/api"), None)
    assert [i.metadata["title"] for i in items] == ["C", "B"]  # no link: no article; cap 2
    assert items[0].url == "https://c.test/a" and items[0].text is None  # the page is fetched
    assert items[0].metadata["transport_ids"] == {"hn": "3"}
    assert items[0].metadata["published_at"] == "1970-01-01T00:05:00Z"
    assert "byline_raw" not in items[0].metadata  # the submitter is not the article's author
    assert cursor == "300"


def test_hn_lists_only_what_is_newer_than_its_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, HITS, [])
    items, cursor = hn.list_items(_Source("https://hn.test/api"), "200")
    assert [i.metadata["title"] for i in items] == ["C"]
    assert cursor == "300"


def test_hn_after_a_cursor_takes_the_oldest_first_so_none_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Next-cycle item 24: over the cap, the newest-first listing moved the cursor past the rest."""
    _serve(monkeypatch, HITS, [])
    first, cursor = hn.list_items(_Source("https://hn.test/api"), "50")
    assert [i.metadata["title"] for i in first] == ["A", "B"]
    assert cursor == "200"
    second, cursor = hn.list_items(_Source("https://hn.test/api"), cursor)
    assert [i.metadata["title"] for i in second] == ["C"]
    assert cursor == "300"


def test_an_article_over_hn_and_over_its_feed_is_one_item() -> None:
    news = ct.content_type("news_article")
    # A trailing slash and a tracking parameter (``www.`` is kept as given: canonical_url).
    over_hn = ct.SourceItem(item_id="3", url="https://c.test/a/?utm_source=hn")
    over_rss = ct.SourceItem(item_id="guid-9", url="https://c.test/a")
    assert ct.item_id(news, over_hn) == ct.item_id(news, over_rss)


# --- the Guardian ---

BODY = "".join(
    f"<p>Paragraph {n} of the report, which says something of its own at some length.</p>"
    for n in range(12)
)
RESULTS = {
    "response": {
        "results": [
            {
                "id": "world/2026/nov/18/a",
                "webUrl": "https://www.theguardian.com/world/2026/nov/18/a",
                "webTitle": "A",
                "webPublicationDate": "2026-11-18T09:00:00Z",
                "fields": {"body": BODY, "byline": "Ana Example and Bo Example", "lang": "en"},
            },
            {
                "id": "world/2026/nov/18/empty",
                "webUrl": "https://www.theguardian.com/world/2026/nov/18/empty",
                "webTitle": "No body",
                "webPublicationDate": "2026-11-18T08:00:00Z",
                "fields": {"body": "", "lang": "en"},
            },
        ]
    }
}


def test_without_a_key_the_guardian_lists_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(guardian.KEY_ENV, raising=False)
    monkeypatch.setattr(downloader, "fetch_url", lambda *a, **k: pytest.fail("fetched anyway"))
    with caplog.at_level(logging.WARNING):
        assert guardian.list_items(_Source("https://guardian.test/search"), "c") == ([], "c")
    # The variable is named in the config and the guide, never in the log: CodeQL reads a key's
    # variable name as a secret (2026-11-18, the operator chose no name in the log).
    assert "API key variable is not set" in caplog.text and "guardian.test/search" in caplog.text
    assert guardian.KEY_ENV not in caplog.text


class _KeyedSource(_Source):
    news = NewsSourceSpec(max_items=2, api_key_env="OTHER_NEWS_API_KEY")


def test_a_sources_api_key_env_names_the_variable_its_key_is_read_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``api_key_env`` on the source (the operator, 2026-11-18: ready for other keyed APIs): the
    key is read from that variable, not the transport's default."""
    monkeypatch.delenv(guardian.KEY_ENV, raising=False)
    monkeypatch.setenv("OTHER_NEWS_API_KEY", "FAKEOTHERKEY")
    seen: List[str] = []
    _serve(monkeypatch, RESULTS, seen)
    items, _cursor = guardian.list_items(_KeyedSource("https://guardian.test/search"), None)
    assert [i.metadata["title"] for i in items] == ["A"]
    assert "api-key=FAKEOTHERKEY" in seen[0]


def test_an_unset_named_key_variable_lists_nothing_and_says_which(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(guardian.KEY_ENV, "FAKEKEYFAKEKEY")  # the default is not a fallback here
    monkeypatch.delenv("OTHER_NEWS_API_KEY", raising=False)
    monkeypatch.setattr(downloader, "fetch_url", lambda *a, **k: pytest.fail("fetched anyway"))
    with caplog.at_level(logging.WARNING):
        assert guardian.list_items(_KeyedSource("https://guardian.test/search"), "c") == ([], "c")
    assert "API key variable is not set" in caplog.text
    assert "OTHER_NEWS_API_KEY" not in caplog.text


@pytest.mark.parametrize("name", ["guardian_api_key", "1KEY", "KEY-NAME", ""])
def test_api_key_env_must_be_an_environment_variable_name(name: str) -> None:
    with pytest.raises(ValueError, match="environment variable name"):
        NewsSourceSpec(api_key_env=name)


def test_the_guardians_items_carry_their_text_and_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(guardian.KEY_ENV, "FAKEKEYFAKEKEY")
    seen: List[str] = []
    _serve(monkeypatch, RESULTS, seen)
    items, cursor = guardian.list_items(_Source("https://guardian.test/search"), None)
    assert [i.metadata["title"] for i in items] == ["A"]  # no body: nothing to write
    item = items[0]
    assert item.text is not None and item.text.startswith("Paragraph 0 of the report")
    assert len(item.text.split("\n\n")) == 12
    assert item.raw_suffix == ".json" and json.loads(item.raw or "")["id"] == item.item_id
    m = item.metadata
    assert m["transport_ids"] == {"guardian": "world/2026/nov/18/a"}
    assert m["byline_raw"] == "Ana Example and Bo Example"
    assert (m["language"], m["language_source"]) == ("en", "page")
    assert m["serveable"] == "metadata"
    assert cursor == "2026-11-18T09:00:00Z"
    assert "api-key=FAKEKEYFAKEKEY" in seen[0]
    assert "FAKEKEYFAKEKEY" not in redact_for_log(seen[0])  # what the downloader logs


def _result(n: int, when: str) -> Dict[str, Any]:
    return {
        "id": f"world/2026/nov/18/r{n}",
        "webUrl": f"https://www.theguardian.com/world/2026/nov/18/r{n}",
        "webTitle": f"R{n}",
        "webPublicationDate": when,
        "fields": {"body": BODY, "lang": "en"},
    }


def test_the_guardian_after_a_cursor_reads_oldest_first_and_pages_past_the_cursors_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Next-cycle item 24. ``from-date`` takes a day, so the day's articles up to the cursor come
    first in an oldest-first listing; the run pages on until its cap is filled."""
    monkeypatch.setenv(guardian.KEY_ENV, "FAKEKEYFAKEKEY")
    pages = {
        1: [_result(1, "2026-11-18T07:00:00Z"), _result(2, "2026-11-18T08:00:00Z")],
        2: [_result(3, "2026-11-18T09:00:00Z"), _result(4, "2026-11-18T10:00:00Z")],
        3: [_result(5, "2026-11-18T11:00:00Z")],
    }
    seen: List[str] = []

    def fake(url: str, *a: Any, **k: Any) -> httpx.Response:
        seen.append(url)
        page = int(url.split("&page=")[1].split("&")[0]) if "&page=" in url else 1
        body = {"response": {"pages": 3, "results": pages[page]}}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(downloader, "fetch_url", fake)
    source = _Source("https://guardian.test/search")
    items, cursor = guardian.list_items(source, "2026-11-18T08:00:00Z")
    assert [i.metadata["title"] for i in items] == ["R3", "R4"]  # cap 2; R1, R2 at or before it
    assert cursor == "2026-11-18T10:00:00Z"
    assert len(seen) == 2 and "order-by=oldest" in seen[0] and "from-date=2026-11-18" in seen[0]
    items, cursor = guardian.list_items(source, cursor)
    assert [i.metadata["title"] for i in items] == ["R5"]
