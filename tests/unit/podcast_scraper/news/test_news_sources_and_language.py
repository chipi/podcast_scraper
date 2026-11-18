"""The news writer's source settings, language and text choice (E05, News 1.0): each source's
cap, URL filters, forced text status and rights; an item's language resolved and gated as an
episode's is; boilerplate removed; the feed's own text taken where the page has less."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

import httpx
import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.news import byline, extract, fetch, rss
from podcast_scraper.rss import downloader
from podcast_scraper.rss.feeds_spec import FeedsSpecDocument, NewsSourceSpec, RssFeedEntry

pytestmark = pytest.mark.unit

EN = (
    "The central bank said on Tuesday that it would keep rates where they are for now, and that "
    "the board would look again at the data before its next meeting in the spring of the year."
)
ES = (
    "El banco central dijo el martes que mantendría los tipos donde están por ahora, y que la "
    "junta volvería a mirar los datos antes de su próxima reunión en la primavera del año."
)


# --- language ---


@pytest.mark.parametrize(
    "text, override, page, feed, want",
    [
        (EN, None, "en", "en", ("en", "page")),
        (EN, None, None, "en-GB", ("en", "rss")),
        (ES, None, "es", "en", ("es", "page")),
        # The text contradicts the declaration (the check reads 200 words or more).
        (" ".join([ES] * 8), None, "en", "en", ("es", "detected")),
        (ES, None, "en", "en", ("en", "page")),  # too short to contradict anything
        (ES, "es", None, None, ("es", "override")),
        (EN, "es", "en", "en", ("es", "override")),  # the operator's override is not second-guessed
        (ES, None, None, None, ("es", "detected")),
        ("Short.", None, None, None, (None, "none")),
    ],
)
def test_an_items_language_is_resolved_as_an_episodes(
    text: str, override: Optional[str], page: Optional[str], feed: Optional[str], want: Any
) -> None:
    assert fetch.resolve_language(text, override=override, page=page, feed=feed) == want


class _Source:
    url = "https://news.test/feed.xml"
    language_override: Optional[str] = None
    news: Optional[NewsSourceSpec] = None


@pytest.fixture
def page(monkeypatch: pytest.MonkeyPatch) -> Iterator[List[str]]:
    """A fake network: robots.txt allows everything, the page is the one ``html`` holds."""
    html: List[str] = []
    monkeypatch.setattr(downloader, "http_get_any_status", lambda *a: (404, b""))
    monkeypatch.setattr(
        downloader,
        "fetch_url",
        lambda url, *a, **k: httpx.Response(
            200, content=html[0].encode(), request=httpx.Request("GET", url)
        ),
    )
    fetch.reset_robots_cache()
    yield html
    fetch.reset_robots_cache()


def _html(lang: str, body: str) -> str:
    paras = "".join(f"<p>{body} ({n})</p>" for n in range(6))
    return f'<html lang="{lang}"><body><article><h1>T</h1>{paras}</article></body></html>'


def test_an_item_in_a_language_we_do_not_ingest_is_skipped(page: List[str]) -> None:
    """config/languages.yaml: ru is described, not enabled, so the item is never written."""
    page.append(_html("ru", EN))
    item = ct.SourceItem(item_id="g", url="https://news.test/a")
    assert fetch.fetch_body(item, _Source()) is None


def test_an_enabled_language_is_written_with_its_source(page: List[str]) -> None:
    page.append(_html("es", ES))
    got = fetch.fetch_body(ct.SourceItem(item_id="g", url="https://news.test/a"), _Source())
    assert got is not None
    assert (got.metadata["language"], got.metadata["language_source"]) == ("es", "page")


# --- the source's settings ---


def test_the_source_writes_its_rights_text_status_and_live_blogs(page: List[str]) -> None:
    page.append(_html("en", EN))
    src = _Source()
    src.news = NewsSourceSpec(
        serveable="summary",
        mineable=False,
        licence="CC-BY-4.0",
        paywalled=True,
        text_status="teaser",
        editorial_type="opinion",
        live_blog_url_pattern="/live/",
    )
    got = fetch.fetch_body(ct.SourceItem(item_id="g", url="https://news.test/live/a"), src)
    assert got is not None
    m = got.metadata
    assert (m["serveable"], m["mineable"], m["licence"], m["paywalled"]) == (
        "summary",
        False,
        "CC-BY-4.0",
        True,
    )
    assert m["text_status"] == "teaser" and m["editorial_type"] == "opinion"
    assert m["live_blog"] is True


_FEED = """<?xml version="1.0"?><rss version="2.0"><channel><title>F</title>
<item><title>A</title><link>https://news.test/articles/d1</link>
<pubDate>Tue, 17 Nov 2026 09:00:00 +0000</pubDate></item>
<item><title>B video</title><link>https://news.test/video/d2</link>
<pubDate>Tue, 17 Nov 2026 08:00:00 +0000</pubDate></item>
<item><title>C paper</title><link>https://news.test/articles/s3</link>
<pubDate>Tue, 17 Nov 2026 07:00:00 +0000</pubDate></item>
<item><title>D</title><link>https://news.test/articles/d4</link>
<pubDate>Tue, 17 Nov 2026 06:00:00 +0000</pubDate></item>
<item><title>E</title><link>https://news.test/articles/d5</link>
<pubDate>Tue, 17 Nov 2026 10:00:00 +0000</pubDate></item>
</channel></rss>"""


@pytest.fixture
def feed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        downloader,
        "fetch_rss_feed_url",
        lambda url, *a, **k: httpx.Response(
            200, content=_FEED.encode(), request=httpx.Request("GET", url)
        ),
    )


def test_the_listing_filters_by_url_and_keeps_the_newest_up_to_the_cap(feed: None) -> None:
    src = _Source()
    src.news = NewsSourceSpec(
        max_items=2, include_url_pattern="/articles/d", exclude_url_pattern="/video/"
    )
    items, cursor = rss.list_items(src, None)
    assert [i.metadata["title"] for i in items] == ["E", "A"]  # newest first, D past the cap
    assert cursor == "2026-11-17T10:00:00Z"


def test_a_feeds_spec_entry_carries_its_news_settings() -> None:
    doc = FeedsSpecDocument.model_validate(
        {
            "feeds": [
                {
                    "url": "https://news.test/feed.xml",
                    "content_type": "news_article",
                    "news": {"max_items": 5, "text_status": "teaser"},
                }
            ]
        }
    )
    entry = doc.feeds[0]
    assert isinstance(entry, RssFeedEntry) and entry.news is not None
    assert entry.news.max_items == 5 and entry.news.serveable == "metadata"
    assert "news" not in entry.override_update_dict()  # identity, never merged into Config
    with pytest.raises(ValueError):
        NewsSourceSpec(include_url_pattern="(")


# --- the text ---


def test_boilerplate_paragraphs_are_dropped_and_long_ones_kept() -> None:
    long_mention = "x " * 130 + "subscribe"
    body = "\n\n".join(
        [
            "The article's first paragraph.",
            "Subscribe to this journal",
            "Rent or buy this article",
            "We'll email you instructions to reset your password",
            long_mention,
        ]
    )
    assert extract.paragraphs(extract.strip_boilerplate(body)) == [
        "The article's first paragraph.",
        " ".join(long_mention.split()),
    ]


def test_the_feeds_own_text_is_taken_when_the_page_has_less() -> None:
    html = '<html lang="en"><body><div>Log in to read. Sign up</div></body></html>'
    feed_html = "<p>" + EN + "</p><p>" + EN + "</p>"
    found = extract.main_text(html, "https://news.test/a", feed_html=feed_html)
    assert found is not None and found.method == extract.FEED
    assert extract.paragraphs(found.text) == [EN, EN]


_PAYWALLED = (
    '<html lang="en"><head><meta name="description" content="Astronomers report a rare pair of '
    'bright black holes less than a billion years after the Big Bang."></head><body><article>'
    "<p>Access Nature and 54 other Nature Portfolio journals</p><p>Subscribe to this journal</p>"
    "<p>32.70 € / 30 days cancel any time</p></article></body></html>"
)


def test_a_page_with_no_usable_text_falls_back_to_its_standfirst() -> None:
    found = extract.main_text(_PAYWALLED, "https://news.test/a")
    assert found is not None and found.method == extract.STANDFIRST
    assert found.text.startswith("Astronomers report a rare pair")


def test_a_source_can_take_the_standfirst_or_the_feed_text() -> None:
    standfirst = extract.main_text(_html("en", EN), "https://news.test/a", text_from="standfirst")
    assert standfirst is None  # the page has no description: nothing, never the page instead
    got = extract.main_text(_PAYWALLED, "https://news.test/a", text_from="standfirst")
    assert got is not None and got.method == extract.STANDFIRST
    feed = extract.main_text(
        _html("en", EN), "https://news.test/a", feed_html=f"<p>{ES}</p>", text_from="feed"
    )
    assert feed is not None and feed.method == extract.FEED and feed.text == ES


def test_the_page_author_is_its_meta_tag() -> None:
    html = '<html><head><meta name="author" content="Ana  Example"></head><body></body></html>'
    assert extract.page_author(html, "https://news.test/a") == "Ana Example"


@pytest.mark.parametrize("raw", ["Al Jazeera Staff", "Ars Contributors", "Rest of World Staff"])
def test_a_desk_credit_is_not_an_author(raw: str) -> None:
    assert byline.parse(raw) == []
