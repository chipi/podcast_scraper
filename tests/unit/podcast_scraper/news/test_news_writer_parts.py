"""The news writer's parts on their own: bylines, extraction, robots.txt, the RSS cursor and the
contract document (E05, News 1.0)."""

from __future__ import annotations

import time
from typing import Any, Iterator, List, Optional, Tuple

import httpx
import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.news import byline, document, extract, fetch, rss
from podcast_scraper.publishers import Publisher, Registry
from podcast_scraper.rss import downloader

pytestmark = pytest.mark.unit


# --- bylines ---


@pytest.mark.parametrize(
    "raw, names",
    [
        ("By Ana Example", ["Ana Example"]),
        ("Ana Example and Cy Example", ["Ana Example", "Cy Example"]),
        ("Ana Example, Bo Example & Cy Example", ["Ana Example", "Bo Example", "Cy Example"]),
        ("Ana Example; Ana Example", ["Ana Example"]),
        ("Ana Example (ana@dailyexample.test)", ["Ana Example"]),
        ("", []),
        (None, []),
    ],
)
def test_a_byline_splits_into_names_in_order(raw: Optional[str], names: List[str]) -> None:
    assert byline.parse(raw) == names


def test_the_publisher_writing_as_itself_is_not_an_author() -> None:
    assert byline.parse("Daily Example and Ana Example", organisations=["Daily Example"]) == [
        "Ana Example"
    ]


# --- extraction ---

_ARTICLE = "".join(
    f"<p>Paragraph {n} of the article says something of its own, number {n}, at some length so "
    f"that it reads as prose and not as a menu entry or a caption.</p>"
    for n in range(1, 7)
)


def _page(body: str, lang: str = "en-GB") -> str:
    return (
        f'<html lang="{lang}"><head><title>T</title></head><body>'
        f"<nav><a href='/'>Home</a></nav><article><h1>Title</h1>{body}</article>"
        "<footer><p>Footer text.</p></footer></body></html>"
    )


def test_the_page_language_is_its_primary_subtag() -> None:
    assert extract.page_language(_page(_ARTICLE, "en-GB")) == "en"
    assert extract.page_language(_page(_ARTICLE, "pt_BR")) == "pt"
    assert extract.page_language("<html><body><p>x</p></body></html>") is None


def test_readability_is_taken_when_trafilatura_finds_too_little(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trafilatura

    monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: "Short.")
    found = extract.main_text(_page(_ARTICLE), "https://news.test/a")
    assert found is not None and found.method == extract.READABILITY
    assert len(found.text) > 2 * len("Short.")


def test_trafilatura_is_kept_when_it_finds_the_article() -> None:
    found = extract.main_text(_page(_ARTICLE), "https://news.test/a")
    assert found is not None and found.method == extract.TRAFILATURA
    assert "Footer text" not in found.text and "Home" not in found.text


def test_a_page_with_no_text_gives_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import trafilatura

    monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: None)
    assert extract.main_text("<html><body></body></html>", "https://news.test/a") is None


@pytest.mark.parametrize(
    "url, text, status",
    [
        ("https://news.test/videos/trail", "x" * 2000, "video"),
        ("https://news.test/av/world-123", "x" * 2000, "video"),
        ("https://news.test/2026/a", "x" * 100, "teaser"),
        ("https://news.test/2026/a", "x" * 2000, "full"),
    ],
)
def test_text_status(url: str, text: str, status: str) -> None:
    assert extract.text_status(text, url) == status


# --- robots.txt (RFC 9309) ---


@pytest.fixture
def robots(monkeypatch: pytest.MonkeyPatch) -> Iterator[List[Tuple[int, bytes]]]:
    answers: List[Tuple[int, bytes]] = []
    monkeypatch.setattr(downloader, "http_get_any_status", lambda url, ua, t: answers.pop(0))
    fetch.reset_robots_cache()
    yield answers
    fetch.reset_robots_cache()


def test_a_disallowed_path_is_refused(robots: List[Tuple[int, bytes]]) -> None:
    robots.append((200, b"User-agent: *\nDisallow: /private/\n"))
    assert fetch.robots_allow("https://news.test/2026/a", "agent")
    assert not fetch.robots_allow("https://news.test/private/a", "agent")  # cached: one fetch


def test_a_missing_robots_txt_allows_everything(robots: List[Tuple[int, bytes]]) -> None:
    robots.append((404, b""))
    assert fetch.robots_allow("https://news.test/anything", "agent")


@pytest.mark.parametrize("status", [500, 503, 0])
def test_an_erroring_or_unreachable_robots_txt_allows_nothing(
    robots: List[Tuple[int, bytes]], status: int
) -> None:
    robots.append((status, b""))
    assert not fetch.robots_allow("https://news.test/anything", "agent")


def test_fetch_body_skips_what_robots_txt_refuses(
    robots: List[Tuple[int, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    robots.append((200, b"User-agent: *\nDisallow: /\n"))
    monkeypatch.setattr(downloader, "fetch_url", lambda *a, **k: pytest.fail("fetched anyway"))
    item = ct.SourceItem(item_id="g", url="https://news.test/a")
    assert fetch.fetch_body(item, None) is None


# --- the RSS transport's cursor ---

_FEED = """<?xml version="1.0"?><rss version="2.0"><channel><title>F</title><language>en</language>
<item><title>New</title><link>https://news.test/new</link><guid>n</guid>
<pubDate>Tue, 17 Nov 2026 08:00:00 +0000</pubDate><author>a@news.test (Ana Example)</author></item>
<item><title>Old</title><link>https://news.test/old</link><guid>o</guid>
<pubDate>Mon, 16 Nov 2026 08:00:00 +0000</pubDate></item>
<item><title>Undated</title><link>https://news.test/undated</link></item>
<item><title>No link</title><guid>x</guid></item>
</channel></rss>"""


class _Source:
    url = "https://news.test/feed.xml"


@pytest.fixture
def feed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, ua: str, timeout: int, **_: Any) -> httpx.Response:
        return httpx.Response(200, content=_FEED.encode(), request=httpx.Request("GET", url))

    monkeypatch.setattr(downloader, "fetch_rss_feed_url", fake)


def test_the_first_listing_returns_every_linked_item_and_the_newest_date(feed: None) -> None:
    items, cursor = rss.list_items(_Source(), None)
    assert [i.metadata["title"] for i in items] == ["New", "Old", "Undated"]
    assert cursor == "2026-11-17T08:00:00Z"
    new = items[0]
    assert new.text is None and new.url == "https://news.test/new"
    assert new.metadata["transport_ids"] == {"rss": "n"}
    assert new.metadata["published_at"] == "2026-11-17T08:00:00Z"
    assert new.metadata["feed_language"] == "en"
    assert items[2].metadata["transport_ids"] == {"rss": "https://news.test/undated"}


def test_a_later_listing_returns_only_what_is_newer_than_the_cursor(feed: None) -> None:
    items, cursor = rss.list_items(_Source(), "2026-11-16T08:00:00Z")
    assert [i.metadata["title"] for i in items] == ["New", "Undated"]
    assert cursor == "2026-11-17T08:00:00Z"


def test_an_unfetchable_feed_keeps_its_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "fetch_rss_feed_url", lambda *a, **k: None)
    assert rss.list_items(_Source(), "2026-11-16T08:00:00Z") == ([], "2026-11-16T08:00:00Z")


def test_iso_reads_feedparsers_utc_struct_time() -> None:
    assert rss._iso(time.strptime("2026-11-17 08:00:00", "%Y-%m-%d %H:%M:%S")) == (
        "2026-11-17T08:00:00Z"
    )
    assert rss._iso(None) is None


# --- the contract document ---

_REG = Registry(
    (
        Publisher(
            id="org:daily", name="Daily Example", domains=("daily.test",), sources=("source:s1",)
        ),
        Publisher(id="org:wire", name="Other Wire", domains=("otherwire.test",)),
    )
)


def _item(url: str, source_id: str = "source:agg", **fields: Any) -> ct.ItemContext:
    return ct.ItemContext(
        content_type=ct.content_type("news_article"),
        item_id="item:news_article:abc",
        source_id=source_id,
        transport="rss",
        times={"published_at": "2026-11-17T08:00:00Z"},
        fields={"url": url, "title": "T", "body_sha256": "f" * 64, **fields},
    )


def test_the_publisher_comes_from_the_source_then_the_domain() -> None:
    by_source = document.shape({}, _item("https://elsewhere.test/a", "source:s1"), registry=_REG)
    assert by_source["publisher_id"] == "org:daily"
    by_domain = document.shape({}, _item("https://m.otherwire.test/a"), registry=_REG)
    assert by_domain["publisher_id"] == "org:wire"
    neither = document.shape({}, _item("https://nowhere.test/a"), registry=_REG)
    assert "publisher_id" not in neither


def test_the_summary_is_the_summary_stages_text_or_absent() -> None:
    done = {"summary": {"title": "S", "bullets": ["One.", "Two.", "Three."], "raw_text": "r"}}
    assert document.shape(done, _item("https://n.test/a"), registry=_REG)["summary"] == "One. Two."
    raw_only = {"summary": {"bullets": [], "raw_text": "Raw."}}
    assert document.shape(raw_only, _item("https://n.test/a"), registry=_REG)["summary"] == "Raw."
    assert "summary" not in document.shape({}, _item("https://n.test/a"), registry=_REG)


def test_the_url_is_canonical_and_unknown_values_are_not_written() -> None:
    doc = document.shape(
        {},
        _item(
            "HTTPS://Daily.TEST/a/?utm_source=x",
            editorial_type="gossip",
            live_blog=True,
            byline_raw="By Daily Example and Ana Example",
        ),
        registry=_REG,
    )
    assert doc["url"] == "https://daily.test/a"
    assert doc["news_article"]["editorial_type"] is None
    assert doc["news_article"]["live_blog"] is True
    assert doc["news_article"]["byline"] == ["Ana Example"]  # the publisher is not an author
    assert doc["serveable"] == "metadata" and doc["mineable"] is True


def test_a_roundup_is_written_under_news_article_and_kept_by_a_rederive() -> None:
    plain = document.shape({}, _item("https://n.test/a"), registry=_REG)
    assert "roundup" not in plain["news_article"]
    doc = document.shape({}, _item("https://n.test/a", roundup=True), registry=_REG)
    assert doc["news_article"]["roundup"] is True
    restored = document.restore_item(doc, "Today: three stories.")
    assert restored.metadata.get("roundup") is True


@pytest.mark.parametrize(
    "method, version",
    [
        (extract.FEED, "feed/1"),
        (extract.STANDFIRST, "standfirst/1"),
        ("guardian", "guardian/api"),
        ("something_else", "something_else/unknown"),
    ],
)
def test_each_extractor_has_its_version(method: str, version: str) -> None:
    assert extract.extractor_version(method) == version


def test_a_library_extractor_carries_its_package_version() -> None:
    from importlib import metadata

    assert extract.extractor_version(extract.TRAFILATURA) == (
        f"trafilatura/{metadata.version('trafilatura')}"
    )
    assert extract.extractor_version(extract.READABILITY) == (
        f"readability/{metadata.version('readability-lxml')}"
    )


def test_the_extraction_is_written_under_news_article_and_kept_by_a_rederive() -> None:
    plain = document.shape({}, _item("https://n.test/a"), registry=_REG)
    assert "extraction" not in plain["news_article"]
    doc = document.shape(
        {},
        _item("https://n.test/a", extraction="trafilatura", extractor_version="trafilatura/2.0.1"),
        registry=_REG,
    )
    own = doc["news_article"]
    assert (own["extraction"], own["extractor_version"]) == ("trafilatura", "trafilatura/2.0.1")
    restored = document.restore_item(doc, "Body.")
    assert restored.metadata.get("extractor_version") == "trafilatura/2.0.1"


def test_an_item_class_is_written_under_news_article_and_kept_by_a_rederive() -> None:
    plain = document.shape({}, _item("https://n.test/a"), registry=_REG)
    assert "item_class" not in plain["news_article"]
    doc = document.shape(
        {},
        _item("https://n.test/a", text_status="teaser", item_class="small_teaser"),
        registry=_REG,
    )
    assert doc["news_article"]["item_class"] == "small_teaser"
    restored = document.restore_item(doc, "A one-line standfirst.")
    assert restored.metadata.get("item_class") == "small_teaser"


@pytest.mark.parametrize("status", [extract.TEXT_TEASER, extract.TEXT_VIDEO])
def test_a_teaser_or_video_item_gets_a_kg_but_no_summary(status: str) -> None:
    """Contract (news-item-1.0 ``text_status``): a teaser or video body is never enriched as if
    whole. A summary of one to three sentences repeats them or is filled in by the model; the
    entities and topics in them are still worth a KG (operator, 2026-11-18)."""
    item = _item("https://n.test/a", text_status=status, language="en")
    assert not ct.runs_for_item(item, ct.SUMMARY)
    assert ct.runs_for_item(item, ct.KG)
    assert ct.runs_for_item(item, ct.TRANSLATION)


def test_a_full_item_gets_its_summary_and_kg() -> None:
    item = _item("https://n.test/a", text_status=extract.TEXT_FULL, language="en")
    assert ct.runs_for_item(item, ct.SUMMARY) and ct.runs_for_item(item, ct.KG)


@pytest.mark.parametrize("language", ["fr", "de", "fr-CA", None])
def test_an_item_not_in_english_is_written_but_not_translated_or_enriched(
    language: Optional[str],
) -> None:
    """News 1.0 does not translate (operator, 2026-11-18). An article in another language, or in
    one nobody could tell, is written as it is; no stage runs English prompts over it."""
    item = _item("https://n.test/a", text_status=extract.TEXT_FULL, language=language)
    for stage in (ct.TRANSLATION, ct.SUMMARY, ct.KG):
        assert not ct.runs_for_item(item, stage)
        assert ct.stage_skip_reason(item, stage) == "translation_off"


def test_the_sources_translate_switch_hands_the_item_to_translation() -> None:
    item = _item("https://n.test/a", text_status=extract.TEXT_FULL, language="fr", translate=True)
    assert all(ct.runs_for_item(item, s) for s in (ct.TRANSLATION, ct.SUMMARY, ct.KG))


@pytest.mark.parametrize("language", ["en", "en-GB", "EN"])
def test_an_english_item_is_enriched(language: str) -> None:
    item = _item("https://n.test/a", text_status=extract.TEXT_FULL, language=language)
    assert ct.runs_for_item(item, ct.SUMMARY) and ct.runs_for_item(item, ct.KG)


def test_a_later_listing_adds_only_the_transports_the_item_lacks() -> None:
    doc = document.shape({}, _item("https://n.test/a", transport_ids={"rss": "g1"}), registry=_REG)
    rss_again = ct.SourceItem(item_id="g2", metadata={"transport_ids": {"rss": "g2"}})
    assert document.also_listed(doc, rss_again) is None  # a transport keeps its first id
    hn = ct.SourceItem(item_id="9", metadata={"transport_ids": {"hn": "9"}})
    updated = document.also_listed(doc, hn)
    assert updated is not None
    assert updated["news_article"]["transport_ids"] == {"rss": "g1", "hn": "9"}
    assert list(updated) == list(doc)
    assert {k: v for k, v in updated.items() if k != "news_article"} == {
        k: v for k, v in doc.items() if k != "news_article"
    }


def test_a_restored_teaser_still_gets_no_summary() -> None:
    """``rederive_only`` re-runs the stages on a restored item: its stored text_status holds."""
    teaser = _item("https://n.test/a", text_status=extract.TEXT_TEASER)
    restored = document.restore_item(document.shape({}, teaser, registry=_REG), "A short teaser.")
    assert not ct.runs_for_item(_item("https://n.test/a", **restored.metadata), ct.SUMMARY)
