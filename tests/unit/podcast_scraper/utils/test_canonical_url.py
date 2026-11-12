"""The canonical URL rules ids are minted from (ontology v3; RFC-134 §2), one case per rule."""

from __future__ import annotations

import pytest

from podcast_scraper.utils.canonical_url import canonical_url

pytestmark = pytest.mark.unit

BASE = "https://example.com/world/2026/nov/12/a-story"


@pytest.mark.parametrize(
    "variant",
    [
        BASE,
        "http://example.com/world/2026/nov/12/a-story",  # scheme
        "HTTPS://Example.COM/world/2026/nov/12/a-story",  # scheme and host case
        "https://example.com:443/world/2026/nov/12/a-story",  # default port
        "https://example.com/world/2026/nov/12/a-story/",  # trailing slash
        "https://example.com/world/2026/nov/12/a-story?utm_source=rss&utm_medium=feed",
        "https://example.com/world/2026/nov/12/a-story?fbclid=abc&gclid=def",
        "https://example.com/world/2026/nov/12/a-story#comments",  # fragment
        "https://amp.example.com/world/2026/nov/12/a-story",  # AMP host
        "https://example.com/world/2026/nov/12/a-story/amp",  # AMP path
        "https://example.com/world/2026/nov/12/a-story/amp/",
        "https://example.com/world/2026/nov/12/a-story?amp",  # AMP switch
        "https://example.com/world/2026/nov/12/a-story?amp=1",
        "https://example.com/world/2026/nov/12/a-story?outputType=amp",
        "  https://example.com/world/2026/nov/12/a-story  ",  # surrounding space
    ],
)
def test_variants_of_one_page_are_one_url(variant: str) -> None:
    assert canonical_url(variant) == BASE


def test_content_parameters_are_kept_and_ordered() -> None:
    a = canonical_url("https://example.com/search?q=rates&page=2&utm_campaign=x")
    b = canonical_url("https://example.com/search?page=2&q=rates")
    assert a == b == "https://example.com/search?page=2&q=rates"


@pytest.mark.parametrize(
    "one, other",
    [
        ("https://example.com/Story", "https://example.com/story"),  # path case is the page's
        ("https://www.example.com/a", "https://example.com/a"),  # www. is not assumed the same
        ("https://example.com:8443/a", "https://example.com/a"),  # a non-default port
        ("https://example.com/a?id=1", "https://example.com/a?id=2"),
        ("https://example.com/campaign", "https://example.com/amp-campaign"),
    ],
)
def test_pages_that_may_differ_stay_apart(one: str, other: str) -> None:
    assert canonical_url(one) != canonical_url(other)


def test_the_root_is_one_page() -> None:
    assert canonical_url("https://example.com") == canonical_url("https://example.com/")


def test_idempotent() -> None:
    url = "http://AMP.Example.com/a/amp/?utm_source=x&b=2&a=1#f"
    once = canonical_url(url)
    assert canonical_url(once) == once == "https://example.com/a?a=1&b=2"


def test_ids_are_minted_from_the_canonical_url() -> None:
    from podcast_scraper import content_types as ct

    by_url = ct.ContentType(
        name="by_url",
        id_scheme="item",
        stage_profile=ct.StageProfile(name="u"),
        default_transport="fixture_feed",
        source_kind="test",
        identity=lambda item: ct.url_identity(item.url or ""),
    )
    linked = ct.SourceItem(item_id="guid-1", url="http://Example.com/a/?utm_source=rss#top")
    clean = ct.SourceItem(item_id="api-77", url="https://example.com/a")
    assert ct.item_id(by_url, linked) == ct.item_id(by_url, clean)
    assert ct.source_id(by_url, "https://amp.example.com/feed/") == ct.source_id(
        by_url, "https://example.com/feed"
    )


@pytest.mark.parametrize("bad", ["", "example.com/a", "/a/b", "https://"])
def test_a_relative_or_hostless_url_has_no_id(bad: str) -> None:
    with pytest.raises(ValueError):
        canonical_url(bad)
