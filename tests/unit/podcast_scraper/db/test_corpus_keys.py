"""The natural keys of ``sources`` and ``content_items`` rows (CRFC-005 §3.1).

The metadata is read from the fixtures the app-validation corpus holds, so a key is checked against
the shape the writers really produce: a podcast episode (``p10_e01``, a local feed id, no guid) and
a news article (``mercados``). A third, prod-shaped episode has a ``sha256:`` feed id and a guid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.db import corpus_keys as keys
from podcast_scraper.kg.pipeline import _podcast_node_id
from podcast_scraper.server.slugs import episode_slug

_CORPUS = Path(__file__).resolve().parents[3] / "fixtures" / "app-validation-corpus" / "public"
_PODCAST = _CORPUS / "feeds/p10/run_20260101-000000/metadata/p10_e01.metadata.json"
_NEWS = (
    _CORPUS
    / "sources/news_article/daily-example/run_20261117-090000/metadata/mercados.metadata.json"
)

_PROD_FEED = "sha256:4c673e4586fe2cd66e590fe07fdad14861e2233d236d26f802a4186f71317d26"
_PROD_GUID = "7c624127-5f6f-42b8-974e-ca4e131a19a1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def podcast() -> dict[str, Any]:
    return _load(_PODCAST)


@pytest.fixture
def news() -> dict[str, Any]:
    return _load(_NEWS)


@pytest.fixture
def prod_podcast() -> dict[str, Any]:
    # The shape of a prod episode (snapshot-prod-20261124, Talk Eastern Europe): a hashed feed id,
    # the RSS guid as the episode id.
    return {
        "feed": {"feed_id": _PROD_FEED, "url": "https://anchor.fm/s/fffeb140/podcast/rss"},
        "episode": {"episode_id": _PROD_GUID, "guid": _PROD_GUID},
    }


class TestPodcastKeys:
    def test_fixture_episode(self, podcast: dict[str, Any]) -> None:
        assert keys.item_natural_id(podcast) == "ep-7072d2e85dcd5233"
        assert keys.source_natural_id(podcast) == "podcast:p10"
        assert keys.raw_feed_id(podcast) == "p10"
        # No guid written: the episode id (a hash of feed, title, date) is the id in its source.
        assert keys.external_item_id(podcast) == "ep-7072d2e85dcd5233"

    def test_prod_shaped_episode(self, prod_podcast: dict[str, Any]) -> None:
        assert keys.item_natural_id(prod_podcast) == _PROD_GUID
        assert keys.raw_feed_id(prod_podcast) == _PROD_FEED
        assert keys.external_item_id(prod_podcast) == _PROD_GUID

    def test_source_id_is_the_one_kg_writes(self, prod_podcast: dict[str, Any]) -> None:
        assert keys.source_natural_id(prod_podcast) == _podcast_node_id(_PROD_FEED)
        assert keys.source_natural_id(prod_podcast).startswith("podcast:")

    def test_values_are_stripped(self) -> None:
        doc = {"feed": {"feed_id": " f "}, "episode": {"episode_id": " e ", "guid": " g "}}
        assert keys.raw_feed_id(doc) == "f"
        assert keys.item_natural_id(doc) == "e"
        assert keys.external_item_id(doc) == "g"

    @pytest.mark.parametrize(
        "fn, doc",
        [
            (keys.item_natural_id, {"episode": {}}),
            (keys.item_natural_id, {}),
            (keys.source_natural_id, {"feed": {"feed_id": ""}}),
            (keys.source_natural_id, {"feed": {"feed_id": "podcast:unknown"}}),
            (keys.raw_feed_id, {"feed": None}),
            (keys.external_item_id, {}),
            (keys.external_item_id, {"episode": {"guid": None}}),
        ],
    )
    def test_a_missing_key_is_refused_not_invented(self, fn: Any, doc: dict[str, Any]) -> None:
        with pytest.raises(keys.KeyNotDerivable):
            fn(doc)


class TestNewsKeys:
    def test_fixture_article(self, news: dict[str, Any]) -> None:
        assert keys.item_natural_id(news) == "item:news_article:b626e4b9538458cb6235"
        assert keys.source_natural_id(news) == "source:125361805bbcfcb84b90"
        assert keys.external_item_id(news) == "item:news_article:b626e4b9538458cb6235"
        assert keys.raw_feed_id(news) is None

    def test_external_id_ignores_the_transport_ids(self, news: dict[str, Any]) -> None:
        news["news_article"]["transport_ids"] = {"hn": "4242", "rss": "guid-1"}
        assert keys.external_item_id(news) == news["item_id"]
        news["news_article"]["transport_ids"] = {"rss": "guid-1", "hn": "4242"}
        assert keys.external_item_id(news) == news["item_id"]

    def test_slug_is_of_source_and_item(self, news: dict[str, Any]) -> None:
        assert keys.item_slug_for(news) == "source125361805bbcfcb84b90-e7dd51d60a"
        assert keys.item_slug_for(news, metadata_relpath="any/run/path") == keys.item_slug_for(news)
        assert keys.item_slug_for(news) == keys.item_slug(
            "source:125361805bbcfcb84b90", "item:news_article:b626e4b9538458cb6235", ""
        )

    def test_missing_ids_are_refused(self, news: dict[str, Any]) -> None:
        del news["item_id"], news["source_id"]
        with pytest.raises(keys.KeyNotDerivable):
            keys.item_natural_id(news)
        with pytest.raises(keys.KeyNotDerivable):
            keys.source_natural_id(news)
        with pytest.raises(keys.KeyNotDerivable):
            keys.external_item_id(news)
        with pytest.raises(keys.KeyNotDerivable):
            keys.item_slug_for(news)


# Values the slug had before its derivation moved into corpus_keys (computed on the pre-move
# server.slugs.episode_slug); the first five are the cases test_app_slugs.py covers.
_SLUGS_BEFORE = [
    (("My Feed", "ep-1", "metadata/x.metadata.json"), "my-feed-2b23ca75f3"),
    (("feed", "ep1", "p"), "feed-1be1fc511d"),
    (("feed", None, "metadata/one.metadata.json"), "feed-e2e336ad9a"),
    (("", "ep1", "p"), "feed-85ac045785"),
    (("Café Crème", "ep1", "p"), "cafe-creme-1f820148d7"),
    (
        (
            "p10",
            "ep-7072d2e85dcd5233",
            "feeds/p10/run_20260101-000000/metadata/p10_e01.metadata.json",
        ),
        "p10-7c1875e8b4",
    ),
    (
        (_PROD_FEED, _PROD_GUID, "x"),
        "sha2564c673e4586fe2cd66e590fe07fdad14861e2233d236d26f802a4186f71317d26-6c5cb7b989",
    ),
    (("feed", "", "metadata/a.metadata.json"), "feed-12d0d22100"),
    (("", None, "sources/news_article/d/run_1/metadata/a.metadata.json"), "feed-f625467f42"),
    (("!!!", "e", "p"), "feed-6804e5b622"),
]


class TestItemSlugFor:
    def test_podcast_is_the_slug_the_catalog_gives(self, podcast: dict[str, Any]) -> None:
        assert keys.item_slug_for(podcast) == "p10-7c1875e8b4"

    def test_prod_shaped_podcast(self, prod_podcast: dict[str, Any]) -> None:
        assert keys.item_slug_for(prod_podcast) == _SLUGS_BEFORE[6][1]

    def test_podcast_without_an_episode_id_falls_back_to_the_path(self) -> None:
        doc = {"feed": {"feed_id": "feed"}, "episode": {}}
        rel = "metadata/one.metadata.json"
        assert keys.item_slug_for(doc, metadata_relpath=rel) == "feed-e2e336ad9a"
        assert keys.item_slug_for(doc, metadata_relpath=rel) != keys.item_slug_for(doc)

    def test_a_second_news_item_has_its_own_slug(self, news: dict[str, Any]) -> None:
        other = dict(news, item_id="item:news_article:0123456789abcdef0123")
        assert keys.item_slug_for(other) == "source125361805bbcfcb84b90-8696a69e39"


class TestItemSlug:
    @pytest.mark.parametrize("args, expected", _SLUGS_BEFORE)
    def test_item_slug_and_episode_slug_keep_their_values(
        self, args: tuple[str, str | None, str], expected: str
    ) -> None:
        assert keys.item_slug(*args) == expected
        assert episode_slug(*args) == expected

    def test_episode_id_wins_over_the_path(self) -> None:
        assert keys.item_slug("feed", "ep1", "a") == keys.item_slug("feed", "ep1", "b")

    def test_without_an_episode_id_the_path_decides(self) -> None:
        one = keys.item_slug("feed", None, "metadata/one.metadata.json")
        assert one != keys.item_slug("feed", None, "metadata/two.metadata.json")
        assert one == keys.item_slug("feed", None, "metadata/one.metadata.json")

    def test_episode_slug_keeps_its_cache(self) -> None:
        assert hasattr(episode_slug, "cache_info")
