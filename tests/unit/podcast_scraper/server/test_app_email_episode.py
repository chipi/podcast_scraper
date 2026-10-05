"""What an email shows next to an episode item (operator 2026-10-05) — app_email_episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server.app_email_episode import enrich_items, episode_display
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import build_catalog_rows

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_CORPUS = _REPO / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _a_slug() -> str:
    return slug_for_row(build_catalog_rows(_CORPUS)[0])


def test_display_facts_come_from_the_catalog() -> None:
    facts = episode_display(_CORPUS, _a_slug())
    assert facts["episode_title"]
    assert facts["podcast_title"]
    allowed = {
        "episode_title",
        "podcast_title",
        "artwork_url",
        "duration_seconds",
        "publish_date",
        "description",
        "summary",
    }
    assert set(facts) <= allowed


def test_unknown_slug_adds_nothing() -> None:
    item = {"episode_slug": "no-such-episode", "deep_link": "/episode/no-such-episode"}
    enrich_items(_CORPUS, [item])
    assert item == {"episode_slug": "no-such-episode", "deep_link": "/episode/no-such-episode"}


def test_fills_in_place_and_never_overwrites_the_producer() -> None:
    slug = _a_slug()
    item = {
        "episode_slug": slug,
        "deep_link": f"/episode/{slug}",
        "episode_title": "Set by producer",
    }
    enrich_items(_CORPUS, [item])
    assert item["episode_title"] == "Set by producer"
    assert item["podcast_title"]


def test_a_topic_link_is_not_dressed_as_its_representative_episode() -> None:
    # Trending items carry a representative episode's slug but link to the TOPIC.
    item = {"episode_slug": _a_slug(), "deep_link": "/topic/topic%3Aai"}
    enrich_items(_CORPUS, [item])
    assert "episode_title" not in item and "artwork_url" not in item


def test_summary_is_cut_at_a_word_to_about_three_lines() -> None:
    from podcast_scraper.server.app_email_episode import _short, SUMMARY_CHARS

    long = "word " * 200
    out = _short(long)
    assert out is not None and out.endswith("…") and len(out) <= SUMMARY_CHARS + 1
    assert not out[:-1].endswith(" ")
    assert _short("  short   text ") == "short text"
    assert _short(None) is None and _short("") is None


def test_an_episode_carries_the_themes_and_storylines_of_its_topics() -> None:
    # operator 2026-10-05: topic, theme and storyline chips wherever an email shows an episode.
    from podcast_scraper.search.topic_clusters import theme_map_by_topic

    themes = theme_map_by_topic(_CORPUS)
    topic = next(iter(themes))
    slug = _a_slug()
    item: dict[str, Any] = {
        "episode_slug": slug,
        "deep_link": f"/episode/{slug}",
        "graph_refs": [{"id": topic, "kind": "topic", "label": topic}],
    }
    enrich_items(_CORPUS, [item])
    assert item["themes"][0]["id"] == themes[topic]["cluster_id"]
    assert item["themes"][0]["id"].startswith("tc:")
    for s in item.get("storylines", []):
        assert s["id"].startswith("topic:")  # the storyline route takes the anchor topic id
