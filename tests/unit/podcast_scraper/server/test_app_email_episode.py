"""What an email shows next to an episode item (operator 2026-10-05) — app_email_episode."""

from __future__ import annotations

from pathlib import Path

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
    allowed = {"episode_title", "podcast_title", "artwork_url", "duration_seconds", "publish_date"}
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
