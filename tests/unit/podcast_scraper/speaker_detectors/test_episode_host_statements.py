"""An episode's own "hosted by X" joins that episode's host pool (synthetic shows)."""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    compose_episode_hosts,
    hosts_stated_in_episode_text,
)

pytestmark = pytest.mark.unit


def test_hosted_by_names_join_the_pool_beside_the_feeds_hosts() -> None:
    pool = compose_episode_hosts(
        ["Ada Wren"],
        [],
        episode_title="Rivers of the Ottoman world",
        episode_description="A conversation hosted by Tobias Holt and Mira Lane.",
        feed_title="River Notes",
    )
    assert pool[0] == "Ada Wren"  # the feed's host first
    assert set(pool) == {"Ada Wren", "Tobias Holt", "Mira Lane"}


@pytest.mark.parametrize(
    "description, expected",
    [
        ("Presented by Tobias Holt.", ["Tobias Holt"]),
        ("With co-host Mira Lane this week.", ["Mira Lane"]),
        ("Hosted by philosopher Ada Wren Please do email us", ["Ada Wren"]),
        ("Hosted by Simplecast, an AdsWizz company.", []),  # one word: not a person's name
        ("Join Ada Wren for a walk.", []),  # the looser phrases are not host statements
    ],
)
def test_the_statements_read(description, expected) -> None:
    assert hosts_stated_in_episode_text(None, description) == expected


def test_a_byline_still_fills_a_feed_that_names_nobody() -> None:
    """Additive: the statement does not switch off the byline rule."""
    pool = compose_episode_hosts(
        [],
        ["Ada Wren"],
        episode_title="Ep",
        episode_description="Hosted by Tobias Holt.",
        feed_title="River Notes",
    )
    assert set(pool) == {"Ada Wren", "Tobias Holt"}
