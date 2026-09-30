"""Another episode's author is not this episode's host (#2197).

Latent Space names no host at feed level, so the episode-level ``<itunes:author>`` fallback seated
"Brandon Anderson, RJ Honicky, and Latent.Space" — one post's byline — as the host of 9 episodes.
The live feed (2026-09-30) carries that tag on ONE of 229 items; 170 say only "Latent.Space".
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.stages.processing import hosts_for_episode
from podcast_scraper.workflow.types import HostDetectionResult

pytestmark = pytest.mark.unit

_ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def _episode(author: str | None) -> SimpleNamespace:
    item = ET.Element("item")
    if author is not None:
        ET.SubElement(item, f"{{{_ITUNES}}}author").text = author
    return SimpleNamespace(item=item)


def _fallback_result(**kw) -> HostDetectionResult:
    """What detection returns when the feed named nobody and the fallback fired."""
    union = {"Brandon Anderson", "RJ Honicky"}
    return HostDetectionResult(
        cached_hosts=union | kw.get("extra", set()),
        heuristics=None,
        feed_title="Latent Space: The AI Engineer Podcast",
        episode_author_hosts=frozenset(union),
    )


def test_an_episode_bylined_only_by_the_show_gets_no_borrowed_host() -> None:
    """THE bug: 8 of the 9 episodes were bylined 'Latent.Space' and still got the two hosts."""
    assert hosts_for_episode(_fallback_result(), _episode("Latent.Space")) == set()


def test_an_episode_gets_its_own_byline_split_into_people() -> None:
    got = hosts_for_episode(
        _fallback_result(), _episode("Brandon Anderson, RJ Honicky, and Latent.Space")
    )
    assert got == {"Brandon Anderson", "RJ Honicky"}


def test_a_different_bylined_episode_gets_its_own_authors() -> None:
    got = hosts_for_episode(_fallback_result(), _episode("Alessio Fanelli and Latent.Space"))
    assert got == {"Alessio Fanelli"}


def test_non_fallback_hosts_are_kept() -> None:
    """Config known_hosts / recurrent hosts are not the fallback's to replace."""
    got = hosts_for_episode(_fallback_result(extra={"swyx"}), _episode("Latent.Space"))
    assert got == {"swyx"}


def test_a_feed_that_states_its_hosts_is_unchanged() -> None:
    result = HostDetectionResult(cached_hosts={"Tyler Cowen"}, heuristics=None)
    assert hosts_for_episode(result, _episode("Someone Else")) == {"Tyler Cowen"}


def test_an_episode_without_an_item_gets_only_the_non_fallback_hosts() -> None:
    assert hosts_for_episode(_fallback_result(), SimpleNamespace()) == set()
