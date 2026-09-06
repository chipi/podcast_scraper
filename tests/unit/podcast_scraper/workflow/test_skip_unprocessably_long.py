"""Episodes longer than the extractor can read are SKIPPED, not truncated (#1975).

An over-long episode does not fail — it truncates. Quote extraction never sees its tail, yet the
episode still emits insights, still scores against the §5i gates, and still looks fine. That is
worse than not having it: a gap is measurable and backfillable the day chunk/map-reduce lands,
whereas silently degraded data pollutes both the corpus and the gates that decide which feeds to
onboard.

The ceiling is derived from ``GI_QUOTE_TRANSCRIPT_MAX_CHARS`` — exactly what the extractor can
read — not chosen by taste. It happens to land within 6% of the independent §5h operator rule of
two hours.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from podcast_scraper import config_constants
from podcast_scraper.workflow.stages import scraping

pytestmark = pytest.mark.unit

_ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def _item(title: str, duration: str | None) -> ET.Element:
    el = ET.Element("item")
    t = ET.SubElement(el, "title")
    t.text = title
    if duration is not None:
        d = ET.SubElement(el, f"{{{_ITUNES}}}duration")
        d.text = duration
    return el


class _Cfg:
    max_episode_seconds = 0


def _drop(items):
    return scraping._drop_unprocessably_long(items, _Cfg())


def test_an_episode_over_the_ceiling_is_dropped() -> None:
    over = config_constants.MAX_PROCESSABLE_EPISODE_SECONDS + 600
    kept = _drop([_item("short", "1800"), _item("epic", str(over))])
    assert [_title(i) for i in kept] == ["short"]


def test_an_episode_at_the_ceiling_is_kept() -> None:
    """The boundary must not silently shrink the corpus by one episode."""
    at = config_constants.MAX_PROCESSABLE_EPISODE_SECONDS
    assert len(_drop([_item("boundary", str(at))])) == 1


def test_unknown_duration_is_kept_not_dropped() -> None:
    """Many feeds omit itunes:duration. Refusing them would shrink the corpus for a metadata gap."""
    kept = _drop([_item("no duration", None), _item("junk", "not-a-number")])
    assert len(kept) == 2


def test_hhmmss_and_mmss_are_both_understood() -> None:
    """A feed writing 3:30:00 must not read as 3 seconds and slip through."""
    assert _drop([_item("three and a half hours", "3:30:00")]) == []
    assert len(_drop([_item("fifty minutes", "50:00")])) == 1


def test_the_ceiling_matches_what_the_extractor_can_actually_read() -> None:
    """Derived, not chosen — so it moves with the budget instead of drifting from it."""
    expected = int(
        (
            config_constants.GI_QUOTE_TRANSCRIPT_MAX_CHARS
            / config_constants.CHARS_PER_MINUTE_OF_SPEECH
        )
        * 60
    )
    assert config_constants.MAX_PROCESSABLE_EPISODE_SECONDS == expected


def test_the_ceiling_is_near_the_two_hour_operator_rule() -> None:
    """§5h independently set 2 hours on editorial grounds; the two should not wildly disagree."""
    assert 6600 <= config_constants.MAX_PROCESSABLE_EPISODE_SECONDS <= 8400


def _title(item: ET.Element) -> str:
    el = item.find("title")
    return (el.text or "") if el is not None else ""
