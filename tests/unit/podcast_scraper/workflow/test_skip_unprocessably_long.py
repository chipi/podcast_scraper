"""Episodes longer than the extractor can read are SKIPPED, not truncated (#1975).

An over-long episode does not fail — it truncates. Quote extraction never sees its tail, yet the
episode still emits insights, still scores against the §5i gates, and still looks fine. That is
worse than not having it: a gap is measurable and backfillable the day chunk/map-reduce lands,
whereas silently degraded data pollutes both the corpus and the gates that decide which feeds to
onboard.

The ceiling is DERIVED from the window the deployment serves — exactly what the extractor can
read — never chosen by taste.

#2050 removed the global fallback. If no StageOption declares a window we do not know it, and a
permanent editorial decision must not rest on a guess: nothing is skipped, and an episode that
turns out not to fit gets a 400 naming the real limit for the provider to clamp against.
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


#: The DGX window, as `prod_dgx_full` materializes it from the summary StageOption.
_SERVED_WINDOW = 32_768


def _ceiling(window: int = _SERVED_WINDOW) -> int:
    budget = config_constants.transcript_budget_chars(
        window, response_tokens=config_constants.GI_QUOTE_RESPONSE_TOKENS
    )
    assert budget is not None
    return int((budget / config_constants.CHARS_PER_MINUTE_OF_SPEECH) * 60)


class _Cfg:
    max_episode_seconds = 0
    llm_served_context_tokens = _SERVED_WINDOW


class _CfgNoWindowDeclared:
    """A profile whose StageOption declares nothing — so the window is unknown."""

    max_episode_seconds = 0
    llm_served_context_tokens = 0


def _drop(items, cfg=None):
    return scraping._drop_unprocessably_long(items, cfg or _Cfg())


def test_an_episode_over_the_ceiling_is_dropped() -> None:
    over = _ceiling() + 600
    kept = _drop([_item("short", "1800"), _item("epic", str(over))])
    assert [_title(i) for i in kept] == ["short"]


def test_an_episode_at_the_ceiling_is_kept() -> None:
    """The boundary must not silently shrink the corpus by one episode."""
    at = _ceiling()
    assert len(_drop([_item("boundary", str(at))])) == 1


def test_unknown_duration_is_kept_not_dropped() -> None:
    """Many feeds omit itunes:duration. Refusing them would shrink the corpus for a metadata gap."""
    kept = _drop([_item("no duration", None), _item("junk", "not-a-number")])
    assert len(kept) == 2


def test_hhmmss_and_mmss_are_both_understood() -> None:
    """A feed writing 3:30:00 must not read as 3 seconds and slip through."""
    assert _drop([_item("three and a half hours", "3:30:00")]) == []
    assert len(_drop([_item("fifty minutes", "50:00")])) == 1


def test_nothing_is_skipped_when_no_window_is_declared() -> None:
    """#2050: a guess is not grounds for a permanent editorial decision.

    Before this, an undeclared profile inherited a ceiling derived from the narrowest model in the
    fleet — so a 1M-token deployment silently refused episodes because of a DGX serving flag.
    """
    ten_hours = str(10 * 3600)
    items = [_item("short", "1800"), _item("enormous", ten_hours)]
    kept = _drop(items, _CfgNoWindowDeclared())
    assert [_title(i) for i in kept] == ["short", "enormous"]


def test_the_ceiling_moves_with_the_served_window() -> None:
    """The whole point of #2050: raise the window (#1985) and the ceiling follows, no code edit."""
    assert _ceiling(65_536) > _ceiling(32_768)
    at_64k = _ceiling(65_536)
    kept = _drop(
        [_item("141 minute dwarkesh", str(141 * 60))],
        type("Cfg", (), {"max_episode_seconds": 0, "llm_served_context_tokens": 65_536})(),
    )
    assert (
        len(kept) == 1
    ), f"a 141-minute episode must fit a 64k window (ceiling {at_64k // 60} min)"


def test_the_ceiling_is_near_the_two_hour_operator_rule_at_the_dgx_window() -> None:
    """§5h independently set 2 hours on editorial grounds; the two should not wildly disagree."""
    assert 6000 <= _ceiling() <= 8400


def _title(item: ET.Element) -> str:
    el = item.find("title")
    return (el.text or "") if el is not None else ""
