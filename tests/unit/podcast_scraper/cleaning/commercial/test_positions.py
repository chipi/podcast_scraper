"""Unit tests for commercial positional heuristics (B8)."""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning.commercial.positions import (
    POSITION_BOOST,
    position_score,
    POSITION_WINDOWS,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("window", list(POSITION_WINDOWS))
def test_position_score_boosts_inside_each_window(window: str) -> None:
    start, end = POSITION_WINDOWS[window]
    mid_ratio = (start + end) / 2
    length = 1000
    assert position_score(int(mid_ratio * length), length) == POSITION_BOOST


def test_position_score_neutral_in_conversation_zone() -> None:
    # 25% is between pre_roll (<=0.15) and mid_roll (>=0.35) -> no boost, no penalty.
    assert position_score(250, 1000) == 0.0


def test_position_score_handles_empty_text() -> None:
    assert position_score(0, 0) == 0.0
