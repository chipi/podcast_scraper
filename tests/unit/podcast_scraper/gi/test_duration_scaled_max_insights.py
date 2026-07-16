"""Duration-scaled insight ceiling (#1191 interim).

A flat 50-insight cap truncated long episodes (a corpus-baked cutoff, see
docs/wip/GI_WHAT_TO_SURFACE.md) while a high flat cap invited padding on short ones. The ceiling now
scales +25 per 30-min unit up to 200 at 4h, floored at the configured gi_max_insights. The value
gate — not this ceiling — removes filler; the full route-and-tag redesign is #1191.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config_constants as cc

pytestmark = pytest.mark.unit

_CPH = cc.GI_CHARS_PER_HOUR


@pytest.mark.parametrize(
    "hours,expected",
    [
        (0.25, 50),  # 15 min -> 1 unit -> floored to 50
        (0.5, 50),  # 30 min -> 1 unit -> 50 (floor)
        (1.0, 50),  # 1h  -> 2 units -> 50 (no invitation to pad a short episode)
        (1.5, 75),  # 3 units
        (2.0, 100),  # 4 units  (fixes "thin at 2h")
        (3.0, 150),  # 6 units
        (4.0, 200),  # 8 units  (hits the ceiling)
        (6.0, 200),  # beyond 4h stays at the ceiling, never unbounded
    ],
)
def test_scales_in_30min_units_up_to_4h(hours: float, expected: int) -> None:
    chars = int(hours * _CPH)
    assert (
        cc.duration_scaled_max_insights(chars, base=cc.GI_DEFAULT_MAX_INSIGHTS) == expected
    ), f"{hours}h transcript should cap at {expected} insights"


def test_never_below_the_configured_floor() -> None:
    # A raised gi_max_insights floor lifts the short-episode ceiling.
    assert cc.duration_scaled_max_insights(1000, base=80) == 80


def test_never_exceeds_the_hard_ceiling() -> None:
    assert cc.duration_scaled_max_insights(10_000_000, base=50) == cc.GI_MAX_INSIGHTS_CEILING
    assert cc.GI_MAX_INSIGHTS_CEILING == 200


def test_empty_transcript_floors_not_crashes() -> None:
    assert cc.duration_scaled_max_insights(0, base=50) == 50
