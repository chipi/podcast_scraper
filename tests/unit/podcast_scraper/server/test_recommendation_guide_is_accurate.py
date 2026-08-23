"""The recommendation guide's numbers must match the shipped code (#69).

`docs/guides/RECOMMENDATION_GUIDE.md` exists so that somebody with no prior knowledge can read one
document and understand how "what should I listen to next" is decided. A tuning table that has
drifted from the code is worse than no table: it reads as authoritative and teaches the wrong
thing, and nobody re-derives a number they have just been told.

So the guide's tuning table is parsed and checked against `DEFAULT_RANKING_CONFIG`, and its ladder
and half-life claims against their constants. Retune anything and this fails until the guide is
updated — which is the point.

Deliberately NOT asserted: the prose. Its accuracy is a review question, not a test question. What
is pinned here is every number a reader might copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from podcast_scraper.server.app_discover_view import _affinity_boost
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    SIGNAL_INTEREST_AFFINITY,
    SIGNAL_RECENCY,
    SIGNAL_SIGNIFICANCE,
    SIGNAL_TREND_VELOCITY,
)
from podcast_scraper.server.app_resurfacing import DAY, LADDER_SECONDS
from podcast_scraper.server.app_user_corpus import (
    DERIVED_HALF_LIFE_DAYS,
    DERIVED_MAX_EPISODES,
    DERIVED_TOP_K,
)

pytestmark = [pytest.mark.unit]

_GUIDE = Path(__file__).resolve().parents[4] / "docs" / "guides" / "RECOMMENDATION_GUIDE.md"


@pytest.fixture(scope="module")
def guide() -> str:
    assert _GUIDE.is_file(), f"the guide is missing: {_GUIDE}"
    return _GUIDE.read_text(encoding="utf-8")


def test_the_guide_exists_and_is_not_a_stub(guide: str) -> None:
    assert len(guide) > 4000, "a stub guide is worse than none — it looks like coverage"
    for heading in ("## The formula", "## What each signal is for", "## Current tuning"):
        assert heading in guide, heading


def test_the_tuning_table_matches_the_shipped_config(guide: str) -> None:
    """Parsed from the table, not hand-restated — restating it here would just move the drift."""
    rows = dict(re.findall(r"^\| `([a-z_]+)` \| (?:yes|\*\*no\*\*) \| ([0-9.]+) \|", guide, re.M))
    assert set(rows) == {
        SIGNAL_SIGNIFICANCE,
        SIGNAL_INTEREST_AFFINITY,
        SIGNAL_RECENCY,
        SIGNAL_TREND_VELOCITY,
    }, rows
    # Only the SCORING signals carry a weight. `discover_pool` is the admission policy — its
    # weight is unused, so comparing it would assert on a number that means nothing. Its
    # parameters are checked by the pool tests instead.
    for signal in DEFAULT_RANKING_CONFIG.signals:
        if signal.name not in rows:
            continue
        assert float(rows[signal.name]) == signal.weight, (
            f"the guide says {signal.name} weight is {rows[signal.name]}, "
            f"the code ships {signal.weight}"
        )


def test_the_guide_states_the_shipped_parameters(guide: str) -> None:
    affinity = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_INTEREST_AFFINITY)
    recency = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_RECENCY)
    significance = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_SIGNIFICANCE)
    for claim in (
        f"`derived_ratio {affinity['derived_ratio']}`",
        f"`half_life_days {int(recency['half_life_days'])}`",
        f"`gi_bonus {significance['gi_bonus']}`",
        f"`kg_bonus {significance['kg_bonus']}`",
    ):
        assert claim in guide, f"the guide does not state {claim}"


def test_the_guide_states_the_real_ladder(guide: str) -> None:
    """Parsed from the sentence, so the guide keeps its own phrasing.

    The first version demanded an exact string and failed on "2 days → 7 → 30 → 90" — better prose
    than the "2 → 7 → 30 → 90" it insisted on. A test that dictates wording rather than facts makes
    the document worse to read, which is the opposite of the job.
    """
    line = next(ln for ln in guide.splitlines() if "come back on a ladder" in ln)
    stated = [int(n) for n in re.findall(r"\d+", line)]
    assert stated == [s // DAY for s in LADDER_SECONDS], (stated, LADDER_SECONDS)


def test_the_guide_states_the_real_derived_interest_constants(guide: str) -> None:
    assert f"**{int(DERIVED_HALF_LIFE_DAYS)}-day half-life**" in guide
    assert f"top-{DERIVED_TOP_K}" in guide
    assert str(DERIVED_MAX_EPISODES) in guide  # the episode bound, quoted in the freeze story


def test_the_saturation_numbers_in_the_guide_are_real(guide: str) -> None:
    """The guide teaches the curve with three worked values; they must be the curve's values."""
    for matches, stated in ((1, "0.5"), (2, "0.75"), (3, "0.875")):
        actual = _affinity_boost(matches, 0, weight=1.0, derived_ratio=0.5, cap=1.0)
        assert round(actual, 6) == float(stated), (matches, actual, stated)
        assert f"{matches} → {stated}" in guide or f"{matches} match" in guide


def test_one_matched_interest_is_still_the_x2_the_guide_promises(guide: str) -> None:
    """The guide explains the 2.0 → 4.0 weight change as behaviour-preserving. If that stops being
    true the explanation becomes a lie, so it is checked rather than trusted."""
    weight = DEFAULT_RANKING_CONFIG.weight_of(SIGNAL_INTEREST_AFFINITY)
    params = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_INTEREST_AFFINITY)
    boost = _affinity_boost(
        1,
        0,
        weight=weight,
        derived_ratio=float(params["derived_ratio"]),
        cap=float(params["cap"]),
    )
    assert round(boost, 6) == 2.0
    assert "×2 boost" in guide
