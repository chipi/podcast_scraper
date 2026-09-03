"""#1930/#1931 — temporal_velocity has to actually power discovery, not just run cleanly.

Three defects, measured on the 1,066-episode corpus:

* **The scalar was four-valued.** ``velocity_last_over_6mo`` scored 0.0 for 9,335 of 9,345 topics,
  6.0 for seven, and took two other values once each. A topic mentioned ONCE in the last month
  scored the maximum (1 / (1/6) = 6.0) and outranked ``monetary policy`` with sixteen mentions.
  #1650 measured the same collapse at 678 episodes; it got worse, not better, with more corpus.
* **Shrinkage alone could not fix the ranking.** Pulling sparse estimates toward flat makes the
  value honest, but every *sustained* topic sits at raw ≈ 1.0, so single-mention spikes still
  outrank them at every prior from 3 to 10. A ratio is the wrong shape for "what is hot".
* **94% of the artifact was zeros.** 8,824 of 9,345 topics had <= 1 non-zero month across the whole
  24-month window — a 128-bucket flat line each, 65 MB total, 2.2s of JSON parse on the first read
  after every ingest so a rail could render twelve rows.

So: the floor removes the topics that cannot carry a trend, the shrinkage makes the retained
scalar honest, and ``trend_score`` — recency-decayed volume scaled by weekly spread — is the field
that actually ranks. These tests pin all three.
"""

from __future__ import annotations

import math

import pytest

from podcast_scraper.enrichment.enrichers.temporal_velocity import (
    _DEFAULT_MIN_TOTAL_MENTIONS,
    _read_min_total,
    _TREND_DECAY_WEEKS,
    _trend_score,
    _velocity,
    _VELOCITY_PRIOR_MENTIONS,
)

pytestmark = pytest.mark.unit


# --- the scalar: honest, not degenerate ------------------------------------------------------


def test_a_single_recent_mention_no_longer_scores_the_maximum() -> None:
    """The headline defect: one mention scored 6.0, the top of the corpus."""
    one_mention = [0, 0, 0, 0, 0, 1]
    raw = 1 / (sum(one_mention) / 6)
    assert raw == 6.0, "sanity: the unshrunk ratio really is the maximum"
    assert _velocity(one_mention) < 3.0, "a single observation must not score near the top"


def test_evidence_earns_back_the_movement() -> None:
    """Shrinkage must be proportional, not a flat penalty — a busy topic keeps its signal."""
    thin = _velocity([0, 0, 0, 0, 0, 1])  # 1 mention
    thick = _velocity([1, 1, 1, 1, 1, 3])  # 8 mentions, same direction
    assert 1.0 < thick, "a genuine rise must still read as rising"
    # Both rise, but the well-evidenced one retains a larger share of its raw movement.
    raw_thin, raw_thick = 6.0, 3 / (8 / 6)
    assert (thick - 1.0) / (raw_thick - 1.0) > (thin - 1.0) / (raw_thin - 1.0)


def test_flat_stays_flat() -> None:
    """1.0 must keep meaning 'flat' — consumers and thresholds depend on it."""
    assert _velocity([2, 2, 2, 2, 2, 2]) == pytest.approx(1.0, abs=0.01)


def test_cooling_still_reads_as_cooling() -> None:
    assert _velocity([3, 3, 3, 3, 3, 1]) < 1.0


def test_empty_history_is_zero_not_a_spike() -> None:
    assert _velocity([0, 0, 0, 0, 0, 0]) == 0.0
    assert _velocity([]) == 0.0


def test_prior_is_a_real_pseudo_count() -> None:
    assert _VELOCITY_PRIOR_MENTIONS > 0


# --- trend_score: the field that ranks -------------------------------------------------------


def _weeks(n: int = 20) -> list[str]:
    return [f"2026-W{i:02d}" for i in range(1, n + 1)]


def test_sustained_presence_beats_a_single_recent_mention() -> None:
    """THE regression. This is the ordering the old scalar got backwards."""
    weeks = _weeks()
    sustained = _trend_score({w: 1 for w in weeks[-8:]}, weeks)
    one_off = _trend_score({weeks[-1]: 1}, weeks)
    assert sustained > one_off


def test_spread_beats_a_single_big_week() -> None:
    """Eight mentions across eight weeks is a running story; eight in one week is an episode."""
    weeks = _weeks()
    spread = _trend_score({w: 1 for w in weeks[-8:]}, weeks)
    burst = _trend_score({weeks[-1]: 8}, weeks)
    assert spread > burst


def test_recent_beats_old_at_equal_volume() -> None:
    weeks = _weeks()
    recent = _trend_score({w: 1 for w in weeks[-5:]}, weeks)
    old = _trend_score({w: 1 for w in weeks[:5]}, weeks)
    assert recent > old


def test_more_of_the_same_ranks_higher() -> None:
    weeks = _weeks()
    assert _trend_score({w: 2 for w in weeks[-6:]}, weeks) > _trend_score(
        {w: 1 for w in weeks[-6:]}, weeks
    )


def test_nothing_in_the_window_scores_zero() -> None:
    """Safe to sort on directly — no None, no negative, no exception."""
    weeks = _weeks()
    assert _trend_score({}, weeks) == 0.0
    assert _trend_score({w: 0 for w in weeks}, weeks) == 0.0
    assert _trend_score({"2019-W01": 5}, weeks) == 0.0, "out-of-window weeks must not count"
    assert _trend_score({"2026-W01": 5}, []) == 0.0


def test_decay_horizon_is_sane() -> None:
    """A topic discussed monthly must still register — the decay is a fade, not a cliff."""
    assert _TREND_DECAY_WEEKS >= 4
    monthly = math.exp(-4 / _TREND_DECAY_WEEKS)
    assert monthly > 0.5, "a mention four weeks ago should keep most of its weight"


# --- the floor: artifact size and honest reporting -------------------------------------------


def test_floor_defaults_to_the_sibling_enrichers_guard() -> None:
    """2 mirrors topic_theme_clusters / guest_coappearance — the two that produce usable output."""
    assert _DEFAULT_MIN_TOTAL_MENTIONS == 2
    assert _read_min_total({}) == 2


def test_floor_is_configurable_and_never_below_one() -> None:
    assert _read_min_total({"min_total_mentions": 5}) == 5
    assert _read_min_total({"min_total_mentions": 0}) == 1
    assert _read_min_total({"min_total_mentions": -3}) == 1


def test_floor_survives_a_junk_config_value() -> None:
    """A bad knob must not take the enricher down or silently disable the floor."""
    assert _read_min_total({"min_total_mentions": "lots"}) == _DEFAULT_MIN_TOTAL_MENTIONS
    assert _read_min_total({"min_total_mentions": None}) == _DEFAULT_MIN_TOTAL_MENTIONS


# --- the axis it is scored against (found by local validation, 2026-09-03) --------------------


def _axis(n: int, start_year: int = 2024) -> list[str]:
    """A contiguous ISO-week axis n entries long."""
    out: list[str] = []
    y, w = start_year, 1
    for _ in range(n):
        out.append(f"{y}-W{w:02d}")
        w += 1
        if w > 52:
            y, w = y + 1, 1
    return out


def test_evidence_off_the_axis_is_silently_deleted_not_decayed() -> None:
    """THE regression. A short axis is a cliff, not a decay.

    ``_trend_score`` skips any week it cannot find on the axis it is handed, so feeding it the
    26-week now-anchored window discards older mentions entirely rather than discounting them.
    That is the decay's job, and doing it twice — once smoothly, once discontinuously — is what
    made the rail degenerate.

    Measured on the 36-episode validation corpus before the fix: ``topic:expert-interviews`` had
    36 mentions spanning 2024-W01..2026-W29, of which the 26-week window held exactly ONE. All
    four surviving topics scored an identical 0.3868 and the rail's order was a four-way tie.
    """
    full = _axis(133)
    short = full[-26:]
    # Sustained: one mention every fourth week across the whole history.
    sustained = {w: 1 for w in full[::4]}

    on_full = _trend_score(sustained, full)
    on_short = _trend_score(sustained, short)
    assert on_full > on_short, "the long axis must see evidence the short one truncates"

    # A single recent mention — the thing that should NOT beat a sustained topic.
    one_off = _trend_score({full[-1]: 1}, full)
    assert on_full > one_off, (
        "a topic discussed across two years lost to a single recent mention — this is the "
        "degeneracy #1931 was supposed to remove"
    )


def test_history_outside_the_short_window_still_separates_topics() -> None:
    """Two topics identical inside 26 weeks, different across history, must not tie."""
    full = _axis(133)
    recent_only = {full[-1]: 1}
    deep = {full[-1]: 1, **{w: 1 for w in full[:60:5]}}

    short = full[-26:]
    assert _trend_score(recent_only, short) == _trend_score(deep, short), (
        "precondition: the short window genuinely cannot tell these apart"
    )
    assert _trend_score(deep, full) > _trend_score(recent_only, full)


def test_identical_histories_still_tie() -> None:
    """Not every tie is a bug — two topics with the same evidence SHOULD score the same.

    The validation corpus has exactly this: `expert-interviews` and `lifelong-learning` both
    appear in all 36 episodes with byte-identical weekly histories.
    """
    full = _axis(133)
    a = {w: 1 for w in full[::3]}
    b = dict(a)
    assert _trend_score(a, full) == _trend_score(b, full)


def test_an_empty_axis_is_still_safe_to_sort_on() -> None:
    assert _trend_score({"2024-W01": 5}, []) == 0.0
    assert _trend_score({}, _axis(10)) == 0.0
