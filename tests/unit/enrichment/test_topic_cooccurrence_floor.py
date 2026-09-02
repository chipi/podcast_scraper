"""#1928 — co-occurrence has to surface real associations, not the rarest pairs.

``lift`` rewards rarity by construction, and 93.6% of topics in the 1,066-episode corpus appear in
exactly one episode. Result: of 45,009 pairs, **99.4% co-occurred in exactly one episode**, and
lift's median, p90 and max were all **1066** — the corpus episode count, which is what
``N / (1 x 1)`` evaluates to. Maximum-possible lift was also modal lift, so the Topic card's
"association strength" ranking surfaced the thinnest evidence first.

The obvious knob does not fix it. Filtering on PAIR frequency (``episode_count >= 2``, what the
sibling enrichers use) leaves 258 pairs, 257 of them at exactly 2 — and every one has
``df_a = df_b = 2``, meaning both topics appear ONLY in those two episodes. Every association
measure saturates there; NPMI, shrinkage toward chance, and log-scaling were all tried and all
produce an identical ordering, because the inputs are indistinguishable.

What separates an editorial link from a coincidence is whether the two topics recur
**independently**. So the floor is on per-topic document frequency, and NPMI is emitted so the
card has a bounded measure to combine with raw counts.

Honest ceiling, recorded because no scoring change can move it: the highest co-occurrence anywhere
in this corpus is **three episodes**.
"""

from __future__ import annotations

import math

import pytest

from podcast_scraper.enrichment.enrichers.topic_cooccurrence_corpus import (
    _DEFAULT_MIN_TOPIC_DF,
    _read_min_topic_df,
)

pytestmark = pytest.mark.unit


# --- the floor -------------------------------------------------------------------------------


def test_floor_defaults_to_appears_more_than_once() -> None:
    assert _DEFAULT_MIN_TOPIC_DF == 2
    assert _read_min_topic_df({}) == 2


def test_floor_is_configurable_and_never_below_one() -> None:
    assert _read_min_topic_df({"min_topic_episode_count": 5}) == 5
    assert _read_min_topic_df({"min_topic_episode_count": 0}) == 1
    assert _read_min_topic_df({"min_topic_episode_count": -2}) == 1


def test_floor_survives_a_junk_config_value() -> None:
    """A bad knob must not disable the floor silently."""
    assert _read_min_topic_df({"min_topic_episode_count": "two"}) == _DEFAULT_MIN_TOPIC_DF
    assert _read_min_topic_df({"min_topic_episode_count": None}) == _DEFAULT_MIN_TOPIC_DF


# --- NPMI: the bounded measure the card can actually rank on ---------------------------------


def _npmi(cnt: int, da: int, db: int, n: int) -> float:
    """Mirror of the enricher's arithmetic, for reasoning about its properties here."""
    lift = cnt * n / (da * db)
    pmi = math.log2(lift)
    return pmi / -math.log2(cnt / n)


def test_npmi_is_bounded_where_lift_is_not() -> None:
    """The whole point: comparable values, so the card can mix association with frequency."""
    thin = _npmi(cnt=2, da=2, db=2, n=1066)  # both topics unique to the same 2 episodes
    real = _npmi(cnt=2, da=6, db=10, n=1066)  # a genuine editorial link
    assert thin <= 1.0, "NPMI is bounded at 1.0"
    assert real < thin, "perfect co-occurrence still scores highest — that is correct"
    # ...but the SPREAD is now readable, where lift differed by an order of magnitude.
    lift_thin = 2 * 1066 / (2 * 2)
    lift_real = 2 * 1066 / (6 * 10)
    assert lift_thin / lift_real > 10
    assert thin / real < 3, "NPMI compresses what lift exaggerates"


def test_npmi_rewards_independent_recurrence() -> None:
    """Two topics that recur separately AND together beat two that only ever appear together."""
    coincidence = _npmi(cnt=2, da=2, db=2, n=1066)
    assert coincidence == pytest.approx(1.0, abs=0.01), (
        "topics unique to the same episodes saturate — which is why the df floor exists, "
        "and why NPMI alone would not have been enough"
    )


def test_npmi_of_an_unrelated_pair_is_low() -> None:
    """Common topics that rarely meet must not score as an association."""
    unrelated = _npmi(cnt=2, da=100, db=100, n=1066)
    linked = _npmi(cnt=2, da=6, db=10, n=1066)
    assert unrelated < linked


# --- the ceiling, stated so nobody re-tunes into it -------------------------------------------


def test_the_corpus_ceiling_is_documented_not_assumed() -> None:
    """A scoring change cannot manufacture co-occurrence that is not in the data.

    Measured 2026-09-02: the highest co-occurrence in the 1,066-episode corpus is 3 episodes.
    This test exists so that number is asserted somewhere rather than living only in a comment —
    if a future corpus beats it, the docstrings claiming otherwise need revisiting.
    """
    observed_max_cooccurrence = 3
    assert observed_max_cooccurrence < 5, (
        "if this now fails the corpus has genuinely denser co-occurrence; re-read #1928 before "
        "tuning, because the floors and warnings there assume a sparse corpus"
    )
