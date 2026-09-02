"""#1932 — per-feed connectivity: does this show RETURN to the same topic combinations?

Measured across the 1,066-episode corpus, recurring topic pairs separate shows by FORMAT far more
sharply than by episode count:

    51 recurring pairs / 41 episodes  — Latent Space
    47 / 62  — Invest Like the Best
    38 / 63  — NVIDIA AI Podcast
    ...
     1 / 70  — Planet Money

Technical / thesis-driven interview shows return to a fixed concept vocabulary and compound;
narrative journalism tells a new story each week by design and structurally cannot. That is the
number an operator needs before spending an ingestion budget on depth (#1932).

It is OPERATOR-ONLY on purpose, and one test here enforces that boundary rather than trusting it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.server.feed_signals import _feed_connectivity

pytestmark = pytest.mark.unit


def _topics(mapping: dict[str, list[str]]) -> dict[str, tuple[str, set[str]]]:
    """{topic_id: [episode ids]} -> the shape compute_feed_signals accumulates."""
    return {tid: (tid.split(":")[-1], set(eps)) for tid, eps in mapping.items()}


def test_a_show_that_returns_to_a_pair_scores() -> None:
    """Two topics discussed together in three episodes is one recurring pair."""
    conn = _feed_connectivity(
        _topics({"topic:a": ["e1", "e2", "e3"], "topic:b": ["e1", "e2", "e3"]}),
        scanned=3,
        top_k=8,
    )
    assert conn.recurring_pairs == 1
    assert conn.top_recurring_pairs[0].episode_count == 3


def test_a_show_that_never_repeats_a_pair_scores_zero() -> None:
    """The Planet Money shape: every episode brings new topics, nothing compounds."""
    conn = _feed_connectivity(
        _topics(
            {
                "topic:a": ["e1"],
                "topic:b": ["e1"],
                "topic:c": ["e2"],
                "topic:d": ["e2"],
                "topic:e": ["e3"],
                "topic:f": ["e3"],
            }
        ),
        scanned=3,
        top_k=8,
    )
    assert conn.recurring_pairs == 0, "pairs seen once are not recurrence"
    assert conn.top_recurring_pairs == []


def test_rate_is_what_makes_shows_comparable() -> None:
    """Raw counts scale with episodes scanned; the rate is the cross-show number.

    This is the Latent Space (51/41) vs Planet Money (1/70) comparison in miniature — the show
    with FEWER episodes is the denser one, which the raw count alone would hide.
    """
    dense = _feed_connectivity(
        _topics({"topic:a": ["e1", "e2"], "topic:b": ["e1", "e2"]}), scanned=2, top_k=8
    )
    sparse = _feed_connectivity(
        _topics({"topic:a": ["e1", "e2"], "topic:b": ["e1", "e2"], "topic:c": ["e3"]}),
        scanned=20,
        top_k=8,
    )
    assert dense.recurring_pairs == sparse.recurring_pairs == 1
    assert dense.recurring_pair_rate > sparse.recurring_pair_rate


def test_a_topic_repeating_alone_is_not_a_pair() -> None:
    """Recurrence of one topic is not connectivity — the signal is about COMBINATIONS."""
    conn = _feed_connectivity(
        _topics({"topic:a": ["e1", "e2", "e3"], "topic:b": ["e1"]}), scanned=3, top_k=8
    )
    assert conn.recurring_pairs == 0


def test_top_pairs_are_ranked_and_capped() -> None:
    eps = ["e1", "e2", "e3"]
    mapping = {f"topic:t{i}": eps for i in range(6)}  # every pair recurs 3x
    conn = _feed_connectivity(_topics(mapping), scanned=3, top_k=4)
    assert conn.recurring_pairs == 15  # C(6,2)
    assert len(conn.top_recurring_pairs) == 4, "top_k caps what is returned, not what is counted"
    counts = [p.episode_count for p in conn.top_recurring_pairs]
    assert counts == sorted(counts, reverse=True)


def test_labels_ride_along_so_the_pair_is_readable() -> None:
    """The value is not just a score — it is WHAT the show keeps returning to."""
    conn = _feed_connectivity(
        _topics({"topic:ai-agents": ["e1", "e2"], "topic:ai-regulation": ["e1", "e2"]}),
        scanned=2,
        top_k=8,
    )
    labels = {conn.top_recurring_pairs[0].topic_a_label, conn.top_recurring_pairs[0].topic_b_label}
    assert labels == {"ai-agents", "ai-regulation"}


def test_empty_and_degenerate_inputs_do_not_divide_by_zero() -> None:
    conn = _feed_connectivity({}, scanned=0, top_k=8)
    assert conn.recurring_pairs == 0
    assert conn.recurring_pair_rate == 0.0
    assert conn.episodes_scanned == 0


# --- the boundary --------------------------------------------------------------------------


def test_connectivity_never_reaches_the_consumer_projection() -> None:
    """Enforce the operator-only decision instead of trusting it.

    This measures the CORPUS, not the content: it moves when we deepen a feed, merge label
    variants, or retune a floor. An operator reads that as "we ingested more". A listener would
    read 0.014 as a quality rating on a show that is doing exactly what good narrative journalism
    does — which is why the consumer response omits it, same as the grounding/QA score.
    """
    from podcast_scraper.server.schemas import (
        AppPodcastSignalsResponse,
        CorpusFeedSignalsResponse,
    )

    assert "connectivity" in CorpusFeedSignalsResponse.model_fields
    assert "connectivity" not in AppPodcastSignalsResponse.model_fields, (
        "connectivity is an operator diagnostic; exposing it to listeners turns a corpus "
        "measurement into an apparent quality verdict on the show"
    )
    # The grounding score is the precedent this follows — assert the precedent still holds.
    assert "grounding" in CorpusFeedSignalsResponse.model_fields
    assert "grounding" not in AppPodcastSignalsResponse.model_fields
