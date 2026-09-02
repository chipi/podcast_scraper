"""#1929 — the linkage cap must not fail silently.

``_average_linkage`` refuses to run above ``_MAX_LINKAGE_TOPICS`` (400) and returns all-singletons
instead. That is a sound cost decision — the merge loop is worst-case O(n^4) in a sync thread that
cannot be cancelled — but every singleton is dropped downstream (``len(members) < 2``), so the
enricher emitted **zero themes** while reporting ``status: ok``, ``partial_reason: None``, no
failed or timeout counters, and nothing in the events stream. An operator, a dashboard and
``GET /api/enrichment/health`` all saw a clean run while the operator's top-down navigation surface
went blank.

Themes power that surface (``graphTopDown.ts``) and the player's Storylines, so a silent empty
removes a capability rather than a decoration.

The cap is also easy to trip by accident, because ``min_pair_episode_count`` reads like a recall
knob and is not one: only topics touching a pair seen in ``>= min_pair`` episodes enter the
linkage, so LOWERING it admits MORE topics. Measured on the 1,066-episode corpus — min_pair=1
puts 9,344 topics in the linkage against a cap of 400 (zero themes); min_pair=2 puts 192 there
(54 themes).

These tests pin that the degradation is now observable in the payload, and that the three very
different causes of "no themes" are distinguishable.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
    _average_linkage,
    _MAX_LINKAGE_TOPICS,
)

pytestmark = pytest.mark.unit


def test_cap_still_degrades_rather_than_burning_a_core() -> None:
    """The behaviour itself is intentional and must not change — only its visibility."""

    def weight(i: int, j: int) -> float:
        return 100.0  # every pair wildly above any threshold

    n = _MAX_LINKAGE_TOPICS + 1
    clusters = _average_linkage(n, weight, 2.0)
    assert len(clusters) == n
    assert all(len(c) == 1 for c in clusters), "past the cap, all-singletons is the contract"


def test_cap_logs_a_warning_naming_the_numbers(caplog: pytest.LogCaptureFixture) -> None:
    """Silence was the bug. The log must say how many topics, and what the cap is."""

    def weight(i: int, j: int) -> float:
        return 100.0

    with caplog.at_level(logging.WARNING):
        _average_linkage(_MAX_LINKAGE_TOPICS + 1, weight, 2.0)

    msgs = [r.getMessage() for r in caplog.records]
    assert msgs, "the cap fired with no log record at all — this is the #1929 defect"
    joined = " ".join(msgs)
    assert str(_MAX_LINKAGE_TOPICS) in joined
    assert str(_MAX_LINKAGE_TOPICS + 1) in joined
    assert "ZERO themes" in joined


def test_the_warning_corrects_the_min_pair_intuition(caplog: pytest.LogCaptureFixture) -> None:
    """Whoever reads this log will reach for min_pair. It must tell them which way to turn."""

    def weight(i: int, j: int) -> float:
        return 100.0

    with caplog.at_level(logging.WARNING):
        _average_linkage(_MAX_LINKAGE_TOPICS + 1, weight, 2.0)

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "min_pair_episode_count" in joined
    assert "worse" in joined, "a LOWER min_pair admits more topics — the log must say so"


def test_below_the_cap_nothing_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A normal run must stay quiet, or the warning stops meaning anything."""

    def weight(i: int, j: int) -> float:
        return 100.0

    with caplog.at_level(logging.WARNING):
        clusters = _average_linkage(10, weight, 2.0)

    assert any(len(c) > 1 for c in clusters), "below the cap the linkage should actually merge"
    assert not [
        r for r in caplog.records if "linkage cap" in r.getMessage()
    ], "the cap warning fired on a healthy run"


# --- the payload half: "no themes" has three causes and they need different responses ---------


def _kg(topic_ids: list[str]) -> dict:
    return {
        "nodes": [{"type": "Topic", "id": t, "properties": {"label": t}} for t in topic_ids],
        "edges": [],
    }


def test_payload_reports_linkage_counters_on_a_healthy_run(tmp_path) -> None:
    """The counters are always present, so an empty surface is never ambiguous."""
    from tests.unit.enrichment.test_deterministic_enrichers import _bundle, _ctx, _run
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )

    bundles = [
        _bundle(tmp_path / "metadata", f"ep-{i}", kg=_kg(["topic:a", "topic:b"])) for i in range(3)
    ]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["linkage_topic_cap"] == _MAX_LINKAGE_TOPICS
    assert data["linkage_skipped"] is False
    assert "partial_reason" in data
    assert "linkage_topic_count" in data


def test_no_cooccurring_topics_is_distinguishable_from_a_cap_skip(tmp_path) -> None:
    """A corpus with nothing to cluster must not look like a capability failure."""
    from tests.unit.enrichment.test_deterministic_enrichers import _bundle, _ctx, _run
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )

    # One episode, so no topic pair can reach min_pair_episode_count=2.
    bundles = [_bundle(tmp_path / "metadata", "ep-1", kg=_kg(["topic:a", "topic:b"]))]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["cluster_count"] == 0
    assert data["partial_reason"] == "no_cooccurring_topics"
    assert data["linkage_skipped"] is False, (
        "an empty corpus must not be reported as a linkage-cap failure"
    )
