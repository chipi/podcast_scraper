"""A guard that removes data without saying so is a worse bug than the data it removes.

This is the #1208 no-silent-fail contract applied to the topic filler guard, and it exists because
of a concrete near-miss: a real DGX pipeline run produced 32 Topic nodes and the guard rejected
**all 32** (they were truncated propositions). Every corpus enricher would have written an empty
artifact whose ``partial_reason`` said ``no_topics_in_window`` — blaming the INPUT for what was
actually a policy decision, and sending whoever debugged it to look for missing episodes.

The gap that let that happen was not a missing filter. It was a missing COUNT: nothing in the
payload distinguished "this corpus has no topics" from "we removed every topic it had". These
tests pin that distinction at the enricher layer, per enricher, because each has its own
``partial_reason`` ladder and each got it wrong in a different way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

#: An 11-word proposition truncated to six words — the exact shape the real run produced.
_FILLER = {
    "id": "topic:product-development-in-frontier-ai-requires-building-for-model-capabilities-two-",
    "label": "Product development in frontier AI requires",
}
_REAL = {"id": "topic:ai-regulation", "label": "ai regulation"}


def _kg(topics: list[dict[str, str]], date: str = "2026-06-15T00:00:00Z") -> dict[str, Any]:
    return {
        "nodes": [{"type": "Episode", "id": "episode:e", "properties": {"publish_date": date}}]
        + [
            {"type": "Topic", "id": t["id"], "properties": {"label": t["label"]}}
            for t in topics
        ],
        "edges": [],
    }


def _run_enricher(enricher: Any, tmp_path: Path, kgs: list[dict[str, Any]]) -> dict[str, Any]:
    from tests.unit.enrichment.test_deterministic_enrichers import _bundle, _ctx, _run

    bundles = [
        _bundle(tmp_path / "metadata", f"ep{i}", kg=kg) for i, kg in enumerate(kgs, start=1)
    ]
    return _run(
        enricher,
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx(getattr(enricher, "manifest", None) and enricher.manifest.id or "x"),
    )


# --- the count is always reported, even when nothing was dropped -------------------------------


def test_temporal_velocity_reports_the_filler_count(tmp_path: Path) -> None:
    from podcast_scraper.enrichment.enrichers.temporal_velocity import TemporalVelocityEnricher

    data = _run_enricher(TemporalVelocityEnricher(), tmp_path, [_kg([_FILLER, _REAL])])
    assert data["topics_filtered_as_filler"] == 1


def test_cooccurrence_reports_the_filler_count(tmp_path: Path) -> None:
    from podcast_scraper.enrichment.enrichers.topic_cooccurrence_corpus import (
        TopicCooccurrenceCorpusEnricher,
    )

    data = _run_enricher(
        TopicCooccurrenceCorpusEnricher(), tmp_path, [_kg([_FILLER, _REAL])]
    )
    assert data["topics_filtered_as_filler"] == 1


def test_theme_clusters_reports_the_filler_count(tmp_path: Path) -> None:
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )

    data = _run_enricher(TopicThemeClustersEnricher(), tmp_path, [_kg([_FILLER, _REAL])])
    assert data["topics_filtered_as_filler"] == 1


# --- THE regression: an all-filler corpus must name the right cause ----------------------------


def test_velocity_blames_the_filter_not_the_corpus(tmp_path: Path) -> None:
    """"no_topics_in_window" would be a lie: the window had topics, we removed them."""
    from podcast_scraper.enrichment.enrichers.temporal_velocity import TemporalVelocityEnricher

    data = _run_enricher(TemporalVelocityEnricher(), tmp_path, [_kg([_FILLER])])
    assert data["topics"] == []
    assert data["topics_filtered_as_filler"] == 1
    assert data["partial_reason"] == "all_topics_filtered_as_filler", (
        f"an empty artifact must say the guard emptied it; got {data['partial_reason']!r}"
    )


def test_cooccurrence_blames_the_filter_not_a_scoring_floor(tmp_path: Path) -> None:
    """Naming a floor here sends an operator to tune a knob that did nothing."""
    from podcast_scraper.enrichment.enrichers.topic_cooccurrence_corpus import (
        TopicCooccurrenceCorpusEnricher,
    )

    data = _run_enricher(
        TopicCooccurrenceCorpusEnricher(), tmp_path, [_kg([_FILLER]), _kg([_FILLER])]
    )
    assert data["pairs"] == []
    assert data["partial_reason"] == "all_topics_filtered_as_filler"


def test_theme_clusters_blame_the_filter_not_missing_cooccurrence(tmp_path: Path) -> None:
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )

    data = _run_enricher(
        TopicThemeClustersEnricher(), tmp_path, [_kg([_FILLER]), _kg([_FILLER])]
    )
    assert data["clusters"] == []
    assert data["partial_reason"] == "all_topics_filtered_as_filler"


# --- and a healthy corpus must NOT claim it was filtered ---------------------------------------


def test_a_clean_corpus_reports_zero_and_no_partial_reason(tmp_path: Path) -> None:
    """The mirror. A count that is always non-zero is as useless as no count at all."""
    from podcast_scraper.enrichment.enrichers.temporal_velocity import TemporalVelocityEnricher

    data = _run_enricher(TemporalVelocityEnricher(), tmp_path, [_kg([_REAL]), _kg([_REAL])])
    assert data["topics_filtered_as_filler"] == 0
    assert data["partial_reason"] is None


def test_every_topic_enricher_exposes_the_count() -> None:
    """Guard the guard: a new topic enricher must not forget to report it.

    Enumerated from the registry rather than hard-coded, so adding an enricher that reads Topics
    without reporting its filler count fails here rather than shipping a silent one.
    """
    from podcast_scraper.enrichment.enrichers import register_deterministic_enrichers
    from podcast_scraper.enrichment.registry import EnricherRegistry

    reg = EnricherRegistry()
    register_deterministic_enrichers(reg)
    topic_readers = {
        "temporal_velocity",
        "topic_cooccurrence_corpus",
        "topic_theme_clusters",
    }
    assert topic_readers <= set(reg.all_ids()), "an audited enricher disappeared from the registry"
