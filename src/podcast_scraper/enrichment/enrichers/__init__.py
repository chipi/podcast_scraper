"""Deterministic enricher zoo (RFC-088 chunk 2).

Seven tier=DETERMINISTIC enrichers covering Topic, Person, Insight, and
temporal signals. None require external models or networks — they read
the core artifacts (``*.kg.json`` + ``*.gi.json`` + ``*.bridge.json`` +
``*.metadata.json``) and write structured JSON envelopes under
``enrichments/`` (corpus-scope) or
``metadata/enrichments/{stem}.<writes>`` (episode-scope).

All seven wrap a sync body with :func:`podcast_scraper.enrichment.protocol.sync_enricher`,
so the executor's async machinery flows uninterrupted without enricher
authors paying the async ceremony tax.
"""

from __future__ import annotations

from podcast_scraper.enrichment.enrichers.grounding_rate import GroundingRateEnricher
from podcast_scraper.enrichment.enrichers.guest_coappearance import GuestCoappearanceEnricher
from podcast_scraper.enrichment.enrichers.insight_density import InsightDensityEnricher
from podcast_scraper.enrichment.enrichers.nli_contradiction import NliContradictionEnricher
from podcast_scraper.enrichment.enrichers.temporal_velocity import TemporalVelocityEnricher
from podcast_scraper.enrichment.enrichers.topic_cooccurrence import TopicCooccurrenceEnricher
from podcast_scraper.enrichment.enrichers.topic_cooccurrence_corpus import (
    TopicCooccurrenceCorpusEnricher,
)
from podcast_scraper.enrichment.enrichers.topic_similarity import TopicSimilarityEnricher
from podcast_scraper.enrichment.enrichers.topic_theme_clusters import TopicThemeClustersEnricher
from podcast_scraper.enrichment.registry import EnricherRegistry

ALL_DETERMINISTIC_ENRICHER_IDS: tuple[str, ...] = (
    "topic_cooccurrence",
    "topic_cooccurrence_corpus",
    "topic_theme_clusters",
    "temporal_velocity",
    "grounding_rate",
    "guest_coappearance",
    "insight_density",
)


def register_deterministic_enrichers(registry: EnricherRegistry) -> None:
    """Register all seven deterministic enrichers on *registry*.

    Idempotent in the sense that the registry's ``register()`` raises
    on duplicate ids — call once per registry instance.
    """
    registry.register(TopicCooccurrenceEnricher())
    registry.register(TopicCooccurrenceCorpusEnricher())
    registry.register(TopicThemeClustersEnricher())
    registry.register(TemporalVelocityEnricher())
    registry.register(GroundingRateEnricher())
    registry.register(GuestCoappearanceEnricher())
    registry.register(InsightDensityEnricher())


__all__ = [
    "ALL_DETERMINISTIC_ENRICHER_IDS",
    "GroundingRateEnricher",
    "GuestCoappearanceEnricher",
    "InsightDensityEnricher",
    "NliContradictionEnricher",
    "TemporalVelocityEnricher",
    "TopicCooccurrenceCorpusEnricher",
    "TopicCooccurrenceEnricher",
    "TopicSimilarityEnricher",
    "TopicThemeClustersEnricher",
    "register_deterministic_enrichers",
]
