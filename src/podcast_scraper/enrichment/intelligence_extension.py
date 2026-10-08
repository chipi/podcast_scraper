"""The private enrichers as an extension (ADR-158 decision 5): storyline themes, temporal velocity,
topic similarity and consensus, and the web person/org enrichers, with their query enricher, scorer
and provider types.

Moves to the private Common package (``intelligence``) at the cutover. Every import stays inside a
function: extensions load wherever the enrichment code builds a registry, and nothing here should
cost anything until it is asked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from podcast_scraper.extensions import EnrichmentContribution, Extension


def _classes() -> Sequence[type]:
    from podcast_scraper.enrichment.enrichers.org_web import OrgWebEnricher
    from podcast_scraper.enrichment.enrichers.person_web import PersonWebEnricher
    from podcast_scraper.enrichment.enrichers.temporal_velocity import TemporalVelocityEnricher
    from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher
    from podcast_scraper.enrichment.enrichers.topic_similarity import TopicSimilarityEnricher
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )

    return (
        TopicThemeClustersEnricher,
        TemporalVelocityEnricher,
        TopicSimilarityEnricher,
        TopicConsensusEnricher,
        PersonWebEnricher,
        OrgWebEnricher,
    )


def _deterministic() -> Sequence[Any]:
    from podcast_scraper.enrichment.enrichers.temporal_velocity import TemporalVelocityEnricher
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )

    return (TopicThemeClustersEnricher(), TemporalVelocityEnricher())


def _ml_wiring(registry: Any, enricher_set: Any) -> None:
    from podcast_scraper.enrichment.ml_wiring import register_ml_enrichers

    register_ml_enrichers(registry, enricher_set)


def _web() -> Sequence[Any]:
    from podcast_scraper.enrichment.enrichers.org_web import OrgWebEnricher
    from podcast_scraper.enrichment.enrichers.person_web import PersonWebEnricher

    return (PersonWebEnricher(), OrgWebEnricher())


def _query_enrichers(corpus_root_provider: Callable[[], Path]) -> Sequence[Any]:
    from podcast_scraper.enrichment.query_enrichers.query_topic_relatedness import (
        QueryTopicRelatednessEnricher,
    )

    return (QueryTopicRelatednessEnricher(corpus_root_provider=corpus_root_provider),)


def _scorers() -> Sequence[Any]:
    from podcast_scraper.enrichment.eval.scorers.topic_similarity import TopicSimilarityScorer

    return (TopicSimilarityScorer(),)


def _provider_types() -> None:
    from podcast_scraper.enrichment.provider_types import consensus  # noqa: F401 — registers


EXTENSION = Extension(
    name="intelligence",
    enrichment=EnrichmentContribution(
        enricher_classes=_classes,
        deterministic=_deterministic,
        ml_wiring=_ml_wiring,
        web=_web,
        query_enrichers=_query_enrichers,
        scorers=_scorers,
        provider_types=_provider_types,
    ),
)
