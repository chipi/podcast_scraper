"""The private enrichers as an extension (ADR-158 decision 5): storyline themes, temporal velocity,
topic similarity and consensus, and the web person/org enrichers, with their query enricher, scorer
and provider types; and the themes and storylines built from them, with the search operators over
them; and the trend stat, photos and logos the public share cards show.

Moves to the private Common package (``intelligence``) at the cutover. Every import stays inside a
function: extensions load wherever the enrichment code builds a registry, and nothing here should
cost anything until it is asked for.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

from podcast_scraper.extensions import EnrichmentContribution, Extension, ShareCardContribution
from podcast_scraper.search.groupings import SearchOperator, TopicGroupings


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


def _lazy(module: str, name: str) -> Callable[..., Any]:
    """*module*.*name*, imported on first call."""

    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(importlib.import_module(module), name)(*args, **kwargs)

    call.__name__ = name
    return call


_THEMES = "podcast_scraper.search.topic_clusters"
_STORYLINES = "podcast_scraper.search.storylines"


def _search_operators() -> dict[str, SearchOperator]:
    from podcast_scraper.search.operators import cluster_hits, consensus_pairs_for_hits

    return {"cluster": cluster_hits, "consensus": consensus_pairs_for_hits}


GROUPINGS = TopicGroupings(
    theme_map_by_topic=_lazy(_THEMES, "theme_map_by_topic"),
    theme_siblings_by_topic=_lazy(_THEMES, "theme_siblings_by_topic"),
    top_themes_by_member_count=_lazy(_THEMES, "top_themes_by_member_count"),
    load_theme_enrichment_map=_lazy(_THEMES, "load_theme_enrichment_map"),
    load_theme_payload=_lazy(_THEMES, "_load_theme_payload"),
    build_topic_clusters_for_corpus=_lazy(_THEMES, "build_topic_clusters_for_corpus"),
    storyline_map_by_topic=_lazy(_STORYLINES, "storyline_map_by_topic"),
    storyline_siblings_by_topic=_lazy(_STORYLINES, "storyline_siblings_by_topic"),
    top_storylines_by_member_count=_lazy(_STORYLINES, "top_storylines_by_member_count"),
    storyline_member_lift=_lazy(_STORYLINES, "storyline_member_lift"),
    storyline_episode_ids=_lazy(_STORYLINES, "storyline_episode_ids"),
    storyline_index_rows=_lazy(_STORYLINES, "storyline_index_rows"),
    search_operators=_search_operators,
)


SHARE_CARDS = ShareCardContribution(
    trends=_lazy("podcast_scraper.server.app_momentum", "share_card_trends"),
    person_image_path=_lazy("podcast_scraper.enrichment.enrichers.person_web", "person_image_path"),
    org_logo_path=_lazy("podcast_scraper.enrichment.enrichers.org_web", "org_logo_path"),
)


EXTENSION = Extension(
    name="intelligence",
    groupings=GROUPINGS,
    share_cards=SHARE_CARDS,
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
