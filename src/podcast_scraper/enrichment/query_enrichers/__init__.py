"""Concrete QueryEnricher implementations (RFC-088 Phase 4)."""

from __future__ import annotations

from podcast_scraper.enrichment.query_enrichers.query_topic_relatedness import (
    QueryTopicRelatednessEnricher,
)
from podcast_scraper.enrichment.query_registry import QueryEnricherRegistry

ALL_DETERMINISTIC_QUERY_ENRICHER_IDS: tuple[str, ...] = ("query_topic_relatedness",)


def register_deterministic_query_enrichers(
    registry: QueryEnricherRegistry, *, corpus_root_provider
) -> None:
    """Register every chunk-5 query enricher on *registry*.

    ``corpus_root_provider`` is a zero-arg callable returning a
    ``pathlib.Path`` — the search route already resolves the corpus
    root per request, so we wire it as a callable rather than freeze
    one path at registry-construction time.
    """
    registry.register(QueryTopicRelatednessEnricher(corpus_root_provider=corpus_root_provider))


__all__ = [
    "ALL_DETERMINISTIC_QUERY_ENRICHER_IDS",
    "QueryTopicRelatednessEnricher",
    "register_deterministic_query_enrichers",
]
