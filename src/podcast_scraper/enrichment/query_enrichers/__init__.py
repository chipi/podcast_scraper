"""Query enrichers (RFC-088 Phase 4). The platform ships none; installed extensions add them
(ADR-162 decision 5) through :func:`podcast_scraper.extensions.enrichment_contributions`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from podcast_scraper.enrichment.query_registry import QueryEnricherRegistry
from podcast_scraper.extensions import enrichment_contributions


def _extension_query_enrichers(corpus_root_provider: Callable[[], Path]) -> list[Any]:
    return [q for c in enrichment_contributions() for q in c.query_enrichers(corpus_root_provider)]


def all_query_enricher_ids() -> tuple[str, ...]:
    """Ids of every installed query enricher."""
    return tuple(q.manifest.id for q in _extension_query_enrichers(lambda: Path(".")))


def __getattr__(name: str) -> Any:
    if name == "ALL_DETERMINISTIC_QUERY_ENRICHER_IDS":
        return all_query_enricher_ids()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_deterministic_query_enrichers(
    registry: QueryEnricherRegistry, *, corpus_root_provider: Callable[[], Path]
) -> None:
    """Register every installed query enricher on *registry*.

    ``corpus_root_provider`` is a zero-arg callable returning a ``pathlib.Path`` — the search route
    resolves the corpus root per request, so it is wired as a callable rather than frozen at
    registry-construction time.
    """
    for enricher in _extension_query_enrichers(corpus_root_provider):
        registry.register(enricher)


__all__ = [
    "ALL_DETERMINISTIC_QUERY_ENRICHER_IDS",
    "all_query_enricher_ids",
    "register_deterministic_query_enrichers",
]
