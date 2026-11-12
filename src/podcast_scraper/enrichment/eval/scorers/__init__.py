"""Accuracy scorers: their registration, from the installed extensions.

Mirrors ``enrichment.enrichers.__init__`` (the runtime side). Each scorer comes with the enricher it
scores, from the extension that owns it (ADR-168 decision 5). The two public shipped (grounding
rate, scalar / tolerance-band; guest co-appearance, set / unordered-pairs precision-recall) left
with their enrichers and GI (E04 move); public ships none of its own.
"""

from __future__ import annotations

from podcast_scraper.enrichment.eval.registry import ScorerRegistry
from podcast_scraper.extensions import enrichment_contributions

BUILTIN_SCORER_ENRICHER_IDS: tuple[str, ...] = ()


def register_builtin_scorers(registry: ScorerRegistry) -> None:
    """Register every accuracy scorer (one per enricher id): the installed extensions'."""
    for contribution in enrichment_contributions():
        for scorer in contribution.scorers():
            registry.register(scorer)


__all__ = [
    "BUILTIN_SCORER_ENRICHER_IDS",
    "register_builtin_scorers",
]
