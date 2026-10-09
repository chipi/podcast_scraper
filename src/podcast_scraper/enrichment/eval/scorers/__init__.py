"""Built-in accuracy scorers + their registration, plus those installed extensions add.

Mirrors ``enrichment.enrichers.__init__`` (the runtime side). The two shipped here cover two of the
metric shapes every other enricher scorer reuses:

* ``grounding_rate``      — scalar / tolerance-band
* ``guest_coappearance``  — set / unordered-pairs precision-recall

Ranking scorers (top-K precision-recall) come with the enrichers they score, from extensions
(ADR-162 decision 5).
"""

from __future__ import annotations

from podcast_scraper.enrichment.eval.registry import ScorerRegistry
from podcast_scraper.enrichment.eval.scorers.grounding_rate import GroundingRateScorer
from podcast_scraper.enrichment.eval.scorers.guest_coappearance import GuestCoappearanceScorer
from podcast_scraper.extensions import enrichment_contributions

BUILTIN_SCORER_ENRICHER_IDS: tuple[str, ...] = (
    "grounding_rate",
    "guest_coappearance",
)


def register_builtin_scorers(registry: ScorerRegistry) -> None:
    """Register every accuracy scorer (one per enricher id): built-in, then extensions'."""
    registry.register(GroundingRateScorer())
    registry.register(GuestCoappearanceScorer())
    for contribution in enrichment_contributions():
        for scorer in contribution.scorers():
            registry.register(scorer)


__all__ = [
    "BUILTIN_SCORER_ENRICHER_IDS",
    "GroundingRateScorer",
    "GuestCoappearanceScorer",
    "register_builtin_scorers",
]
