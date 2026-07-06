"""Built-in accuracy scorers + their registration.

Mirrors ``enrichment.enrichers.__init__`` (the runtime side): import the scorer
classes, expose the id list, and register them on a :class:`ScorerRegistry`
with one explicit call. Adding a scorer = 1 file here + 3 lines below, exactly
like adding an enricher.

The three shipped today cover the three metric shapes every other enricher
scorer will reuse:

* ``grounding_rate``      — scalar / tolerance-band
* ``topic_similarity``    — ranking / top-K precision-recall
* ``guest_coappearance``  — set / unordered-pairs precision-recall
"""

from __future__ import annotations

from podcast_scraper.enrichment.eval.registry import ScorerRegistry
from podcast_scraper.enrichment.eval.scorers.grounding_rate import GroundingRateScorer
from podcast_scraper.enrichment.eval.scorers.guest_coappearance import GuestCoappearanceScorer
from podcast_scraper.enrichment.eval.scorers.topic_similarity import TopicSimilarityScorer

BUILTIN_SCORER_ENRICHER_IDS: tuple[str, ...] = (
    "grounding_rate",
    "topic_similarity",
    "guest_coappearance",
)


def register_builtin_scorers(registry: ScorerRegistry) -> None:
    """Register every built-in accuracy scorer on *registry* (one per enricher id)."""
    registry.register(GroundingRateScorer())
    registry.register(TopicSimilarityScorer())
    registry.register(GuestCoappearanceScorer())


__all__ = [
    "BUILTIN_SCORER_ENRICHER_IDS",
    "GroundingRateScorer",
    "GuestCoappearanceScorer",
    "TopicSimilarityScorer",
    "register_builtin_scorers",
]
