"""Accuracy-scorer registry — the grading counterpart to ``EnricherRegistry``.

A flat registry of :class:`AccuracyScorer` instances keyed by
``manifest.enricher_id`` (one scorer per enricher). Mirrors
:class:`podcast_scraper.enrichment.registry.EnricherRegistry` so the two sides
feel identical: register / get / list, fresh instance per test.
"""

from __future__ import annotations

import logging

from podcast_scraper.enrichment.eval.protocol import AccuracyScorer

logger = logging.getLogger(__name__)


class ScorerRegistry:
    """A flat registry of ``AccuracyScorer`` instances keyed by enricher id."""

    def __init__(self) -> None:
        self._scorers: dict[str, AccuracyScorer] = {}

    def register(self, scorer: AccuracyScorer) -> None:
        """Register a scorer under its ``manifest.enricher_id``.

        Raises ``ValueError`` if a scorer is already registered for that
        enricher id (exactly one scorer per enricher). Use ``clear()`` between
        test runs.
        """
        eid = scorer.manifest.enricher_id
        if eid in self._scorers:
            raise ValueError(f"scorer already registered for enricher: {eid!r}")
        self._scorers[eid] = scorer

    def get(self, enricher_id: str) -> AccuracyScorer:
        """Lookup by enricher id (raises ``KeyError`` if absent)."""
        return self._scorers[enricher_id]

    def has(self, enricher_id: str) -> bool:
        """True when a scorer is registered for this enricher id."""
        return enricher_id in self._scorers

    def all_enricher_ids(self) -> list[str]:
        """All enricher ids with a registered scorer (insertion order)."""
        return list(self._scorers.keys())

    def clear(self) -> None:
        """Clear all registered scorers (test fixture cleanup)."""
        self._scorers.clear()


__all__ = ["ScorerRegistry"]
