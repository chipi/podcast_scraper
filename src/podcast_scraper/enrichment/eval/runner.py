"""Runner — grade a batch of enricher outputs against gold, produce metrics.

Closes the loop: ``outputs`` (what enrichers wrote) + ``gold`` (authored
``expected_enrichment`` blocks) → per-enricher :class:`ScoreResult`. The
metrics extracted here are what gets written to ``data/eval`` and later read by
``eval.admission`` to drive the gate. One scorer per enricher id (from the
:class:`ScorerRegistry`); enrichers with no output or no gold are ``skipped``
(the gate's ``on_missing_data`` policy — not a 0 score — decides those).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from podcast_scraper.enrichment.eval.protocol import ScoreResult
from podcast_scraper.enrichment.eval.registry import ScorerRegistry

logger = logging.getLogger(__name__)


def run_scorers(
    registry: ScorerRegistry,
    outputs: Mapping[str, dict[str, Any]],
    gold: Mapping[str, dict[str, Any]],
    config: Mapping[str, dict[str, Any]] | None = None,
) -> list[ScoreResult]:
    """Grade every enricher that has a registered scorer.

    ``outputs`` / ``gold`` / ``config`` are keyed by enricher id. A scorer that
    raises is caught and downgraded to a ``skipped`` result (grading a single
    enricher must never abort the batch).
    """
    cfg = config or {}
    results: list[ScoreResult] = []
    for eid in registry.all_enricher_ids():
        out = outputs.get(eid)
        g = gold.get(eid)
        if out is None or g is None:
            results.append(
                ScoreResult(
                    enricher_id=eid,
                    skipped=True,
                    notes="no output" if out is None else "no gold",
                )
            )
            continue
        try:
            results.append(registry.get(eid).score(output=out, gold=g, config=cfg.get(eid)))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("enrichment eval: scorer %s raised (%s); skipping", eid, exc)
            results.append(ScoreResult(enricher_id=eid, skipped=True, notes=f"scorer error: {exc}"))
    return results


def metrics_by_enricher(results: list[ScoreResult]) -> dict[str, dict[str, float]]:
    """Project scored (non-skipped) results to the gate's input shape.

    ``{enricher_id: {metric: value}}`` — exactly what ``admission`` expects, so
    an in-memory eval can feed the gate directly without a ``data/eval`` round
    trip.
    """
    return {r.enricher_id: r.metrics for r in results if not r.skipped}


__all__ = ["metrics_by_enricher", "run_scorers"]
