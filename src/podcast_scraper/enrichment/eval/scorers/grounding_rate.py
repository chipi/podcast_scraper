"""Reference scorer — ``grounding_rate`` (scalar / tolerance-band shape).

Grades the corpus grounding rate (grounded insights / total insights, pooled
across persons) against an authored expected value within a tolerance band.
Emits ``within_tolerance`` (a 0/1 metric a floor-gate reads: ``>= 1.0``) plus
``abs_error`` for drill-down. This is the template every *scalar* enricher
scorer follows (insight_density, temporal_velocity, …).
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.eval.protocol import ScoreResult, ScorerManifest

_DEFAULT_TOLERANCE = 0.10


class GroundingRateScorer:
    """Scalar accuracy scorer for the ``grounding_rate`` enricher."""

    manifest = ScorerManifest(
        enricher_id="grounding_rate",
        version="1.0.0",
        metrics=("within_tolerance", "abs_error"),
        description="Pooled grounding rate vs expected within a tolerance band.",
        gold_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "expected_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "tolerance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["expected_rate"],
        },
    )

    def score(
        self,
        *,
        output: dict[str, Any],
        gold: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Compare pooled grounding rate to ``gold['expected_rate']``."""
        persons = output.get("persons")
        expected = gold.get("expected_rate")
        if not isinstance(persons, list) or not isinstance(expected, (int, float)):
            return ScoreResult(
                enricher_id="grounding_rate",
                skipped=True,
                notes="missing persons output or expected_rate gold",
            )
        grounded = sum(int(p.get("grounded_insights", 0)) for p in persons)
        total = sum(int(p.get("total_insights", 0)) for p in persons)
        if total <= 0:
            return ScoreResult(
                enricher_id="grounding_rate", skipped=True, notes="no insights in output"
            )
        actual = grounded / total
        tolerance = float(gold.get("tolerance", _DEFAULT_TOLERANCE))
        abs_error = abs(actual - float(expected))
        within = 1.0 if abs_error <= tolerance else 0.0
        return ScoreResult(
            enricher_id="grounding_rate",
            metrics={"within_tolerance": within, "abs_error": round(abs_error, 4)},
            sample_count=len(persons),
            details={"actual_rate": round(actual, 4), "expected_rate": float(expected)},
        )


__all__ = ["GroundingRateScorer"]
