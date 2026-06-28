"""``fixed_scripted`` — deterministic ``NliScorer`` with scripted scores.

CI-safe: returns the configured ``default`` score for every pair (no
external dependencies). Used by integration tests + smoke runs to
exercise the ``nli_contradiction`` enricher's resilience plumbing
without loading DeBERTa.

Params (all optional):

* ``default_contradiction`` — contradiction probability for every
  scored pair. Default 0.05 (below the default 0.5 threshold so the
  enricher emits zero pairs by default, a quiet fixture).
* ``default_neutral`` / ``default_entailment`` — probabilities for
  the other two NLI classes. Default 0.85 / 0.10.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.provider_types.registry import register_provider_type
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScore


def _make_scorer(params: dict[str, Any]) -> FixedNliScorer:
    def _float_or(default: float, key: str) -> float:
        raw = params.get(key, default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return default
        if not 0.0 <= v <= 1.0:
            return default
        return v

    default = NliScore(
        contradiction=_float_or(0.05, "default_contradiction"),
        neutral=_float_or(0.85, "default_neutral"),
        entailment=_float_or(0.10, "default_entailment"),
    )
    return FixedNliScorer(default=default)


register_provider_type(
    name="fixed_scripted",
    protocol="NliScorer",
    description="Deterministic NLI scorer with scripted output (CI-safe, no model deps).",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "default_contradiction": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.05,
            },
            "default_neutral": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.85,
            },
            "default_entailment": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.10,
            },
        },
    },
    factory=_make_scorer,
)


__all__: list[str] = []
