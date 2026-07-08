"""``fixed_consensus`` — deterministic ``ConsensusScorer`` (CI-safe, no model deps).

Returns a configured default :class:`ConsensusSignal` for every pair. Used by
integration tests + smoke runs to exercise the ``topic_consensus`` enricher's
plumbing without loading MiniLM / DeBERTa.

Params (all optional):

* ``default_cosine`` — cosine for every pair. Default 0.0 (below the enricher's
  default 0.70 gate → emits zero pairs, a quiet fixture).
* ``default_contradiction`` — contradiction for every pair. Default 0.0.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.provider_types.registry import register_provider_type
from podcast_scraper.enrichment.scorers.consensus import FixedConsensusScorer
from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal


def _make_scorer(params: dict[str, Any]) -> FixedConsensusScorer:
    def _float_or(default: float, key: str) -> float:
        raw = params.get(key, default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return default
        return v if 0.0 <= v <= 1.0 else default

    default = ConsensusSignal(
        cosine=_float_or(0.0, "default_cosine"),
        contradiction=_float_or(0.0, "default_contradiction"),
    )
    return FixedConsensusScorer(default=default)


register_provider_type(
    name="fixed_consensus",
    protocol="ConsensusScorer",
    description="Deterministic composite consensus scorer (CI-safe, no model deps).",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "default_cosine": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.0},
            "default_contradiction": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.0,
            },
        },
    },
    factory=_make_scorer,
)


__all__: list[str] = []
