"""``deberta_local`` — local DeBERTa cross-encoder NLI scorer.

Wraps :class:`DeBERTaNliScorer` which lazy-loads
``cross-encoder/nli-deberta-v3-small`` via sentence-transformers'
``CrossEncoder``. Model load is deferred to first ``.score()`` call
so the CLI stays importable on ``.[dev]``-only installs.

Per AGENTS.md → "What 'no LLM in CI' actually means": this is a LOCAL
model (no paid API), it's fine to use in CI when sentence-transformers
is installed (typically only the nightly / stack-test jobs install
``.[ml]``). For the default ``.[dev]`` CI path, integration tests use
the ``fixed_scripted`` provider type instead.

Params:

* ``model`` — cross-encoder model id. Defaults to
  ``"cross-encoder/nli-deberta-v3-small"`` (the original chunk-4
  default). Override for fine-tuned variants or model-card revisions.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.provider_types.registry import register_provider_type
from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer


def _make_scorer(params: dict[str, Any]) -> DeBERTaNliScorer:
    model_raw = params.get("model")
    if isinstance(model_raw, str) and model_raw:
        return DeBERTaNliScorer(model_id=model_raw)
    return DeBERTaNliScorer()


register_provider_type(
    name="deberta_local",
    protocol="NliScorer",
    description="Local DeBERTa NLI cross-encoder (lazy-loaded, requires [ml] extra).",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "model": {
                "type": "string",
                "default": "cross-encoder/nli-deberta-v3-small",
                "description": "HuggingFace cross-encoder model id (NLI task head).",
            },
        },
    },
    factory=_make_scorer,
)


__all__: list[str] = []
