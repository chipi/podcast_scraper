"""``fake_for_test`` — deterministic ``EmbeddingProvider`` with no external deps.

CI-safe: backed by :class:`HashEmbedder` (deterministic hash → vector
mapping) so integration tests can exercise the full ``topic_similarity``
pipeline without pulling in sentence-transformers / model weights.

Params (all optional):

* ``dim`` — output vector dimensionality. Default 32. Larger dimensions
  give crisper cosine distances when assertion thresholds matter.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.provider_types.registry import register_provider_type
from podcast_scraper.enrichment.scorers.embedding import (
    HashEmbedder,
    TopicEmbeddingProvider,
)


def _make_fake_provider(params: dict[str, Any]) -> TopicEmbeddingProvider:
    dim_raw = params.get("dim", 32)
    try:
        dim = int(dim_raw)
    except (TypeError, ValueError):
        dim = 32
    if dim < 4 or dim > 1024:
        dim = 32
    embedder = HashEmbedder(dim=dim)
    return TopicEmbeddingProvider(embed_text=embedder)


register_provider_type(
    name="fake_for_test",
    protocol="EmbeddingProvider",
    description="Deterministic hash-based embedder (CI-safe, no model deps).",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dim": {
                "type": "integer",
                "minimum": 4,
                "maximum": 1024,
                "default": 32,
                "description": "Vector dimensionality emitted by the deterministic hash embedder.",
            },
        },
    },
    factory=_make_fake_provider,
)


__all__: list[str] = []
