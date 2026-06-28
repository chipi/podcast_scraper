"""EmbeddingProvider implementations registered with the provider-type registry."""

from __future__ import annotations

# Import side-effects register each type with the global registry.
from podcast_scraper.enrichment.provider_types.embedding import (  # noqa: F401
    fake_for_test,
    sentence_transformer_local,
)
