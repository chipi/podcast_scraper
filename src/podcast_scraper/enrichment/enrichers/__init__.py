"""Deterministic enrichers (RFC-088 chunk 2): the public ones, plus those extensions add.

tier=DETERMINISTIC enrichers need no external models or networks — they read the core artifacts
(``*.kg.json`` + ``*.metadata.json``, and whatever an extension's stages write) and write structured
JSON envelopes under ``enrichments/`` (corpus-scope) or ``metadata/enrichments/{stem}.<writes>``
(episode-scope). The one here is the public platform's; the rest come from installed extensions
(ADR-168 decision 5) through :func:`podcast_scraper.extensions.enrichment_contributions`. The four
that read ``gi.json`` (grounding rate, guest co-appearance, insight density and sentiment) left
with GI (E04 move).

All wrap a sync body with :func:`podcast_scraper.enrichment.protocol.sync_enricher`, so the
executor's async machinery flows uninterrupted without enricher authors paying the async ceremony
tax.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.enrichers.topic_cooccurrence_corpus import (
    TopicCooccurrenceCorpusEnricher,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.extensions import enrichment_contributions

#: The platform's own deterministic enrichers, in registration order.
PUBLIC_DETERMINISTIC_ENRICHER_IDS: tuple[str, ...] = ("topic_cooccurrence_corpus",)

#: The platform's own enricher classes (class-level ``manifest``, no instantiation).
PUBLIC_ENRICHER_CLASSES: tuple[type, ...] = (TopicCooccurrenceCorpusEnricher,)


def _extension_deterministic() -> list[Any]:
    return [e for c in enrichment_contributions() for e in c.deterministic()]


def all_deterministic_enricher_ids() -> tuple[str, ...]:
    """Public deterministic ids, then those installed extensions add."""
    return PUBLIC_DETERMINISTIC_ENRICHER_IDS + tuple(
        e.manifest.id for e in _extension_deterministic()
    )


def __getattr__(name: str) -> Any:
    # Kept as a name for existing callers; it depends on what is installed, so it is computed.
    if name == "ALL_DETERMINISTIC_ENRICHER_IDS":
        return all_deterministic_enricher_ids()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_deterministic_enrichers(registry: EnricherRegistry) -> None:
    """Register every deterministic enricher — the platform's and installed extensions' — on
    *registry*. The registry's ``register()`` raises on duplicate ids: call once per registry."""
    registry.register(TopicCooccurrenceCorpusEnricher())
    for enricher in _extension_deterministic():
        registry.register(enricher)


__all__ = [
    "ALL_DETERMINISTIC_ENRICHER_IDS",
    "PUBLIC_DETERMINISTIC_ENRICHER_IDS",
    "PUBLIC_ENRICHER_CLASSES",
    "TopicCooccurrenceCorpusEnricher",
    "all_deterministic_enricher_ids",
    "register_deterministic_enrichers",
]
