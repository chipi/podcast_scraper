"""``--with-web`` wiring: register WEB-tier enrichers (wave-G).

The web analogue of ``ml_wiring``. Called by the CLI only when ``--with-web`` is set — WEB-tier
enrichers reach an external source (Wikipedia) at fetch time, so they are opt-in and never part of
the airgapped CI profile. Returns the registered enricher ids so the CLI can enable + opt-in them
(they live in no profile's default set by design).
"""

from __future__ import annotations

import logging

from podcast_scraper.enrichment.enrichers.person_web import PersonWebEnricher
from podcast_scraper.enrichment.registry import EnricherRegistry

logger = logging.getLogger(__name__)


def register_web_enrichers(registry: EnricherRegistry) -> list[str]:
    """Register the WEB-tier enrichers into ``registry``. Returns their manifest ids."""
    enrichers = [PersonWebEnricher()]
    ids: list[str] = []
    for enricher in enrichers:
        registry.register(enricher)
        ids.append(enricher.manifest.id)
        logger.info("enrichment: --with-web: registered %r", enricher.manifest.id)
    return ids


__all__ = ["register_web_enrichers"]
