"""WEB-tier enricher registration (wave-G): the web analogue of the ML wiring.

The CLI registers WEB-tier enrichers always — registration is harmless, nothing fetches until a
run — and profile membership decides whether they run, so the airgapped CI profile never
fetches. The platform ships none; installed extensions add them (ADR-158). Returns the registered
ids so the CLI can enable and opt them in under ``--with-web``.
"""

from __future__ import annotations

import logging

from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.extensions import enrichment_contributions

logger = logging.getLogger(__name__)


def register_web_enrichers(registry: EnricherRegistry) -> list[str]:
    """Register every installed WEB-tier enricher on *registry*; return their ids."""
    ids: list[str] = []
    for contribution in enrichment_contributions():
        for enricher in contribution.web():
            if enricher.manifest.id in registry.all_ids():
                continue
            registry.register(enricher)
            ids.append(enricher.manifest.id)
            logger.info("enrichment: --with-web: registered %r", enricher.manifest.id)
    return ids


__all__ = ["register_web_enrichers"]
