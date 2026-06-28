"""Provider-type registry (RFC-088 v2 enrichment-config surface).

See :mod:`podcast_scraper.enrichment.provider_types.registry` for the
registry API and architecture rationale. Importing this package is
side-effecting — every shipped provider type registers on import.
"""

from __future__ import annotations

# Side-effect: import the protocol subpackages so their types register.
from podcast_scraper.enrichment.provider_types import embedding, nli  # noqa: F401
from podcast_scraper.enrichment.provider_types.registry import (
    ProviderType,
    ProviderTypeRegistry,
    get_global_registry,
    register_provider_type,
)

__all__ = [
    "ProviderType",
    "ProviderTypeRegistry",
    "get_global_registry",
    "register_provider_type",
]
