"""Provider-type registry (RFC-088 v2 enrichment-config surface).

See :mod:`podcast_scraper.enrichment.provider_types.registry` for the
registry API and architecture rationale. Importing this package is
side-effecting — every shipped provider type registers on import.
"""

from __future__ import annotations

# Side-effect: import the protocol subpackages so their types register.
from podcast_scraper.enrichment.provider_types import embedding, nli  # noqa: F401
from podcast_scraper.enrichment.provider_types.registry import (
    get_global_registry,
    ProviderType,
    ProviderTypeRegistry,
    register_provider_type,
)
from podcast_scraper.extensions import enrichment_contributions

# Installed extensions register theirs too (ADR-158), after the registry above is importable.
for _contribution in enrichment_contributions():
    if _contribution.provider_types is not None:
        _contribution.provider_types()

__all__ = [
    "ProviderType",
    "ProviderTypeRegistry",
    "get_global_registry",
    "register_provider_type",
]
