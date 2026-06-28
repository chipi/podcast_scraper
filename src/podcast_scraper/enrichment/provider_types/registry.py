"""Provider-type registry — maps YAML ``provider.type`` strings to constructors.

The CLI's ``--with-ml`` flag walks the active :class:`EnricherSet`,
finds every enricher with a ``provider_requirement`` declared on its
manifest, looks up the matching provider type by name from the
operator YAML's ``enrichers.<id>.provider.type`` field, and
constructs the runtime instance via this registry. The constructed
provider/scorer is then injected into the enricher.

Adding a new provider type ships under
``provider_types/<protocol>/<name>.py`` with one
:func:`register_provider_type` call at module top. Registration happens
at import time (the protocol-module ``__init__.py`` imports each
type file). The YAML schema's per-type ``params`` validation is
composed from the ``params_schema`` arg each registration declares.

Three reasons this is a registry instead of a switch in the CLI:

1. **Profile YAML stays self-describing.** Operators read profile
   YAMLs (or operator YAML) to know what runs, with which provider,
   with which knobs — everything in one file.
2. **The UI form is auto-generated.** ``GET /api/enrichment/provider-types``
   returns the registry's known types per protocol; the UI's
   ``EnrichmentConfigEditor.vue`` builds dropdowns + parameter
   forms from the ``params_schema`` of each. Adding a new type = 0
   viewer code.
3. **CI vs production substitution is trivial.** The deterministic
   ``fake_for_test`` and ``fixed_scripted`` types ship under
   ``[dev]`` so CI integration tests exercise the full enricher
   surface without [ml] extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# A factory takes the YAML-derived params dict (already validated by
# ``params_schema`` at YAML-load time) and returns a constructed
# provider / scorer instance ready to inject.
ProviderFactory = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ProviderType:
    """Registered metadata for one provider type.

    The registry stores these by ``name``. ``protocol`` is the matching
    :class:`podcast_scraper.enrichment.protocol.ProviderRequirement.protocol`
    string so a single registry can index multiple protocols.
    """

    name: str  # YAML ``provider.type`` key — e.g. "sentence_transformer_local"
    protocol: str  # ProviderRequirement.protocol — e.g. "EmbeddingProvider"
    description: str  # human label for the UI dropdown
    params_schema: dict[str, Any]  # JSON-Schema fragment for params
    factory: ProviderFactory


@dataclass
class ProviderTypeRegistry:
    """Process-scoped registry. Singleton via the module-level helpers."""

    _types: dict[str, ProviderType] = field(default_factory=dict, init=False)

    def register(self, ptype: ProviderType) -> None:
        """Register a provider type; raises ``ValueError`` on duplicate name."""
        if ptype.name in self._types:
            raise ValueError(f"provider type already registered: {ptype.name!r}")
        self._types[ptype.name] = ptype

    def get(self, name: str) -> ProviderType:
        """Look up a registered provider type by name (raises ``KeyError``)."""
        return self._types[name]

    def list_for_protocol(self, protocol: str) -> list[ProviderType]:
        """Every registered type matching the protocol string, name-sorted."""
        return sorted(
            (t for t in self._types.values() if t.protocol == protocol),
            key=lambda t: t.name,
        )

    def all_types(self) -> list[ProviderType]:
        """Every registered type (name-sorted)."""
        return sorted(self._types.values(), key=lambda t: t.name)

    def instantiate(self, name: str, config: dict[str, Any]) -> Any:
        """Construct a provider instance for the given type + params."""
        ptype = self.get(name)
        return ptype.factory(config or {})

    def clear(self) -> None:
        """Test-fixture cleanup."""
        self._types.clear()


_GLOBAL = ProviderTypeRegistry()


def register_provider_type(
    *,
    name: str,
    protocol: str,
    description: str,
    params_schema: dict[str, Any],
    factory: ProviderFactory,
) -> None:
    """Register a provider type on the process-scoped global registry."""
    _GLOBAL.register(
        ProviderType(
            name=name,
            protocol=protocol,
            description=description,
            params_schema=params_schema,
            factory=factory,
        )
    )


def get_global_registry() -> ProviderTypeRegistry:
    """Return the process-scoped registry (lazy-populated on first import)."""
    return _GLOBAL


__all__ = [
    "ProviderFactory",
    "ProviderType",
    "ProviderTypeRegistry",
    "get_global_registry",
    "register_provider_type",
]
