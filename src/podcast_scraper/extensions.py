"""Installed extensions: how private packages add to the platform (ADR-158 decision 4).

The platform names nothing private. A package that wants to add routes or take part in account
deletion publishes an :class:`Extension` under the ``podcast_scraper.extensions`` entry-point group;
the server and the account-deletion path read :func:`load_extensions` and act on what they find.
Absent package, absent feature — the platform runs, tests and deploys without any of them.

An extension module must import with the platform's core dependencies only: the pipeline image
has no web stack, and enrichment loads extensions too. So routers are a callable the server invokes,
and hooks import what they need inside their own bodies.

Until the cutover the modules that will become those packages still live in this tree, listed in
``_IN_TREE``. Each is optional: a module the split has already removed is skipped, which is exactly
what ``scripts/tools/split_probe.py`` checks. An extension found both in-tree and through an entry
point is loaded once, by name.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

    from podcast_scraper.server.app_user_store import User

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "podcast_scraper.extensions"

#: Where a router mounts. ``operator`` is the operator read plane (``/api``, full posture only;
#: also the curated public operator plane when ``operator_public`` is set), ``app`` the consumer
#: plane (``/api/app``), ``internal`` the service-to-service plane (``/internal``), ``root`` the
#: app root (well-known documents).
Plane = Literal["operator", "app", "internal", "root"]

_PREFIX: dict[str, str] = {
    "operator": "/api",
    "app": "/api/app",
    "internal": "/internal",
    "root": "",
}

#: ``(data_dir, user) -> {count_name: n}``. Runs before the user's directory is removed; must be
#: idempotent. The counts reach the deletion log and the admin response, never an address or token.
AccountDeletedHook = Callable[[Path, "User"], dict[str, int]]

#: ``(data_dir, user, provider)``. Runs once, right after sign-up creates the account (Google, Apple
#: or email link). An app records its own first-use state here.
AccountCreatedHook = Callable[[Path, "User", str], None]

#: Modules that become private packages at the cutover, each exposing ``EXTENSION``.
_IN_TREE: tuple[str, ...] = (
    "podcast_scraper.server.app_mcp_extension",
    "podcast_scraper.server.app_player_extension",
    "podcast_scraper.enrichment.intelligence_extension",
)


@dataclass(frozen=True)
class RouterMount:
    """One router and the plane it mounts on."""

    router: APIRouter
    plane: Plane
    #: Operator-plane routers only: also mount on the curated public operator surface (RFC-108).
    operator_public: bool = False

    @property
    def prefix(self) -> str:
        return _PREFIX[self.plane]


def _no_routers() -> Sequence[RouterMount]:
    return ()


def _none() -> Sequence[Any]:
    return ()


def _no_query_enrichers(corpus_root_provider: Callable[[], Path]) -> Sequence[Any]:
    return ()


@dataclass(frozen=True)
class EnrichmentContribution:
    """Enrichers an extension adds (ADR-158 decision 5). Each part is a callable so importing the
    extension stays cheap; the enrichment code calls them where it builds its registries."""

    #: Every enricher class the extension owns. Their class-level ``manifest`` feeds the accuracy
    #: gate, the config schema and the profile sets, without instantiating anything.
    enricher_classes: Callable[[], Sequence[type]] = _none
    #: Deterministic enricher instances, registered wherever the platform registers its own.
    deterministic: Callable[[], Sequence[Any]] = _none
    #: ``--with-ml`` wiring: ``(enricher_registry, enricher_set) -> None``.
    ml_wiring: Callable[[Any, Any], None] | None = None
    #: WEB-tier enricher instances (registered always; profile membership decides if they run).
    web: Callable[[], Sequence[Any]] = _none
    #: Query enricher instances, built per search registry: ``(corpus_root_provider) -> [...]``.
    query_enrichers: Callable[[Callable[[], Path]], Sequence[Any]] = _no_query_enrichers
    #: Accuracy scorer instances for the eval gate.
    scorers: Callable[[], Sequence[Any]] = _none
    #: Registers the extension's provider types on the global provider-type registry.
    provider_types: Callable[[], None] | None = None


@dataclass(frozen=True)
class Extension:
    """What one installed package adds to the platform. Every part is optional."""

    name: str
    #: Called by the server only, so building the routers may import the web stack.
    routers: Callable[[], Sequence[RouterMount]] = _no_routers
    account_deleted: Sequence[AccountDeletedHook] = field(default_factory=tuple)
    account_created: Sequence[AccountCreatedHook] = field(default_factory=tuple)
    enrichment: EnrichmentContribution | None = None


_override: list[Extension] | None = None


def _from_module(module_name: str) -> Extension | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module_name.startswith(exc.name):
            return None  # removed by the split: absent package, absent feature
        raise
    ext = getattr(module, "EXTENSION", None)
    if not isinstance(ext, Extension):
        raise TypeError(f"{module_name}.EXTENSION is not an Extension")
    return ext


def load_extensions() -> list[Extension]:
    """Every installed extension, in-tree ones first, one per name."""
    if _override is not None:
        return list(_override)
    found: dict[str, Extension] = {}
    for module_name in _IN_TREE:
        ext = _from_module(module_name)
        if ext is not None:
            found.setdefault(ext.name, ext)
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        ext = ep.load()
        if callable(ext) and not isinstance(ext, Extension):
            ext = ext()
        if not isinstance(ext, Extension):
            raise TypeError(f"entry point {ep.name} ({ep.value}) is not an Extension")
        if ext.name in found:
            logger.debug("extension %s already loaded; skipping entry point %s", ext.name, ep.value)
            continue
        found[ext.name] = ext
    return list(found.values())


def enrichment_contributions() -> list[EnrichmentContribution]:
    """Every installed extension's enrichment contribution, in load order."""
    return [ext.enrichment for ext in load_extensions() if ext.enrichment is not None]


def run_account_created(data_dir: Path, user: User, provider: str) -> None:
    """Every extension's ``account_created`` hook, in load order."""
    for ext in load_extensions():
        for hook in ext.account_created:
            hook(data_dir, user, provider)


@contextmanager
def use_extensions(extensions: Sequence[Extension]) -> Iterator[None]:
    """Replace discovery with *extensions* for the duration (tests: a fake app, or none)."""
    global _override
    previous = _override
    _override = list(extensions)
    try:
        yield
    finally:
        _override = previous


__all__ = [
    "AccountCreatedHook",
    "AccountDeletedHook",
    "ENTRY_POINT_GROUP",
    "EnrichmentContribution",
    "Extension",
    "Plane",
    "RouterMount",
    "enrichment_contributions",
    "load_extensions",
    "run_account_created",
    "use_extensions",
]
