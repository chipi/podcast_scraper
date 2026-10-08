"""Installed extensions: how private packages add to the platform (ADR-158 decision 4).

The platform names nothing private. A package that wants to add routes or take part in account
deletion publishes an :class:`Extension` under the ``podcast_scraper.extensions`` entry-point group;
the server and the account-deletion path read :func:`load_extensions` and act on what they find.
Absent package, absent feature — the platform runs, tests and deploys without any of them.

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
from typing import Callable, Iterator, Literal, Sequence, TYPE_CHECKING

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

#: Modules that become private packages at the cutover, each exposing ``EXTENSION``.
_IN_TREE: tuple[str, ...] = ("podcast_scraper.server.app_mcp_extension",)


@dataclass(frozen=True)
class RouterMount:
    router: APIRouter
    plane: Plane
    #: Operator-plane routers only: also mount on the curated public operator surface (RFC-108).
    operator_public: bool = False

    @property
    def prefix(self) -> str:
        return _PREFIX[self.plane]


@dataclass(frozen=True)
class Extension:
    name: str
    routers: Sequence[RouterMount] = ()
    account_deleted: Sequence[AccountDeletedHook] = field(default_factory=tuple)


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
    "AccountDeletedHook",
    "ENTRY_POINT_GROUP",
    "Extension",
    "Plane",
    "RouterMount",
    "load_extensions",
    "use_extensions",
]
