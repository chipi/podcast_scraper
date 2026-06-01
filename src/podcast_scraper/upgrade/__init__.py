"""Managed corpus upgrade-path framework (#862).

A small, ordered, idempotent migration runner for moving a deployed corpus across
releases (the first concrete step being the 2.6 → 2.7 FAISS → LanceDB migration,
#858). Two design rules make it forward-compatible with a future database backend
replacing files-on-disk:

1. **Storage-agnostic state.** *Where* the current version and the applied-migration
   ledger live is behind the ``StateStore`` protocol (``state.py``). Today that is
   JSON on disk (``FilesystemStateStore``); introducing a DB means adding one
   ``DbStateStore`` that reads/writes a ``schema_migrations`` table — the runner and
   CLI never change.
2. **Opaque migration steps.** A ``Migration`` (``migration.py``) exposes
   ``apply`` / ``verify`` / ``plan`` and nothing about storage. A step may rewrite
   files today and target a DB tomorrow; the runner only sequences them and records
   the ledger.

The runner is ledger-driven (Rails/Django-style): a migration runs unless its id is
already recorded, so it is safe to re-run and maps directly onto a DB migrations
table when that day comes.
"""

from __future__ import annotations

from .migration import Migration, MigrationContext, MigrationResult
from .registry import get_migrations
from .runner import UpgradeRunner, UpgradeStatus
from .state import FilesystemStateStore, StateStore

__all__ = [
    "Migration",
    "MigrationContext",
    "MigrationResult",
    "StateStore",
    "FilesystemStateStore",
    "UpgradeRunner",
    "UpgradeStatus",
    "get_migrations",
]
