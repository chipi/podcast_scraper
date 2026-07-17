"""Upgrade state: current version + applied-migration ledger (#862).

``StateStore`` is the single seam that makes the framework storage-agnostic. The
filesystem implementation keeps the ledger in ``upgrade_ledger.json`` at the corpus
root and reads the corpus's produced-by code version from the existing
``corpus_manifest.json`` (``corpus_version.py``, #796) — it does **not** invent a
second version concept. A future database backend implements the same four methods
against a ``schema_migrations`` table; nothing above this file changes.

Version reporting precedence: the ledger's recorded version (set as migrations
advance the corpus) wins; absent that, the manifest's produced-by code version; else
``None`` (unstamped corpus).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Protocol, Set

from ..corpus_version import corpus_code_version, read_produced_by

LEDGER_FILE = "upgrade_ledger.json"


class StateStore(Protocol):
    """Persists the upgrade ledger + current version, independent of storage medium."""

    def current_version(self) -> Optional[str]:
        """Return the corpus's current version, or ``None`` if unstamped."""
        ...

    def applied_migration_ids(self) -> Set[str]:
        """Return the set of migration ids already applied."""
        ...

    def record_applied(self, migration_id: str, *, to_version: str, at: str) -> None:
        """Record *migration_id* as applied and advance the version to *to_version*."""
        ...


class FilesystemStateStore:
    """``StateStore`` backed by a JSON ledger at the corpus root."""

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = Path(corpus_root)
        self.ledger_path = self.corpus_root / LEDGER_FILE

    def _read_ledger(self) -> dict:
        if not self.ledger_path.is_file():
            return {}
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_ledger(self, data: dict) -> None:
        self.corpus_root.mkdir(parents=True, exist_ok=True)
        # Atomic tmp + replace: a mid-write kill otherwise truncates the ledger,
        # which _read_ledger treats as "no migrations applied" -> full replay
        # (review 2026-07-17 low/ledger-atomic).
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.ledger_path)

    def current_version(self) -> Optional[str]:
        """Ledger version if set, else the manifest produced-by code version."""
        ledger = self._read_ledger()
        ver = ledger.get("version")
        if isinstance(ver, str) and ver.strip():
            return ver.strip()
        return corpus_code_version(read_produced_by(self.corpus_root))

    def applied_migration_ids(self) -> Set[str]:
        """Migration ids recorded in the ledger's ``applied`` list."""
        ledger = self._read_ledger()
        applied = ledger.get("applied")
        if not isinstance(applied, list):
            return set()
        return {
            str(entry["id"]) for entry in applied if isinstance(entry, dict) and entry.get("id")
        }

    def applied_records(self) -> List[dict]:
        """Full applied-migration records (id, to_version, at) in recorded order."""
        ledger = self._read_ledger()
        applied = ledger.get("applied")
        return [e for e in applied if isinstance(e, dict)] if isinstance(applied, list) else []

    def record_applied(self, migration_id: str, *, to_version: str, at: str) -> None:
        """Append *migration_id* to the ledger and advance the recorded version."""
        ledger = self._read_ledger()
        applied = ledger.get("applied")
        if not isinstance(applied, list):
            applied = []
        applied = [e for e in applied if not (isinstance(e, dict) and e.get("id") == migration_id)]
        applied.append({"id": migration_id, "to_version": to_version, "at": at})
        ledger["applied"] = applied
        ledger["version"] = to_version
        self._write_ledger(ledger)
