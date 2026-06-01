"""Upgrade runner: sequence pending migrations, record the ledger (#862).

Ledger-driven: a migration is pending iff its id is not in the ``StateStore``'s
applied set. Migrations run in id order; each successful apply is recorded and the
corpus version advanced. A migration that raises stops the run — earlier steps stay
recorded (forward progress is preserved and the ledger reflects reality), so a fixed
re-run resumes from the failed step. This is exactly how a DB migrations table
behaves, so the contract survives a future move off the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from packaging.version import InvalidVersion, Version

from .migration import Migration, MigrationContext, MigrationResult
from .registry import get_migrations
from .state import StateStore


@dataclass
class UpgradeStatus:
    """Snapshot of where a corpus sits relative to the registered migrations."""

    current_version: Optional[str]
    target_version: Optional[str]
    applied: List[str]
    pending: List[Migration]
    up_to_date: bool = field(init=False)

    def __post_init__(self) -> None:
        """Derive ``up_to_date`` from whether anything is pending."""
        self.up_to_date = not self.pending


class UpgradeRunner:
    """Plans and applies pending migrations against a ``StateStore``."""

    def __init__(
        self, state_store: StateStore, migrations: Optional[List[Migration]] = None
    ) -> None:
        self.state = state_store
        source = migrations if migrations is not None else get_migrations()
        # Id order is a runner invariant — migrations must apply deterministically in
        # sequence regardless of how the list was supplied.
        self.migrations = sorted(source, key=lambda m: m.id)

    @staticmethod
    def _as_version(raw: str) -> Version:
        try:
            return Version(raw)
        except InvalidVersion:
            return Version("0")

    def _target_version(self) -> Optional[str]:
        versions = [m.to_version for m in self.migrations if m.to_version]
        return max(versions, key=self._as_version) if versions else None

    def pending(self) -> List[Migration]:
        """Registered migrations whose id is not yet recorded as applied."""
        applied = self.state.applied_migration_ids()
        return [m for m in self.migrations if m.id not in applied]

    def status(self) -> UpgradeStatus:
        """Current vs target version, applied ids, and the pending migration list."""
        applied = self.state.applied_migration_ids()
        return UpgradeStatus(
            current_version=self.state.current_version(),
            target_version=self._target_version(),
            applied=sorted(applied),
            pending=self.pending(),
        )

    def run(
        self,
        ctx: MigrationContext,
        *,
        to_version: Optional[str] = None,
        now: str = "",
    ) -> List[MigrationResult]:
        """Apply pending migrations in order (optionally only up to *to_version*).

        Records each applied migration in the ledger unless ``ctx.dry_run``. Stops at
        the first migration that raises, after collecting results for the steps that
        did run. ``now`` is the timestamp stamped into ledger records (passed in so
        the runner stays deterministic / clock-free).
        """
        results: List[MigrationResult] = []
        ceiling = self._as_version(to_version) if to_version is not None else None
        for migration in self.pending():
            if ceiling is not None and self._as_version(migration.to_version) > ceiling:
                continue
            result = migration.apply(ctx)
            results.append(result)
            if result.applied and not ctx.dry_run:
                self.state.record_applied(migration.id, to_version=migration.to_version, at=now)
        return results

    def verify(self, ctx: MigrationContext) -> List[Tuple[str, bool, str]]:
        """Run ``verify`` for every applied migration; ``[(id, ok, message)]``."""
        applied = self.state.applied_migration_ids()
        out: List[Tuple[str, bool, str]] = []
        for migration in self.migrations:
            if migration.id in applied:
                ok, msg = migration.verify(ctx)
                out.append((migration.id, ok, msg))
        return out
