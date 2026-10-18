"""Upgrade runner: sequence pending migrations, record the ledger (#862).

Ledger-driven: a migration is pending iff its id is not in the ``StateStore``'s
applied set. Migrations run in id order; each successful apply is recorded and the
corpus version advanced. A migration that raises stops the run — earlier steps stay
recorded (forward progress is preserved and the ledger reflects reality), so a fixed
re-run resumes from the failed step. This is exactly how a DB migrations table
behaves, so the contract survives a future move off the filesystem.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def _in_regime(migration_id: str) -> bool:
    from .outcome_hook import in_regime

    return in_regime(migration_id)


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
        #: Regime migrations the outcome hook prepared for this run (``outcome_hook.prepare``),
        #: by id; a regime migration with none is refused (``upgrade/outcome_hook.py``).
        self.prepared: Dict[str, Any] = {}
        self.snapshot: Optional[str] = None

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
        hold_lock: bool = True,
    ) -> List[MigrationResult]:
        """Apply pending migrations in order (optionally only up to *to_version*).

        Records each applied migration in the ledger unless ``ctx.dry_run``. Stops at
        the first migration that raises, after collecting results for the steps that
        did run. ``now`` is the timestamp stamped into ledger records (passed in so
        the runner stays deterministic / clock-free).

        ``hold_lock=False`` says the CALLER already holds the corpus lock. The CLI does, because
        its pre-upgrade snapshot — a full copytree, minutes on a real corpus — has to be inside the
        same lock as the migrations: an ingest writing during the copy yields an inconsistent
        snapshot, and the operator has already been told that directory is the rollback. Re-taking
        it here would not merely be redundant: the lock is ``timeout=0``, so the second acquire in
        the same process reads its own live PID from the holder file and refuses.
        """
        results: List[MigrationResult] = []
        ceiling = self._as_version(to_version) if to_version is not None else None

        # HOLD THE CORPUS LOCK FOR THE WHOLE RUN. A migration is a read-modify-write over every
        # artifact in the corpus; an ingest running concurrently rewrites some of them underneath
        # it, and the two writers interleave silently. m0009 additionally records a `file_sha_after`
        # per episode for the undo to check — a sha for a file the pipeline rewrites moments later
        # makes that episode permanently unrollbackable, which is a worse outcome than refusing.
        #
        # The rollback already refuses on contention (`role_ledger.undo_from_ledger`). Locking the
        # forward direction too is what makes that symmetric: before this, the runbook's "confirm
        # no ingest job is running" was the ONLY thing standing between a migration and a race.
        #
        # Contention raises RuntimeError and is NOT swallowed — the operator stops the ingest and
        # re-runs. Only "there is no lock to take" degrades to unlocked. A dry run reads and writes
        # nothing, so it does not need the lock and must not be blocked by an ingest.
        with contextlib.ExitStack() as stack:
            if not ctx.dry_run and hold_lock:
                try:
                    from ..utils.corpus_lock import corpus_parent_lock

                    stack.enter_context(corpus_parent_lock(ctx.corpus_root))
                except (ImportError, OSError):
                    pass
            for migration in self.pending():
                if ceiling is not None and self._as_version(migration.to_version) > ceiling:
                    continue
                if not ctx.dry_run and _in_regime(migration.id):
                    result = self._apply_measured(migration, ctx, now)
                else:
                    result = migration.apply(ctx)
                    if result.applied and not ctx.dry_run:
                        self.state.record_applied(
                            migration.id, to_version=migration.to_version, at=now
                        )
                results.append(result)
        return results

    def _apply_measured(self, migration: Migration, ctx: MigrationContext, now: str) -> Any:
        """A regime migration: applied and recorded inside the outcome hook's measurement."""
        from . import outcome_hook

        def apply() -> MigrationResult:
            result = migration.apply(ctx)
            if result.applied:
                self.state.record_applied(migration.id, to_version=migration.to_version, at=now)
            return result

        prepared = self.prepared.get(migration.id)
        if prepared is None:
            # Decided again here, inside the lock and after every earlier migration of the run
            # (criteria R1): only a plan that is still empty runs unmeasured.
            if outcome_hook.episode_plan(migration, ctx.corpus_root):
                raise outcome_hook.HookRefused(
                    f"{migration.id}: its plan is not empty here; run it alone with --to"
                )
            return outcome_hook.apply_empty(migration, ctx.corpus_root, apply)
        return outcome_hook.measure(prepared, ctx.corpus_root, apply, self.snapshot)

    def verify(self, ctx: MigrationContext) -> List[Tuple[str, bool, str]]:
        """Run ``verify`` for every applied migration; ``[(id, ok, message)]``."""
        applied = self.state.applied_migration_ids()
        out: List[Tuple[str, bool, str]] = []
        for migration in self.migrations:
            if migration.id in applied:
                ok, msg = migration.verify(ctx)
                out.append((migration.id, ok, msg))
        return out
