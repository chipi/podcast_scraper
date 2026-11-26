"""Corpus migrations on rows (RFC-135).

A :class:`RowMigration` changes rows in the ``corpus`` schema only through :class:`Rows`, which
reads each row's whole state (``to_jsonb``) before and after the write and appends a receipt.
:func:`run_migration` applies one in a single transaction: its rows, receipts, ledger row in
``corpus_migrations``, the ``corpus_revision`` bump and its ``migration_runs`` row commit together
or not at all. A dry run is the same apply rolled back, then recorded as a ``dry_run`` run.
:func:`undo_migration` restores the last apply's receipts, all or nothing: a row that no longer
holds its receipt's ``after`` refuses the whole undo. Both hold the ``corpus`` lease exclusively
and refuse, never wait, while anyone else holds it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Set, Tuple

from sqlalchemy import and_, delete, func, insert, select, Table, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db import leases
from podcast_scraper.db.engines import migrator_engine  # noqa: F401 - re-exported for the CLI
from podcast_scraper.db.schema import (
    corpus_metadata,
    corpus_migrations,
    CORPUS_SCHEMA,
    migration_receipts,
    migration_runs,
)
from podcast_scraper.obs.events import emit_event

from . import outcome_hook
from .migration import Migration, MigrationContext, MigrationResult

logger = logging.getLogger(__name__)

#: Per-item surfaces in Postgres refreshed inside the migration's transaction (RFC-135 §6): each
#: gets the open connection and the changed item ids. Their writers register here.
IN_TRANSACTION_SURFACES: list[Tuple[str, Callable[[Connection, frozenset[int]], None]]] = []
#: Surfaces refreshed after the commit (LanceDB, the corpus-wide builds): each gets the engine and
#: the changed item ids. A failure leaves the migration applied and is recorded per surface.
AFTER_COMMIT_SURFACES: list[Tuple[str, Callable[[Engine, frozenset[int]], None]]] = []


@dataclass
class RowScoreboard:
    """The outcome hook's scoreboard on rows (RFC-135 §5): ``score_items`` gives each served
    item's counts keyed by its natural id, with the metric names of the file scoreboard;
    ``gating_metrics`` the ones a migration's outcome is held to."""

    score_items: Callable[[Connection], dict[str, dict[str, int]]]
    gating_metrics: Callable[[], list[str]]


#: The registered row scoreboard (the database session's loaders over Common's scoring); while it
#: is ``None`` a migration on rows is gated on its plan only and the measure is recorded skipped.
SCOREBOARD: Optional[RowScoreboard] = None


class RowsRefused(RuntimeError):
    """The run was refused before it wrote anything (applied already, not applied, lease held)."""


class UndoRefused(RuntimeError):
    """A row changed since the apply; the undo restored nothing."""

    def __init__(self, refused: list[str]) -> None:
        super().__init__(f"{len(refused)} row(s) changed since the apply: {refused[:5]}")
        self.refused = refused


class PlanStopped(outcome_hook.HookStopped):
    """The plan differs from the clone's and no accept file names the difference (exit 4)."""

    def __init__(self, message: str, diff: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.diff = diff


@dataclass
class RowResult:
    """What ``apply_rows`` changed: the item ids drive the derived surfaces."""

    item_ids: Set[int] = field(default_factory=set)
    message: str = ""
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RunOutcome:
    """One recorded run: its ``migration_runs`` row and what it wrote (or would have)."""

    run_id: int
    kind: str
    status: str
    receipts: int
    item_ids: frozenset[int]
    result: dict[str, Any]
    #: After-commit surfaces that failed: the migration stays applied (exit 6).
    after_commit_failed: list[str] = field(default_factory=list)
    #: The measure's failures against the outcome: the migration stays applied (exit 5).
    outcome_failed: list[str] = field(default_factory=list)


def _table(name: str) -> Table:
    return corpus_metadata.tables[f"{CORPUS_SCHEMA}.{name}"]


def _pk(table: Table) -> list[str]:
    return [c.name for c in table.primary_key.columns]


def _key_values(table: Table, key: Mapping[str, Any]) -> list[Any]:
    names = _pk(table)
    if set(key) != set(names):
        raise KeyError(f"{table.name}: key {sorted(key)} is not its primary key {names}")
    return [key[n] for n in names]


def _where(table: Table, key: Mapping[str, Any]) -> Any:
    return and_(*(table.c[n] == v for n, v in key.items()))


def _row_name(table: Table, key: Mapping[str, Any]) -> str:
    return f"{table.name}:" + json.dumps(
        _key_values(table, key), separators=(",", ":"), default=str
    )


#: Prefix of an item's plan key until :meth:`Rows.plan` names it by its natural id; an item deleted
#: by the run keeps it (``item_id:<id>``).
_ITEM = "item_id:"


def _meta(table: Table, key: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    """The plan row a row belongs to: its item, else the row itself."""
    item = state.get("item_id")
    return f"{_ITEM}{item}" if isinstance(item, int) else _row_name(table, key)


def _sha(state: Mapping[str, Any]) -> str:
    raw = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _state(conn: Connection, table: Table, key: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The row's whole state, locked for the rest of the transaction; ``None`` when absent."""
    cond = " AND ".join(f't."{n}" = :k{i}' for i, n in enumerate(_pk(table)))
    state: Optional[dict[str, Any]] = conn.execute(
        text(f"SELECT to_jsonb(t) FROM {CORPUS_SCHEMA}.{table.name} AS t WHERE {cond} FOR UPDATE"),
        {f"k{i}": v for i, v in enumerate(_key_values(table, key))},
    ).scalar_one_or_none()
    return state


class Rows:
    """The only way a row migration writes: every change leaves a receipt (RFC-135 §1)."""

    def __init__(self, conn: Connection, migration_id: int, run_id: int) -> None:
        self.conn = conn
        self._migration_id = migration_id
        self._run_id = run_id
        self.receipts = 0
        self.by_table: dict[str, int] = {}
        # The plan's evidence (RFC-135 §5): the sha256 of every row read before its write, and
        # what was written, each under the item it belongs to.
        self.reads: dict[str, dict[str, str]] = {}
        self.writes: dict[str, dict[str, int]] = {}
        self._seen: Set[str] = set()

    def _read(self, t: Table, key: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        state = _state(self.conn, t, key)
        name = _row_name(t, key)
        # The first read is the input; later reads of the same row see this run's own writes.
        if state is not None and name not in self._seen:
            self._seen.add(name)
            self.reads.setdefault(_meta(t, key, state), {})[name] = _sha(state)
        return state

    def get(self, table: str, **key: Any) -> Optional[dict[str, Any]]:
        """The row's state (locked), or ``None``."""
        return self._read(_table(table), key)

    def update(self, table: str, key: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
        """Update one row; ``False``, and no receipt, when nothing changed."""
        t = _table(table)
        before = self._read(t, key)
        if before is None:
            raise LookupError(f"{table} {dict(key)}: no such row")
        self.conn.execute(update(t).where(_where(t, key)).values(**values))
        after = _state(self.conn, t, key)
        if after == before:
            return False
        self._receipt(t, key, before, after)
        return True

    def insert(self, table: str, values: Mapping[str, Any]) -> dict[str, Any]:
        """Insert one row; returns its primary key."""
        t = _table(table)
        pk = list(t.primary_key.columns)
        got = self.conn.execute(insert(t).values(**values).returning(*pk)).one()
        key = {c.name: getattr(got, c.name) for c in pk}
        self._receipt(t, key, None, _state(self.conn, t, key))
        return key

    def delete(self, table: str, key: Mapping[str, Any]) -> bool:
        """Delete one row; ``False`` when it was absent."""
        t = _table(table)
        before = self._read(t, key)
        if before is None:
            return False
        self.conn.execute(delete(t).where(_where(t, key)))
        self._receipt(t, key, before, None)
        return True

    def plan(self) -> list[dict[str, Any]]:
        """The run's plan in the outcome hook's shape: one row per item it read or wrote, keyed by
        the item's natural id (the row scoreboard's key), with the receipts per table (``counts``)
        and the sha256 of each row read (``inputs``)."""
        keys = set(self.reads) | set(self.writes)
        ids = [int(k[len(_ITEM) :]) for k in keys if k.startswith(_ITEM)]
        items = _table("content_items")
        natural = dict(
            self.conn.execute(
                select(items.c.item_id, items.c.natural_id).where(items.c.item_id.in_(ids))
            ).all()
        )

        def meta(k: str) -> str:
            if k.startswith(_ITEM):
                return str(natural.get(int(k[len(_ITEM) :]), k))
            return k

        rows = [
            {
                "meta": meta(k),
                "counts": dict(sorted(self.writes.get(k, {}).items())),
                "inputs": dict(sorted(self.reads.get(k, {}).items())),
            }
            for k in keys
        ]
        return sorted(rows, key=lambda r: str(r["meta"]))

    def _receipt(
        self,
        table: Table,
        key: Mapping[str, Any],
        before: Optional[dict[str, Any]],
        after: Optional[dict[str, Any]],
    ) -> None:
        self.receipts += 1
        self.by_table[table.name] = self.by_table.get(table.name, 0) + 1
        counts = self.writes.setdefault(_meta(table, key, before or after or {}), {})
        counts[table.name] = counts.get(table.name, 0) + 1
        self.conn.execute(
            insert(migration_receipts).values(
                migration_id=self._migration_id,
                run_id=self._run_id,
                seq=self.receipts,
                target_table=table.name,
                target_key=json.dumps(_key_values(table, key), separators=(",", ":"), default=str),
                before=before,
                after=after,
            )
        )


@dataclass
class RowContext:
    """What ``apply_rows`` gets: the open transaction's connection and :class:`Rows`."""

    conn: Connection
    rows: Rows
    dry_run: bool
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("upgrade.rows"))


class RowMigration(Migration):
    """A corpus migration over rows. Subclasses implement :meth:`apply_rows` and, for a checkable
    outcome, :meth:`verify_rows`; they write only through ``ctx.rows``."""

    kind = "rows"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Row migrations run through :func:`run_migration`, not the file runner."""
        raise TypeError(f"{self.id} is a migration on rows: run it with upgrade.rows.run_migration")

    @abstractmethod
    def apply_rows(self, ctx: RowContext) -> RowResult:
        """Change rows through ``ctx.rows`` (idempotent: a second apply changes nothing)."""
        raise NotImplementedError

    def verify_rows(self, conn: Connection) -> Tuple[bool, str]:
        """SQL assertions that the change is present; default: none defined."""
        return True, "no verification defined"


def bump_corpus_revision(conn: Connection) -> bool:
    """Advance the corpus revision counter; ``False`` while its table does not exist yet."""
    found = conn.execute(text(f"SELECT to_regclass('{CORPUS_SCHEMA}.corpus_revision')")).scalar()
    if found is None:
        return False
    conn.execute(
        text(
            f"UPDATE {CORPUS_SCHEMA}.corpus_revision "
            "SET revision = revision + 1, updated_at = now() WHERE id = 1"
        )
    )
    return True


def _ledger_row(conn: Connection, natural_id: str) -> Optional[Any]:
    return conn.execute(
        select(corpus_migrations)
        .where(corpus_migrations.c.natural_id == natural_id)
        .with_for_update()
    ).one_or_none()


def _new_run(conn: Connection, natural_id: str, kind: str, **fields: Any) -> int:
    run_id: int = conn.execute(
        insert(migration_runs)
        .values(natural_id=natural_id, kind=kind, status="ok", **fields)
        .returning(migration_runs.c.run_id)
    ).scalar_one()
    return run_id


def _record(
    engine: Engine, natural_id: str, kind: str, status: str, result: dict[str, Any], **fields: Any
) -> int:
    """A run row in a transaction of its own: a dry run, a refusal or a failure."""
    with engine.begin() as conn:
        run_id: int = conn.execute(
            insert(migration_runs)
            .values(
                natural_id=natural_id,
                kind=kind,
                status=status,
                result=result,
                finished_at=func.now(),
                **fields,
            )
            .returning(migration_runs.c.run_id)
        ).scalar_one()
    return run_id


def _set_ledger(conn: Connection, natural_id: str, status: str, run_id: int, **fields: Any) -> int:
    stmt = pg_insert(corpus_migrations).values(
        natural_id=natural_id, status=status, last_run_id=run_id, finished_at=func.now(), **fields
    )
    migration_id: int = conn.execute(
        stmt.on_conflict_do_update(
            index_elements=[corpus_migrations.c.natural_id],
            set_={"status": status, "last_run_id": run_id, "finished_at": func.now(), **fields},
        ).returning(corpus_migrations.c.migration_id)
    ).scalar_one()
    return migration_id


def _lease(engine: Engine, natural_id: str, kind: str) -> Tuple[str, int]:
    holder = f"upgrade:{uuid.uuid4().hex[:12]}"
    try:
        held = leases.acquire(
            engine, [leases.LeaseRequest(leases.CORPUS, leases.EXCLUSIVE)], holder
        )
    except leases.LeaseRefused as exc:
        _record(engine, natural_id, kind, "refused", {"error": str(exc)})
        raise RowsRefused(str(exc)) from exc
    return holder, held


def _emit(outcome: RunOutcome, natural_id: str) -> None:
    emit_event(
        "corpus_migration.run",
        migration=natural_id,
        run_id=outcome.run_id,
        kind=outcome.kind,
        status=outcome.status,
        receipts=outcome.receipts,
        items=len(outcome.item_ids),
        after_commit_failed=outcome.after_commit_failed or None,
        outcome_failed=len(outcome.outcome_failed) or None,
    )


def _after_commit(
    engine: Engine, run_id: int, result: dict[str, Any], items: frozenset[int]
) -> list[str]:
    """Run every after-commit surface; record each result on the run; return the failed names."""
    if not AFTER_COMMIT_SURFACES:
        return []
    done: dict[str, str] = {}
    for name, refresh in AFTER_COMMIT_SURFACES:
        try:
            refresh(engine, items)
            done[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - recorded; the step is re-run on its own
            logger.exception("after-commit surface %s failed", name)
            done[name] = f"failed: {type(exc).__name__}: {exc}"
    result["after_commit"] = done
    with engine.begin() as conn:
        conn.execute(
            update(migration_runs).where(migration_runs.c.run_id == run_id).values(result=result)
        )
    return [n for n, r in done.items() if r != "ok"]


@dataclass
class Regime:
    """What the outcome hook's static checks established for one row migration (RFC-135 §5):
    the shipped outcome and, for a plan that differs from the clone's, the operator's accept."""

    outcome: dict[str, Any]
    outcome_sha: str
    code_digest: str
    accept: Optional[dict[str, Any]] = None


def prepare_regime(
    migration: RowMigration,
    applied: Set[str],
    pending_ids: list[str],
    *,
    outcome_file: Optional[Path] = None,
    accept_file: Optional[Path] = None,
) -> Regime:
    """The static checks of the file regime, for a row migration: a GO outcome measured on clean
    code, this migration first pending, prod's applied set the clone's, the installed code the
    measured code (``HookRefused``, exit 3). The plan is compared inside the run's transaction."""
    outcome, sha = outcome_hook.load_outcome(migration.id, outcome_file)
    if pending_ids and pending_ids[0] != migration.id:
        raise outcome_hook.HookRefused(f"{migration.id}: run {pending_ids[0]} first")
    if set(outcome["applied_before"]) != set(applied):
        missing = sorted(set(outcome["applied_before"]) - set(applied))
        extra = sorted(set(applied) - set(outcome["applied_before"]))
        raise outcome_hook.HookRefused(
            f"{migration.id}: prod's applied migrations differ from the clone's "
            f"(missing {missing}, extra {extra})"
        )
    digest = outcome_hook.migration_digest(migration)
    if digest != outcome["code_digest"]:
        raise outcome_hook.HookRefused(
            f"{migration.id}: the installed code is not the code that was measured"
        )
    accept = None
    if accept_file is not None:
        try:
            accept = json.loads(Path(accept_file).read_bytes())
        except (OSError, ValueError) as exc:
            raise outcome_hook.HookStopped(
                f"{migration.id}: the accept file {accept_file} is unreadable ({exc})"
            ) from exc
        if not isinstance(accept, dict):
            raise outcome_hook.HookStopped(f"{migration.id}: the accept file is not an object")
    return Regime(outcome, sha, digest, accept)


def _check_plan(
    migration: RowMigration, regime: Regime, plan: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The plan's difference from the clone's, accepted; ``HookStopped`` (exit 4) when it differs
    and no accept file names the difference."""
    diff = outcome_hook.plan_diff(regime.outcome["plan"], plan)
    if not diff:
        return diff
    if regime.accept is None:
        lines = "\n".join(f"  {e['kind']:10} {e['meta']} {e.get('fields', '')}" for e in diff)
        raise PlanStopped(
            f"{migration.id}: prod's plan differs from the clone's in {len(diff)} item(s); "
            f"each needs a cause (--accept-plan-diff):\n{lines}",
            diff,
        )
    outcome_hook.check_accept(regime.accept, migration.id, regime.outcome_sha, diff)
    return diff


def _measure(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
    metrics: list[str],
    regime: Optional[Regime],
    diff: list[dict[str, Any]],
) -> dict[str, Any]:
    """The file regime's comparison (``outcome_hook.measure``) on the row scoreboard: this run's
    deltas against the clone's, an accepted row that differs only in its inputs held to the
    clone's exactly, a differing one only to no rise, and no inconsistency left after."""
    prod = outcome_hook.deltas(before, after, metrics)
    totals_after = {m: sum(e.get(m, 0) for e in after.values()) for m in metrics}
    measured: dict[str, Any] = {
        "prod_deltas": prod,
        "totals_before": {m: sum(e.get(m, 0) for e in before.values()) for m in metrics},
        "totals_after": totals_after,
        "failures": [],
    }
    if regime is None:
        return measured
    inputs_only = {e["meta"] for e in diff if outcome_hook._inputs_only(e)}
    differing = {e["meta"] for e in diff} - inputs_only
    fails = outcome_hook.compare(prod, regime.outcome["episode_deltas"], differing, metrics)
    if totals_after.get("S1c_inconsistent", 0):
        fails.append(f"S1c is {totals_after['S1c_inconsistent']} after, not 0")
    measured["failures"] = fails
    measured["clone_deltas"] = regime.outcome["episode_deltas"]
    return measured


def run_migration(
    engine: Engine,
    migration: RowMigration,
    *,
    dry_run: bool = False,
    regime: Optional[Regime] = None,
) -> RunOutcome:
    """Apply ``migration``, or dry-run it, under the exclusive ``corpus`` lease (RFC-135 §3).

    Raises :class:`RowsRefused` when it is applied already or the lease is held, and the outcome
    hook's ``HookStopped`` when ``regime``'s plan differs; either, or any error inside the
    transaction, rolls everything back, is recorded as a run and re-raised. An after-commit
    surface that fails leaves the migration applied and is named in ``after_commit_failed``.
    """
    kind = "dry_run" if dry_run else "apply"
    stamps = {
        "code_digest": regime.code_digest if regime else None,
        "outcome_sha256": regime.outcome_sha if regime else None,
    }
    holder, held = _lease(engine, migration.id, kind)
    plan: list[dict[str, Any]] = []
    try:
        # The except clauses run after the connection is back in the pool: a migrator engine has
        # one connection, and recording the refusal or failure needs it.
        try:
            with engine.connect() as conn, conn.begin() as tx:
                leases.heartbeat(conn, holder, held)
                row = _ledger_row(conn, migration.id)
                if row is not None and row.status == "applied":
                    raise RowsRefused(f"{migration.id} is applied already")
                run_id = _new_run(conn, migration.id, kind, **stamps)
                migration_id = _set_ledger(
                    conn, migration.id, "applied", run_id, to_version=migration.to_version or None
                )
                board = SCOREBOARD
                # Scored in the transaction, before and after: a scoreboard that fails rolls the
                # apply back, so a migration on rows is never left applied and unmeasured.
                scored = board.score_items(conn) if board is not None else {}
                rows = Rows(conn, migration_id, run_id)
                res = migration.apply_rows(RowContext(conn=conn, rows=rows, dry_run=dry_run))
                plan = rows.plan()
                diff = _check_plan(migration, regime, plan) if regime is not None else []
                items = frozenset(res.item_ids)
                for _name, refresh in IN_TRANSACTION_SURFACES:
                    refresh(conn, items)
                if board is not None:
                    measured = _measure(
                        scored, board.score_items(conn), board.gating_metrics(), regime, diff
                    )
                else:
                    measured = {"skipped": "no row scoreboard registered: gated on the plan only"}
                result: dict[str, Any] = {
                    "measure": measured,
                    "message": res.message,
                    "counts": res.counts,
                    "receipts": rows.receipts,
                    "receipts_by_table": rows.by_table,
                    "items": len(items),
                    "item_ids": sorted(items),
                    "corpus_revision_bumped": bump_corpus_revision(conn),
                }
                ok, msg = migration.verify_rows(conn)
                result["verify"] = {"ok": ok, "message": msg}
                if dry_run:
                    tx.rollback()
                else:
                    conn.execute(
                        update(migration_runs)
                        .where(migration_runs.c.run_id == run_id)
                        .values(plan=plan, result=result, finished_at=func.now())
                    )
        except RowsRefused as exc:
            _record(engine, migration.id, kind, "refused", {"error": str(exc)})
            raise
        except outcome_hook.HookStopped as exc:
            refused = {
                "error": str(exc),
                "plan_diff": exc.diff if isinstance(exc, PlanStopped) else None,
            }
            _record(engine, migration.id, kind, "refused", refused, plan=plan, **stamps)
            raise
        except Exception as exc:
            _record(engine, migration.id, kind, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            raise
        if dry_run:
            run_id = _record(engine, migration.id, "dry_run", "ok", result, plan=plan, **stamps)
            outcome = RunOutcome(run_id, "dry_run", "ok", result["receipts"], items, result)
        else:
            failed = _after_commit(engine, run_id, result, items)
            outcome = RunOutcome(run_id, "apply", "ok", result["receipts"], items, result, failed)
        outcome.outcome_failed = list(measured.get("failures") or [])
        _emit(outcome, migration.id)
        return outcome
    finally:
        leases.release(engine, holder)


def rerun_after_commit(engine: Engine, natural_id: str) -> RunOutcome:
    """Run the after-commit surfaces again for the items the last apply or undo of ``natural_id``
    changed (each step is idempotent), under the exclusive ``corpus`` lease; a ``rerun`` run."""
    holder, _held = _lease(engine, natural_id, "rerun")
    try:
        try:
            with engine.begin() as conn:
                row = _ledger_row(conn, natural_id)
                if row is None or row.last_run_id is None:
                    raise RowsRefused(f"{natural_id} has no run to complete")
                last = conn.execute(
                    select(migration_runs.c.result).where(
                        migration_runs.c.run_id == row.last_run_id
                    )
                ).scalar_one()
        except RowsRefused as exc:
            _record(engine, natural_id, "rerun", "refused", {"error": str(exc)})
            raise
        items = frozenset((last or {}).get("item_ids") or [])
        result: dict[str, Any] = {"of": "after_commit", "for_run": row.last_run_id}
        run_id = _record(engine, natural_id, "rerun", "ok", result)
        failed = _after_commit(engine, run_id, result, items)
        outcome = RunOutcome(run_id, "rerun", "ok", 0, items, result, failed)
        _emit(outcome, natural_id)
        return outcome
    finally:
        leases.release(engine, holder)


def _restore(
    conn: Connection,
    table: Table,
    key: dict[str, Any],
    state: Optional[dict[str, Any]],
    exists: bool,
) -> None:
    """Put ``state`` back on the row ``key`` names: delete it for ``None``, else update it in
    place (a delete and insert would cascade to its children) or insert it when it is absent."""
    if state is None:
        conn.execute(delete(table).where(_where(table, key)))
        return
    name = f"{CORPUS_SCHEMA}.{table.name}"
    record = f"jsonb_populate_record(NULL::{name}, CAST(:s AS jsonb))"
    if not exists:
        conn.execute(text(f"INSERT INTO {name} SELECT * FROM {record}"), {"s": json.dumps(state)})
        return
    cols = ", ".join(f'"{c.name}"' for c in table.columns)
    picked = ", ".join(f'r."{c.name}"' for c in table.columns)
    cond = " AND ".join(f'"{n}" = :k{i}' for i, n in enumerate(_pk(table)))
    conn.execute(
        text(f"UPDATE {name} SET ({cols}) = (SELECT {picked} FROM {record} AS r) WHERE {cond}"),
        {"s": json.dumps(state), **{f"k{i}": v for i, v in enumerate(_key_values(table, key))}},
    )


def undo_migration(engine: Engine, natural_id: str, *, dry_run: bool = False) -> RunOutcome:
    """Restore the last apply of ``natural_id`` from its receipts, all or nothing (RFC-135 §4).

    Raises :class:`RowsRefused` when it is not applied or the lease is held, and
    :class:`UndoRefused`, with nothing restored, when any row changed since the apply.
    """
    kind = "dry_run" if dry_run else "undo"
    holder, held = _lease(engine, natural_id, kind)
    try:
        try:
            with engine.connect() as conn, conn.begin() as tx:
                leases.heartbeat(conn, holder, held)
                row = _ledger_row(conn, natural_id)
                if row is None or row.status != "applied":
                    raise RowsRefused(f"{natural_id} is not applied")
                receipts = conn.execute(
                    select(migration_receipts)
                    .where(migration_receipts.c.run_id == row.last_run_id)
                    .order_by(migration_receipts.c.seq.desc())
                ).all()
                refused: list[str] = []
                restored = 0
                items: set[int] = set()
                for r in receipts:
                    table = _table(r.target_table)
                    key = dict(zip(_pk(table), json.loads(r.target_key)))
                    now = _state(conn, table, key)
                    if now == r.before:
                        restored += 1
                        continue
                    if now != r.after:
                        refused.append(f"{r.target_table} {r.target_key}")
                        continue
                    _restore(conn, table, key, r.before, exists=now is not None)
                    restored += 1
                    item = (r.before or r.after or {}).get("item_id")
                    if isinstance(item, int):
                        items.add(item)
                if refused:
                    raise UndoRefused(refused)
                run_id = _new_run(conn, natural_id, "undo")
                _set_ledger(conn, natural_id, "undone", run_id)
                frozen = frozenset(items)
                for _name, refresh in IN_TRANSACTION_SURFACES:
                    refresh(conn, frozen)
                result: dict[str, Any] = {
                    "of": "undo",
                    "receipts": len(receipts),
                    "restored": restored,
                    "items": len(frozen),
                    "item_ids": sorted(frozen),
                    "undid_run": row.last_run_id,
                    "corpus_revision_bumped": bump_corpus_revision(conn),
                }
                if dry_run:
                    tx.rollback()
                else:
                    conn.execute(
                        update(migration_runs)
                        .where(migration_runs.c.run_id == run_id)
                        .values(result=result, finished_at=func.now())
                    )
        except UndoRefused as exc:
            _record(engine, natural_id, kind, "refused", {"of": "undo", "refused": exc.refused})
            raise
        except RowsRefused as exc:
            _record(engine, natural_id, kind, "refused", {"of": "undo", "error": str(exc)})
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _record(engine, natural_id, kind, "failed", {"of": "undo", "error": error})
            raise
        if dry_run:
            run_id = _record(engine, natural_id, "dry_run", "ok", result)
            outcome = RunOutcome(run_id, "dry_run", "ok", result["receipts"], frozen, result)
        else:
            failed = _after_commit(engine, run_id, result, frozen)
            outcome = RunOutcome(run_id, "undo", "ok", result["receipts"], frozen, result, failed)
        _emit(outcome, natural_id)
        return outcome
    finally:
        leases.release(engine, holder)


class DbStateStore:
    """The upgrade ledger in ``corpus_migrations`` (the ``StateStore`` protocol, ADR-169)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def applied_migration_ids(self) -> Set[str]:
        """Ids whose ledger row is ``applied``."""
        with self._engine.connect() as conn:
            return set(
                conn.execute(
                    select(corpus_migrations.c.natural_id).where(
                        corpus_migrations.c.status == "applied"
                    )
                ).scalars()
            )

    def current_version(self) -> Optional[str]:
        """The ``to_version`` of the highest applied id, or ``None``."""
        with self._engine.connect() as conn:
            version: Optional[str] = conn.execute(
                select(corpus_migrations.c.to_version)
                .where(corpus_migrations.c.status == "applied")
                .order_by(corpus_migrations.c.natural_id.desc())
                .limit(1)
            ).scalar_one_or_none()
        return version

    def record_applied(self, migration_id: str, *, to_version: str, at: str) -> None:
        """Not used: :func:`run_migration` writes the ledger in the migration's own transaction."""
        raise NotImplementedError("the ledger row is written by run_migration, in its transaction")

    def record_reverted(self, migration_id: str) -> bool:
        """Not used: :func:`undo_migration` writes the ledger in the undo's own transaction."""
        raise NotImplementedError("the ledger row is written by undo_migration, in its transaction")
