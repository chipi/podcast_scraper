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

import json
import logging
import os
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Set, Tuple

from sqlalchemy import and_, create_engine, delete, func, insert, select, Table, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db import leases
from podcast_scraper.db.engines import unbounded_connect_args
from podcast_scraper.db.schema import (
    corpus_metadata,
    corpus_migrations,
    CORPUS_SCHEMA,
    migration_receipts,
    migration_runs,
)
from podcast_scraper.db.settings import DatabaseNotConfigured, ENV_MIGRATOR_URL, normalize_url

from .migration import Migration, MigrationContext, MigrationResult

logger = logging.getLogger(__name__)

#: Per-item surfaces in Postgres refreshed inside the migration's transaction (RFC-135 §6): each
#: gets the open connection and the changed item ids. Their writers register here.
IN_TRANSACTION_SURFACES: list[Tuple[str, Callable[[Connection, frozenset[int]], None]]] = []
#: Surfaces refreshed after the commit (LanceDB, the corpus-wide builds): each gets the engine and
#: the changed item ids. A failure leaves the migration applied and is recorded per surface.
AFTER_COMMIT_SURFACES: list[Tuple[str, Callable[[Engine, frozenset[int]], None]]] = []


class RowsRefused(RuntimeError):
    """The run was refused before it wrote anything (applied already, not applied, lease held)."""


class UndoRefused(RuntimeError):
    """A row changed since the apply; the undo restored nothing."""

    def __init__(self, refused: list[str]) -> None:
        super().__init__(f"{len(refused)} row(s) changed since the apply: {refused[:5]}")
        self.refused = refused


def migrator_engine(url: Optional[str] = None) -> Engine:
    """A ``migrator`` engine for one corpus migration one-shot (RFC-135 §3); the caller disposes it.

    Only one-shots get the ``migrator`` URL (CRFC-005 §7). One connection, no statement timeout as
    for schema migrations, and a 5 s lock timeout so a row another writer holds fails the run
    instead of hanging it.
    """
    if url is None:
        url = (os.environ.get(ENV_MIGRATOR_URL) or "").strip()
        if not url:
            raise DatabaseNotConfigured(
                f"{ENV_MIGRATOR_URL} is not set; a corpus migration on rows runs as migrator"
            )
    args = unbounded_connect_args()
    args["options"] += " -c lock_timeout=5000"
    return create_engine(
        normalize_url(url),
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args=args,
    )


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

    def get(self, table: str, **key: Any) -> Optional[dict[str, Any]]:
        """The row's state (locked), or ``None``."""
        return _state(self.conn, _table(table), key)

    def update(self, table: str, key: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
        """Update one row; ``False``, and no receipt, when nothing changed."""
        t = _table(table)
        before = _state(self.conn, t, key)
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
        before = _state(self.conn, t, key)
        if before is None:
            return False
        self.conn.execute(delete(t).where(_where(t, key)))
        self._receipt(t, key, before, None)
        return True

    def _receipt(
        self,
        table: Table,
        key: Mapping[str, Any],
        before: Optional[dict[str, Any]],
        after: Optional[dict[str, Any]],
    ) -> None:
        self.receipts += 1
        self.by_table[table.name] = self.by_table.get(table.name, 0) + 1
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


def _after_commit(
    engine: Engine, run_id: int, result: dict[str, Any], items: frozenset[int]
) -> None:
    if not AFTER_COMMIT_SURFACES:
        return
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


def run_migration(
    engine: Engine,
    migration: RowMigration,
    *,
    dry_run: bool = False,
    code_digest: Optional[str] = None,
    outcome_sha256: Optional[str] = None,
) -> RunOutcome:
    """Apply ``migration``, or dry-run it, under the exclusive ``corpus`` lease (RFC-135 §3).

    Raises :class:`RowsRefused` when it is applied already or the lease is held; any error inside
    the transaction rolls everything back, is recorded as a ``failed`` run and re-raised.
    """
    kind = "dry_run" if dry_run else "apply"
    stamps = {"code_digest": code_digest, "outcome_sha256": outcome_sha256}
    holder, held = _lease(engine, migration.id, kind)
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
                rows = Rows(conn, migration_id, run_id)
                res = migration.apply_rows(RowContext(conn=conn, rows=rows, dry_run=dry_run))
                items = frozenset(res.item_ids)
                for _name, refresh in IN_TRANSACTION_SURFACES:
                    refresh(conn, items)
                result: dict[str, Any] = {
                    "message": res.message,
                    "counts": res.counts,
                    "receipts": rows.receipts,
                    "receipts_by_table": rows.by_table,
                    "items": len(items),
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
                        .values(result=result, finished_at=func.now())
                    )
        except RowsRefused as exc:
            _record(engine, migration.id, kind, "refused", {"error": str(exc)})
            raise
        except Exception as exc:
            _record(engine, migration.id, kind, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            raise
        if dry_run:
            run_id = _record(engine, migration.id, "dry_run", "ok", result, **stamps)
            return RunOutcome(run_id, "dry_run", "ok", result["receipts"], items, result)
        _after_commit(engine, run_id, result, items)
        return RunOutcome(run_id, "apply", "ok", result["receipts"], items, result)
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
            return RunOutcome(run_id, "dry_run", "ok", result["receipts"], frozen, result)
        _after_commit(engine, run_id, result, frozen)
        return RunOutcome(run_id, "undo", "ok", result["receipts"], frozen, result)
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
