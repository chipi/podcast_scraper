"""Run and inspect the Alembic histories (ADR-169, ADR-172), never at app start.

The kernel's history runs once per plane schema; the ``shared`` schema's once (the rows both
planes read and write); each installed extension's module history (``Extension.database``) on the
planes it names, with its own version table. ``upgrade --plane P`` runs ``shared``, then the
kernel, then each module, each in its own transaction; a failed step leaves the earlier ones
where they got to.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import Connection, create_engine, Engine, MetaData, select, text
from sqlalchemy.schema import CreateSchema

from podcast_scraper import build_info
from podcast_scraper.db.engines import unbounded_connect_args
from podcast_scraper.db.schema import metadata as kernel_metadata, schema_history, SHARED_SCHEMA
from podcast_scraper.db.settings import normalize_url, validate_plane

_HERE = Path(__file__).parent
_PLANE_SCRIPTS = _HERE / "migrations"
_SHARED_SCRIPTS = _HERE / "migrations_shared"
_VERSION_TABLE = "alembic_version"
_MODULE_NAME = re.compile(r"^[a-z][a-z0-9]{1,30}$")
_RESERVED = frozenset({"kernel", "shared", "alembic", "schema"})


@dataclass(frozen=True)
class History:
    """One Alembic history: where its scripts are and which version table it keeps."""

    name: str
    scripts: Path
    version_table: str = _VERSION_TABLE


KERNEL = History("kernel", _PLANE_SCRIPTS)
SHARED = History("shared", _SHARED_SCRIPTS)


def _default(schema: str | None) -> History:
    return SHARED if schema == SHARED_SCHEMA else KERNEL


def _config(
    schema: str | None = None,
    connection: Connection | None = None,
    history: History | None = None,
) -> Config:
    hist = history or _default(schema)
    cfg = Config()
    cfg.set_main_option("script_location", str(hist.scripts))
    cfg.attributes["version_table"] = hist.version_table
    if schema is not None:
        cfg.attributes["plane"] = schema
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def module_history(module: object) -> History:
    """The :class:`History` of an extension's ``DbModule``, after :func:`validate_module`."""
    validate_module(module)
    name = getattr(module, "name")
    return History(name, Path(getattr(module, "scripts")), f"{_VERSION_TABLE}_{name}")


def validate_module(module: object) -> None:
    """Refuse a module whose name, revisions or tables break the ADR-172 rules.

    Names are short lowercase words, not a reserved one; every revision id and every table starts
    with ``<name>_``; tables carry no schema (the plane places them), so one module cannot write
    into ``shared`` or reach another plane, and its revisions cannot collide with the kernel's in
    ``schema_history``.
    """
    name = getattr(module, "name")
    if not _MODULE_NAME.match(name) or name in _RESERVED:
        raise ValueError(f"module name {name!r}: lowercase letters and digits, not {_RESERVED}")
    prefix = f"{name}_"
    for table in getattr(module, "metadata").tables.values():
        if table.schema is not None or not table.name.startswith(prefix):
            raise ValueError(
                f"module {name}: table {table.fullname!r} must be named {prefix}… with no schema"
            )
    scripts = ScriptDirectory(str(getattr(module, "scripts")))
    for rev in scripts.walk_revisions():
        if not rev.revision.startswith(prefix):
            raise ValueError(f"module {name}: revision {rev.revision!r} must start with {prefix}")
    for plane in getattr(module, "planes"):
        validate_plane(plane)


def modules(plane: str) -> list[History]:
    """The installed extensions' module histories that run on ``plane``: a module after every one
    it names in ``after``, otherwise in load order. A module named in ``after`` that does not run
    on the plane, or a cycle, is refused."""
    from podcast_scraper.extensions import load_extensions

    found = {
        ext.database.name: ext.database
        for ext in load_extensions()
        if ext.database is not None and plane in ext.database.planes
    }
    ordered: list[History] = []
    placed: set[str] = set()

    def place(name: str, chain: tuple[str, ...]) -> None:
        if name in placed:
            return
        if name in chain:
            raise ValueError(f"modules run after each other in a cycle: {' -> '.join(chain)}")
        module = found[name]
        for dep in module.after:
            if dep not in found:
                raise ValueError(f"module {name} runs after {dep}, which is not on plane {plane}")
            place(dep, (*chain, name))
        ordered.append(module_history(module))
        placed.add(name)

    for name in found:
        place(name, ())
    return ordered


def with_kernel(module_metadata: MetaData) -> MetaData:
    """The kernel's tables and a module's in one ``MetaData``, so the module's references to
    ``users`` resolve: for Alembic's ``compare_metadata``, ``sorted_tables`` and test cleanup."""
    combined = MetaData()
    for source in (kernel_metadata, module_metadata):
        for table in source.tables.values():
            table.to_metadata(combined)
    return combined


def plane_histories(plane: str) -> list[History]:
    """Every history that runs on ``plane``: the kernel's, then each module's."""
    return [KERNEL, *modules(plane)]


def head_revision(schema: str | None = None, history: History | None = None) -> str:
    """The head of a shipped history: by default the kernel's (any plane), or ``shared``'s."""
    head = ScriptDirectory.from_config(_config(schema, history=history)).get_current_head()
    assert head is not None
    return head


def current_revision(
    connection: Connection, schema: str, history: History | None = None
) -> str | None:
    """The history's applied revision in the schema, or None when it has never been upgraded."""
    hist = history or _default(schema)
    ctx = MigrationContext.configure(
        connection, opts={"version_table_schema": schema, "version_table": hist.version_table}
    )
    return ctx.get_current_revision()


def readiness(
    connection: Connection, schema: str, history: History | None = None
) -> tuple[str, str | None, str]:
    """``(state, current, head)``: ``at_head``, ``ahead``, ``behind`` or ``unknown`` (ADR-172).

    A revision the code does not ship is ``ahead`` only when this code's head was applied to the
    schema before it (its ``schema_history`` row): a later release migrated past it, expand-only.
    The code cannot walk the newer revisions themselves; they are not in its script directory.
    """
    hist = history or _default(schema)
    head = head_revision(schema, hist)
    current = current_revision(connection, schema, hist)
    if current == head:
        return "at_head", current, head
    if current is None:
        return "behind", current, head
    scripts = ScriptDirectory.from_config(_config(schema, history=hist))
    try:
        scripts.get_revision(current)
    except CommandError:
        if (
            _can_read_history(connection, schema)
            and connection.execute(
                select(schema_history.c.id)
                .where(schema_history.c.revision == head)
                .limit(1)
                .execution_options(schema_translate_map={None: schema})
            ).first()
        ):
            return "ahead", current, head
        return "unknown", current, head
    return "behind", current, head


def _ensure_schema(conn: Connection, schema: str) -> None:
    """Create the schema only when it is missing.

    In prod infra pre-creates the schemas and the migrating role owns them without holding CREATE
    on the database; Postgres checks that privilege before the IF NOT EXISTS short-circuit, so an
    unconditional ``CREATE SCHEMA IF NOT EXISTS`` would fail there.
    """
    present = conn.execute(
        text("SELECT 1 FROM pg_namespace WHERE nspname = :name"), {"name": schema}
    ).first()
    if present is None:
        conn.execute(CreateSchema(schema, if_not_exists=True))


def _engine(url: str) -> Engine:
    return create_engine(normalize_url(url), connect_args=unbounded_connect_args())


def _upgrade_schema(
    engine: Engine, schema: str, revision: str, history: History | None = None
) -> list[str]:
    """One history on one schema, with its ``schema_history`` rows, in one transaction."""
    applied: list[tuple[str, int]] = []
    with engine.begin() as conn:
        _ensure_schema(conn, schema)
    with engine.connect() as conn:
        cfg = _config(schema, conn, history)
        cfg.attributes["applied"] = applied
        command.upgrade(cfg, revision)
        _record_history(conn, schema, applied)
        conn.commit()
    return [rev for rev, _ in applied]


def upgrade_all(url: str, plane: str) -> dict[str, list[str]]:
    """``shared``, the plane's kernel, then each installed module, to head: applied, by history."""
    validate_plane(plane)
    histories = plane_histories(plane)
    engine = _engine(url)
    try:
        applied = {SHARED.name: _upgrade_schema(engine, SHARED_SCHEMA, "head", SHARED)}
        for hist in histories:
            applied[hist.name] = _upgrade_schema(engine, plane, "head", hist)
        return applied
    finally:
        engine.dispose()


def upgrade(url: str, plane: str, revision: str = "head") -> list[str]:
    """Bring ``shared`` to its head, the plane's kernel to ``revision``, and, at ``head``, each
    installed module to its head (schemas created if missing).

    Appends one ``schema_history`` row per revision each run applied (none when already there),
    in the same transaction as that upgrade. Returns the kernel's applied revisions, oldest first;
    :func:`upgrade_all` reports every history's.
    """
    if revision == "head":
        return upgrade_all(url, plane)[KERNEL.name]
    validate_plane(plane)
    engine = _engine(url)
    try:
        _upgrade_schema(engine, SHARED_SCHEMA, "head")
        return _upgrade_schema(engine, plane, revision)
    finally:
        engine.dispose()


def upgrade_shared(url: str) -> list[str]:
    """Bring ``shared`` to its head; the revisions applied, oldest first."""
    engine = _engine(url)
    try:
        return _upgrade_schema(engine, SHARED_SCHEMA, "head")
    finally:
        engine.dispose()


def _record_history(conn: Connection, schema: str, applied: list[tuple[str, int]]) -> None:
    """Insert the run's applied revisions, when the schema has a ``schema_history`` table yet."""
    if not applied or not _has_history_table(conn, schema):
        return
    release, source = build_info.release()
    conn.execute(
        schema_history.insert().execution_options(schema_translate_map={None: schema}),
        [
            {"revision": rev, "release": release, "release_source": source, "duration_ms": ms}
            for rev, ms in applied
        ],
    )


def _has_history_table(conn: Connection, schema: str) -> bool:
    found = conn.execute(
        text("SELECT to_regclass(:name)"), {"name": f'"{schema}".schema_history'}
    ).scalar()
    return found is not None


def _can_read_history(conn: Connection, schema: str) -> bool:
    """The table exists and this role may read it (a deployment that grants the plane roles no
    SELECT on ``shared.schema_history`` reads as unknown, never as a crash)."""
    return bool(
        _has_history_table(conn, schema)
        and conn.execute(
            text("SELECT has_table_privilege(:name, 'SELECT')"),
            {"name": f'"{schema}".schema_history'},
        ).scalar()
    )


def history(conn: Connection, schema: str, limit: int = 5) -> list[dict[str, object]]:
    """The schema's last ``limit`` ``schema_history`` rows, newest first (``[]`` before any)."""
    if not _has_history_table(conn, schema):
        return []
    rows = conn.execute(
        schema_history.select()
        .order_by(schema_history.c.id.desc())
        .limit(limit)
        .execution_options(schema_translate_map={None: schema})
    ).mappings()
    return [dict(r) for r in rows]


def _status(url: str, schema: str) -> tuple[str | None, str]:
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            return current_revision(conn, schema), head_revision(schema)
    finally:
        engine.dispose()


def status(url: str, plane: str) -> tuple[str | None, str]:
    """``(current, head)`` of the kernel's history on one plane."""
    validate_plane(plane)
    return _status(url, plane)


def shared_status(url: str) -> tuple[str | None, str]:
    """``(current, head)`` for the ``shared`` schema."""
    return _status(url, SHARED_SCHEMA)


def module_status(url: str, plane: str) -> list[tuple[str, str | None, str]]:
    """``(module, current, head)`` for each installed module history on ``plane``."""
    validate_plane(plane)
    found = modules(plane)
    if not found:
        return []
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            return [
                (h.name, current_revision(conn, plane, h), head_revision(plane, h)) for h in found
            ]
    finally:
        engine.dispose()


def history_rows(url: str, schema: str, limit: int = 5) -> list[dict[str, object]]:
    """``history`` over a fresh connection to ``url``, with ``applied_at`` in UTC."""
    if schema != SHARED_SCHEMA:
        validate_plane(schema)
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            rows = history(conn, schema, limit)
    finally:
        engine.dispose()
    for row in rows:
        row["applied_at"] = row["applied_at"].astimezone(timezone.utc)  # type: ignore[attr-defined]
    return rows


def _render(schema: str, start: str | None, label: str, history: History | None = None) -> str:
    head = head_revision(schema, history)
    if start == head:
        return f"-- {label} {schema} is already at {head}; nothing to run\n"
    cfg = _config(schema, history=history)
    buffer = io.StringIO()
    cfg.output_buffer = buffer
    command.upgrade(cfg, f"{start or 'base'}:{head}", sql=True)
    return (
        f'-- {label} {schema}: {start or "base"} -> {head}. '
        f'The schema "{schema}" must already exist.\n' + buffer.getvalue()
    )


def upgrade_sql(url: str | None, plane: str, *, from_revision: str | None = None) -> str:
    """The SQL ``upgrade`` would run for ``plane``, as text; nothing is executed.

    ``shared`` first, then the plane's kernel, then each installed module, as ``upgrade`` runs
    them. Each starts from its current revision, read with a read-only query when a URL is
    configured, else from base; ``from_revision`` overrides the kernel's start. Already at head:
    a comment line.
    """
    validate_plane(plane)
    found = modules(plane)
    start: str | None = None
    shared_start: str | None = None
    module_starts: dict[str, str | None] = {}
    if url:
        engine = create_engine(
            normalize_url(url),
            connect_args=unbounded_connect_args(read_only=True),
        )
        try:
            with engine.connect() as conn:
                start = current_revision(conn, plane)
                shared_start = current_revision(conn, SHARED_SCHEMA)
                module_starts = {h.name: current_revision(conn, plane, h) for h in found}
        finally:
            engine.dispose()
    if from_revision is not None:
        start = from_revision
    rendered = _render(SHARED_SCHEMA, shared_start, "schema") + _render(plane, start, "plane")
    for h in found:
        rendered += _render(plane, module_starts.get(h.name), f"module {h.name} on plane", h)
    return rendered
