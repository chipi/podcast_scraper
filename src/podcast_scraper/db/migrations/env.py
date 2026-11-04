"""Alembic environment: one kernel history, run once per plane schema (ADR-169).

Driven by ``podcast_scraper.db.migrate``, which passes the plane (and, online, the open
connection) in ``config.attributes``; there is no alembic.ini. Offline mode (``--sql``) renders the
SQL to ``config.output_buffer`` and touches no database.
"""

import time
from typing import Any, Collection, Mapping

from alembic import context
from alembic.runtime.migration import MigrationContext, MigrationInfo

from podcast_scraper.db.schema import metadata

config = context.config
plane = config.attributes["plane"]

_last_step_end = time.monotonic()


def _record_applied(
    ctx: MigrationContext, step: MigrationInfo, heads: Collection[Any], run_args: Mapping[str, Any]
) -> None:
    """Note each upgrade step with its duration; ``migrate.upgrade`` writes them to
    ``schema_history`` once the run is done (the table does not exist yet while 0001 applies)."""
    global _last_step_end
    now = time.monotonic()
    applied = config.attributes.get("applied")
    if applied is not None and step.is_upgrade:
        applied.append((step.up_revision_id, round((now - _last_step_end) * 1000)))
    _last_step_end = now


if context.is_offline_mode():
    context.configure(
        url="postgresql+psycopg://",
        target_metadata=metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=plane,
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    # Online, kernel tables that carry no schema land in the plane (a safety net: the revisions
    # also name the plane on every object, which is what the offline render relies on).
    connection = config.attributes["connection"].execution_options(
        schema_translate_map={None: plane}
    )
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table_schema=plane,
        on_version_apply=_record_applied,
    )
    with context.begin_transaction():
        context.run_migrations()
