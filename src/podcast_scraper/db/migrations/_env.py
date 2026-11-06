"""The Alembic environment both histories run (ADR-169): the kernel's, once per plane schema, and
the ``shared`` schema's.

Driven by ``podcast_scraper.db.migrate``, which passes the schema (and, online, the open
connection) in ``config.attributes``; there is no alembic.ini. Offline mode (``--sql``) renders the
SQL to ``config.output_buffer`` and touches no database.
"""

from __future__ import annotations

import time
from typing import Any, Collection, Mapping

from alembic import context
from alembic.runtime.migration import MigrationContext, MigrationInfo
from sqlalchemy import MetaData


def run(metadata: MetaData) -> None:
    """Run the history in ``config.attributes["plane"]`` (the schema), online or offline."""
    config = context.config
    schema = config.attributes["plane"]
    last_step_end = [time.monotonic()]

    def record_applied(
        ctx: MigrationContext,
        step: MigrationInfo,
        heads: Collection[Any],
        run_args: Mapping[str, Any],
    ) -> None:
        """Note each upgrade step with its duration; ``migrate.upgrade`` writes them to
        ``schema_history`` once the run is done (the table does not exist yet while the first
        revision applies)."""
        now = time.monotonic()
        applied = config.attributes.get("applied")
        if applied is not None and step.is_upgrade:
            applied.append((step.up_revision_id, round((now - last_step_end[0]) * 1000)))
        last_step_end[0] = now

    if context.is_offline_mode():
        context.configure(
            url="postgresql+psycopg://",
            target_metadata=metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            version_table_schema=schema,
        )
        with context.begin_transaction():
            context.run_migrations()
        return
    # Online, tables that carry no schema land in the given one (a safety net: the revisions also
    # name the schema on every object, which is what the offline render relies on).
    connection = config.attributes["connection"].execution_options(
        schema_translate_map={None: schema}
    )
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table_schema=schema,
        on_version_apply=record_applied,
    )
    with context.begin_transaction():
        context.run_migrations()
