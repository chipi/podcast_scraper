"""Alembic environment: one kernel history, run once per plane schema (ADR-169).

Driven by ``podcast_scraper.db.migrate``, which passes the plane (and, online, the open
connection) in ``config.attributes``; there is no alembic.ini. Offline mode (``--sql``) renders the
SQL to ``config.output_buffer`` and touches no database.
"""

from alembic import context

from podcast_scraper.db.schema import metadata

config = context.config
plane = config.attributes["plane"]

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
    )
    with context.begin_transaction():
        context.run_migrations()
