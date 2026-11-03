"""Alembic environment: one kernel history, run once per plane schema (ADR-169).

Driven by ``podcast_scraper.db.migrate``, which passes the open connection and the plane in
``config.attributes``; there is no offline mode and no alembic.ini.
"""

from alembic import context

from podcast_scraper.db.schema import metadata

config = context.config
connection = config.attributes["connection"]
plane = config.attributes["plane"]

# Kernel tables carry no schema; this maps them (and nothing else) onto the plane.
connection = connection.execution_options(schema_translate_map={None: plane})
context.configure(
    connection=connection,
    target_metadata=metadata,
    version_table_schema=plane,
)
with context.begin_transaction():
    context.run_migrations()
