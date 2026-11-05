"""The database layer (ADR-169): settings, the per-process engine registry, kernel tables.

App state lives in Postgres, reached through SQLAlchemy Core. One database, one schema per API
plane; the kernel tables are defined without a schema and mapped onto the process's plane schema
(``schema_translate_map``), so the same history runs once per plane.
"""

from podcast_scraper.db.engines import (
    init_engines,
    plane_schema,
    read_with_retry,
    reset_engines,
    ro_engine,
    rw_engine,
)
from podcast_scraper.db.settings import DatabaseNotConfigured, DatabaseSettings, PLANES

__all__ = [
    "DatabaseNotConfigured",
    "DatabaseSettings",
    "PLANES",
    "init_engines",
    "plane_schema",
    "read_with_retry",
    "reset_engines",
    "ro_engine",
    "rw_engine",
]
