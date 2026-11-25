"""The database layer (ADR-169): settings, the per-process engine registry, kernel tables.

App state lives in Postgres, reached through SQLAlchemy Core. One database, one schema per API
plane; the kernel tables are defined without a schema and mapped onto the process's plane schema
(``schema_translate_map``), so the same history runs once per plane. Corpus reads go through their
own ``corpus_ro`` engine (CRFC-005 §7).
"""

from podcast_scraper.db.engines import (
    corpus_ro_engine,
    database_available,
    init_corpus_engine,
    init_engines,
    plane_schema,
    read_connection,
    read_with_retry,
    reset_engines,
    ro_engine,
    rw_engine,
)
from podcast_scraper.db.settings import (
    CorpusSettings,
    DatabaseNotConfigured,
    DatabaseSettings,
    PLANES,
)

__all__ = [
    "CorpusSettings",
    "DatabaseNotConfigured",
    "DatabaseSettings",
    "PLANES",
    "corpus_ro_engine",
    "database_available",
    "init_corpus_engine",
    "init_engines",
    "plane_schema",
    "read_connection",
    "read_with_retry",
    "reset_engines",
    "ro_engine",
    "rw_engine",
]
