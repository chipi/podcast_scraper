"""Tests keep their file corpora; the catalog reads rows (r-intents).

The catalog (``server.catalog_cache``) reads the corpus rows only. Hundreds of tests build a corpus
as files, under ``tmp_path`` or ``tests/fixtures``, and then call the catalog or a route over it.
:func:`corpus_rows_patch` makes the catalog import the corpus root it is asked for into the
per-process test database (``tests._database``) first, with the importer the cut-over uses
(``importers.corpus_items.import_corpus``), so those tests run the rows path unchanged.

The rows hold one root at a time, because the catalog reads every row: a read of another root, a
metadata file written or edited since the import, or a write by anything else (the row token moved)
empties the corpus tables and imports again. ``import_corpus`` refuses a changed item rather than
updating it, so it is never asked to.

The patch is session-wide, so module-scoped fixtures that read the catalog get rows too;
:func:`corpus_rows_fixture` clears the catalog caches around each test. Both live outside
``tests/conftest.py`` so the Player's suite, which replaces public's conftest with its own, can
register them too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from tests._database import worker_database_url

_NO_DATABASE = "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
_NAMESPACES = ("app_catalog_rows", "app_slug_index")


def _stamp(root: Path) -> float:
    return max((p.stat().st_mtime for p in root.rglob("*.metadata.json")), default=0.0)


def _empty_corpus_tables(conn: Connection) -> None:
    from podcast_scraper.db.schema import corpus_metadata, corpus_revision

    for table in reversed(corpus_metadata.sorted_tables):
        if table is corpus_revision:
            conn.execute(corpus_revision.update().values(revision=0))
        else:
            conn.execute(table.delete())


def corpus_rows_patch() -> Iterator[None]:
    """Register as a session-scoped autouse fixture."""
    from podcast_scraper.db import engines
    from podcast_scraper.db.importers.corpus_items import import_corpus
    from podcast_scraper.db.settings import CorpusSettings
    from podcast_scraper.server import catalog_cache

    read = catalog_cache._catalog_rows
    token = catalog_cache.catalog_token
    key_of = catalog_cache._key
    loaded: dict[str, Optional[object]] = {"root": None, "stamp": None, "token": None}
    # Its own engine: module-scoped fixtures read the catalog before the app engines are set up.
    writer: list[Engine] = []

    def connect() -> None:
        url = worker_database_url()
        if url is None:
            raise AssertionError(_NO_DATABASE)
        engines.init_corpus_engine(CorpusSettings(url=url))
        if not writer:
            writer.append(create_engine(url))

    def ensure(root: Path) -> None:
        connect()
        key = str(Path(root).resolve())
        stamp = _stamp(Path(root))
        if (key, stamp, token()) == (loaded["root"], loaded["stamp"], loaded["token"]):
            return
        with writer[0].begin() as conn:
            _empty_corpus_tables(conn)
            import_corpus(conn, Path(key))
        loaded.update(root=key, stamp=stamp, token=token())

    def key(root: Path) -> str:
        # ``cached_catalog`` evaluates its key before its token: importing here, the token it
        # stores the rows under is the one after the import, so the next read is a hit.
        ensure(root)
        return key_of(root)

    def rows(root: Path) -> list:
        ensure(root)
        return read(root)

    def current_token() -> float:
        connect()
        if loaded["root"] is not None:
            ensure(Path(str(loaded["root"])))
        return token()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(catalog_cache, "_key", key)
        patch.setattr(catalog_cache, "_catalog_rows", rows)
        patch.setattr(catalog_cache, "catalog_token", current_token)
        yield
    for engine in writer:
        engine.dispose()


def corpus_rows_fixture() -> Iterator[None]:
    """Register as a function-scoped autouse fixture after the database fixture."""
    from podcast_scraper import perf_cache

    # An entry a former test left for the same root can carry a token that reads as current.
    for namespace in _NAMESPACES:
        perf_cache.clear(namespace)
    yield
    for namespace in _NAMESPACES:
        perf_cache.clear(namespace)
