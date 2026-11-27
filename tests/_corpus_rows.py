"""Tests keep their file corpora; the catalog reads rows (r-intents).

The catalog (``server.catalog_cache``) reads the corpus rows only. Hundreds of tests build a corpus
as files, under ``tmp_path`` or ``tests/fixtures``, and then call the catalog or a route over it.
:func:`corpus_rows_patch` makes the catalog import the corpus root it is asked for into the
per-process test database (``tests._database``) first, with the importer the cut-over uses
(``importers.corpus_items.import_corpus``), so those tests run the rows path unchanged.

The rows hold one root at a time, because the catalog reads every row: a read of another root, a
file under it written, edited or deleted since the import (its path, size and mtime), or a write by
anything else (the row token moved) empties the corpus tables, but the job and lease rows, and
imports again. ``import_corpus`` refuses a changed item rather than
updating it, so it is never asked to.

The patch is session-wide, so module-scoped fixtures that read the catalog get rows too;
:func:`corpus_rows_fixture` clears the catalog caches around each test. The change check costs
token queries the API never makes: a test that counts statements wraps its measured request in
:func:`corpus_rows_settled`, which puts the API's own catalog functions back while it is open.
All three live outside ``tests/conftest.py`` so the Player's suite, which replaces public's
conftest with its own, can register and use them too.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from tests._database import worker_database_url

_NO_DATABASE = "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
_NAMESPACES = ("app_catalog_rows", "app_slug_index")
#: The API's own catalog functions, while :func:`corpus_rows_patch` replaces them.
_REAL: dict[str, Any] = {}


#: Rows ``import_corpus`` never writes; a test's own job and lease rows outlive a re-import.
_KEPT = ("jobs", "job_leases")
#: One check-and-import at a time: a route's worker threads and the perf cache's rebuild thread
#: read the catalog at once, and two imports interleaved commit one's rows between the other's
#: deletes (a ForeignKeyViolation on ``cluster_ids``, Player's suite, 2026-11-27).
_ENSURE = threading.Lock()


def _stamp(root: Path) -> int:
    """Every file under ``root``, by path, size and mtime: the importer reads more than metadata
    (transcripts, GI and KG, artwork), and a delete or a same-mtime edit moves no newest mtime."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return hash(tuple(files))


def _empty_corpus_tables(conn: Connection) -> None:
    from podcast_scraper.db.schema import corpus_metadata, corpus_revision

    for table in reversed(corpus_metadata.sorted_tables):
        if table.name in _KEPT:
            continue
        if table is corpus_revision:
            conn.execute(corpus_revision.update().values(revision=0))
        else:
            conn.execute(table.delete())


def corpus_rows_patch() -> Iterator[None]:
    """Register as a session-scoped autouse fixture."""
    from podcast_scraper.db import engines
    from podcast_scraper.db.importers.corpus_items import import_corpus
    from podcast_scraper.db.settings import CorpusSettings
    from podcast_scraper.search import corpus_search
    from podcast_scraper.search.two_tier_indexer import import_search_chunks
    from podcast_scraper.server import catalog_cache

    read = catalog_cache._catalog_rows
    # The token without its hold: a test reads the rows it just imported.
    token = catalog_cache._read_token
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
        with _ENSURE:
            connect()
            key = str(Path(root).resolve())
            stamp = _stamp(Path(root))
            if (key, stamp, token()) == (loaded["root"], loaded["stamp"], loaded["token"]):
                return
            with writer[0].begin() as conn:
                _empty_corpus_tables(conn)
                import_corpus(conn, Path(key))
                import_search_chunks(conn, Path(key))
            loaded.update(root=key, stamp=stamp, token=token())

    def key(root: Path) -> str:
        # ``cached_catalog`` evaluates its key before its token: importing here, the token it
        # stores the rows under is the one after the import, so the next read is a hit.
        ensure(root)
        return key_of(root)

    def rows(root: Path) -> list:
        ensure(root)
        return read(root)

    hydrate = corpus_search._enrich_lift_and_slice

    def search_rows(filtered: Any, output_dir: Path, *args: Any, **kwargs: Any) -> Any:
        # A search hydrates its transcript hits from search_chunks: the corpus it searches is
        # imported first, as a catalog read imports it.
        ensure(Path(output_dir))
        return hydrate(filtered, output_dir, *args, **kwargs)

    def current_token() -> float:
        connect()
        if loaded["root"] is not None:
            ensure(Path(str(loaded["root"])))
        return token()

    _REAL.update(_key=key_of, _catalog_rows=read, catalog_token=catalog_cache.catalog_token)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(catalog_cache, "_key", key)
        patch.setattr(catalog_cache, "_catalog_rows", rows)
        patch.setattr(catalog_cache, "catalog_token", current_token)
        patch.setattr(corpus_search, "_enrich_lift_and_slice", search_rows)
        yield
    _REAL.clear()
    for engine in writer:
        engine.dispose()


@contextmanager
def corpus_rows_settled() -> Iterator[None]:
    """While open, the catalog runs the API's own functions: no import and no change check, and
    the token's hold starts empty, so a request reads the token once as a cold API would. The
    corpus must already be imported (a read before the block)."""
    from podcast_scraper.server import catalog_cache

    if not _REAL:
        raise RuntimeError("corpus_rows_settled needs corpus_rows_patch active")
    patched = {name: getattr(catalog_cache, name) for name in _REAL}
    for name, real in _REAL.items():
        setattr(catalog_cache, name, real)
    catalog_cache._held_token._held = None
    try:
        yield
    finally:
        for name, value in patched.items():
            setattr(catalog_cache, name, value)


def corpus_rows_fixture() -> Iterator[None]:
    """Register as a function-scoped autouse fixture after the database fixture."""
    from podcast_scraper import perf_cache

    # An entry a former test left for the same root can carry a token that reads as current.
    for namespace in _NAMESPACES:
        perf_cache.clear(namespace)
    yield
    for namespace in _NAMESPACES:
        perf_cache.clear(namespace)


def every_hit_has_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a search test over an index built from synthetic segments, with no corpus to import:
    each transcript hit gets a served row, so the hit is kept and the test sees what the index
    returned. Search hydrates from ``search_chunks`` only and drops a hit with no row; that is
    ``tests/unit/podcast_scraper/search/test_search_hydrates_from_rows.py``'s to pin."""
    from podcast_scraper.search import corpus_search

    def rows(ids: list) -> list:
        return [
            {
                "natural_id": i,
                "served": 1,
                "withheld": False,
                "text": "",
                "start_ms": None,
                "end_ms": None,
                "title": None,
                "source_title": None,
                "speakers": [],
            }
            for i in ids
        ]

    monkeypatch.setattr(corpus_search, "_chunk_rows", rows)
