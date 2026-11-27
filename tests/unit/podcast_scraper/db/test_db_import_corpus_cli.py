"""``podcast_scraper db import-corpus``: a file corpus into the corpus schema, one transaction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine, func, select

from podcast_scraper.db.cli_handlers import parse_db_argv, run_db_cli
from podcast_scraper.db.schema import content_items, search_chunks
from tests._corpus_rows import _empty_corpus_tables
from tests._database import worker_database_url

LOG = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _database_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = worker_database_url()
    if url is None:
        pytest.fail("set PODCAST_TEST_DATABASE_URL (make test-db-up)")
    monkeypatch.setenv("PODCAST_DATABASE_URL", url)
    yield url
    # The command commits; the worker's database is shared with every later test.
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            _empty_corpus_tables(conn)
    finally:
        engine.dispose()


def _corpus(root: Path) -> Path:
    meta = root / "metadata"
    meta.mkdir(parents=True)
    for stem, title in (("a", "First"), ("b", None)):
        feed = {"feed_id": "show", "title": "A show"} if title else {"feed_id": "show"}
        doc = {
            "feed": feed,
            "episode": {"episode_id": stem, "title": title or "Second"},
        }
        (meta / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (meta / "c.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": "untitled"}, "episode": {"episode_id": "c", "title": "C"}}),
        encoding="utf-8",
    )
    return root


def _items(url: str) -> int:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(content_items)).scalar_one())
    finally:
        engine.dispose()


def test_it_writes_the_corpus_and_prints_what_it_refused(
    tmp_path: Path, _database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _items(_database_url)
    code = run_db_cli(parse_db_argv(["import-corpus", "--root", str(_corpus(tmp_path))]), LOG)
    out = capsys.readouterr().out
    assert code == 0
    assert "wrote sources 1, items 2," in out
    assert "refused: 1" in out and "REFUSED metadata/c.metadata.json" in out
    assert _items(_database_url) == before + 2


def test_a_dry_run_writes_nothing(
    tmp_path: Path, _database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _items(_database_url)
    args = parse_db_argv(["import-corpus", "--root", str(_corpus(tmp_path)), "--dry-run"])
    assert run_db_cli(args, LOG) == 0
    assert "would write sources 1, items 2," in capsys.readouterr().out
    assert _items(_database_url) == before


def test_a_missing_root_is_an_error(tmp_path: Path) -> None:
    assert run_db_cli(parse_db_argv(["import-corpus", "--root", str(tmp_path / "no")]), LOG) == 1


def _chunks(url: str) -> int:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(search_chunks)).scalar_one())
    finally:
        engine.dispose()


def test_it_writes_the_search_chunks_and_a_dry_run_leaves_none(
    tmp_path: Path, _database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.unit.podcast_scraper.db.test_search_chunks_writer import _episode

    _episode(tmp_path)
    before = _chunks(_database_url)
    args = parse_db_argv(["import-corpus", "--root", str(tmp_path), "--dry-run"])
    assert run_db_cli(args, LOG) == 0
    assert "search chunks: written 1" in capsys.readouterr().out
    assert _chunks(_database_url) == before
    assert run_db_cli(parse_db_argv(["import-corpus", "--root", str(tmp_path)]), LOG) == 0
    assert "search chunks: written 1" in capsys.readouterr().out
    assert _chunks(_database_url) > before
