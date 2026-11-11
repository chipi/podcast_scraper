"""Where a local test run gets its database when ``PODCAST_TEST_DATABASE_URL`` is unset.

Order: an exported URL, nothing in CI, this checkout's ``make test-db-up`` container, an embedded
PostgreSQL started for the run, else one clear error. Starting a real server is covered by running
the suite itself with the variable unset.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from podcast_scraper.db import embedded
from tests import _database


class FakeServer:
    def __init__(self, pgdata: Path, *, settings=None, delete_on_stop=False) -> None:
        self.pgdata = pgdata
        self.settings = settings
        self.delete_on_stop = delete_on_stop
        self.stopped = 0

    def start(self) -> str:
        return f"postgresql://postgres@/postgres?host={self.pgdata}"

    def stop(self) -> None:
        self.stopped += 1


@pytest.fixture
def unset(monkeypatch):
    monkeypatch.delenv(_database.ENV_TEST_URL, raising=False)
    monkeypatch.delenv("CI", raising=False)
    yield monkeypatch
    _database._session_server.clear()


def test_an_exported_url_is_used_as_is(unset) -> None:
    unset.setenv(_database.ENV_TEST_URL, "postgresql://x@h/postgres")
    unset.setattr(_database, "_container_url", lambda: pytest.fail("must not look"))
    assert _database.provide_session_database() is None
    assert _database.os.environ[_database.ENV_TEST_URL] == "postgresql://x@h/postgres"


def test_ci_is_never_provided_for(unset) -> None:
    unset.setenv("CI", "true")
    unset.setattr(_database, "_container_url", lambda: pytest.fail("must not look"))
    assert _database.provide_session_database() is None
    assert _database.ENV_TEST_URL not in _database.os.environ


def test_this_checkouts_container_comes_first(unset) -> None:
    unset.setattr(_database, "_container_url", lambda: "postgresql://p:p@127.0.0.1:5999/postgres")
    unset.setattr(_database, "EmbeddedPostgres", lambda *a, **k: pytest.fail("must not start"))
    assert "container" in (_database.provide_session_database() or "")
    assert _database.os.environ[_database.ENV_TEST_URL].endswith(":5999/postgres")


def test_without_a_container_an_embedded_server_is_started_and_released(unset) -> None:
    unset.setattr(_database, "_container_url", lambda: "")
    unset.setattr(_database, "EmbeddedPostgres", FakeServer)
    assert "embedded" in (_database.provide_session_database() or "")
    (server,) = _database._session_server
    assert server.delete_on_stop and server.settings["fsync"] == "off"
    assert _database.os.environ[_database.ENV_TEST_URL] == server.start()
    _database.release_session_database()
    assert server.stopped == 1 and not _database._session_server
    _database.shutil.rmtree(server.pgdata, ignore_errors=True)  # the real server deletes its own


def test_neither_is_one_error_saying_how_to_get_one(unset) -> None:
    unset.setattr(_database, "_container_url", lambda: "")
    real_import = builtins.__import__

    def no_embedded(name, *a, **kw):
        if name == "embedded_postgres":
            raise ImportError(name)
        return real_import(name, *a, **kw)

    unset.setattr(builtins, "__import__", no_embedded)
    with pytest.raises(embedded.EmbeddedPostgresMissing) as exc:
        _database.provide_session_database()
    assert str(exc.value) == _database.NO_DATABASE
    assert "pip install -e '.[dev]'" in str(exc.value) and "make test-db-up" in str(exc.value)
    assert _database.ENV_TEST_URL not in _database.os.environ


def test_container_url_is_empty_when_the_script_fails(unset) -> None:
    unset.setattr(
        _database.subprocess,
        "run",
        lambda *a, **k: _database.subprocess.CompletedProcess(a, 1, "", "no such container"),
    )
    assert _database._container_url() == ""
