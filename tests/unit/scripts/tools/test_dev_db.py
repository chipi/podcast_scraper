"""``scripts/tools/dev_db.py`` — the dev database ``make serve-api`` runs the API against.

The decisions it makes, with the server, the CLI and the child process faked: which server it
manages (embedded by default, native on request, none when ``PODCAST_DATABASE_URL`` is set), that
fixtures load once per data dir, and that a server it started is stopped however the run ends.
Starting a real embedded / native PostgreSQL is covered by running ``make serve-api`` itself.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "dev_db.py"


def _load():
    spec = importlib.util.spec_from_file_location("dev_db", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dev_db: Any = _load()


class FakeServer(dev_db.Server):
    def __init__(self, pgdata: Path) -> None:
        self.pgdata = pgdata
        self.started = self.stopped = 0

    def start(self) -> str:
        self.started += 1
        return "postgresql://postgres@/postgres?host=/sock"

    def stop(self) -> None:
        self.stopped += 1


class FakeProc:
    def __init__(self, rc: int) -> None:
        self.rc = rc
        self.signals: list[int] = []

    def wait(self) -> int:
        return self.rc

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)


def _cli_ok(calls: list[list[str]]):
    def fake(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert env["PODCAST_DATABASE_URL"]
        out = "plane x: head (head head)\napplied: nothing\n"
        if args[0] == "import":
            out = "store users: read 10, inserted 10, already present 0, mismatched 0\n"
        return subprocess.CompletedProcess(args, 0, out, "")

    return fake


def test_embedded_is_the_default_and_native_is_chosen_by_env(tmp_path: Path) -> None:
    base = {"PODCAST_DEV_DB_DIR": str(tmp_path)}
    assert isinstance(dev_db.choose(base), dev_db.EmbeddedServer)
    native = dev_db.choose({**base, "PODCAST_DEV_DB": "native", "PODCAST_DEV_DB_BIN": "/pg/bin"})
    assert isinstance(native, dev_db.NativeServer) and native.bin_dir == "/pg/bin"
    # Separate data dirs: one cluster is never opened by the other PostgreSQL build.
    assert native.pgdata != dev_db.choose(base).pgdata
    with pytest.raises(SystemExit):
        dev_db.choose({**base, "PODCAST_DEV_DB": "docker"})


def test_a_set_database_url_is_used_as_is(tmp_path: Path, monkeypatch) -> None:
    assert dev_db.choose({"PODCAST_DATABASE_URL": "postgresql://h/db"}) is None
    calls: list[list[str]] = []
    monkeypatch.setattr(dev_db, "_cli", _cli_ok(calls))
    seen: dict[str, str] = {}

    def popen(cmd, env):
        seen.update(env)
        return FakeProc(0)

    rc = dev_db.run(["serve"], {"PODCAST_DATABASE_URL": "postgresql://h/db"}, popen=popen)
    assert rc == 0 and seen["PODCAST_DATABASE_URL"] == "postgresql://h/db"
    # Both planes migrated, and no fixtures loaded into a database it does not own.
    assert calls == [["upgrade", "--plane", "player"], ["upgrade", "--plane", "operator"]]


def test_fixtures_load_once_per_data_dir(tmp_path: Path, monkeypatch) -> None:
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    server = FakeServer(tmp_path / "pgdata")
    monkeypatch.setattr(dev_db, "choose", lambda env: server)
    calls: list[list[str]] = []
    monkeypatch.setattr(dev_db, "_cli", _cli_ok(calls))
    env = {"PODCAST_DEV_DB_FIXTURES": str(fixtures)}
    for _ in range(2):
        assert dev_db.run(["serve"], env, popen=lambda cmd, env: FakeProc(0)) == 0
    imports = [c for c in calls if c[0] == "import"]
    assert imports == [["import", "--from-files", str(fixtures), "--plane", "player"]]
    assert (tmp_path / "pgdata.fixtures-loaded").is_file()


def test_no_fixtures_when_turned_off(tmp_path: Path) -> None:
    assert dev_db.fixtures_dir({"PODCAST_DEV_DB_FIXTURES": "0"}) is None
    assert dev_db.fixtures_dir({}) == dev_db.DEFAULT_FIXTURES
    assert dev_db.DEFAULT_FIXTURES.is_dir()


@pytest.mark.parametrize("child_rc", [0, 3])
def test_the_server_is_stopped_when_the_api_exits(tmp_path: Path, monkeypatch, child_rc) -> None:
    server = FakeServer(tmp_path / "pgdata")
    monkeypatch.setattr(dev_db, "choose", lambda env: server)
    monkeypatch.setattr(dev_db, "_cli", _cli_ok([]))
    env = {"PODCAST_DEV_DB_FIXTURES": "0"}
    assert dev_db.run(["serve"], env, popen=lambda cmd, env: FakeProc(child_rc)) == child_rc
    assert (server.started, server.stopped) == (1, 1)


def test_a_failed_migration_stops_the_server_and_never_starts_the_api(
    tmp_path: Path, monkeypatch
) -> None:
    server = FakeServer(tmp_path / "pgdata")
    monkeypatch.setattr(dev_db, "choose", lambda env: server)
    monkeypatch.setattr(
        dev_db, "_cli", lambda args, env: subprocess.CompletedProcess(args, 1, "", "boom")
    )

    def popen(cmd, env):
        raise AssertionError("the API must not start on an unmigrated database")

    assert dev_db.run(["serve"], {}, popen=popen) == 1
    assert server.stopped == 1


def test_native_keeps_the_socket_path_short(tmp_path: Path) -> None:
    short = dev_db.NativeServer(tmp_path / "d", None)
    assert short._socket_dir() == short.pgdata
    deep = dev_db.NativeServer(tmp_path / ("x" * 120), None)
    assert len(str(deep._socket_dir())) < 60
    # Stable across runs (a second `make serve` finds the same directory).
    assert deep._socket_dir() == dev_db.NativeServer(deep.pgdata, None)._socket_dir()


def test_usage() -> None:
    assert dev_db.main([]) == 2
    assert dev_db.main(["run", "serve"]) == 2


def test_embedded_missing_says_how_to_get_it(tmp_path: Path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_embedded(name, *a, **kw):
        if name == "embedded_postgres":
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_embedded)
    with pytest.raises(SystemExit) as exc:
        dev_db.EmbeddedServer(tmp_path / "pgdata").start()
    assert "pip install -e '.[dev]'" in str(exc.value) and "PODCAST_DEV_DB=native" in str(exc.value)


def test_serve_api_runs_through_dev_db() -> None:
    makefile = (_SCRIPT.parents[2] / "Makefile").read_text()
    assert "DEV_DB_RUN = $(PYTHON) scripts/tools/dev_db.py run --" in makefile
    recipe = makefile.split("\nserve-api:", 1)[1].split("\nserve-ui:", 1)[0]
    assert "$(DEV_DB_RUN)" in recipe and "cli serve" in recipe
    assert "docker" not in recipe.lower()
