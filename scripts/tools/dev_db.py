#!/usr/bin/env python3
"""The dev database ``make serve`` runs the API against, with no docker: ``dev_db.py run -- CMD``.

Starts a PostgreSQL for this checkout, migrates both planes (``db upgrade``, which brings
``shared`` to its head first), loads the dev fixtures once, runs ``CMD`` with
``PODCAST_DATABASE_URL`` set, and stops the database when ``CMD`` exits (Ctrl+C included).

Where the server comes from (``PODCAST_DEV_DB``):

- ``embedded`` (default): the ``embedded-postgres`` wheel from the ``[dev]`` extra, PostgreSQL 18.6.
- ``native``: your own PostgreSQL; ``initdb`` and ``pg_ctl`` on ``PATH`` (or in
  ``PODCAST_DEV_DB_BIN``).
- ``PODCAST_DATABASE_URL`` already set: that database is used as is; migrated, no fixtures, never
  started or stopped here.

Both managed servers keep their data in ``PODCAST_DEV_DB_DIR`` (default ``.devdata/postgres``, which
git ignores), listen on a Unix socket only (no TCP port), and survive restarts. Fixtures
(``PODCAST_DEV_DB_FIXTURES``, default the viewer validation corpus's ``.app``; ``0`` = none) are
imported with ``db import --from-files`` into the player plane once per data dir: a re-import would
report the graph events you add while using the app as mismatches. Delete the data dir to start
over.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from podcast_scraper.db.embedded import EmbeddedPostgres, EmbeddedPostgresMissing

REPO = Path(__file__).resolve().parents[2]
PLANES = ("player", "operator")
DEFAULT_FIXTURES = REPO / "tests/fixtures/viewer-validation-corpus/v3/.app"
# Unix-domain socket paths are limited to ~104 bytes (macOS) / 108 (Linux), socket name included.
_SOCKET_PATH_LIMIT = 90


def _log(msg: str) -> None:
    print(f"[dev-db] {msg}", file=sys.stderr, flush=True)


def _cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "podcast_scraper.cli", "db", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class Server:
    """A managed server: ``start`` returns its URL; ``stop`` undoes only what ``start`` did."""

    def start(self) -> str:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class EmbeddedServer(Server):
    def __init__(self, pgdata: Path) -> None:
        self.pgdata = pgdata
        self._pg = EmbeddedPostgres(pgdata)

    def start(self) -> str:
        try:
            return self._pg.start()
        except EmbeddedPostgresMissing as exc:
            raise SystemExit(
                f"[dev-db] {exc}, or PODCAST_DEV_DB=native to use your own PostgreSQL"
            ) from None

    def stop(self) -> None:
        self._pg.stop()


class NativeServer(Server):
    def __init__(self, pgdata: Path, bin_dir: str | None) -> None:
        self.pgdata = pgdata
        self.bin_dir = bin_dir
        self._started = False

    def _exe(self, name: str) -> str:
        return str(Path(self.bin_dir) / name) if self.bin_dir else name

    def _socket_dir(self) -> Path:
        if len(str(self.pgdata)) <= _SOCKET_PATH_LIMIT:
            return self.pgdata
        digest = hashlib.sha256(str(self.pgdata).encode()).hexdigest()[:10]
        return Path(tempfile.gettempdir()) / f"pgdev-{os.getuid()}-{digest}"

    def _running_socket_dir(self) -> Path | None:
        """The socket dir of a server already running on this data dir, from postmaster.pid."""
        pid_file = self.pgdata / "postmaster.pid"
        status = subprocess.run(
            [self._exe("pg_ctl"), "-D", str(self.pgdata), "status"],
            capture_output=True,
            check=False,
        )
        if status.returncode != 0 or not pid_file.is_file():
            return None
        lines = pid_file.read_text().splitlines()
        return Path(lines[4]) if len(lines) > 4 and lines[4] else None

    def start(self) -> str:
        if not (self.pgdata / "PG_VERSION").is_file():
            self.pgdata.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    self._exe("initdb"),
                    "-D",
                    str(self.pgdata),
                    "-U",
                    "postgres",
                    "-A",
                    "trust",
                    "-E",
                    "UTF8",
                ],
                check=True,
                capture_output=True,
            )
        sock = self._running_socket_dir()
        if sock is None:
            sock = self._socket_dir()
            sock.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    self._exe("pg_ctl"),
                    "-D",
                    str(self.pgdata),
                    "-w",
                    "-l",
                    str(self.pgdata / "server.log"),
                    "-o",
                    f'-h "" -k {sock}',
                    "start",
                ],
                check=True,
                capture_output=True,
            )
            self._started = True
        else:
            _log(f"reusing the server already running on {self.pgdata}")
        return f"postgresql://postgres@/postgres?host={sock}"

    def stop(self) -> None:
        if self._started:
            subprocess.run(
                [self._exe("pg_ctl"), "-D", str(self.pgdata), "-m", "fast", "stop"],
                capture_output=True,
                check=False,
            )


def choose(env: dict[str, str]) -> Server | None:
    """The server to manage, or None when ``PODCAST_DATABASE_URL`` names one already."""
    if env.get("PODCAST_DATABASE_URL", "").strip():
        return None
    base = Path(env.get("PODCAST_DEV_DB_DIR") or REPO / ".devdata/postgres")
    mode = env.get("PODCAST_DEV_DB", "embedded").strip() or "embedded"
    if mode == "embedded":
        return EmbeddedServer(base / "pgdata")
    if mode == "native":
        return NativeServer(base / "pgdata-native", env.get("PODCAST_DEV_DB_BIN") or None)
    raise SystemExit(f"[dev-db] PODCAST_DEV_DB must be embedded or native, not {mode!r}")


def fixtures_dir(env: dict[str, str]) -> Path | None:
    raw = env.get("PODCAST_DEV_DB_FIXTURES")
    if raw is None:
        return DEFAULT_FIXTURES
    if raw.strip() in ("", "0"):
        return None
    return Path(raw)


def prepare(url: str, env: dict[str, str], marker: Path | None, fixtures: Path | None) -> int:
    """Migrate both planes, then import the fixtures once (``marker`` records it). 0 = ready."""
    for plane in PLANES:
        r = _cli(["upgrade", "--plane", plane], env)
        if r.returncode != 0:
            _log(f"db upgrade --plane {plane} failed (exit {r.returncode}):\n{r.stdout}{r.stderr}")
            return r.returncode
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("plane ", "applied:"))]
        _log("; ".join(lines) or f"{plane}: migrated")
    if marker is None or fixtures is None or marker.exists():
        return 0
    if not fixtures.is_dir():
        _log(f"fixtures dir {fixtures} does not exist")
        return 2
    r = _cli(["import", "--from-files", str(fixtures), "--plane", "player"], env)
    if r.returncode != 0:
        _log(f"fixture import from {fixtures} failed (exit {r.returncode}):\n{r.stdout}{r.stderr}")
        return r.returncode
    for line in r.stdout.splitlines():
        if line.startswith("store ") and ": read 0," not in line:
            _log(line)
    marker.write_text(f"{fixtures}\n")
    _log(f"fixtures loaded from {fixtures} (once; delete {marker.parent} to start over)")
    return 0


def run(
    cmd: list[str], env: dict[str, str], popen: Callable[..., subprocess.Popen] = subprocess.Popen
) -> int:
    server = choose(env)
    child_env = dict(env)
    if server is None:
        url, marker = env["PODCAST_DATABASE_URL"], None
        _log("using PODCAST_DATABASE_URL as is (not started or stopped here, no fixtures)")
    else:
        url = server.start()
        marker = server.pgdata.parent / f"{server.pgdata.name}.fixtures-loaded"
        _log(f"{type(server).__name__.removesuffix('Server').lower()} PostgreSQL at {url}")
    child_env["PODCAST_DATABASE_URL"] = url
    try:
        rc = prepare(url, child_env, marker, fixtures_dir(env))
        if rc != 0:
            return rc
        proc = popen(cmd, env=child_env)

        def forward(signum: int, _frame: object) -> None:
            proc.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                # Ctrl+C reaches the whole process group: CMD got it too. Wait for it to finish.
                continue
    finally:
        if server is not None:
            server.stop()
            _log("stopped")


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "run" or argv[1] != "--":
        print("usage: dev_db.py run -- CMD [ARGS...]", file=sys.stderr)
        return 2
    return run(argv[2:], dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
