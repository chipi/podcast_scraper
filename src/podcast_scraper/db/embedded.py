"""A PostgreSQL from the ``embedded-postgres`` wheel (``[dev]`` extra), no docker.

Two users: the dev database ``make serve`` runs against (``scripts/tools/dev_db.py``) and the test
database a local pytest run starts when none is given (``tests/conftest.py``). It listens on a
Unix socket only, never a TCP port. Not used in deployments.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

MISSING_HINT = "embedded-postgres is not installed: pip install -e '.[dev]'"


class EmbeddedPostgresMissing(RuntimeError):
    """The ``embedded-postgres`` wheel is not importable (a venv without the ``[dev]`` extra)."""


class EmbeddedPostgres:
    """One server on ``pgdata``: ``start`` returns its URL, ``stop`` stops it.

    ``settings`` are written with ``ALTER SYSTEM`` and need a restart, so a start that has them
    starts the server twice. ``delete_on_stop`` removes ``pgdata`` when the server stops, including
    at interpreter exit.
    """

    def __init__(
        self,
        pgdata: Path,
        *,
        settings: Mapping[str, str] | None = None,
        delete_on_stop: bool = False,
    ) -> None:
        self.pgdata = pgdata
        self.settings = dict(settings or {})
        self.delete_on_stop = delete_on_stop
        self._srv: Any = None

    def start(self) -> str:
        try:
            import embedded_postgres
        except ImportError:
            raise EmbeddedPostgresMissing(MISSING_HINT) from None
        self.pgdata.mkdir(parents=True, exist_ok=True)
        if self.settings:
            srv = embedded_postgres.get_server(self.pgdata, cleanup_mode="stop")
            srv.psql("".join(f"ALTER SYSTEM SET {k} = '{v}';\n" for k, v in self.settings.items()))
            srv.cleanup()
        mode = "delete" if self.delete_on_stop else "stop"
        self._srv = embedded_postgres.get_server(self.pgdata, cleanup_mode=mode)
        return str(self._srv.get_uri())

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.cleanup()
            self._srv = None
