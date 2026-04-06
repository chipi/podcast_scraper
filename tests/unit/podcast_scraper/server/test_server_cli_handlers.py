"""Unit tests for ``podcast serve`` CLI handlers (RFC-062)."""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.server import cli_handlers

_LOG = logging.getLogger("test_server_cli_handlers")


@pytest.mark.unit
def test_parse_serve_argv_defaults() -> None:
    ns = cli_handlers.parse_serve_argv(["--output-dir", "/tmp/x"])
    assert ns.command == "serve"
    assert ns.output_dir == "/tmp/x"
    assert ns.host == "127.0.0.1"
    assert ns.port == 8000
    assert ns.reload is False
    assert ns.no_static is False


@pytest.mark.unit
def test_run_serve_missing_uvicorn(tmp_path: Path) -> None:
    args = Namespace(
        output_dir=str(tmp_path),
        host="127.0.0.1",
        port=8000,
        reload=False,
        no_static=False,
        command="serve",
    )
    real_import = __import__

    def fake_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, globals_, locals_, tuple(fromlist), level)

    with patch("builtins.__import__", side_effect=fake_import):
        assert cli_handlers.run_serve(args, _LOG) == 1


@pytest.mark.unit
def test_run_serve_bad_output_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    args = Namespace(
        output_dir=str(missing),
        host="127.0.0.1",
        port=8000,
        reload=False,
        no_static=False,
        command="serve",
    )
    uv = ModuleType("uvicorn")
    with patch.dict(sys.modules, {"uvicorn": uv}):
        assert cli_handlers.run_serve(args, _LOG) == 2


@pytest.mark.unit
def test_run_serve_starts_uvicorn_non_reload(tmp_path: Path) -> None:
    args = Namespace(
        output_dir=str(tmp_path),
        host="127.0.0.1",
        port=8000,
        reload=False,
        no_static=True,
        command="serve",
    )
    uv = ModuleType("uvicorn")
    uv.run = MagicMock()
    fake_app = object()
    with patch.dict(sys.modules, {"uvicorn": uv}):
        with patch("podcast_scraper.server.app.create_app", return_value=fake_app) as cap:
            assert cli_handlers.run_serve(args, _LOG) == 0
    cap.assert_called_once()
    uv.run.assert_called_once()
    call_kw = uv.run.call_args.kwargs
    assert call_kw["host"] == "127.0.0.1"
    assert call_kw["port"] == 8000
    assert call_kw["reload"] is False


@pytest.mark.unit
def test_run_serve_reload_uses_factory_string(tmp_path: Path) -> None:
    args = Namespace(
        output_dir=str(tmp_path),
        host="127.0.0.1",
        port=8000,
        reload=True,
        no_static=False,
        command="serve",
    )
    uv = ModuleType("uvicorn")
    uv.run = MagicMock()
    with patch.dict(sys.modules, {"uvicorn": uv}):
        assert cli_handlers.run_serve(args, _LOG) == 0
    uv.run.assert_called_once()
    pos = uv.run.call_args[0]
    assert pos[0] == "podcast_scraper.server.app:create_app_for_uvicorn"
