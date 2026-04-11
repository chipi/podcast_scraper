"""Unit tests for memray CLI/service re-exec helpers."""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

import pytest

from podcast_scraper.monitor.memray_util import (
    maybe_reexec_memray_cli,
    maybe_reexec_memray_service,
    MEMRAY_ACTIVE_ENV,
    resolve_memray_output_cli,
    resolve_memray_output_service,
)


def test_resolve_memray_output_cli_explicit() -> None:
    args = argparse.Namespace(memray_output="/tmp/custom.bin", output_dir="/out")
    p = resolve_memray_output_cli(args, ["https://a.com/f.xml"])
    assert str(p).endswith("custom.bin")


def test_resolve_memray_output_service_explicit() -> None:
    p = resolve_memray_output_service("/data", "/tmp/svc.bin")
    assert str(p).endswith("svc.bin")


@patch.dict(os.environ, {MEMRAY_ACTIVE_ENV: "1"}, clear=False)
def test_maybe_reexec_memray_cli_skips_when_active() -> None:
    args = argparse.Namespace(memray=True, memray_output=None, output_dir=None)
    maybe_reexec_memray_cli(args, ["https://x"], ["https://x"])


def test_maybe_reexec_memray_cli_skips_when_disabled() -> None:
    args = argparse.Namespace(memray=False, memray_output=None, output_dir=None)
    maybe_reexec_memray_cli(args, [], [])


@patch("podcast_scraper.monitor.memray_util.shutil.which", return_value=None)
def test_maybe_reexec_memray_cli_exits_when_missing_binary(_mock_which: object) -> None:
    args = argparse.Namespace(memray=True, memray_output=None, output_dir="/tmp/o")
    with pytest.raises(SystemExit) as exc:
        maybe_reexec_memray_cli(args, ["--rss", "https://a.com/f.xml"], ["https://a.com/f.xml"])
    assert exc.value.code == 1


@patch.dict(os.environ, {MEMRAY_ACTIVE_ENV: "1"}, clear=False)
def test_maybe_reexec_memray_service_skips_when_active() -> None:
    assert maybe_reexec_memray_service(memray=True, output_dir="/o", memray_output=None) is None


def test_maybe_reexec_memray_service_skips_when_disabled() -> None:
    assert maybe_reexec_memray_service(memray=False, output_dir="/o", memray_output=None) is None


@patch("podcast_scraper.monitor.memray_util.shutil.which", return_value=None)
def test_maybe_reexec_memray_service_error_when_missing_binary(_mock_which: object) -> None:
    err = maybe_reexec_memray_service(memray=True, output_dir="/o", memray_output=None)
    assert err is not None
    assert "memray" in err.lower()


@patch("podcast_scraper.monitor.memray_util.os.execvpe")
@patch("podcast_scraper.monitor.memray_util.shutil.which", return_value="/fake/memray")
def test_maybe_reexec_memray_cli_invokes_execvpe(_mock_which: object, mock_exec: object) -> None:
    args = argparse.Namespace(
        memray=True,
        memray_output="/tmp/m.bin",
        output_dir="/ignored",
    )
    maybe_reexec_memray_cli(args, ["--dry-run"], ["https://example.com/feed.xml"])
    mock_exec.assert_called_once()
    cargs = mock_exec.call_args.args or mock_exec.call_args[0]
    _bin, cmd, _env = cargs
    assert _bin == "/fake/memray"
    assert cmd[0] == "/fake/memray"
    assert cmd[1] == "run"
    assert cmd[2] == "-o"
    assert str(cmd[3]).endswith("m.bin")
    assert cmd[4] == "-f"
    assert cmd[5] == "-m"
    assert cmd[6] == "podcast_scraper.cli"
    assert list(cmd[7:]) == ["--dry-run"]


def test_maybe_reexec_memray_service_invokes_execvpe() -> None:
    with patch("sys.argv", ["/fake/service.py", "--config", "/app/c.yaml"]):
        with patch("podcast_scraper.monitor.memray_util.shutil.which", return_value="/fake/memray"):
            with patch("podcast_scraper.monitor.memray_util.os.execvpe") as mock_exec:
                maybe_reexec_memray_service(
                    memray=True, output_dir="/out", memray_output="/tmp/x.bin"
                )
                mock_exec.assert_called_once()
                cargs = mock_exec.call_args.args or mock_exec.call_args[0]
                _bin, cmd, _env = cargs
                assert _bin == "/fake/memray"
                assert "podcast_scraper.service" in cmd
                assert "--config" in cmd
                assert "/app/c.yaml" in cmd
