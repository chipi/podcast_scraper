"""Unit tests for optional py-spy stdin listener."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from podcast_scraper.monitor.py_spy_listener import start_py_spy_stdin_listener


def test_start_py_spy_stdin_listener_disabled() -> None:
    assert start_py_spy_stdin_listener(output_dir="/tmp/out", enabled=False) is None


@patch("podcast_scraper.monitor.py_spy_listener.sys.stdin")
@patch("podcast_scraper.monitor.py_spy_listener.sys.platform", "linux")
def test_start_py_spy_stdin_listener_skips_without_tty(mock_stdin: MagicMock) -> None:
    mock_stdin.isatty.return_value = False
    assert start_py_spy_stdin_listener(output_dir="/tmp/out", enabled=True) is None


@patch("podcast_scraper.monitor.py_spy_listener.sys.stdin")
@patch("podcast_scraper.monitor.py_spy_listener.sys.platform", "win32")
def test_start_py_spy_stdin_listener_skips_on_windows(mock_stdin: MagicMock) -> None:
    mock_stdin.isatty.return_value = True
    assert start_py_spy_stdin_listener(output_dir="/tmp/out", enabled=True) is None


@patch("podcast_scraper.monitor.py_spy_listener.shutil.which", return_value=None)
@patch("podcast_scraper.monitor.py_spy_listener.sys.stdin")
@patch("podcast_scraper.monitor.py_spy_listener.sys.platform", "linux")
def test_start_py_spy_stdin_listener_no_py_spy_binary(
    mock_stdin: MagicMock,
    _mock_which: object,
) -> None:
    mock_stdin.isatty.return_value = True
    assert start_py_spy_stdin_listener(output_dir="/tmp/out", enabled=True) is None
