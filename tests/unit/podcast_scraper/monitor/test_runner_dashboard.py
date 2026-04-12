"""Tests for monitor stderr vs file logging (RFC-065 / profile freeze)."""

from __future__ import annotations

import io
import sys

import pytest

from podcast_scraper.monitor.runner import MONITOR_FILE_LOG_ENV, monitor_use_rich_live_on_stderr

pytestmark = [pytest.mark.unit]


def test_monitor_use_rich_when_stderr_tty_and_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MONITOR_FILE_LOG_ENV, raising=False)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    # StringIO.isatty() is False — simulate TTY
    class _FakeTTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    assert monitor_use_rich_live_on_stderr() is True


def test_monitor_forces_file_when_env_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MONITOR_FILE_LOG_ENV, "1")

    class _FakeTTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    assert monitor_use_rich_live_on_stderr() is False


def test_monitor_uses_rich_when_stderr_not_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MONITOR_FILE_LOG_ENV, raising=False)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert monitor_use_rich_live_on_stderr() is False
