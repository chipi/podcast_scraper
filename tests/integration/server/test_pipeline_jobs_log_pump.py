"""Unit coverage for the pipeline job log pump (#666 review #10)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from podcast_scraper.server.pipeline_jobs import (
    _LOG_MAX_BYTES_DEFAULT,
    _pump_subprocess_to_log,
    job_log_max_bytes,
)

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


class _FakeStream:
    """Minimal ``asyncio.StreamReader``-shaped stub for pump tests.

    Emits the queued chunks in order; returns ``b""`` when drained (EOF).
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._remaining: list[bytes] = list(chunks)

    async def read(self, n: int) -> bytes:
        if not self._remaining:
            return b""
        return self._remaining.pop(0)


def _run_pump(stream: Any, log_abs: Path, *, max_bytes: int, job_id: str = "job") -> None:
    """Drive the pump under ``asyncio.run``.

    ``stream`` is cast to ``asyncio.StreamReader`` — ``_FakeStream`` /
    ``_ExplodingStream`` only implement the ``async read(n)`` surface the
    pump actually touches.
    """
    asyncio.run(
        _pump_subprocess_to_log(
            cast(asyncio.StreamReader, stream),
            log_abs,
            max_bytes=max_bytes,
            job_id=job_id,
        )
    )


def test_pump_writes_all_bytes_under_cap(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    _run_pump(_FakeStream([b"hello\n", b"world\n"]), log, max_bytes=1024)
    assert log.read_bytes() == b"hello\nworld\n"


def test_pump_truncates_at_cap_and_writes_marker(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    # 3 chunks of 100 bytes each = 300 total; cap at 150 → expect 150 bytes of
    # payload + a trailing truncation marker.
    _run_pump(_FakeStream([b"A" * 100, b"B" * 100, b"C" * 100]), log, max_bytes=150)
    data = log.read_bytes()
    assert data.startswith(b"A" * 100 + b"B" * 50)
    assert b"LOG TRUNCATED at 150 bytes" in data
    # No raw 'C' chunk made it through — those bytes were dropped.
    assert b"C" * 100 not in data


def test_pump_drains_stream_after_truncation(tmp_path: Path) -> None:
    """After hitting the cap the pump must keep reading so the subprocess
    does not deadlock on a full pipe, even though writes are discarded."""
    log = tmp_path / "job.log"
    _run_pump(_FakeStream([b"X" * 200, b"Y" * 500]), log, max_bytes=100)
    data = log.read_bytes()
    assert data.startswith(b"X" * 100)
    assert b"LOG TRUNCATED" in data


def test_pump_uncapped_when_max_bytes_zero(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    big = b"Z" * (200 * 1024)
    _run_pump(_FakeStream([big]), log, max_bytes=0)
    assert log.read_bytes() == big


def test_pump_handles_empty_stream(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    _run_pump(_FakeStream([]), log, max_bytes=1024)
    assert log.read_bytes() == b""


def test_pump_creates_parent_dirs(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "job-dir" / "job.log"
    _run_pump(_FakeStream([b"ok\n"]), log, max_bytes=1024)
    assert log.read_bytes() == b"ok\n"


def test_pump_survives_read_exceptions(tmp_path: Path) -> None:
    """If ``stream.read`` raises, the pump writes what it has and exits
    cleanly instead of dropping the whole job to an uncaught exception."""
    log = tmp_path / "job.log"

    class _ExplodingStream:
        async def read(self, _: int) -> bytes:
            raise RuntimeError("simulated stream error")

    _run_pump(_ExplodingStream(), log, max_bytes=1024)
    assert log.exists()
    assert log.read_bytes() == b""


def test_job_log_max_bytes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_JOB_LOG_MAX_BYTES", raising=False)
    assert job_log_max_bytes() == _LOG_MAX_BYTES_DEFAULT


def test_job_log_max_bytes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_JOB_LOG_MAX_BYTES", "4096")
    assert job_log_max_bytes() == 4096


def test_job_log_max_bytes_zero_means_uncapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_JOB_LOG_MAX_BYTES", "0")
    assert job_log_max_bytes() == 0


def test_job_log_max_bytes_unparsable_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PODCAST_JOB_LOG_MAX_BYTES", "abc")
    with caplog.at_level("WARNING"):
        assert job_log_max_bytes() == _LOG_MAX_BYTES_DEFAULT
    assert any("PODCAST_JOB_LOG_MAX_BYTES" in rec.message for rec in caplog.records)
