"""Unit tests for ``jobs_log_path`` (no FastAPI — runs under ``.[dev]`` only)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.server import jobs_log_path as jlp


def test_read_job_log_tail_missing_file() -> None:
    assert jlp.read_job_log_tail_utf8("/no/such/file/ever.log", 96_000) == ("", False)


def test_read_job_log_tail_getsize_oserror() -> None:
    with patch.object(jlp.os.path, "getsize", side_effect=OSError("nope")):
        assert jlp.read_job_log_tail_utf8("/fake/path", 4096) == ("", False)


def test_read_job_log_tail_small_file_not_truncated(tmp_path: Path) -> None:
    p = tmp_path / "j.log"
    p.write_text("line1\nline2\n", encoding="utf-8")
    text, truncated = jlp.read_job_log_tail_utf8(str(p), 96_000)
    assert truncated is False
    assert "line2" in text


def test_read_job_log_tail_clamps_max_bytes() -> None:
    """``max_bytes`` below 1024 is clamped up to 1024 for the read window."""
    raw = b"x" * 1500
    tail = raw[-1024:]

    class _Fh:
        def __init__(self) -> None:
            self._pos = 0

        def seek(self, pos: int) -> None:
            self._pos = pos
            assert pos == len(raw) - 1024

        def read(self) -> bytes:
            assert self._pos == len(raw) - 1024
            return tail

        def __enter__(self) -> _Fh:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    with patch.object(jlp.os.path, "getsize", return_value=len(raw)):
        with patch.object(jlp.io, "open", return_value=_Fh()):
            text, truncated = jlp.read_job_log_tail_utf8("/fake/path", max_bytes=100)
    assert truncated is True
    assert len(text) == 1024


def test_read_job_log_tail_truncated_no_strip_when_newline_is_final_char() -> None:
    """Branch ``nl + 1 < len(text)`` false: tail is only a fragment ending in newline."""
    window = b"frag\n"
    size = 50_000

    class _Fh:
        def seek(self, pos: int) -> None:
            assert pos == size - 4096

        def read(self) -> bytes:
            return window

        def __enter__(self) -> _Fh:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    with patch.object(jlp.os.path, "getsize", return_value=size):
        with patch.object(jlp.io, "open", return_value=_Fh()):
            text, truncated = jlp.read_job_log_tail_utf8("/fake/path", max_bytes=4096)
    assert truncated is True
    assert text == "frag\n"


def test_read_job_log_tail_truncated_strips_first_fragment_line() -> None:
    """After seek, drop bytes before the first newline so the tail starts on a line boundary."""
    window = b"fragment-of-line\nfull-line\n"
    size = 50_000

    class _Fh:
        def seek(self, pos: int) -> None:
            assert pos == size - 4096

        def read(self) -> bytes:
            return window

        def __enter__(self) -> _Fh:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    with patch.object(jlp.os.path, "getsize", return_value=size):
        with patch.object(jlp.io, "open", return_value=_Fh()):
            text, truncated = jlp.read_job_log_tail_utf8("/fake/path", max_bytes=4096)
    assert truncated is True
    assert text == "full-line\n"


def test_read_job_log_tail_open_oserror_returns_empty() -> None:
    with patch.object(jlp.os.path, "getsize", return_value=10):
        with patch.object(jlp.io, "open", side_effect=OSError("boom")):
            assert jlp.read_job_log_tail_utf8("/fake/path", 4096) == ("", False)


def _run(coro):  # noqa: ANN001
    return asyncio.run(coro)


async def _resolved(corpus: Path, job_id: str) -> str:
    return await jlp.resolve_pipeline_job_log_path(corpus, job_id)


def test_resolved_job_log_path_job_missing(tmp_path: Path) -> None:
    with patch.object(jlp, "get_job", return_value=None):
        with pytest.raises(jlp.JobLogPathError) as ei:
            _run(_resolved(tmp_path, "missing-uuid"))
    assert ei.value.status_code == 404


def test_resolved_job_log_path_invalid_relative(tmp_path: Path) -> None:
    log_rel = "../../outside.log"
    with patch.object(
        jlp,
        "get_job",
        return_value={"job_id": "j1", "log_relpath": log_rel},
    ):
        with pytest.raises(jlp.JobLogPathError) as ei:
            _run(_resolved(tmp_path, "j1"))
    assert ei.value.status_code == 400


def test_resolved_job_log_path_file_missing(tmp_path: Path) -> None:
    rel = ".viewer/jobs/j1.log"
    (tmp_path / ".viewer" / "jobs").mkdir(parents=True)
    with patch.object(
        jlp,
        "get_job",
        return_value={"job_id": "j1", "log_relpath": rel},
    ):
        with pytest.raises(jlp.JobLogPathError) as ei:
            _run(_resolved(tmp_path, "j1"))
    assert ei.value.status_code == 404


def test_resolved_job_log_path_rejects_when_normpath_if_under_root_returns_none(
    tmp_path: Path,
) -> None:
    """``normpath_if_under_root`` immediately before ``isfile`` must reject non-corpus paths."""
    rel = ".viewer/jobs/j1.log"
    good_verified = str((tmp_path / rel).resolve())
    with patch.object(
        jlp,
        "get_job",
        return_value={"job_id": "j1", "log_relpath": rel},
    ):
        with patch.object(jlp, "safe_relpath_under_corpus_root", return_value=good_verified):
            with patch.object(jlp, "normpath_if_under_root", return_value=None):
                with pytest.raises(jlp.JobLogPathError) as ei:
                    _run(_resolved(tmp_path, "j1"))
    assert ei.value.status_code == 400


def test_resolved_job_log_path_success(tmp_path: Path) -> None:
    rel = ".viewer/jobs/j1.log"
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ok\n", encoding="utf-8")
    with patch.object(
        jlp,
        "get_job",
        return_value={"job_id": "j1", "log_relpath": rel},
    ):
        got = _run(_resolved(tmp_path, "j1"))
    assert os.path.isfile(got)
    assert got.endswith("j1.log")
