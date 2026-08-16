"""Tests for the three job-error-surfacing fixes (feat/long-term-fixes).

Fix 1 — preflight provider-secret check at submit → 400 naming the key.
Fix 2 — _finalize_job + _patch_error_reason_from_log set real error_reason.
Fix 3 — _pump_subprocess_to_log drops docker pull-progress lines.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("fastapi")

from podcast_scraper.server.jobs import (
    _finalize_job,
    _parse_error_reason_from_log,
    _patch_error_reason_from_log,
    _pump_subprocess_to_log,
)
from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_mutate
from podcast_scraper.server.routes.jobs import _check_provider_secrets

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStream:
    """Minimal asyncio.StreamReader stub."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._remaining = list(chunks)

    async def read(self, n: int) -> bytes:  # noqa: ARG002
        if not self._remaining:
            return b""
        return self._remaining.pop(0)


def _run_pump(chunks: list[bytes], log_path: Path, *, max_bytes: int = 0) -> None:
    asyncio.run(
        _pump_subprocess_to_log(
            cast(asyncio.StreamReader, _FakeStream(chunks)),
            log_path,
            max_bytes=max_bytes,
            job_id="test-job",
        )
    )


def _seed_job(
    corpus: Path,
    *,
    job_id: str = "aaaa-bbbb",
    status: str = "running",
    error_reason: str | None = None,
    log_relpath: str | None = None,
) -> None:
    def fn(jobs: list) -> None:
        jobs.append(
            {
                "job_id": job_id,
                "command_type": "full_incremental_pipeline",
                "status": status,
                "created_at": "2026-01-01T00:00:00Z",
                "started_at": "2026-01-01T00:00:01Z",
                "ended_at": None,
                "pid": None,
                "argv_summary": "[]",
                "exit_code": None,
                "log_relpath": log_relpath or f".viewer/jobs/{job_id}.log",
                "error_reason": error_reason,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(corpus, fn)


# ---------------------------------------------------------------------------
# Fix 3 — docker pull-progress filter
# ---------------------------------------------------------------------------


def test_pump_drops_docker_pull_progress_lines(tmp_path: Path) -> None:
    """Pull-progress lines are filtered; the real error survives within the cap."""
    pull_noise = (
        b"Pulling from library/python\n"
        b"1234567890ab: Waiting\n"
        b"abcdef012345: Downloading\n"
        b"abcdef012345: Verifying Checksum\n"
        b"abcdef012345: Download complete\n"
        b"abcdef012345: Pull complete\n"
        b"Already exists\n"
    )
    real_error = b"ERROR CRITICAL: Deepgram API key required for transcription\n"
    log = tmp_path / "job.log"
    # cap large enough that the error must survive; without filtering,
    # the pull noise would have dominated.
    _run_pump([pull_noise + real_error], log, max_bytes=0)
    content = log.read_bytes()
    assert real_error in content
    # Pull lines must not appear in the log.
    assert b"Waiting" not in content
    assert b"Verifying Checksum" not in content
    assert b"Pull complete" not in content


def test_pump_pull_filter_counts_toward_cap_only_real_lines(tmp_path: Path) -> None:
    """Pull lines are excluded from the byte cap so the real output is not truncated."""
    # 5 KiB of pull noise + a real 100-byte error line; cap at 200 bytes.
    # Without filtering the real line would be truncated; with filtering it survives.
    pull_noise = b"abcdef012345: Downloading\n" * 200  # ~5 KiB
    real_error = b"ValueError: Deepgram API key required for transcription_provider='deepgram'\n"
    log = tmp_path / "job.log"
    _run_pump([pull_noise + real_error], log, max_bytes=200)
    content = log.read_bytes()
    # Real error line must be present despite the small cap.
    assert real_error in content


def test_pump_non_pull_lines_still_count_toward_cap(tmp_path: Path) -> None:
    """Regular (non-docker-pull) lines are NOT filtered and do count toward the cap."""
    real_lines = b"INFO: processing\n" * 100  # ~1.7 KiB
    log = tmp_path / "job.log"
    _run_pump([real_lines], log, max_bytes=100)
    content = log.read_bytes()
    assert b"LOG TRUNCATED" in content


def test_pump_partial_line_at_eof_is_flushed(tmp_path: Path) -> None:
    """A partial line without a trailing newline is written at EOF."""
    log = tmp_path / "job.log"
    _run_pump([b"no newline at end"], log, max_bytes=0)
    assert log.read_bytes() == b"no newline at end"


def test_pump_pull_lines_across_chunk_boundary(tmp_path: Path) -> None:
    """A pull line split across two chunks is still filtered correctly."""
    # Split "Pulling from library/python\n" across two chunks.
    line = b"Pulling from library/python\n"
    split_at = 10
    chunk1 = line[:split_at]
    chunk2 = line[split_at:] + b"INFO: real output\n"
    log = tmp_path / "job.log"
    _run_pump([chunk1, chunk2], log, max_bytes=0)
    content = log.read_bytes()
    assert b"Pulling" not in content
    assert b"real output" in content


# ---------------------------------------------------------------------------
# Fix 2 — error_reason parsing
# ---------------------------------------------------------------------------


def test_parse_error_reason_config_key_error(tmp_path: Path) -> None:
    """Config-validation 'API key required' line is returned as-is."""
    log = tmp_path / "j.log"
    log.write_text(
        "INFO: loading config\n"
        "ValueError: Deepgram API key required for transcription_provider='deepgram'. "
        "Set DEEPGRAM_API_KEY environment variable or deepgram_api_key in config.\n",
        encoding="utf-8",
    )
    reason = _parse_error_reason_from_log(log)
    assert reason is not None
    assert "Deepgram API key required" in reason


def test_parse_error_reason_traceback(tmp_path: Path) -> None:
    """Last traceback exception line is returned when no key-required line present."""
    log = tmp_path / "j.log"
    log.write_text(
        "INFO: start\n"
        "Traceback (most recent call last):\n"
        "  File 'foo.py', line 10, in bar\n"
        "    raise RuntimeError('pipeline exploded')\n"
        "RuntimeError: pipeline exploded\n",
        encoding="utf-8",
    )
    reason = _parse_error_reason_from_log(log)
    assert reason == "RuntimeError: pipeline exploded"


def test_parse_error_reason_error_line(tmp_path: Path) -> None:
    """Last ERROR line is used when no traceback present."""
    log = tmp_path / "j.log"
    log.write_text(
        "INFO: processing\n" "ERROR: feed fetch failed: connection refused\n" "INFO: done\n",
        encoding="utf-8",
    )
    reason = _parse_error_reason_from_log(log)
    assert reason is not None
    assert "feed fetch failed" in reason


def test_parse_error_reason_empty_log(tmp_path: Path) -> None:
    """Empty log yields None (caller falls back to exit_code_N)."""
    log = tmp_path / "j.log"
    log.write_text("", encoding="utf-8")
    assert _parse_error_reason_from_log(log) is None


def test_parse_error_reason_missing_file() -> None:
    """Missing log file yields None."""
    assert _parse_error_reason_from_log(Path("/no/such/file.log")) is None


def test_parse_error_reason_max_length(tmp_path: Path) -> None:
    """Returned reason is capped at 300 characters."""
    log = tmp_path / "j.log"
    long_msg = "A" * 500
    log.write_text(f"ERROR: {long_msg}\n", encoding="utf-8")
    reason = _parse_error_reason_from_log(log)
    assert reason is not None
    assert len(reason) <= 300


def test_patch_error_reason_updates_exit_code_placeholder(tmp_path: Path) -> None:
    """_patch_error_reason_from_log replaces exit_code_N with parsed cause."""
    log_relpath = ".viewer/jobs/test-job.log"
    (tmp_path / ".viewer" / "jobs").mkdir(parents=True)
    log_abs = tmp_path / log_relpath
    log_abs.write_text(
        "INFO: config load\n"
        "ValueError: Deepgram API key required for transcription_provider='deepgram'. "
        "Set DEEPGRAM_API_KEY\n",
        encoding="utf-8",
    )
    _seed_job(tmp_path, job_id="test-job", status="failed", error_reason="exit_code_1")

    _patch_error_reason_from_log(tmp_path, "test-job", log_relpath)

    from podcast_scraper.server.jobs import get_job

    rec = get_job(tmp_path, "test-job")
    assert rec is not None
    assert "Deepgram API key required" in (rec.get("error_reason") or "")


def test_patch_error_reason_does_not_overwrite_non_exit_code_reason(tmp_path: Path) -> None:
    """_patch_error_reason_from_log leaves non-placeholder reasons intact."""
    log_relpath = ".viewer/jobs/test-job2.log"
    (tmp_path / ".viewer" / "jobs").mkdir(parents=True)
    log_abs = tmp_path / log_relpath
    log_abs.write_text("ERROR: something else\n", encoding="utf-8")
    _seed_job(
        tmp_path,
        job_id="test-job2",
        status="failed",
        error_reason="spawn_failed: OSError",
        log_relpath=log_relpath,
    )

    _patch_error_reason_from_log(tmp_path, "test-job2", log_relpath)

    from podcast_scraper.server.jobs import get_job

    rec = get_job(tmp_path, "test-job2")
    assert rec is not None
    # Non-placeholder reason must be unchanged.
    assert rec.get("error_reason") == "spawn_failed: OSError"


def test_finalize_job_returns_log_relpath_on_failure(tmp_path: Path) -> None:
    """_finalize_job returns the log_relpath when exit_code != 0."""
    log_relpath = ".viewer/jobs/fail-job.log"
    _seed_job(tmp_path, job_id="fail-job", log_relpath=log_relpath)

    result = asyncio.run(_finalize_job(tmp_path, "fail-job", exit_code=1, cancelled=False))
    assert result == log_relpath


def test_finalize_job_returns_none_on_success(tmp_path: Path) -> None:
    """_finalize_job returns None when exit_code == 0 (no error to parse)."""
    _seed_job(tmp_path, job_id="ok-job")

    result = asyncio.run(_finalize_job(tmp_path, "ok-job", exit_code=0, cancelled=False))
    assert result is None


# ---------------------------------------------------------------------------
# Fix 1 — preflight provider-secret check
# ---------------------------------------------------------------------------


def test_check_provider_secrets_raises_for_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator YAML with cloud_balanced profile → ValueError when DEEPGRAM_API_KEY absent."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    # Write a minimal operator yaml that selects cloud_balanced.
    op_yaml = tmp_path / "viewer_operator.yaml"
    op_yaml.write_text("profile: cloud_balanced\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)deepgram.*key required|key required.*deepgram"):
        _check_provider_secrets(op_yaml)


def test_check_provider_secrets_passes_when_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config with key set → no exception raised."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key-for-test")
    op_yaml = tmp_path / "viewer_operator.yaml"
    op_yaml.write_text("profile: cloud_balanced\n", encoding="utf-8")

    # Should not raise.
    _check_provider_secrets(op_yaml)


def test_check_provider_secrets_passes_for_profile_without_key_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """airgapped profile (local whisper, no cloud LLM) needs no API keys."""
    for k in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    op_yaml = tmp_path / "viewer_operator.yaml"
    op_yaml.write_text("profile: airgapped\n", encoding="utf-8")

    # airgapped uses local whisper + local summarizer — no cloud key required.
    _check_provider_secrets(op_yaml)


def test_check_provider_secrets_passes_for_missing_operator_yaml(
    tmp_path: Path,
) -> None:
    """Missing operator YAML is not a submit-time error (pipeline handles it)."""
    op_yaml = tmp_path / "nonexistent.yaml"
    # Should not raise.
    _check_provider_secrets(op_yaml)


def test_submit_pipeline_job_400_when_provider_secret_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/jobs → 400 naming the missing key; no container spawned."""
    from fastapi.testclient import TestClient

    from podcast_scraper.server.app import create_app
    from podcast_scraper.server.operator_paths import viewer_operator_yaml_path

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    corpus = tmp_path
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)

    # Write operator yaml selecting cloud_balanced.
    op_yaml = viewer_operator_yaml_path(app, corpus)
    op_yaml.parent.mkdir(parents=True, exist_ok=True)
    op_yaml.write_text("profile: cloud_balanced\n", encoding="utf-8")

    spawned: list[bool] = []

    async def _never_spawn(*_args, **_kwargs):  # noqa: ANN002, ANN003
        spawned.append(True)
        raise AssertionError("should not have spawned")

    app.state.jobs_subprocess_factory = _never_spawn

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/api/jobs", params={"path": str(corpus)})

    assert r.status_code == 400
    detail = r.json().get("detail", "")
    assert "deepgram" in detail.lower() or "key required" in detail.lower()
    assert not spawned, "Container was spawned despite missing secret"


def test_submit_pipeline_job_202_when_provider_secret_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/jobs → 202 when required secret is present (not over-rejected)."""
    from fastapi.testclient import TestClient

    from podcast_scraper.server.app import create_app
    from podcast_scraper.server.operator_paths import viewer_operator_yaml_path

    monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-test-key")

    corpus = tmp_path
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)

    op_yaml = viewer_operator_yaml_path(app, corpus)
    op_yaml.parent.mkdir(parents=True, exist_ok=True)
    op_yaml.write_text("profile: cloud_balanced\n", encoding="utf-8")

    class _FakeProc:
        pid = 12345

        async def wait(self) -> int:
            return 0

    async def _fake_factory(argv, corpus_root, log_abs):  # noqa: ANN001, ARG001
        log_abs.parent.mkdir(parents=True, exist_ok=True)
        log_abs.write_bytes(b"ok\n")
        return _FakeProc()

    app.state.jobs_subprocess_factory = _fake_factory

    client = TestClient(app)
    r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 202
