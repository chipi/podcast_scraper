"""Unit tests for ``pipeline_job_registry`` (read/write under temp dirs)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.pipeline_job_registry import (
    jobs_registry_path,
    read_jobs,
    with_jobs_locked_mutate,
    with_jobs_locked_read,
    write_jobs_atomic,
)


def test_read_jobs_dedupes_duplicate_job_id_latest_wins(tmp_path: Path) -> None:
    p = jobs_registry_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"job_id": "same", "status": "running", "note": "old"})
        + "\n"
        + json.dumps({"job_id": "same", "status": "cancelled", "note": "new"})
        + "\n",
        encoding="utf-8",
    )
    rows = read_jobs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "cancelled"
    assert rows[0]["note"] == "new"


def test_read_jobs_skips_malformed_lines(tmp_path: Path) -> None:
    p = jobs_registry_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"job_id": "ok", "status": "queued"}) + "\nnot-json{{{",
        encoding="utf-8",
    )
    rows = read_jobs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["job_id"] == "ok"


def test_write_jobs_atomic_roundtrip(tmp_path: Path) -> None:
    jobs = [{"job_id": "a", "status": "succeeded"}, {"job_id": "b", "status": "failed"}]
    write_jobs_atomic(tmp_path, jobs)
    back = read_jobs(tmp_path)
    assert {j["job_id"] for j in back} == {"a", "b"}


def test_with_jobs_locked_read_no_persist(tmp_path: Path) -> None:
    write_jobs_atomic(tmp_path, [{"job_id": "x", "status": "queued"}])

    def peek(jobs: list) -> int:
        jobs.clear()
        return 0

    with_jobs_locked_read(tmp_path, peek)
    assert read_jobs(tmp_path)[0]["job_id"] == "x"


def test_with_jobs_locked_mutate_persists(tmp_path: Path) -> None:
    def bump(jobs: list) -> None:
        jobs.append({"job_id": "new", "status": "queued"})

    with_jobs_locked_mutate(tmp_path, bump)
    ids = {j["job_id"] for j in read_jobs(tmp_path)}
    assert "new" in ids
