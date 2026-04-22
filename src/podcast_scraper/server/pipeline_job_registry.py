"""JSONL job registry under ``<corpus>/.viewer/jobs.jsonl`` (RFC-077 Phase 2)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from filelock import FileLock

VIEWER_DIR = ".viewer"
JOBS_FILE = "jobs.jsonl"
LOCK_NAME = "jobs.jsonl.lock"
LOCK_TIMEOUT_S = 30.0

_T = TypeVar("_T")


def _dedupe_job_rows_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per ``job_id``: later lines win (repairs duplicate JSONL rows).

    The registry is normally rewritten as a single line per job, but interrupted
    writes or older tooling can leave multiple lines with the same ``job_id``.
    """
    no_id: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for j in rows:
        jid = str(j.get("job_id", "")).strip()
        if not jid:
            no_id.append(dict(j))
            continue
        if jid not in by_id:
            order.append(jid)
        by_id[jid] = dict(j)
    return no_id + [by_id[k] for k in order]


def viewer_dir(corpus_root: Path) -> Path:
    """Return ``<corpus>/.viewer``."""
    return corpus_root / VIEWER_DIR


def jobs_registry_path(corpus_root: Path) -> Path:
    """Return the JSONL registry path under the corpus."""
    return viewer_dir(corpus_root) / JOBS_FILE


def jobs_lock_path(corpus_root: Path) -> Path:
    """Return the file lock path adjacent to the registry."""
    return viewer_dir(corpus_root) / LOCK_NAME


@contextmanager
def _locked_registry(corpus_root: Path) -> Iterator[None]:
    vdir = viewer_dir(corpus_root)
    vdir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(jobs_lock_path(corpus_root)), timeout=LOCK_TIMEOUT_S)
    with lock:
        yield


def read_jobs(corpus_root: Path) -> list[dict[str, Any]]:
    """Load job records from disk (skips malformed JSON lines)."""
    path = jobs_registry_path(corpus_root)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return _dedupe_job_rows_by_id(out)


def write_jobs_atomic(corpus_root: Path, jobs: list[dict[str, Any]]) -> None:
    """Replace the registry file atomically via a temp file."""
    path = jobs_registry_path(corpus_root)
    viewer_dir(corpus_root).mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    lines = "\n".join(json.dumps(j, sort_keys=True) for j in jobs)
    if lines:
        lines += "\n"
    tmp.write_text(lines, encoding="utf-8")
    tmp.replace(path)


def with_jobs_locked_mutate(corpus_root: Path, fn: Callable[[list[dict[str, Any]]], _T]) -> _T:
    """Run ``fn(jobs: list[dict]) -> result`` under a file lock; persist ``jobs`` after ``fn``."""
    with _locked_registry(corpus_root):
        jobs = read_jobs(corpus_root)
        result = fn(jobs)
        write_jobs_atomic(corpus_root, jobs)
        return result


def with_jobs_locked_read(corpus_root: Path, fn: Callable[[list[dict[str, Any]]], _T]) -> _T:
    """Run ``fn(jobs)`` under lock without persisting mutations to ``jobs``."""
    with _locked_registry(corpus_root):
        jobs = read_jobs(corpus_root)
        return fn(jobs)
