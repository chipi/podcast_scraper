"""Pipeline job log path resolution and tail reads.

No FastAPI dependency so this module imports under ``.[dev]`` only (CI unit job).
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path

from podcast_scraper.server.pipeline_jobs import get_job
from podcast_scraper.utils.path_validation import (
    normpath_if_under_root,
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)


class JobLogPathError(Exception):
    """Invalid or missing job log path (HTTP layer maps to status + detail)."""

    __slots__ = ("status_code", "detail")

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def read_job_log_tail_utf8(abs_path: str, max_bytes: int) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for the last ``max_bytes`` of the file."""
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        return "", False
    mb = max(1024, min(int(max_bytes), 512_000))
    try:
        with io.open(abs_path, "rb") as fh:
            if size <= mb:
                raw = fh.read()
                truncated = False
            else:
                fh.seek(size - mb)
                raw = fh.read()
                truncated = True
    except OSError:
        return "", False
    text = raw.decode("utf-8", errors="replace")
    if truncated and "\n" in text:
        nl = text.find("\n")
        if nl >= 0 and nl + 1 < len(text):
            text = text[nl + 1 :]
    return text, truncated


async def resolve_pipeline_job_log_path(corpus: Path, job_id: str) -> str:
    """Resolve registry row to an absolute log path under *corpus*; raise ``JobLogPathError``."""
    rec = await asyncio.to_thread(get_job, corpus, job_id)
    if rec is None:
        raise JobLogPathError(404, "Job not found.")
    rel = str(rec.get("log_relpath") or f".viewer/jobs/{job_id}.log").strip()
    root_res = safe_resolve_directory(corpus)
    if root_res is None:
        raise JobLogPathError(400, "Invalid corpus path.")
    root_s = os.path.normpath(str(root_res))
    verified = safe_relpath_under_corpus_root(root_res, rel.replace("\\", "/"))
    if not verified:
        raise JobLogPathError(400, "Invalid log path.")
    log_path = normpath_if_under_root(os.path.normpath(verified), root_s)
    if not log_path:
        raise JobLogPathError(400, "Invalid log path.")
    if not os.path.isfile(log_path):
        raise JobLogPathError(404, "Log file not present yet.")
    return log_path
