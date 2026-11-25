"""Append-only search-activity log (PRD-033 FR6.2 — #888 follow-up).

Records one line per search — an ISO-8601 UTC timestamp plus the detected intent
(``query_type``) only, **no raw query text** — under ``<corpus>/search/query_log.jsonl``.
The Dashboard reads it for an honest *search-volume-over-time* signal (the original
FR6.2 "query volume by topic over time" is not achievable without per-query topic
tagging; this is the supported subset, labelled as search activity, not query-by-topic).

Append is **best-effort**: a logging failure must never break the search read path.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_LOG_RELPATH = "search/query_log.jsonl"

# One backup generation, so the log never exceeds ~2x this. A line is ~110 bytes, so the cap
# holds well over the 30-day dashboard window at any realistic search volume.
MAX_LOG_BYTES = 8 * 1024 * 1024
_CHUNK = 64 * 1024


def query_log_path(corpus_dir: Path | str) -> Path:
    """Path to the search-activity log for *corpus_dir*."""
    return Path(corpus_dir) / _LOG_RELPATH


def append_query_event(
    corpus_dir: Path | str,
    query_type: str,
    *,
    now: Optional[datetime] = None,
) -> None:
    """Append one search event (timestamp + intent). Best-effort — never raises.

    Thin wrapper over the canonical event emitter (ADR-119): one JSONL line to the
    ``search/query_log.jsonl`` file sink, with the shared ``{ts, schema, event_type}``
    envelope. A shipping agent tails the file → VictoriaLogs (or anywhere).
    """
    from ..obs.events import emit_event

    _rotate_if_oversized(query_log_path(corpus_dir))
    emit_event(
        "search_query",
        sink="file",
        path=query_log_path(corpus_dir),
        ts=(now or datetime.now(timezone.utc)).isoformat(),
        query_type=str(query_type or ""),
    )


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".1")


def _rotate_if_oversized(path: Path) -> None:
    """Move an over-cap log to ``<name>.1`` (replacing the previous backup). Never raises.

    The size is re-checked under an flock so concurrent writers rotate once; a writer that
    already holds the old file open just finishes its line there (appends stay O_APPEND).
    """
    try:
        if path.stat().st_size <= MAX_LOG_BYTES:
            return
        lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if path.stat().st_size > MAX_LOG_BYTES:
                os.replace(path, _backup_path(path))
        finally:
            os.close(lock_fd)
    except OSError as exc:
        logger.debug("query-log rotation failed for %s: %s", path, exc)


def _lines_newest_first(path: Path) -> Iterator[str]:
    """Lines of *path* from the end, reading backwards in chunks."""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        carry = b""
        while pos > 0:
            step = min(_CHUNK, pos)
            pos -= step
            fh.seek(pos)
            parts = (fh.read(step) + carry).split(b"\n")
            carry = parts[0]
            for raw in reversed(parts[1:]):
                if raw.strip():
                    yield raw.decode("utf-8", errors="replace")
        if carry.strip():
            yield carry.decode("utf-8", errors="replace")


def _event_date(line: str) -> Optional[str]:
    """The UTC ``YYYY-MM-DD`` of a log line, or None if unparsable."""
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return None
    ts = obj.get("ts") if isinstance(obj, dict) else None
    if not isinstance(ts, str) or len(ts) < 10:
        return None
    day = ts[:10]
    # Cheap shape check: YYYY-MM-DD.
    if len(day) == 10 and day[4] == "-" and day[7] == "-":
        return day
    return None


def read_query_activity(
    corpus_dir: Path | str,
    *,
    days: int = 30,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Daily search counts for the last *days* days (UTC-date buckets, zero-filled).

    Returns ``{"total": int, "buckets": [{"date": "YYYY-MM-DD", "count": int}, …]}``
    with one bucket per day, oldest→newest, ending at *today* (UTC). ``total`` is the
    sum over the returned window. Missing/empty log → all-zero buckets.
    """
    span = max(1, int(days))
    end = today or datetime.now(timezone.utc).date()
    window = [end - timedelta(days=offset) for offset in range(span - 1, -1, -1)]
    counts: Dict[str, int] = {d.isoformat(): 0 for d in window}

    oldest = window[0].isoformat()
    path = query_log_path(corpus_dir)
    try:
        for source in (path, _backup_path(path)):
            if not source.exists():
                continue
            reached_older = False
            for line in _lines_newest_first(source):
                day = _event_date(line)
                if day is None:
                    continue
                if day < oldest:
                    reached_older = True
                    break
                if day in counts:
                    counts[day] += 1
            if reached_older:
                break
    except OSError as exc:  # pragma: no cover - disk edge
        logger.debug("query-log read failed for %s: %s", corpus_dir, exc)

    buckets: List[Dict[str, Any]] = [
        {"date": d.isoformat(), "count": counts[d.isoformat()]} for d in window
    ]
    return {"total": sum(b["count"] for b in buckets), "buckets": buckets}
