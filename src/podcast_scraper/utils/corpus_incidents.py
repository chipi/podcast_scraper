"""Append-only corpus incident log (JSONL) for operator triage (GitHub #557)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from podcast_scraper.utils.log_redaction import redact_for_log

IncidentScope = Literal["episode", "feed", "batch"]
IncidentCategory = Literal["policy", "soft", "hard"]

_locks_guard = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        if path not in _path_locks:
            _path_locks[path] = threading.Lock()
        return _path_locks[path]


def append_corpus_incident(
    log_path: Optional[str],
    *,
    scope: IncidentScope,
    category: IncidentCategory,
    message: str,
    exception_type: str,
    stage: Optional[str] = None,
    feed_url: Optional[str] = None,
    episode_id: Optional[str] = None,
    episode_idx: Optional[int] = None,
) -> None:
    """Append one JSON line to ``corpus_incidents.jsonl`` (thread-safe per path).

    Args:
        log_path: Absolute or relative path to the JSONL file; if None or empty, no-op.
        scope: Whether the incident refers to an episode, a whole feed, or batch-level work.
        category: ``policy`` (limits, documented skips), ``soft`` (operational / recoverable),
            or ``hard`` (unexpected).
        message: Human-readable detail (redacted before write).
        exception_type: ``type(exc).__name__`` or ``"Message"`` when no exception object.
        stage: Optional pipeline stage (e.g. ``transcription``, ``rss``).
        feed_url: Optional RSS URL for the feed in scope.
        episode_id: Optional stable episode id when scope is ``episode``.
        episode_idx: Optional 0-based episode index in the feed.
    """
    if not log_path or not str(log_path).strip():
        return
    path = str(Path(log_path))
    record: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "occurred_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "scope": scope,
        "category": category,
        "exception_type": exception_type,
        "message": redact_for_log(message, max_len=2000),
    }
    if stage:
        record["stage"] = stage
    if feed_url:
        record["feed_url"] = feed_url
    if episode_id:
        record["episode_id"] = episode_id
    if episode_idx is not None:
        record["episode_idx"] = int(episode_idx)

    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    lock = _lock_for(path)
    with lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
