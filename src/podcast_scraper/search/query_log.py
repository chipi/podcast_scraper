"""Append-only search-activity log (PRD-033 FR6.2 — #888 follow-up).

Records one line per search — an ISO-8601 UTC timestamp plus the detected intent
(``query_type``) only, **no raw query text** — under ``<corpus>/search/query_log.jsonl``.
The Dashboard reads it for an honest *search-volume-over-time* signal (the original
FR6.2 "query volume by topic over time" is not achievable without per-query topic
tagging; this is the supported subset, labelled as search activity, not query-by-topic).

Append is **best-effort**: a logging failure must never break the search read path.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_RELPATH = "search/query_log.jsonl"


def query_log_path(corpus_dir: Path | str) -> Path:
    """Path to the search-activity log for *corpus_dir*."""
    return Path(corpus_dir) / _LOG_RELPATH


def append_query_event(
    corpus_dir: Path | str,
    query_type: str,
    *,
    now: Optional[datetime] = None,
) -> None:
    """Append one search event (timestamp + intent). Best-effort — never raises."""
    try:
        ts = (now or datetime.now(timezone.utc)).isoformat()
        path = query_log_path(corpus_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": ts, "query_type": str(query_type or "")}) + "\n")
    except OSError as exc:  # pragma: no cover - disk/permission edge
        logger.debug("query-log append failed for %s: %s", corpus_dir, exc)


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

    path = query_log_path(corpus_dir)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    day = _event_date(line)
                    if day is not None and day in counts:
                        counts[day] += 1
        except OSError as exc:  # pragma: no cover - disk edge
            logger.debug("query-log read failed for %s: %s", corpus_dir, exc)

    buckets: List[Dict[str, Any]] = [
        {"date": d.isoformat(), "count": counts[d.isoformat()]} for d in window
    ]
    return {"total": sum(b["count"] for b in buckets), "buckets": buckets}
