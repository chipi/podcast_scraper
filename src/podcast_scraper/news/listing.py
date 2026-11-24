"""What one listing of a news source left behind: the backlog over its cap, how far its cursor is
behind, and items that fell out of its feed before they were taken (next-cycle item 24).

Each transport calls :func:`report` once per listing. The event ``news_source_listing`` goes to the
log stream (VictoriaLogs), so a source over its cap shows a backlog night after night and its
catch-up is read from the same events. A feed whose oldest entry is already newer than the cursor
lost whatever was published in between (nothing only when exactly its whole window was published
since the cursor): that is a warning, so it reaches GlitchTip as well.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hours_since(iso: Optional[str], now: datetime) -> Optional[float]:
    if not iso:
        return None
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return round((now - then).total_seconds() / 3600, 2)


def report(
    transport: str,
    source: Any,
    *,
    cursor_before: Optional[str],
    cursor_after: Optional[str],
    waiting: int,
    taken: int,
    oldest_listed: Optional[str] = None,
    complete: bool = True,
    now: Optional[datetime] = None,
) -> None:
    """Log one listing: ``waiting`` items passed the filters after the cursor, ``taken`` of them
    were taken (the rest is the backlog). ``oldest_listed`` is the oldest dated entry the feed
    still lists, compared with ``cursor_before`` for lost items; ``complete`` is False when the
    listing stopped before it saw everything after the cursor (so the backlog is a lower bound).
    Never raises."""
    from ..obs.events import emit_event

    url = str(getattr(source, "url", "") or "")
    moment = now or datetime.now(timezone.utc)
    lost_between = None
    if cursor_before and oldest_listed and oldest_listed > cursor_before:
        lost_between = [cursor_before, oldest_listed]
        logger.warning(
            "news %s: %s: the feed's oldest entry (%s) is newer than the cursor (%s); "
            "anything published in between was never listed (none, only if the feed's whole "
            "window was published since the cursor)",
            transport,
            url,
            oldest_listed,
            cursor_before,
        )
    emit_event(
        "news_source_listing",
        transport=transport,
        source=url,
        cursor_before=cursor_before,
        cursor_after=cursor_after,
        waiting=waiting,
        taken=taken,
        backlog=max(waiting - taken, 0),
        backlog_complete=complete,
        cursor_lag_hours=_hours_since(cursor_after, moment),
        lost_between=lost_between,
    )
