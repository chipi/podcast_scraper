"""Best-effort audit log for operator write actions (#1071).

One record per mutating operator request, and per security-relevant event. Best-effort: failures
never block the request. Written only when an audit path is configured (``app.state.audit_path``).

Where it goes (ADR-169): a plane's ``audit_log`` table when the process has a database, else the
JSONL file at the audit path. The control-plane api runs without a database by design, so its trail
stays the file it always was.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import exc as sa_exc, select

from podcast_scraper.db import database_available, read_with_retry, rw_engine
from podcast_scraper.db.schema import audit_log
from podcast_scraper.utils.privileges import match_data_owner

logger = logging.getLogger(__name__)


def append_audit(audit_path: Path | None, record: dict[str, Any]) -> None:
    """Append ``record`` (stamped with ``ts``) to the audit log; never raises."""
    if audit_path is None:
        return
    stamp = int(time.time())
    stamped = {"ts": stamp, **record}
    if database_available():
        try:
            with rw_engine().begin() as conn:
                conn.execute(
                    audit_log.insert().values(
                        ts=datetime.fromtimestamp(stamp, tz=timezone.utc), record=stamped
                    )
                )
        except (sa_exc.SQLAlchemyError, OSError):
            logger.warning("audit: could not record %s", record.get("event"), exc_info=True)
        return
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(stamped, ensure_ascii=False)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        match_data_owner(audit_path)
    except OSError:
        pass


def read_audit(audit_path: Path | None) -> list[dict[str, Any]]:
    """Every audit record, oldest first, from wherever :func:`append_audit` writes them."""
    if database_available():

        def read(engine: Any) -> list[dict[str, Any]]:
            with engine.connect() as conn:
                rows = conn.execute(select(audit_log.c.record).order_by(audit_log.c.id)).scalars()
                return [r for r in rows if isinstance(r, dict)]

        return read_with_retry(read)
    if audit_path is None or not audit_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def audit_event(request: Any, event: str, **fields: Any) -> None:
    """Best-effort audit of a security-relevant ``event`` using the app's configured audit path.

    Convenience over :func:`append_audit` for route handlers — reads the app's ``audit_path``
    (``None`` on a bare deploy → no-op). Used for MCP credential/consent events + auth denials.
    """
    append_audit(getattr(request.app.state, "audit_path", None), {"event": event, **fields})
