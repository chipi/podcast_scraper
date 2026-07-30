"""Structured JSON log formatter (opt-in via ``cfg.json_logs``).

One JSON object per log record on stdout, carrying the same correlation join keys
(``run_id`` / ``episode_id`` / ``trace_id``) that :class:`~podcast_scraper.utils.correlation.
CorrelationFormatter` stamps on the plain-text format — so a shipping agent (Alloy → VictoriaLogs)
gets pre-parsed fields instead of a regex, and a log line still joins to its trace / cost event /
Sentry issue by the shared ids (#1053 / ADR-119).

Historically ``cfg.json_logs=True`` referenced this module, which did not exist — setting the flag
raised ``ModuleNotFoundError`` at logging setup. This ships it. Stdlib only; no 3rd-party dep.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

# Standard ``LogRecord`` attributes we render explicitly or deliberately omit, so that any *extra*
# fields a caller attached via ``logger.info(..., extra={...})`` are surfaced without duplication.
_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single-line JSON object with correlation ids."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise *record* to a single-line JSON string with correlation keys."""
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Correlation join keys — same source as CorrelationFormatter, guarded so logging never
        # depends on a run being active (or OTEL being installed).
        try:
            from .correlation import correlation_fields, current_trace_id

            payload.update(correlation_fields())  # run_id / episode_id when set
            trace_id = current_trace_id()
            if trace_id and trace_id != "-":
                payload["trace_id"] = trace_id
        except Exception:  # pragma: no cover - never let correlation break a log line
            pass

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Pass through any structured extras the caller attached.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_") and key not in payload:
                try:
                    json.dumps(value)  # only include JSON-serializable extras
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        return json.dumps(payload, default=str, ensure_ascii=False)
