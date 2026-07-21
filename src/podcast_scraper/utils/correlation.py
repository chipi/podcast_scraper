"""Run / episode correlation ids — the join key across every o11y signal (#1053).

The point: stamp the SAME id on the Loki cost event, the Loki log lines, the Sentry
event, and the Langfuse trace for a run, so an agent (or human) can pull every signal
for one run/episode and correlate them.

- ``run_id`` is constant for a whole run and is a **process global**: the pipeline runs
  as a per-run subprocess, so one run == one process, and every worker thread reads the
  same value (Config is frozen, so it can't live there; contextvars don't propagate into
  the summarization worker pool). Resolve it **once** at run start via :func:`set_run_id`.
- ``episode_id`` varies per episode and episodes summarise in parallel, so it's a
  ``ContextVar`` set inside each worker's episode scope (Tier 2).

All getters are cheap and side-effect free; nothing here imports a 3rd-party SDK.
"""

from __future__ import annotations

import contextvars
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

# Process-global run id (set once at run start; read from any thread).
_RUN_ID: Optional[str] = None

# Per-episode id — set within each episode's (possibly worker-thread) scope.
_EPISODE_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "podcast_episode_id", default=None
)


def resolve_run_id(raw: Optional[str]) -> str:
    """Map a configured ``run_id`` (or ``"auto"`` / unset) to a concrete, stable value.

    A real value is used as-is; ``"auto"`` / empty / ``None`` becomes a UTC timestamp id
    so a run always has *one* identifier the signals can share.
    """
    candidate = (raw or "").strip()
    if candidate and candidate.lower() != "auto":
        return candidate
    return "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def set_run_id(run_id: Optional[str]) -> None:
    """Set the process-global run id (call once at run start)."""
    global _RUN_ID
    _RUN_ID = (run_id or "").strip() or None


def get_run_id() -> Optional[str]:
    """The current run id, or ``None`` if a run hasn't started."""
    return _RUN_ID


def set_episode_id(episode_id: Optional[str]) -> None:
    """Set the current episode id for this context (worker scope)."""
    _EPISODE_ID.set((episode_id or "").strip() or None)


def get_episode_id() -> Optional[str]:
    """The current episode id for this context, or ``None``."""
    return _EPISODE_ID.get()


def correlation_fields() -> Dict[str, str]:
    """The non-empty correlation ids, ready to stamp onto a signal."""
    fields: Dict[str, str] = {}
    run_id = _RUN_ID
    if run_id:
        fields["run_id"] = run_id
    episode_id = _EPISODE_ID.get()
    if episode_id:
        fields["episode_id"] = episode_id
    return fields


def _current_trace_id() -> str:
    """Active OTEL trace id (hex) or ``"-"``. Guarded — no ``[otel]`` / no span → ``"-"``.

    So a plain log line carries the same ``trace_id`` as its trace in VictoriaTraces,
    letting an operator pivot log ↔ trace (ADR-119 correlation).
    """
    try:
        from opentelemetry import trace as _otel_trace

        ctx = _otel_trace.get_current_span().get_span_context()
        if getattr(ctx, "is_valid", False):
            return format(ctx.trace_id, "032x")
    except Exception:  # noqa: BLE001 — no OTEL installed / no active span
        pass
    return "-"


class CorrelationFormatter(logging.Formatter):
    """A ``logging.Formatter`` that injects ``run_id`` / ``episode_id`` / ``trace_id``
    onto every record at format time (#1053, ADR-119), so a format string can reference
    ``%(run_id)s`` / ``%(trace_id)s`` without any record ever raising ``KeyError`` — and
    every log line carries the join keys, queryable in VictoriaLogs. Defaults to ``"-"``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Stamp ``run_id`` / ``episode_id`` / ``trace_id`` onto the record, then format."""
        record.run_id = _RUN_ID or "-"
        record.episode_id = _EPISODE_ID.get() or "-"
        record.trace_id = _current_trace_id()
        return super().format(record)


def _reset_for_tests() -> None:
    """Test hook: clear both ids."""
    global _RUN_ID
    _RUN_ID = None
    _EPISODE_ID.set(None)
