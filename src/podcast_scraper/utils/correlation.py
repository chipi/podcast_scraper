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

import contextlib
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


# #1873: the feed and the resolved profile join a signal to HOW it was running. They live
# beside run_id/episode_id because every surface (logs, events, spans, Sentry) already reads
# this module for the other two — a fifth surface added later gets them for free, and the
# per-surface drift that made #1873 necessary cannot recur.
#
# Feed is a ContextVar, not a global: a batch walks many feeds in one process, and a global
# would leak the previous feed onto the next one's signals. Profile is a global because it is
# resolved once per run — a per-feed profile pin (#1872) produces a separate child config, and
# the run that reports it is the one the operator asked about.
_FEED_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "podcast_scraper_feed_id", default=None
)
# Mirror of the ContextVar for thread-pool workers. ContextVars do NOT propagate into
# ThreadPoolExecutor workers, and the pipeline runs whole stages in per-stage pools
# (transcription, summarization, processing) — so the ContextVar alone left every event,
# span and Sentry tag raised from a worker with NO feed, which is most of them.
#
# A plain global is CORRECT here because a batch walks feeds strictly sequentially (both
# multi-feed loops call run_pipeline per feed in order), so there is never more than one
# feed in flight per process. The ContextVar stays and wins when set, for async/server
# contexts where concurrent feeds would be possible; the global is only the fallback.
_FEED_ID_FALLBACK: Optional[str] = None
_PROFILE: Optional[str] = None


def set_feed_id(feed_id: Optional[str]) -> None:
    """Set the feed (URL or slug) the current context is processing."""
    global _FEED_ID_FALLBACK
    value = (feed_id or "").strip() or None
    _FEED_ID.set(value)
    _FEED_ID_FALLBACK = value


def get_feed_id() -> Optional[str]:
    """Current feed id, or ``None`` — ContextVar first, then the thread-visible fallback."""
    return _FEED_ID.get() or _FEED_ID_FALLBACK


def set_profile(profile: Optional[str]) -> None:
    """Set the resolved profile name for this run."""
    global _PROFILE
    _PROFILE = (profile or "").strip() or None


def get_profile() -> Optional[str]:
    """Resolved profile name for this run, or ``None``."""
    return _PROFILE


def set_episode_id(episode_id: Optional[str]) -> None:
    """Set the current episode id for this context (worker scope)."""
    _EPISODE_ID.set((episode_id or "").strip() or None)


def get_episode_id() -> Optional[str]:
    """The current episode id for this context, or ``None``."""
    return _EPISODE_ID.get()


@contextlib.contextmanager
def episode_scope(episode_id: Optional[str]):
    """Bind ``episode_id`` for the duration of one episode's processing, then restore.

    Uses the ``ContextVar`` token so the previous value is restored on exit — this matters on a
    reused worker thread, where a bare :func:`set_episode_id` would otherwise leak one episode's id
    onto the next. Wrap each per-episode unit of work (download / transcription / metadata-gen) in
    this so every log line, cost event, Sentry tag, and Langfuse span for that episode carries its
    id (#1053). Never raises on a falsy id — it simply binds ``None``.
    """
    token = _EPISODE_ID.set((episode_id or "").strip() or None)
    try:
        yield
    finally:
        _EPISODE_ID.reset(token)


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


def current_trace_id() -> str:
    """Public accessor for the active OTEL trace id (hex) or ``"-"``.

    Same value :class:`CorrelationFormatter` stamps — for callers (e.g. the API
    request-access-log middleware) that want the trace id inline in a message so a log
    line pivots to its trace in VictoriaTraces (ADR-119). ``"-"`` when no active span.
    """
    return _current_trace_id()


class CorrelationFormatter(logging.Formatter):
    """A ``logging.Formatter`` that injects ``run_id`` / ``episode_id`` / ``feed_id`` /
    ``profile`` / ``trace_id``
    onto every record at format time (#1053, ADR-119), so a format string can reference
    ``%(run_id)s`` / ``%(trace_id)s`` without any record ever raising ``KeyError`` — and
    every log line carries the join keys, queryable in VictoriaLogs. Defaults to ``"-"``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Stamp ``run_id`` / ``episode_id`` / ``trace_id`` onto the record, then format.

        The stamped values are coerced to ``str``: these are join-key fields consumed as
        ``%(run_id)s`` and captured by log handlers, so they must be plain strings. Coercion is
        idempotent for real ids and is a hard requirement for xdist report serialization — a
        heavily-mocked unit test can leave a ``Mock`` in the episode/run context, and a raw ``Mock``
        on a captured record makes ``pytest-json-report`` fail to serialize the report over execnet
        (worker crash under ``-n``; harmless in prod where ids are always strings). #1355.
        """
        record.run_id = str(_RUN_ID) if _RUN_ID else "-"
        record.episode_id = str(_EPISODE_ID.get() or "-")
        record.feed_id = str(_FEED_ID.get() or "-")
        record.profile = str(_PROFILE or "-")
        record.trace_id = str(_current_trace_id())
        return super().format(record)


def _reset_for_tests() -> None:
    """Test hook: clear both ids."""
    global _RUN_ID
    _RUN_ID = None
    _EPISODE_ID.set(None)
