"""Canonical observability event emitter — the vendor-neutral emit side (ADR-119).

The app produces structured JSONL events through ONE function; *shipping* them to a
backend is an external, pluggable concern (a collection agent — Alloy today — tails
process stdout and the corpus JSONL files and forwards to VictoriaLogs/Loki/anywhere).
Nothing here imports a Grafana/Loki/vendor SDK. The contract is just the canonical
envelope on two channels.

Envelope (every event)::

    {"ts": "<iso8601-utc>", "schema": 1, "event_type": "<name>", ...fields}

Two sinks (``sink=``):

- ``"log"`` — one JSON line to the ``podcast_scraper.events`` logger → process
  stdout. Use for events emitted where there is no persistent corpus to write to:
  the ephemeral pipeline-runner containers (``llm_cost``, ``ml_inference``,
  ``pipeline_stage``). A collection agent captures the container stdout.
- ``"file"`` — append the line to a per-corpus JSONL file. Use for serve-side
  events tied to a corpus that must persist for volume roll-ups even with no agent
  attached (``search_query``, ``listen``, ``job``).

Telemetry MUST NEVER break the caller: every path is wrapped and swallows errors.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Bump when the envelope shape changes in a non-additive way (consumers key on it).
EVENT_SCHEMA = 1

# Dedicated logger so a shipping agent / handler can target the event stream
# distinctly from ordinary app logs if it wants. Propagates to the root by default
# (→ stdout), which is all the reference Alloy sink needs.
_event_logger = logging.getLogger("podcast_scraper.events")
_fallback_logger = logging.getLogger(__name__)

# Default corpus-relative path per file-sink event type. Callers may override with
# an explicit ``path=``. Unknown event types fall back to ``events/<type>.jsonl``.
_FILE_FOR: dict[str, str] = {
    "search_query": "search/query_log.jsonl",
    "listen": "listen.jsonl",  # usually combined with a per-user subdir via path=
    "job": ".viewer/jobs.jsonl",
}


def _trace_context() -> dict[str, str]:
    """Current OTEL trace/span ids (hex) if a span is active, else ``{}``.

    Guarded — opentelemetry is only present with the ``[otel]`` extra and there may
    be no active span. Stamping ``trace_id``/``span_id`` onto every event is the
    correlation key that joins a log/event to its trace in VictoriaTraces (ADR-119).
    """
    try:
        from opentelemetry import trace as _otel_trace

        ctx = _otel_trace.get_current_span().get_span_context()
        if getattr(ctx, "is_valid", False):
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }
    except Exception:  # noqa: BLE001 — no OTEL installed / no active span
        pass
    return {}


def emit_event(
    event_type: str,
    *,
    sink: str = "log",
    corpus_dir: Optional[Path | str] = None,
    path: Optional[Path | str] = None,
    logger: Optional[logging.Logger] = None,
    ts: Optional[str] = None,
    **fields: Any,
) -> Optional[str]:
    """Emit one canonical JSONL observability event. Best-effort — never raises.

    Args:
        event_type: the event discriminator (e.g. ``"llm_cost"``, ``"search_query"``).
        sink: ``"log"`` (→ stdout, for pipeline/ephemeral contexts) or ``"file"``
            (→ a persistent corpus JSONL, for serve-side events).
        corpus_dir: corpus root for ``sink="file"`` (path derived via ``_FILE_FOR``).
        path: explicit output file for ``sink="file"`` (wins over ``corpus_dir``);
            e.g. a per-user ``.../users/<uid>/listen.jsonl``.
        logger: override the log-sink logger (defaults to ``podcast_scraper.events``);
            lets a caller keep its own logger name for continuity / filtering. Every
            logger propagates to stdout, which is all the shipping agent needs.
        **fields: event payload; ``None`` values are dropped so the envelope stays lean.

    Returns:
        the serialized JSON line on success (handy for a caller that also echoes it),
        or ``None`` if emission failed.
    """
    try:
        record: dict[str, Any] = {
            "ts": ts or datetime.now(timezone.utc).isoformat(),
            "schema": EVENT_SCHEMA,
            "event_type": event_type,
        }
        record.update({k: v for k, v in fields.items() if v is not None})
        record.update(_trace_context())  # trace↔event correlation (no-op without a span)
        line = json.dumps(record, default=str, ensure_ascii=False)

        if sink == "file":
            out = Path(path) if path is not None else _corpus_path(corpus_dir, event_type)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        else:  # "log" (default)
            (logger or _event_logger).info("%s", line)
        return line
    except Exception:  # noqa: BLE001 — telemetry must never break the caller
        _fallback_logger.debug("emit_event(%s) failed", event_type, exc_info=True)
        return None


def _corpus_path(corpus_dir: Optional[Path | str], event_type: str) -> Path:
    if corpus_dir is None:
        raise ValueError("sink='file' requires corpus_dir= or path=")
    rel = _FILE_FOR.get(event_type, f"events/{event_type}.jsonl")
    return Path(corpus_dir) / rel
