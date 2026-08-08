"""Per-tool observability for the remote MCP surface (#1505).

Every registered tool call emits three signals, all best-effort and NEVER able to break the
call (ADR-120 — telemetry never downs the app):

- **trace**: an OTel span ``mcp.tool.<name>`` (child of the inbound ``POST /mcp`` server span),
  attributed with the tool, the authenticated ``user_id``, and ``ok``.
- **log**: a structured line (tool, user_id, duration_ms, ok) with ``trace_id`` correlation.
- **metric**: ``mcp_tool_calls_total{tool,ok}`` + ``mcp_tool_duration_seconds{tool}``.

Privacy (mirrors the MCP no-leak review): only the tool NAME, the ``user_id``, ``ok`` and timing
leave the process — never query args, corpus content, or transcripts.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .auth import current_mcp_user

_LOGGER = logging.getLogger("podcast_scraper.mcp.tool")

# Prometheus metrics — module-level singletons; a missing prometheus_client (or duplicate
# registration under test reload) degrades to no-op rather than breaking tool calls.
try:  # pragma: no cover - exercised via the metrics endpoint
    from prometheus_client import Counter, Histogram

    _CALLS: Optional[Any] = Counter(
        "mcp_tool_calls_total", "MCP tool calls by tool + outcome", ["tool", "ok"]
    )
    _DURATION: Optional[Any] = Histogram(
        "mcp_tool_duration_seconds", "MCP tool call wall-clock duration", ["tool"]
    )
except Exception:  # noqa: BLE001 - metrics are optional; never break import
    _CALLS = None
    _DURATION = None


class _ToolCall:
    """Mutable handle so the caller can hand back the result envelope for the ``ok`` verdict."""

    __slots__ = ("tool", "ok")

    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.ok = True

    def set_result(self, result: Any) -> None:
        # Tools return the uniform ``{ok, data, note}`` envelope; a raised exception (ok stays
        # False via the except branch below) or ``ok=False`` both count as not-ok.
        if isinstance(result, dict) and "ok" in result:
            self.ok = bool(result.get("ok"))


@contextmanager
def _maybe_span(tool: str) -> Iterator[Any]:
    """Yield an OTel span for the tool call, or ``None`` when tracing isn't wired.

    The span context manager is resolved OUTSIDE any ``except`` so an exception thrown into the
    ``yield`` (a failing tool body) propagates cleanly — yielding from within an ``except`` would
    raise ``RuntimeError: generator didn't stop after throw()``.
    """
    span_cm = None
    try:
        from opentelemetry import trace

        span_cm = trace.get_tracer("podcast_scraper.mcp").start_as_current_span(f"mcp.tool.{tool}")
    except Exception:  # noqa: BLE001 - no OTel API installed → run un-spanned
        span_cm = None
    if span_cm is None:
        yield None
        return
    with span_cm as span:
        yield span


@contextmanager
def observe_tool_call(tool: str) -> Iterator[_ToolCall]:
    """Wrap a tool call so it emits span + structured log + metric. Best-effort throughout."""
    call = _ToolCall(tool)
    start = time.perf_counter()
    with _maybe_span(tool) as span:
        try:
            yield call
        except Exception:
            call.ok = False
            raise
        finally:
            duration_s = time.perf_counter() - start
            try:
                user = current_mcp_user.get()
            except Exception:  # noqa: BLE001
                user = None
            _emit(tool, call.ok, duration_s, user, span)


def _emit(tool: str, ok: bool, duration_s: float, user: Optional[str], span: Any) -> None:
    if span is not None:
        try:
            span.set_attribute("mcp.tool", tool)
            span.set_attribute("mcp.ok", ok)
            if user:
                span.set_attribute("mcp.user_id", user)
        except Exception:  # noqa: BLE001
            pass
    if _CALLS is not None and _DURATION is not None:
        try:
            _CALLS.labels(tool=tool, ok=str(ok).lower()).inc()
            _DURATION.labels(tool=tool).observe(duration_s)
        except Exception:  # noqa: BLE001
            pass
    try:
        _LOGGER.info(
            "mcp tool call tool=%s user=%s ok=%s duration_ms=%.1f",
            tool,
            user or "-",
            ok,
            duration_s * 1000.0,
        )
    except Exception:  # noqa: BLE001
        pass
