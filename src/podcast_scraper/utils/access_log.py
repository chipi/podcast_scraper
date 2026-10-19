"""Pure-ASGI request access log with trace correlation, shared by the api and the MCP server."""

from __future__ import annotations

import logging
import threading
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from podcast_scraper.utils.correlation import current_trace_id

# Requests currently being served: id -> (method, path, monotonic start). Read by the memory
# watchdog (server/mem_watchdog.py) to name what was running when memory climbed.
_INFLIGHT: dict[int, tuple[str, str, float]] = {}
_INFLIGHT_LOCK = threading.Lock()


def inflight_requests() -> list[tuple[str, str, float]]:
    """A snapshot of the requests in flight: (method, path, monotonic start)."""
    with _INFLIGHT_LOCK:
        return list(_INFLIGHT.values())


class AccessLogMiddleware:
    """Pure-ASGI request access log with trace correlation (ADR-119, G1 correlation).

    Deliberately NOT a Starlette ``BaseHTTPMiddleware`` (``@app.middleware("http")``): that
    wrapper buffers the response and BREAKS endpoint background tasks — which silently
    dropped queue-add / favourite persistence (caught by the auth-queue + library-saved
    e2e). A pure ASGI middleware wraps ``send`` without touching the body or the
    background-task lifecycle.

    Logs ONE line per request. The trace id is captured at ``http.response.start`` — the
    span is still active there (uvicorn's own access log runs AFTER the span closes, so it
    can only ever see ``"-"``) — so a VictoriaLogs line pivots to its VictoriaTraces span.
    ``trace=-`` when tracing is off, so it is a no-op-safe structured access log in every
    environment; ``/health`` + ``/metrics`` are skipped to keep the log readable.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._logger = logging.getLogger("podcast.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or path.endswith("/health") or path == "/metrics":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        captured = {"status": 0, "trace": "-"}
        key = id(scope)
        with _INFLIGHT_LOCK:
            _INFLIGHT[key] = (scope.get("method", "?"), path, time.monotonic())

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                captured["trace"] = current_trace_id()
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.pop(key, None)
        self._logger.info(
            "%s %s -> %s in %.1fms trace=%s",
            scope.get("method", "?"),
            path,
            captured["status"],
            (time.perf_counter() - start) * 1000.0,
            captured["trace"],
        )
