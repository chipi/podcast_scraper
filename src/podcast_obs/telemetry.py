"""Self-observability for the obs MCP server (player-obs).

The control plane that reads every other service's signals emitted none of its own: no traces,
no metrics, no GlitchTip project, and only uvicorn's access line (mostly its own 30 s healthcheck)
in VictoriaLogs. This module gives it the same signals as the content MCP (player-mcp):

- **errors**: Sentry SDK → its own GlitchTip project, from ``PODCAST_SENTRY_DSN_OBS``.
- **traces**: an OTLP ``TracerProvider`` (``OTEL_TRACES_EXPORTER=otlp`` + endpoint), inbound ASGI
  server spans, and one ``mcp.tool.<name>`` span per tool call.
- **logs**: a request access line and a tool-call line, each carrying ``trace_id=``.
- **metrics**: ``mcp_tool_calls_total{tool,ok}`` + ``mcp_tool_duration_seconds{tool}`` (the same
  names as player-mcp; the scrape labels ``service=player-obs``), every tool pre-registered at 0.
- **analytics**: an Umami ``obs_tool:<name>`` event, filed under the caller's analytics id.

It mirrors :mod:`podcast_scraper.mcp.telemetry` rather than importing it: the obs image ships only
``src/podcast_obs``. Every path is best-effort — telemetry never breaks a tool call or startup, and
each signal is a no-op when its env or its package is absent (stdio, dev, tests).

Privacy: only the tool name, ``ok``, timing, the user id (logs/spans) and the analytics id
(Umami) leave the process — never tool arguments or results.
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
import os
import threading
import time
import urllib.request
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterable, Iterator, Optional
from urllib.parse import urlparse

from .auth import current_obs_participant, current_obs_user

logger = logging.getLogger(__name__)
_TOOL_LOGGER = logging.getLogger("podcast_obs.tool")
_ACCESS_LOGGER = logging.getLogger("podcast_obs.access")

SERVICE_NAME = "player-obs"
_SENTRY_DSN_ENV = "PODCAST_SENTRY_DSN_OBS"
_DISCOVERY_PATH = "/.well-known/oauth-protected-resource"

# Umami: the same site + collect URL as player-mcp; the event name keeps the two MCPs apart.
_UMAMI_WEBSITE_ID = "PODCAST_MCP_UMAMI_WEBSITE_ID"
_UMAMI_COLLECT_URL = "PODCAST_MCP_UMAMI_URL"
_UMAMI_HOSTNAME = "PODCAST_MCP_UMAMI_HOSTNAME"
# Umami's bot filter drops server-ish UAs silently (200 {"beep":"boop"}, nothing stored); a browser
# UA with the sender's name appended passes (probed against the homelab Umami 2026-10-16).
_UMAMI_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36 podcast-obs/1.0"
)
_UMAMI_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="umami-obs")
_UMAMI_INFLIGHT = threading.BoundedSemaphore(8)

# A registry of its own: the same metric names as podcast_scraper.mcp.telemetry would collide in
# the default registry whenever both modules load in one process (the test suite does).
try:  # pragma: no cover - exercised via the metrics endpoint
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        GCCollector,
        Histogram,
        PlatformCollector,
        ProcessCollector,
    )

    _registry = CollectorRegistry()
    ProcessCollector(registry=_registry)
    PlatformCollector(registry=_registry)
    GCCollector(registry=_registry)
    REGISTRY: Optional[Any] = _registry
    _CALLS: Optional[Any] = Counter(
        "mcp_tool_calls_total",
        "MCP tool calls by tool + outcome",
        ["tool", "ok"],
        registry=REGISTRY,
    )
    _DURATION: Optional[Any] = Histogram(
        "mcp_tool_duration_seconds",
        "MCP tool call wall-clock duration",
        ["tool"],
        registry=REGISTRY,
    )
except Exception:  # noqa: BLE001 - metrics are optional; never break import
    REGISTRY = None
    _CALLS = None
    _DURATION = None


# ---------------------------------------------------------------- process setup


def configure_logging() -> None:
    """INFO to stderr in the format the other player containers use, when nothing configured it.

    Without this the ``podcast_obs`` loggers sat at Python's WARNING default, so the admission and
    tool lines were never written.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def init_sentry() -> bool:
    """Send unhandled errors (and ERROR logs) to GlitchTip when ``PODCAST_SENTRY_DSN_OBS`` is set.

    Its own GlitchTip project, so obs failures don't land among the api's or the content MCP's.
    """
    dsn = os.environ.get(_SENTRY_DSN_ENV, "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("PODCAST_ENV", "").strip() or "dev",
            release=os.environ.get("PODCAST_RELEASE", "").strip() or None,
            traces_sample_rate=0.0,
            send_default_pii=False,
            # This process holds read tokens for every backend; frame locals must not leave it.
            include_local_variables=False,
        )
        sentry_sdk.set_tag("component", "obs")
        return True
    except Exception:  # noqa: BLE001 - error reporting never blocks serving
        logger.debug("obs sentry init skipped", exc_info=True)
        return False


def init_otel() -> bool:
    """Install an OTLP ``TracerProvider`` when ``OTEL_TRACES_EXPORTER=otlp`` + an endpoint are set.

    ``Resource.create()`` reads ``OTEL_SERVICE_NAME`` + ``OTEL_RESOURCE_ATTRIBUTES``; the exporter
    reads the traces endpoint and protocol. The endpoint var is also obs's own read base for
    VictoriaTraces, so the opt-in is ``OTEL_TRACES_EXPORTER``, not the endpoint.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
    if os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower() != "otlp":
        return False
    if not (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    ):
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if isinstance(trace.get_tracer_provider(), TracerProvider):
            return True
        provider = TracerProvider(resource=Resource.create())
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        return True
    except Exception:  # noqa: BLE001 - tracing never blocks serving
        logger.debug("obs otel init skipped", exc_info=True)
        return False


def current_trace_id() -> str:
    """The active span's 32-hex trace id, or ``-``."""
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:  # noqa: BLE001
        pass
    return "-"


# ---------------------------------------------------------------- tool calls


def preregister_tools(names: Iterable[str]) -> None:
    """Create every tool's series at 0, so an unused tool reads 0 rather than absent."""
    if _CALLS is None or _DURATION is None:
        return
    for name in names:
        try:
            _CALLS.labels(tool=name, ok="true")
            _CALLS.labels(tool=name, ok="false")
            _DURATION.labels(tool=name)
        except Exception:  # noqa: BLE001
            pass


def _request_state() -> dict[str, Any]:
    """``scope["state"]`` of the HTTP request carrying this tool call, or ``{}`` (stdio, tests).

    FastMCP runs a tool in the task of the request that opened the MCP session, so ContextVars
    (and the OTel current span) there belong to that first request: every call in a session was
    traced and attributed to it. The SDK's ``request_ctx`` holds the actual CallTool request.
    """
    try:
        from mcp.server.lowlevel.server import request_ctx

        scope = getattr(request_ctx.get().request, "scope", None) or {}
        state = scope.get("state")
        return state if isinstance(state, dict) else {}
    except Exception:  # noqa: BLE001 - no request in flight
        return {}


@contextmanager
def _maybe_span(tool: str, parent: Any = None) -> Iterator[Any]:
    # Resolved outside any ``except`` so a failing tool body propagates cleanly through the yield.
    span_cm = None
    try:
        from opentelemetry import trace

        span_cm = trace.get_tracer("podcast_obs").start_as_current_span(
            f"mcp.tool.{tool}", context=parent
        )
    except Exception:  # noqa: BLE001
        span_cm = None
    if span_cm is None:
        yield None
        return
    with span_cm as span:
        yield span


def observed(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap a tool so each call emits span + log + metric + Umami event.

    ``functools.wraps`` keeps the signature and docstring FastMCP builds the tool schema from.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        tool = fn.__name__
        ok = False
        state = _request_state()
        start = time.perf_counter()
        with _maybe_span(tool, state.get("obs_otel_context")) as span:
            try:
                result = fn(*args, **kwargs)
                ok = not (isinstance(result, dict) and result.get("ok") is False)
                return result
            finally:
                _emit(tool, ok, time.perf_counter() - start, span, state.get("obs_principal"))

    return wrapper


def _emit(
    tool: str, ok: bool, duration_s: float, span: Any, principal: Optional[dict] = None
) -> None:
    if principal:
        user, participant = principal.get("user_id"), principal.get("analytics_id")
    else:
        try:
            user = current_obs_user.get()
            participant = current_obs_participant.get()
        except Exception:  # noqa: BLE001
            user, participant = None, None
    if span is not None:
        try:
            span.set_attribute("mcp.tool", tool)
            span.set_attribute("mcp.ok", ok)
            if user:
                span.set_attribute("mcp.user_id", user)
            if participant:
                span.set_attribute("mcp.participant", participant)
        except Exception:  # noqa: BLE001
            pass
    if _CALLS is not None and _DURATION is not None:
        try:
            _CALLS.labels(tool=tool, ok=str(ok).lower()).inc()
            _DURATION.labels(tool=tool).observe(duration_s)
        except Exception:  # noqa: BLE001
            pass
    try:
        _TOOL_LOGGER.info(
            "mcp tool call tool=%s user=%s participant=%s ok=%s duration_ms=%.1f trace_id=%s",
            tool,
            user or "-",
            participant or "-",
            ok,
            duration_s * 1000.0,
            current_trace_id(),
        )
    except Exception:  # noqa: BLE001
        pass
    _emit_umami(tool, ok, participant)


def _umami_hostname() -> str:
    host = os.environ.get(_UMAMI_HOSTNAME, "").strip() or "obs"
    if "://" in host:
        host = urlparse(host).hostname or host
    return host


def _emit_umami(tool: str, ok: bool, participant: Optional[str]) -> None:
    """Fire-and-forget ``obs_tool:<tool>``; dropped (never queued) past 8 sends in flight."""
    website = os.environ.get(_UMAMI_WEBSITE_ID, "").strip()
    url = os.environ.get(_UMAMI_COLLECT_URL, "").strip()
    if not website or not url:
        return
    body = json.dumps(
        {
            "type": "event",
            "payload": {
                "website": website,
                "hostname": _umami_hostname(),
                "name": f"obs_tool:{tool}",
                "data": {"tool": tool, "ok": ok},
                **({"id": participant} if participant else {}),
            },
        }
    ).encode()
    if not _UMAMI_INFLIGHT.acquire(blocking=False):
        return

    def _send() -> None:
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": _UMAMI_USER_AGENT},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=4)  # noqa: S310 - collect URL from deploy env
        except Exception:  # noqa: BLE001 - analytics never breaks a tool call
            pass
        finally:
            _UMAMI_INFLIGHT.release()

    try:
        _UMAMI_POOL.submit(_send)
    except Exception:  # noqa: BLE001
        _UMAMI_INFLIGHT.release()


# ---------------------------------------------------------------- ASGI stack

_Send = Callable[[dict[str, Any]], Awaitable[None]]
_Receive = Callable[[], Awaitable[dict[str, Any]]]


def _is_healthcheck(scope: dict[str, Any]) -> bool:
    # The compose healthcheck GETs the discovery doc from inside the container every 30 s; real
    # clients arrive through Caddy and the docker bridge, never from loopback.
    client = scope.get("client") or ("", 0)
    return scope.get("path") == _DISCOVERY_PATH and client[0] in {"127.0.0.1", "::1"}


class AccessLogMiddleware:
    """One ``METHOD path -> status in Nms trace_id=<id>`` line per request (not /metrics).

    Sits inside the OTel server span, so the trace id is live when the response starts.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: _Receive, send: _Send) -> None:
        if scope.get("type") != "http" or scope.get("path") == "/metrics":
            await self._app(scope, receive, send)
            return
        start = time.perf_counter()
        seen = {"status": 0, "trace": "-"}
        try:
            from opentelemetry import context as otel_context

            # The request's server span, for the tool span to parent on (see _request_state).
            scope.setdefault("state", {})["obs_otel_context"] = otel_context.get_current()
        except Exception:  # noqa: BLE001
            pass

        async def _send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                seen["status"] = message.get("status", 0)
                seen["trace"] = current_trace_id()
            await send(message)

        try:
            await self._app(scope, receive, _send)
        finally:
            _ACCESS_LOGGER.info(
                "%s %s -> %s in %.1fms trace_id=%s",
                scope.get("method", "?"),
                scope.get("path", ""),
                seen["status"],
                (time.perf_counter() - start) * 1000.0,
                seen["trace"],
            )


def _with_metrics(app: Any) -> Any:
    """Serve Prometheus ``/metrics`` without a bearer.

    Internal only: the container port is bound to host loopback and the obs Caddy vhost allows
    just ``/mcp`` and ``/.well-known/*``; Alloy scrapes the docker-network IP.
    """
    if REGISTRY is None:
        return app
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    async def _asgi(scope: dict[str, Any], receive: _Receive, send: _Send) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/metrics":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", CONTENT_TYPE_LATEST.encode())],
                }
            )
            await send({"type": "http.response.body", "body": generate_latest(REGISTRY)})
            return
        await app(scope, receive, send)

    return _asgi


def _instrument_asgi(app: Any) -> Any:
    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        return OpenTelemetryMiddleware(app, excluded_urls="/metrics$")
    except Exception:  # noqa: BLE001 - no instrumentor → serve un-instrumented
        logger.debug("obs ASGI instrumentation skipped", exc_info=True)
        return app


def http_stack(authed_app: Any) -> Any:
    """``authed_app`` (auth-wrapped MCP app) inside: access log, ``/metrics``, OTel spans.

    The container's own healthcheck skips all three, so it neither traces nor logs.
    """
    instrumented = _instrument_asgi(_with_metrics(AccessLogMiddleware(authed_app)))

    async def _asgi(scope: dict[str, Any], receive: _Receive, send: _Send) -> None:
        if scope.get("type") == "http" and _is_healthcheck(scope):
            await authed_app(scope, receive, send)
            return
        await instrumented(scope, receive, send)

    return _asgi
