"""Optional in-app OpenTelemetry tracing bootstrap (ADR-119 "traces" signal).

Enable-when-env-present, mirroring :mod:`podcast_scraper.utils.sentry_init` and
:mod:`podcast_scraper.utils.langfuse_tracing`: a TRUE no-op unless ``OTEL_TRACES_EXPORTER=otlp``
AND an OTLP traces endpoint are set. When enabled it installs a global ``TracerProvider`` that
exports OTLP/HTTP to the configured endpoint (VictoriaTraces) and instruments the outbound HTTP
libraries, so provider calls (LLM / diarization / media) become spans — grouped by
``service.name`` + ``deployment.environment`` from ``OTEL_RESOURCE_ATTRIBUTES``.

Why in-app instead of the ``opentelemetry-instrument`` launcher: that launcher's argparse
abbreviation-matches the CLI's own ``--config`` flag, so it can't wrap ``python -m …cli --config``.
Self-init works for every entrypoint (cli, serve) with no wrapper. Both approaches no-op without
the OTEL env, so the packaged image can use either. Every path is guarded — tracing never breaks
the app.
"""

from __future__ import annotations

import atexit
import functools
import importlib
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

_LOGGER = logging.getLogger(__name__)
_initialized = False

# Hard cap on the span flush at process exit. A batch / containerised process is SIGTERM'd on
# deploy; the default ``BatchSpanProcessor.shutdown`` keeps retrying the OTLP export and, when the
# endpoint is briefly unreachable during the stop, logs "Failed to export span batch due to timeout"
# — which the Sentry logging integration then ships to GlitchTip as an application error (73 of them
# on 2026-07-28's prod deploy). Bounding the flush makes it fail fast and quietly: tracing is
# best-effort, and a span dropped at shutdown must never surface as an error. Overridable via env.
_SHUTDOWN_FLUSH_MS = int(os.environ.get("OTEL_BSP_SHUTDOWN_FLUSH_MS", "2000") or "2000")


def _bounded_shutdown(provider: Any) -> None:
    """Flush pending spans within ``_SHUTDOWN_FLUSH_MS``, then shut the provider down — swallowing
    any failure. Registered in place of the bare ``provider.shutdown`` so a slow/unreachable OTLP
    endpoint at process exit cannot hang the stop or raise a GlitchTip-visible error."""
    try:
        provider.force_flush(timeout_millis=_SHUTDOWN_FLUSH_MS)
    except Exception:  # noqa: BLE001 — telemetry must never break shutdown
        pass
    try:
        provider.shutdown()
    except Exception:  # noqa: BLE001
        pass


def otel_tracing_enabled() -> bool:
    """True when OTLP trace export is requested via env (the enable signal)."""
    if os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower() != "otlp":
        return False
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )


def _instrument_http() -> None:
    """Instrument the outbound HTTP libs so provider calls become client spans."""
    for modpath, cls in (
        ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
        ("opentelemetry.instrumentation.urllib3", "URLLib3Instrumentor"),
    ):
        try:
            getattr(importlib.import_module(modpath), cls)().instrument()
        except Exception:  # noqa: BLE001 — a missing/failed instrumentor must not break tracing
            _LOGGER.debug("otel instrument %s skipped", cls, exc_info=True)


def init_otel() -> bool:
    """Initialise OTLP tracing if the env asks for it. Returns True when activated, else False.

    Idempotent no-op on repeat calls, when the env signal is absent, or when the ``[otel]`` extra
    is not installed. ``Resource.create()`` picks up ``OTEL_SERVICE_NAME`` + ``OTEL_RESOURCE_
    ATTRIBUTES``; the OTLP exporter picks up ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` + protocol.
    """
    global _initialized
    if _initialized or not otel_tracing_enabled():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        _LOGGER.debug("otel tracing requested but the [otel] extra is not installed")
        return False

    try:
        provider = TracerProvider(resource=Resource.create())
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _instrument_http()
        atexit.register(_bounded_shutdown, provider)
    except Exception:  # noqa: BLE001 — telemetry must never break the app
        _LOGGER.debug("otel init failed", exc_info=True)
        return False

    _initialized = True
    _LOGGER.info(
        "otel tracing initialised (service=%s environment=%s)",
        os.environ.get("OTEL_SERVICE_NAME", "?"),
        os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "?"),
    )
    return True


@contextmanager
def episode_span(
    *,
    run_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    feed_id: Optional[str] = None,
    name: str = "episode.process",
) -> Iterator[Any]:
    """Root span for ONE episode — the missing link that makes traces pivotable.

    Without a per-episode root span the provider HTTP spans (transcription / diarization / gemini,
    auto-instrumented by :func:`init_otel`) are parentless and carry no correlation, and
    ``emit_event`` fires outside any span so its events get no ``trace_id``. Wrapping the episode in
    this span (a) gives same-thread HTTP spans a parent, (b) stamps ``run_id`` / ``episode_id`` /
    ``feed_id`` as span attributes so an agent can query VictoriaTraces by run and pivot run→trace,
    and (c) lets ``emit_event``'s ``_trace_context`` join same-thread events to that trace. (Stages
    that run in their own executor threads don't inherit the span's context — see the module note.)

    A TRUE no-op unless OTEL tracing is active (env signal + the ``[otel]`` extra); never raises.
    """
    if not otel_tracing_enabled():
        yield None
        return
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return
    # #1873: a span should be readable without pivoting to logs to learn how the run was
    # configured. Profile + the resolved stage routing come from the shared context, so any
    # future span gets them without its caller passing anything.
    _ctx: dict[str, Any] = {}
    try:
        from podcast_scraper.obs.events import get_run_context as _get_run_context
        from podcast_scraper.utils import correlation as _corr

        _ctx = {k: v for k, v in _get_run_context().items() if v is not None}
        if not feed_id:
            feed_id = _corr.get_feed_id()
    except Exception:  # noqa: BLE001 - telemetry context is best-effort
        _ctx = {}
    attributes = {
        k: v
        for k, v in (
            ("run_id", run_id),
            ("episode_id", episode_id),
            ("feed_id", feed_id),
            *_ctx.items(),
        )
        if v
    }
    try:
        tracer = trace.get_tracer("podcast_scraper.pipeline")
        with tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span
    except Exception:  # noqa: BLE001 — telemetry must never break episode processing
        _LOGGER.debug("episode_span failed", exc_info=True)
        yield None


def wrap_with_current_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` so it runs under the OTEL context active at wrap time.

    ``ThreadPoolExecutor`` workers do NOT inherit the submitting thread's OTEL context, so a stage
    that runs in a nested executor emits its logs/events with NO active span — ``trace_id`` renders
    ``"-"`` and ``emit_event``'s ``_trace_context`` can't join them to the episode's root span.
    Capturing the current context at submit and attaching it inside the worker restores the join
    key (``run_id`` is process-global and ``episode_id`` is passed explicitly, but ``trace_id`` is
    read from the *current span*, which is thread-local). TRUE no-op — returns ``fn`` unchanged —
    when OTEL tracing is off or the API is unavailable; never raises.
    """
    if not otel_tracing_enabled():
        return fn
    try:
        from opentelemetry import context as _otel_context
    except ImportError:
        return fn
    ctx = _otel_context.get_current()

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        token = _otel_context.attach(ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            _otel_context.detach(token)

    return _wrapped


def _reset_for_tests() -> None:
    """Test hook: allow re-init after env changes."""
    global _initialized
    _initialized = False
