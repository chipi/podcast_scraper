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
import importlib
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_LOGGER = logging.getLogger(__name__)
_initialized = False


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
        atexit.register(provider.shutdown)
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
    attributes = {
        k: v for k, v in (("run_id", run_id), ("episode_id", episode_id), ("feed_id", feed_id)) if v
    }
    try:
        tracer = trace.get_tracer("podcast_scraper.pipeline")
        with tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span
    except Exception:  # noqa: BLE001 — telemetry must never break episode processing
        _LOGGER.debug("episode_span failed", exc_info=True)
        yield None


def _reset_for_tests() -> None:
    """Test hook: allow re-init after env changes."""
    global _initialized
    _initialized = False
