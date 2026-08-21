"""Tests for the in-app OTLP tracing bootstrap — the enable gate (no-op unless env asks)."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

_ENV = (
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


def _fresh(monkeypatch, **env):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import podcast_scraper.utils.otel_init as m

    return importlib.reload(m)


def test_disabled_by_default(monkeypatch):
    m = _fresh(monkeypatch)
    assert m.otel_tracing_enabled() is False
    assert m.init_otel() is False  # true no-op


def test_requires_both_exporter_and_endpoint(monkeypatch):
    # exporter set but no endpoint -> still disabled
    m = _fresh(monkeypatch, OTEL_TRACES_EXPORTER="otlp")
    assert m.otel_tracing_enabled() is False
    # both -> enabled
    m = _fresh(
        monkeypatch,
        OTEL_TRACES_EXPORTER="otlp",
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://backend:10428/insert/opentelemetry/v1/traces",
    )
    assert m.otel_tracing_enabled() is True


def test_non_otlp_exporter_disabled(monkeypatch):
    m = _fresh(
        monkeypatch,
        OTEL_TRACES_EXPORTER="console",
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://backend:10428",
    )
    assert m.otel_tracing_enabled() is False


def test_episode_span_noop_when_disabled(monkeypatch):
    # The root-span helper is a TRUE no-op when tracing is off: it yields None and never raises, so
    # wrapping every episode in it costs nothing on the packaged image (no OTEL env). The enabled
    # path — the span carrying run_id/episode_id/feed_id — is proven live against VictoriaTraces
    # (kept out of unit since it needs the optional [otel] extra; see the integration trace test).
    m = _fresh(monkeypatch)
    assert m.otel_tracing_enabled() is False
    with m.episode_span(run_id="r", episode_id="e", feed_id="https://f") as span:
        assert span is None


def test_wrap_with_current_context_noop_when_disabled(monkeypatch):
    """With OTEL off, the wrapper returns the callable UNCHANGED (identity) — zero overhead."""
    m = _fresh(monkeypatch)
    assert m.otel_tracing_enabled() is False

    def fn(x: int) -> int:
        return x * 2

    assert m.wrap_with_current_context(fn) is fn
    assert m.wrap_with_current_context(fn)(3) == 6


def test_wrap_with_current_context_propagates_span_into_worker_thread(monkeypatch):
    """The fix for `trace=-`: a ThreadPoolExecutor worker does NOT inherit the caller's active span,
    so it loses the trace_id; the wrapper re-attaches the submit-time context inside the worker."""
    from concurrent.futures import ThreadPoolExecutor

    m = _fresh(
        monkeypatch,
        OTEL_TRACES_EXPORTER="otlp",
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://backend:10428/insert/opentelemetry/v1/traces",
    )
    assert m.otel_tracing_enabled() is True

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer("test.wrap_with_current_context")

    def worker_trace_id() -> int:
        # int() is explicit: some opentelemetry versions type trace_id as Any (no-any-return under
        # a no-ML .[dev] venv, where the richer stubs aren't pulled in); it is always an int.
        return int(trace.get_current_span().get_span_context().trace_id)

    with tracer.start_as_current_span("root"):
        expected = trace.get_current_span().get_span_context().trace_id
        assert expected != 0  # a real, recording span
        with ThreadPoolExecutor(max_workers=1) as ex:
            naive = ex.submit(worker_trace_id).result()
            wrapped = ex.submit(m.wrap_with_current_context(worker_trace_id)).result()

    assert naive != expected  # naive worker lost the span context (the bug)
    assert wrapped == expected  # wrapper restored it (the fix)
