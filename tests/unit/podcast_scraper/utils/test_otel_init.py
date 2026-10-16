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


class _FakeLoggingInstrumentor:
    """Stands in for opentelemetry's LoggingInstrumentor (unit tests use [dev] extras only)."""

    instrumented = False
    calls: list = []

    @property
    def is_instrumented_by_opentelemetry(self):
        return type(self).instrumented

    def instrument(self, **kw):
        type(self).calls.append(kw)
        type(self).instrumented = True


def _fake_logging_module(monkeypatch, instrumented=False):
    import sys
    import types

    _FakeLoggingInstrumentor.instrumented = instrumented
    _FakeLoggingInstrumentor.calls = []
    mod = types.ModuleType("opentelemetry.instrumentation.logging")
    mod.LoggingInstrumentor = _FakeLoggingInstrumentor
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.logging", mod)


def test_log_correlation_off_leaves_logging_alone(monkeypatch):
    m = _fresh(monkeypatch)
    monkeypatch.delenv("OTEL_PYTHON_LOG_CORRELATION", raising=False)
    _fake_logging_module(monkeypatch)
    m._instrument_logging()
    assert _FakeLoggingInstrumentor.calls == []


def test_log_correlation_on_instruments_logging_with_trace_format(monkeypatch):
    # player-mcp starts without the opentelemetry-instrument launcher; with the env flag set,
    # init_otel's logging step must instrument logging itself (trace_id/span_id in the format).
    m = _fresh(monkeypatch, OTEL_PYTHON_LOG_CORRELATION="true")
    _fake_logging_module(monkeypatch)
    m._instrument_logging()
    assert _FakeLoggingInstrumentor.calls == [{"set_logging_format": True}]


def test_log_correlation_skipped_when_launcher_already_did_it(monkeypatch):
    # api/pipeline: the launcher instrumented logging first; a second instrument() only warns.
    m = _fresh(monkeypatch, OTEL_PYTHON_LOG_CORRELATION="true")
    _fake_logging_module(monkeypatch, instrumented=True)
    m._instrument_logging()
    assert _FakeLoggingInstrumentor.calls == []


def test_default_service_name_fills_only_an_unset_name(monkeypatch):
    # Every process names itself: the CLI is the pipeline, the server passes podcast-api.
    # Runs even when tracing is off, because the dev log/metric push reads the same name.
    m = _fresh(monkeypatch)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    m.init_otel()
    import os

    assert os.environ["OTEL_SERVICE_NAME"] == "podcast-pipeline"
    monkeypatch.setenv("OTEL_SERVICE_NAME", "  ")
    m.init_otel(default_service_name="podcast-api")
    assert os.environ["OTEL_SERVICE_NAME"] == "podcast-api"


def test_explicit_service_name_is_never_overridden(monkeypatch):
    # prod compose sets OTEL_SERVICE_NAME for every service (player-api, player-mcp, ...)
    m = _fresh(monkeypatch)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "player-api")
    m.init_otel(default_service_name="podcast-api")
    import os

    assert os.environ["OTEL_SERVICE_NAME"] == "player-api"


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["mcp", "--transport", "http"], "player-mcp"),
        (["serve", "--port", "8000"], "podcast-api"),
        (["--rss", "https://example.com/feed.xml"], "podcast-pipeline"),
        ([], "podcast-pipeline"),
    ],
)
def test_cli_names_the_process_by_subcommand(argv, expected):
    # main() initialises tracing before dispatching, so the subcommand picks the default name
    # (prod sets OTEL_SERVICE_NAME explicitly; this is what an unset dev process calls itself).
    from podcast_scraper import cli

    assert cli._default_service_name(argv) == expected


def test_under_the_auto_instrument_launcher_init_otel_adds_nothing(monkeypatch):
    """The api image starts under ``opentelemetry-instrument`` and ALSO calls init_otel(). A second
    provider + a second HTTP instrumentation only logged "Overriding of current TracerProvider is
    not allowed" and "Attempting to instrument while already instrumented" (2026-10-15 prod: 28
    lines across podcast-api, operator-api, player-api per deploy). An SDK provider already in
    place means tracing is on: init_otel keeps it and instruments nothing again."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    m = _fresh(
        monkeypatch,
        OTEL_TRACES_EXPORTER="otlp",
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://backend:10428/insert/opentelemetry/v1/traces",
    )
    # Under the launcher the exporter is not needed, so its import must not decide the answer.
    # CI 2026-10-15 (run 38796124599): an earlier test's MagicMock `google` package (#1799) made
    # `google.protobuf`, and with it the OTLP exporter, unimportable, and init_otel returned False.
    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    existing = TracerProvider()
    set_calls, http_calls = [], []
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: existing)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: set_calls.append(p))
    monkeypatch.setattr(m, "_instrument_http", lambda: http_calls.append(1))

    assert m.init_otel() is True
    assert set_calls == [] and http_calls == []
    assert m.init_otel() is False  # idempotent
