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
