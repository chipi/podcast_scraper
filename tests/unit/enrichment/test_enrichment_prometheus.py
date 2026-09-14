"""Enrichment must reach VictoriaMetrics, not only the corpus files (#2071).

Measured on prod 2026-09-14: `{__name__=~"enrichment.*"}` and `{__name__=~"enricher.*"}` were
both EMPTY while the pipeline had podcast_pipeline_run_* histograms. So enrichment could be
inspected on demand via /api/enrichment/*, but nothing could ALERT on it — not "this enricher
has failed its last N runs", not "enrichment stopped happening".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from podcast_scraper.server import enrichment_run_prometheus as erp  # noqa: E402

_SUMMARY = {
    "duration_ms": 45032,
    "status": "ok",
    "run_id": "r-1",
    "per_enricher": {
        "org_web": {
            "duration_ms": 37207,
            "records_written": 0,  # ok status, zero rows — the org_web failure mode
            "retries": 0,
            "runs_ok": 1,
            "runs_failed": 0,
            "cost_usd": 0.0,
        },
        "person_web": {
            "duration_ms": 1334,
            "records_written": 1,
            "retries": 2,
            "runs_ok": 0,
            "runs_failed": 1,
            "cost_usd": 0.25,
        },
    },
}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    enr = tmp_path / "enrichments"
    enr.mkdir(parents=True)
    (enr / "run_summary.json").write_text(json.dumps(_SUMMARY), encoding="utf-8")
    return tmp_path


def _scrape() -> str:
    raw: bytes = prometheus_client.generate_latest(prometheus_client.REGISTRY)
    return raw.decode()


def test_disabled_by_default_emits_nothing(corpus: Path, monkeypatch) -> None:
    """Same gate as the pipeline exporter: silent unless PODCAST_METRICS_ENABLED."""
    monkeypatch.delenv("PODCAST_METRICS_ENABLED", raising=False)
    erp._PROM_STATE.clear()
    erp._PROM_STATE["done"] = False
    erp.observe_enrichment_terminal_metrics(corpus, {"status": "succeeded"})
    assert erp._PROM_STATE.get("runs") is None


def test_per_enricher_outcomes_are_alertable(corpus: Path, monkeypatch) -> None:
    """The series that answers 'is this enricher healthy' must carry id + outcome labels."""
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    erp.observe_enrichment_terminal_metrics(corpus, {"status": "succeeded"})
    out = _scrape()
    assert (
        'podcast_enrichment_enricher_runs_total{enricher_id="person_web",outcome="failed"}' in out
    )
    assert 'podcast_enrichment_enricher_runs_total{enricher_id="org_web",outcome="ok"}' in out
    assert 'podcast_enrichment_runs_total{status="succeeded"}' in out
    # Throughput: "ok but wrote nothing" is only visible with a records series.
    assert 'podcast_enrichment_records_written_total{enricher_id="person_web"}' in out
    assert 'podcast_enrichment_retries_total{enricher_id="person_web"}' in out


def test_a_missing_summary_never_raises(tmp_path: Path, monkeypatch) -> None:
    """Telemetry is additive: no run_summary must not fail the job's finalize path."""
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    erp.observe_enrichment_terminal_metrics(tmp_path, {"status": "succeeded"})  # must not raise


def test_non_terminal_status_is_ignored(corpus: Path, monkeypatch) -> None:
    """A running job has no totals to record yet."""
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    before = _scrape().count("podcast_enrichment_runs_total")
    erp.observe_enrichment_terminal_metrics(corpus, {"status": "running"})
    assert _scrape().count("podcast_enrichment_runs_total") == before


# --- traces (#2071) ----------------------------------------------------------------------------
# VictoriaTraces knew enrichment only as HTTP routes (GET /api/enrichment/events) while the
# pipeline had a real domain span, episode.process. The WEB tier's outbound Wikipedia / Wikidata
# calls were therefore parentless auto-instrumented spans attributable to nothing — exactly where
# it hurt, since org_web spent 45s on five organizations against a rate-limited upstream.


def test_enrichment_span_is_a_noop_when_otel_is_off(monkeypatch) -> None:
    """Must be a TRUE no-op without OTEL — enrichment cannot depend on tracing being wired."""
    from podcast_scraper.utils import otel_init

    monkeypatch.setattr(otel_init, "otel_tracing_enabled", lambda: False)
    with otel_init.enrichment_span(run_id="r", enricher_id="org_web", tier="web") as span:
        assert span is None


def test_enrichment_span_never_raises(monkeypatch) -> None:
    """A broken tracer must yield None, not raise — telemetry cannot fail a run."""
    from podcast_scraper.utils import otel_init

    monkeypatch.setattr(otel_init, "otel_tracing_enabled", lambda: True)

    class _BrokenTrace:
        @staticmethod
        def get_tracer(_name):
            raise RuntimeError("tracer exploded")

    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry", type(sys)("opentelemetry"))
    sys.modules["opentelemetry"].trace = _BrokenTrace  # type: ignore[attr-defined]

    with otel_init.enrichment_span(run_id="r", enricher_id="org_web", tier="web") as span:
        assert span is None, "a failing tracer must degrade to None, never propagate"


def test_enrichment_span_stamps_correlation_attributes(monkeypatch) -> None:
    """run_id / enricher_id / tier are what make run -> trace pivotable, as for episode.process."""
    from podcast_scraper.utils import otel_init

    monkeypatch.setattr(otel_init, "otel_tracing_enabled", lambda: True)
    captured: dict = {}

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Tracer:
        @staticmethod
        def start_as_current_span(name, attributes=None):
            captured["name"] = name
            captured["attributes"] = dict(attributes or {})
            return _Span()

    class _Trace:
        @staticmethod
        def get_tracer(_name):
            return _Tracer

    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry", type(sys)("opentelemetry"))
    sys.modules["opentelemetry"].trace = _Trace  # type: ignore[attr-defined]

    with otel_init.enrichment_span(run_id="r-1", enricher_id="org_web", tier="web"):
        pass

    assert captured["name"] == "enrichment.enricher"
    assert captured["attributes"]["run_id"] == "r-1"
    assert captured["attributes"]["enricher_id"] == "org_web"
    assert captured["attributes"]["tier"] == "web"
