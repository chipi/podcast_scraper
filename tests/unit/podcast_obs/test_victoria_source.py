"""Phase-A: VictoriaLogs/Metrics/Traces sources + GlitchTip base-url + config fields."""

from __future__ import annotations

from podcast_obs.config import ObservabilityConfig, TargetConfig
from podcast_obs.sources import sentry, victoria


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


# --- VictoriaLogs events -----------------------------------------------------------


def test_events_not_configured() -> None:
    r = victoria.events(_t(), "pipeline_stage")
    assert r["ok"] is False and r["configured"] is False


def test_events_builds_logsql_with_surface_and_correlation(monkeypatch) -> None:
    seen = {}

    def _fake(url, **kw):
        seen["url"] = url
        seen["query"] = kw["params"]["query"]
        return [{"event_type": "pipeline_stage", "stage": "asr"}]

    monkeypatch.setattr(victoria, "get_ndjson", _fake)
    r = victoria.events(
        _t(victorialogs_url="http://homelab:9428"),
        "pipeline_stage",
        surface="pipeline",
        run_id="run-1",
        episode_id="ep-1",
    )
    assert r["ok"] and r["data"]["count"] == 1
    q = seen["query"]
    # The event type is matched against `_msg`, NOT an `event_type` field: emit_event ships with
    # `_msg_field: event_type` (dev_push.py), so VictoriaLogs stores the type as the built-in `_msg`
    # message field and keeps no `event_type` field. Live-verified — an `event_type:` filter returns
    # zero rows against real pushed data, which this assertion previously (wrongly) allowed.
    assert '_msg:"pipeline_stage"' in q
    assert "event_type:" not in q
    assert 'surface:"pipeline"' in q and 'component:"pipeline"' in q  # surface OR component
    assert 'run_id:"run-1"' in q and 'episode_id:"ep-1"' in q
    assert seen["url"].endswith("/select/logsql/query")


# --- VictoriaMetrics ---------------------------------------------------------------


def test_metrics_instant_parses_series(monkeypatch) -> None:
    monkeypatch.setattr(
        victoria,
        "get_json",
        lambda url, **_: {
            "data": {"result": [{"metric": {"service": "podcast-api"}, "value": [0, "3.5"]}]}
        },
    )
    r = victoria.metrics_instant(_t(victoriametrics_url="http://homelab:8428"), "up")
    assert r["ok"] and r["data"]["series"][0]["value"] == [0, "3.5"]


def test_red_metrics_builds_three_queries(monkeypatch) -> None:
    queries = []

    def _fake(url, **kw):
        queries.append(kw["params"]["query"])
        return {"data": {}}

    monkeypatch.setattr(victoria, "get_json", _fake)
    r = victoria.red_metrics(_t(victoriametrics_url="http://h:8428"), "podcast-api")
    assert r["ok"]
    assert any("http_requests_total" in q for q in queries)
    assert any('status=~"5.."' in q for q in queries)  # error rate
    assert any("histogram_quantile(0.95" in q for q in queries)  # p95


# --- VictoriaTraces ----------------------------------------------------------------


def test_traces_recent_not_configured() -> None:
    assert victoria.traces_recent(_t(), "podcast-api")["configured"] is False


def test_trace_by_id_hits_jaeger_endpoint(monkeypatch) -> None:
    seen = {}

    def _fake(url, **_):
        seen["url"] = url
        return {"data": ["span"]}

    monkeypatch.setattr(victoria, "get_json", _fake)
    r = victoria.trace_by_id(_t(victoriatraces_url="http://h:10428"), "abc123")
    assert r["ok"] and r["data"]["trace"] == ["span"]
    assert seen["url"].endswith("/select/jaeger/api/traces/abc123")


def test_traces_by_run_not_configured() -> None:
    assert victoria.traces_by_run(_t(), "run-1")["configured"] is False


def test_traces_by_run_filters_by_run_id_tag(monkeypatch) -> None:
    # The run→trace pivot: filter the Jaeger API by the run_id span tag (stamped by the
    # episode.process root span), scoped to the pipeline service.
    seen = {}

    def _fake(url, **kw):
        seen["url"] = url
        seen["params"] = kw["params"]
        return {"data": [{"traceID": "t1"}]}

    monkeypatch.setattr(victoria, "get_json", _fake)
    r = victoria.traces_by_run(_t(victoriatraces_url="http://h:10428"), "run-9")
    assert r["ok"] and r["data"]["count"] == 1 and r["data"]["run_id"] == "run-9"
    assert seen["url"].endswith("/select/jaeger/api/traces")
    assert seen["params"]["service"] == "pipeline"
    assert seen["params"]["tags"] == '{"run_id": "run-9"}'


# --- GlitchTip base-url (errors source) --------------------------------------------


def test_sentry_api_base_defaults_to_saas() -> None:
    assert sentry._api_base(_t()) == "https://sentry.io/api/0"


def test_sentry_api_base_uses_glitchtip_url() -> None:
    t = _t(sentry_url="http://homelab:8090")
    assert sentry._api_base(t) == "http://homelab:8090/api/0"


# --- config wiring -----------------------------------------------------------------


def test_env_populates_victoria_and_sentry_url(monkeypatch) -> None:
    for k in ("VICTORIALOGS_URL", "VICTORIAMETRICS_URL", "VICTORIATRACES_URL", "SENTRY_URL"):
        monkeypatch.setenv(f"PODCAST_OBS_{k}", f"http://h/{k.lower()}")
    monkeypatch.delenv("PODCAST_OBS_CONFIG", raising=False)
    t = ObservabilityConfig.from_env().target()
    assert t.victorialogs_url == "http://h/victorialogs_url"
    assert t.victoriametrics_url == "http://h/victoriametrics_url"
    assert t.victoriatraces_url == "http://h/victoriatraces_url"
    assert t.sentry_url == "http://h/sentry_url"
