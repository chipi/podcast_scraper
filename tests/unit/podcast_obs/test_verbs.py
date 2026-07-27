"""Phase-B: the obs_surface / obs_investigate verbs — observe a surface, drill on a join key."""

from __future__ import annotations

from podcast_obs import aggregate
from podcast_obs.config import TargetConfig
from podcast_obs.result import ok


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


def _ok(source):
    return lambda *a, **k: ok(source, {"hit": source})


def test_surface_pipeline_fans_signals_no_http_metrics(monkeypatch) -> None:
    monkeypatch.setattr(aggregate.victoria, "recent_logs", _ok("logs"))
    monkeypatch.setattr(aggregate.victoria, "traces_recent", _ok("traces"))
    monkeypatch.setattr(aggregate.victoria, "events", _ok("events"))
    monkeypatch.setattr(aggregate.sentry, "recent_errors", _ok("errors"))
    r = aggregate.surface(_t(), "pipeline")
    assert r["ok"]
    sig = r["data"]["signals"]
    # pipeline is a subprocess → NO http RED metrics (job=None); has pipeline_stage + cost extras
    assert set(sig) == {"errors", "logs", "traces", "pipeline_stage", "cost"}
    assert r["data"]["trace_service"] == "pipeline"  # live-verified Jaeger service
    assert r["data"]["job"] is None


def test_surface_api_has_metrics_no_pipeline_extras(monkeypatch) -> None:
    for fn in ("red_metrics", "recent_logs", "traces_recent", "events"):
        monkeypatch.setattr(aggregate.victoria, fn, _ok(fn))
    monkeypatch.setattr(aggregate.sentry, "recent_errors", _ok("errors"))
    r = aggregate.surface(_t(), "api")
    assert set(r["data"]["signals"]) == {"metrics", "errors", "logs", "traces"}
    assert r["data"]["trace_service"] == "podcast-api"
    assert r["data"]["job"] == "api"


def test_investigate_requires_a_key() -> None:
    r = aggregate.investigate(_t())
    assert r["ok"] is False and "one of" in r["error"]


def test_investigate_by_episode_id(monkeypatch) -> None:
    monkeypatch.setattr(aggregate.victoria, "events", _ok("events"))
    monkeypatch.setattr(aggregate.victoria, "recent_logs", _ok("logs"))
    r = aggregate.investigate(_t(), episode_id="ep-9")
    assert r["ok"] and r["data"]["episode_id"] == "ep-9"
    assert set(r["data"]["signals"]) == {"ep_pipeline_stage", "ep_cost", "ep_logs"}


def test_investigate_by_trace_id(monkeypatch) -> None:
    monkeypatch.setattr(aggregate.victoria, "trace_by_id", _ok("trace"))
    monkeypatch.setattr(aggregate.victoria, "recent_logs", _ok("logs"))
    r = aggregate.investigate(_t(), trace_id="abc")
    assert set(r["data"]["signals"]) == {"trace", "trace_logs"}


def test_investigate_by_run_id_includes_pipeline_stage(monkeypatch) -> None:
    # run correlators now come from VictoriaLogs (cost=events, errors/logs=recent_logs) + the
    # pipeline_stage events probe; langfuse trace + enrichment stay as optional supplements.
    monkeypatch.setattr(aggregate.victoria, "events", _ok("events"))
    monkeypatch.setattr(aggregate.victoria, "recent_logs", _ok("logs"))
    monkeypatch.setattr(aggregate.langfuse, "trace_by_run", _ok("trace"))
    monkeypatch.setattr(aggregate.enrichment, "recent_events", _ok("enr"))
    r = aggregate.investigate(_t(), run_id="run-9")
    assert "pipeline_stage" in r["data"]["signals"]
    assert "trace" in r["data"]["signals"] and "cost" in r["data"]["signals"]
    assert "errors" in r["data"]["signals"] and "logs" in r["data"]["signals"]
