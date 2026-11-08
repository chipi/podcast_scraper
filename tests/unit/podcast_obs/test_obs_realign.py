"""The o11y MCP realigned with what prod emits (2026-11-08 review).

- One scrape job (``job="api"``) covers the podcast, player and operator APIs, told apart by
  the ``service`` label. ``obs_surface(player|operator)`` asked for ``job="player"`` /
  ``job="operator"``, which do not exist, and ``obs_surface(api)`` summed all three.
- ``prod_correlate("")`` dropped every run filter and returned the last day of everything.
- The jobs registry rows carry ``trigger`` (manual / nightly / ...), which the run listing dropped.
- Runs now emit ``pipeline_run.*`` events; the pipeline surface and a run's investigation did not
  look at them.
"""

from __future__ import annotations

from podcast_obs import aggregate
from podcast_obs.config import TargetConfig
from podcast_obs.result import ok
from podcast_obs.sources import prod_api, victoria


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


def test_red_metrics_scopes_to_one_service(monkeypatch) -> None:
    queries = []

    def _get_json(url, **kw):
        queries.append(kw["params"]["query"])
        return {"data": {}}

    monkeypatch.setattr(victoria, "get_json", _get_json)
    victoria.red_metrics(_t(victoriametrics_url="http://h:8428"), "api", service="player-api")
    assert queries and all('job="api",service="player-api"' in q for q in queries)


def test_each_api_surface_reads_its_own_service(monkeypatch) -> None:
    seen = {}

    def _red(target, job, *, window="5m", service=None):
        seen["job"], seen["service"] = job, service
        return ok("metrics", {})

    monkeypatch.setattr(aggregate.victoria, "red_metrics", _red)
    for fn in ("recent_logs", "traces_recent", "events"):
        monkeypatch.setattr(aggregate.victoria, fn, lambda *a, **k: ok(fn, {}))
    monkeypatch.setattr(aggregate.sentry, "recent_errors", lambda *a, **k: ok("errors", {}))
    for fn in ("events", "stats", "active"):
        monkeypatch.setattr(aggregate.umami, fn, lambda *a, **k: ok(fn, {}))
    for name, service in (
        ("api", "podcast-api"),
        ("player", "player-api"),
        ("operator", "operator-public-api"),
    ):
        aggregate.surface(_t(), name)
        assert seen == {"job": "api", "service": service}, name


def test_correlate_refuses_an_empty_run_id() -> None:
    r = aggregate.correlate(_t(), "")
    assert r["ok"] is False
    r = aggregate.correlate(_t(), "   ")
    assert r["ok"] is False


def test_the_run_listing_keeps_the_trigger() -> None:
    assert "trigger" in prod_api._RUN_FIELDS


def test_the_pipeline_surface_and_a_run_include_the_run_events(monkeypatch) -> None:
    asked = []

    def _events(target, event_type, **kw):
        asked.append(event_type)
        return ok("events", {})

    monkeypatch.setattr(aggregate.victoria, "events", _events)
    for fn in ("recent_logs", "traces_recent", "traces_by_run", "trace_by_id"):
        monkeypatch.setattr(aggregate.victoria, fn, lambda *a, **k: ok(fn, {}))
    monkeypatch.setattr(aggregate.sentry, "recent_errors", lambda *a, **k: ok("errors", {}))
    aggregate.surface(_t(), "pipeline")
    assert "pipeline_run.*" in asked
    asked.clear()
    aggregate.investigate(_t(), run_id="r-1")
    assert "pipeline_run.*" in asked
