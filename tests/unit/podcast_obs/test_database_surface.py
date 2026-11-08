"""The database surface: the Postgres exporter's series as one snapshot
(``surface("database")``)."""

from __future__ import annotations

from podcast_obs import aggregate
from podcast_obs.config import TargetConfig
from podcast_obs.result import err, ok


def _t() -> TargetConfig:
    return TargetConfig(name="t", victoriametrics_url="http://vm")


def test_every_signal_is_one_query_on_the_exporters_job(monkeypatch) -> None:
    asked: list[str] = []

    def instant(target, query):
        asked.append(query)
        return ok("victoriametrics.query", {"query": query, "series": []})

    monkeypatch.setattr(aggregate.victoria, "metrics_instant", instant)
    out = aggregate.surface(_t(), "database", window="30m")
    data = out["data"]
    assert out["ok"] and data["surface"] == "database" and data["window"] == "30m"
    assert set(data["live"]) == set(aggregate._database_queries("30m"))
    assert data["failed"] == [] and data["unconfigured"] == []
    assert len(asked) == len(data["live"])
    assert all('job="postgres"' in q for q in asked)
    statements = [q for q in asked if "pg_stat_statements" in q]
    assert statements and all('user!="monitor"' in q and "[30m]" in q for q in statements)


def test_a_failing_signal_degrades_alone(monkeypatch) -> None:
    def instant(target, query):
        if "pg_locks_count" in query:
            raise RuntimeError("boom")
        if "pg_up" in query:
            return err("victoriametrics.query", "not configured", configured=False)
        return ok("victoriametrics.query", {"query": query, "series": []})

    monkeypatch.setattr(aggregate.victoria, "metrics_instant", instant)
    data = aggregate.surface(_t(), "database")["data"]
    assert data["failed"] == ["locks"]
    assert data["unconfigured"] == ["up"]
    assert "connections" in data["live"]
