"""The three prod obs-MCP defects found by calling every tool inside player-obs (2026-10-27).

- ``prod_recent_traces`` timed out: Langfuse's full trace listing took 5-11 s against a 10 s
  timeout; asking for ``fields=core,metrics`` took 0.5-0.7 s with latency and cost still present.
- ``prod_cost_today`` returned 500 raw ``llm_cost`` events (766 KB) instead of a total.
- ``obs_analytics`` without ``frontend`` reported every signal "unconfigured" on prod, where the
  sites exist per frontend (player, operator) and both were live.
"""

from __future__ import annotations

import pytest

from podcast_obs import aggregate, mcp_server
from podcast_obs.config import ObservabilityConfig, TargetConfig
from podcast_obs.result import ok
from podcast_obs.sources import langfuse, umami, victoria


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="prod", **kw)


def test_recent_traces_asks_langfuse_for_core_and_metrics_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    def _fake(url, **kw):
        seen.update(kw.get("params") or {})
        return {"data": [{"id": "t1", "latency": 1.0, "totalCost": 0.01}]}

    monkeypatch.setattr(langfuse, "get_json", _fake)
    r = langfuse.recent_traces(_t(langfuse_public_key="pk", langfuse_secret_key="sk"), limit=3)
    assert r["ok"] is True
    assert seen == {"limit": 3, "fields": "core,metrics"}
    assert r["data"]["traces"][0]["latency"] == 1.0


def test_cost_rollup_sums_per_model_in_victorialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _fake(url, **kw):
        seen["query"] = kw["params"]["query"]
        return [
            {
                "provider": "vllm",
                "model": "qwen",
                "calls": "960",
                "cost_usd": "0",
                "prompt_tokens": "6248315",
                "completion_tokens": "663568",
            },
            {
                "provider": "openai",
                "model": "gpt",
                "calls": "4",
                "cost_usd": "0.0123",
                "prompt_tokens": "1000",
                "completion_tokens": "200",
            },
        ]

    monkeypatch.setattr(victoria, "get_ndjson", _fake)
    r = victoria.cost_rollup(_t(victorialogs_url="http://vl"), window="24h")
    assert r["ok"] is True
    q = seen["query"]
    assert q.startswith("_time:24h event_type:llm_cost | stats by (provider, model)")
    # No request_id grouping: most prod events carry none, and grouping folds them into one.
    assert "request_id" not in q
    data = r["data"]
    assert data["total_calls"] == 964
    assert data["total_cost_usd"] == 0.0123
    assert [m["model"] for m in data["by_model"]] == ["gpt", "qwen"]  # costliest first
    assert data["by_model"][1]["prompt_tokens"] == 6248315


def test_cost_rollup_not_configured() -> None:
    r = victoria.cost_rollup(_t())
    assert r["ok"] is False and r["configured"] is False


def test_prod_cost_today_is_the_rollup_not_raw_events(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_raw(*_a, **_k):
        raise AssertionError("prod_cost_today read the raw llm_cost events")

    monkeypatch.setattr(victoria, "events", _no_raw)
    monkeypatch.setattr(victoria, "cost_rollup", lambda t, window: ok("x", {"window": window}))
    cfg = ObservabilityConfig(
        targets={"prod": _t(victorialogs_url="http://vl")}, default_target="prod"
    )
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(cfg)}
    assert tools["prod_cost_today"]()["data"] == {"window": "24h"}


def _stub_umami(monkeypatch: pytest.MonkeyPatch) -> list:
    sites: list = []

    def _probe(site, **_):
        sites.append(site.umami_website_id)
        return ok("umami", {"site": site.umami_website_id})

    for name in ("events", "stats", "active"):
        monkeypatch.setattr(umami, name, _probe)
    return sites


def test_analytics_without_frontend_answers_every_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    sites = _stub_umami(monkeypatch)
    target = _t(umami_url="http://u", umami_website_ids={"player": "wp", "operator": "wo"})
    data = aggregate.analytics(target)["data"]
    assert set(data["frontends"]) == {"operator", "player"}
    assert data["unconfigured"] == [] and data["failed"] == []
    assert "player.page_analytics" in data["live"] and "operator.user_actions" in data["live"]
    assert sorted(set(sites)) == ["wo", "wp"]


def test_analytics_with_frontend_or_single_site_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sites = _stub_umami(monkeypatch)
    per = _t(umami_url="http://u", umami_website_ids={"player": "wp", "operator": "wo"})
    one = aggregate.analytics(per, frontend="player")["data"]
    assert one["frontend"] == "player" and "frontends" not in one
    single = aggregate.analytics(_t(umami_url="http://u", umami_website_id="w1"))["data"]
    assert "frontends" not in single and sorted(single["live"]) == [
        "active_users",
        "page_analytics",
        "user_actions",
    ]
    assert set(sites) == {"wp", "w1"}
