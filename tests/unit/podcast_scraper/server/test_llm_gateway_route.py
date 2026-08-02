"""Unit tests for GET /api/ops/llm-gateway (#53 / #1357, ADR-142).

The handler reads the prod LiteLLM gateway's per-key spend from homelab VictoriaMetrics via
``podcast_obs``. These tests stub ``podcast_obs`` so no network is touched — covering the
configured/aggregation path and the honest-degradation paths (unwired VM, query error).
"""

from __future__ import annotations

import podcast_obs.config as obs_config
import podcast_obs.sources.victoria as victoria
from podcast_obs.config import TargetConfig
from podcast_scraper.server.routes import llm_gateway


class _Cfg:
    """Stand-in for ObservabilityConfig whose .target() returns a fixed TargetConfig."""

    def __init__(self, target: TargetConfig) -> None:
        self._t = target

    def target(self) -> TargetConfig:
        return self._t


def _vector(query: str, alias: str, value: float) -> dict:
    """A metrics_instant() ``ok`` envelope with one series (the shape victoria.py returns)."""
    return {
        "ok": True,
        "source": "victoriametrics.metrics",
        "data": {
            "query": query,
            "series": [
                {
                    "metric": {"key_alias": alias, "box": "prod"},
                    "value": [1_700_000_000, str(value)],
                }
            ],
        },
    }


def _patch_target(monkeypatch, target: TargetConfig) -> None:
    monkeypatch.setattr(
        obs_config.ObservabilityConfig, "load", classmethod(lambda cls: _Cfg(target))
    )


def test_unconfigured_when_no_victoriametrics_url(monkeypatch) -> None:
    _patch_target(monkeypatch, TargetConfig(name="t", victoriametrics_url=None))
    out = llm_gateway.ops_llm_gateway()
    assert out == {"configured": False, "reachable": False, "keys": []}


def test_aggregates_spend_budget_burn_per_key(monkeypatch) -> None:
    _patch_target(monkeypatch, TargetConfig(name="t", victoriametrics_url="http://homelab:8428"))

    def fake_instant(_target, query: str) -> dict:
        if "spend_usd" in query:
            return _vector(query, "proj-podcast-prod", 0.42)
        if "max_budget_usd" in query:
            return _vector(query, "proj-podcast-prod", 25)
        if "burn_ratio" in query:
            return _vector(query, "proj-podcast-prod", 0.0168)
        return {"ok": False, "source": "vm", "error": "unexpected query"}

    monkeypatch.setattr(victoria, "metrics_instant", fake_instant)
    out = llm_gateway.ops_llm_gateway()
    assert out["configured"] is True
    assert out["reachable"] is True
    assert out["keys"] == [
        {
            "key_alias": "proj-podcast-prod",
            "spend_usd": 0.42,
            "max_budget_usd": 25.0,
            "burn_ratio": 0.0168,
        }
    ]


def test_reachable_false_when_a_query_errors(monkeypatch) -> None:
    _patch_target(monkeypatch, TargetConfig(name="t", victoriametrics_url="http://homelab:8428"))

    def flaky_instant(_target, query: str) -> dict:
        if "spend_usd" in query:
            return _vector(query, "proj-podcast-prod", 1.0)
        return {"ok": False, "source": "vm", "error": "boom"}

    monkeypatch.setattr(victoria, "metrics_instant", flaky_instant)
    out = llm_gateway.ops_llm_gateway()
    # configured + the spend series still surfaced, but reachable=False flags the partial read.
    assert out["configured"] is True
    assert out["reachable"] is False
    assert out["keys"] == [{"key_alias": "proj-podcast-prod", "spend_usd": 1.0}]


def test_sorted_by_spend_desc(monkeypatch) -> None:
    _patch_target(monkeypatch, TargetConfig(name="t", victoriametrics_url="http://homelab:8428"))

    def multi_key(_target, query: str) -> dict:
        if "spend_usd" not in query:
            return {"ok": True, "source": "vm", "data": {"query": query, "series": []}}
        return {
            "ok": True,
            "source": "vm",
            "data": {
                "query": query,
                "series": [
                    {"metric": {"key_alias": "low"}, "value": [1, "0.1"]},
                    {"metric": {"key_alias": "high"}, "value": [1, "9.9"]},
                ],
            },
        }

    monkeypatch.setattr(victoria, "metrics_instant", multi_key)
    out = llm_gateway.ops_llm_gateway()
    assert [k["key_alias"] for k in out["keys"]] == ["high", "low"]
