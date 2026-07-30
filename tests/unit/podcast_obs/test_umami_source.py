"""Umami source: not-configured degradation, stats/events/active parsing, and login exchange."""

from __future__ import annotations

import pytest

from podcast_obs.config import TargetConfig
from podcast_obs.sources import umami

pytestmark = [pytest.mark.unit]


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


def _cfg(**kw) -> TargetConfig:
    return _t(umami_url="http://h:3001", umami_website_id="w1", umami_token="TOK", **kw)


def test_not_configured_without_url_or_website() -> None:
    assert umami.stats(_t())["configured"] is False
    assert umami.events(_t(umami_url="http://h:3001"))["configured"] is False  # website missing


def test_not_configured_without_any_creds() -> None:
    # url + website present, but no token and no username/password -> honestly "not wired".
    r = umami.events(_t(umami_url="http://h:3001", umami_website_id="w1"))
    assert r["configured"] is False


def test_stats_parses_and_hits_endpoint(monkeypatch) -> None:
    seen: dict = {}

    def _fake(url, **kw):
        seen.update(url=url, headers=kw.get("headers"), params=kw.get("params"))
        return {"pageviews": {"value": 10}, "visitors": {"value": 4}}

    monkeypatch.setattr(umami, "get_json", _fake)
    r = umami.stats(_cfg(), window="24h")
    assert r["ok"] and r["data"]["stats"]["pageviews"]["value"] == 10
    assert seen["url"].endswith("/api/websites/w1/stats")
    assert seen["headers"]["Authorization"] == "Bearer TOK"
    assert "startAt" in seen["params"] and "endAt" in seen["params"]


def test_events_uses_event_metrics_type(monkeypatch) -> None:
    seen: dict = {}

    def _fake(url, **kw):
        seen.update(url=url, params=kw.get("params"))
        return [{"x": "search", "y": 5}, {"x": "explore", "y": 3}]

    monkeypatch.setattr(umami, "get_json", _fake)
    r = umami.events(_cfg())
    assert r["ok"] and r["data"]["count"] == 2
    assert seen["url"].endswith("/api/websites/w1/metrics")
    assert seen["params"]["type"] == "event"


def test_login_exchange_when_no_token(monkeypatch) -> None:
    seen: dict = {}

    def _login(url, **kw):
        seen.update(login_url=url, body=kw.get("json"))
        return {"token": "LOGIN_TOK"}

    def _get(url, **kw):
        seen["auth"] = (kw.get("headers") or {}).get("Authorization")
        return {"x": 2}

    monkeypatch.setattr(umami, "post_json", _login)
    monkeypatch.setattr(umami, "get_json", _get)
    r = umami.active(
        _t(umami_url="http://h:3001", umami_website_id="w1", umami_username="u", umami_password="p")
    )
    assert r["ok"]
    assert seen["login_url"].endswith("/api/auth/login")
    assert seen["body"] == {"username": "u", "password": "p"}
    assert seen["auth"] == "Bearer LOGIN_TOK"


def test_login_failure_reports_error_not_unconfigured(monkeypatch) -> None:
    def _boom(url, **kw):
        raise RuntimeError("401 bad creds")

    monkeypatch.setattr(umami, "post_json", _boom)
    r = umami.stats(
        _t(umami_url="http://h:3001", umami_website_id="w1", umami_username="u", umami_password="x")
    )
    assert r["ok"] is False and r.get("configured") is not False  # a real failure, not "not wired"
