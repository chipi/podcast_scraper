"""The operator resilience API — GET status + POST reset. The ADR-113 gap, exposed over HTTP.

GET /api/resilience is a normal read (MCP + dashboards query it); POST /api/ops/resilience/reset is
under the operator-guarded prefix so a configured deploy admin-gates the reset.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.utils import llm_circuit_breaker as breaker
from podcast_scraper.utils.llm_circuit_breaker import LLMCircuitBreakerConfig, record_failure

pytestmark = pytest.mark.integration


@pytest.fixture()
def client():
    breaker.reset_all()
    yield TestClient(create_app())
    breaker.reset_all()


def test_get_resilience_returns_the_snapshot(client) -> None:
    r = client.get("/api/resilience")
    assert r.status_code == 200
    data = r.json()
    assert "llm_breakers" in data and "rss" in data and "fuses" in data
    assert data["any_open"] is False
    for provider in ("openai", "gemini", "anthropic"):
        assert provider in data["llm_breakers"]


def test_get_resilience_shows_an_open_breaker(client) -> None:
    cfg = LLMCircuitBreakerConfig(
        enabled=True, failure_threshold=1, window_seconds=30.0, cooldown_seconds=60.0
    )
    record_failure("gemini", cfg, 503)
    data = client.get("/api/resilience").json()
    assert data["llm_breakers"]["gemini"]["open"] is True
    assert "gemini" in data["llm_breakers_open"]
    assert data["any_open"] is True


def test_post_reset_force_closes_breakers(client) -> None:
    cfg = LLMCircuitBreakerConfig(
        enabled=True, failure_threshold=1, window_seconds=30.0, cooldown_seconds=600.0
    )
    record_failure("openai", cfg, 429)
    assert client.get("/api/resilience").json()["llm_breakers"]["openai"]["open"] is True

    rr = client.post("/api/ops/resilience/reset?scope=all")
    assert rr.status_code == 200
    assert "llm_breakers" in rr.json()["reset"]
    assert client.get("/api/resilience").json()["llm_breakers"]["openai"]["open"] is False
