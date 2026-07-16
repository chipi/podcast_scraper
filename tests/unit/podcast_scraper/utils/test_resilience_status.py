"""The operator surface for resilience: SEE what's open, and RESET it. Closes the ADR-113 gap.

Drives the two functions every wrapper (server API, o11y MCP tool, operator UI) sits on: a status
snapshot that lists every provider breaker (open or closed) and the RSS breaker, and a reset that
force-closes them early instead of waiting out the cooldown.
"""

from __future__ import annotations

import pytest

from podcast_scraper.utils import llm_circuit_breaker as breaker
from podcast_scraper.utils.llm_circuit_breaker import LLMCircuitBreakerConfig, record_failure
from podcast_scraper.utils.resilience_status import reset_resilience, resilience_snapshot

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_breakers():
    breaker.reset_all()
    yield
    breaker.reset_all()


def test_snapshot_lists_every_provider_even_when_all_closed() -> None:
    snap = resilience_snapshot()
    for provider in ("openai", "anthropic", "gemini", "mistral", "deepseek", "grok", "ollama"):
        assert provider in snap["llm_breakers"], f"{provider} missing — a closed breaker must show"
        assert snap["llm_breakers"][provider]["open"] is False
    assert snap["any_open"] is False
    assert snap["llm_breakers_open"] == []


def test_snapshot_reports_an_open_breaker_with_cooldown() -> None:
    cfg = LLMCircuitBreakerConfig(
        enabled=True, failure_threshold=1, window_seconds=30.0, cooldown_seconds=60.0
    )
    record_failure("gemini", cfg, 503)  # one 503 trips it (threshold 1)

    snap = resilience_snapshot()
    g = snap["llm_breakers"]["gemini"]
    assert g["open"] is True
    assert g["cooldown_remaining_seconds"] > 0
    assert "gemini" in snap["llm_breakers_open"]
    assert snap["any_open"] is True


def test_reset_force_closes_the_breaker() -> None:
    cfg = LLMCircuitBreakerConfig(
        enabled=True, failure_threshold=1, window_seconds=30.0, cooldown_seconds=600.0
    )
    record_failure("openai", cfg, 429)
    assert resilience_snapshot()["llm_breakers"]["openai"]["open"] is True

    result = reset_resilience("all")
    assert "llm_breakers" in result["reset"]
    assert (
        resilience_snapshot()["llm_breakers"]["openai"]["open"] is False
    ), "the operator reset must force-close early, not wait the 600s cooldown"


def test_reset_scope_llm_only_leaves_rss_alone() -> None:
    result = reset_resilience("llm")
    assert result["reset"] == ["llm_breakers"]
    assert result["scope"] == "llm"


def test_snapshot_surfaces_the_fuse_budgets_from_cfg() -> None:
    class _Cfg:
        llm_max_calls_per_episode = 500
        llm_max_calls_per_run = 8000

    snap = resilience_snapshot(_Cfg())
    assert snap["fuses"]["llm_max_calls_per_episode"] == 500
    assert snap["fuses"]["llm_max_calls_per_run"] == 8000
    assert "fix-the-cause-and-rerun" in snap["fuses"]["note"]
