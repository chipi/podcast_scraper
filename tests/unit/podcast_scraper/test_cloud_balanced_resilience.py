"""cloud_balanced RFC-111 (#1482) gateway-resilience wiring: a transient homelab:4001 -> OpenRouter
connection failure (APIConnectionError) must (1) hold/retry for minutes, not the ~12s default, and
(2) fail over to DIRECT DeepSeek if the gateway stays down. This asserts the PROFILE materializes
both — resolved config values, not wall-clock behaviour (the sleep-for-minutes case belongs to
Part A's llm_resilience unit tests, which assert resolved values only).
"""

from __future__ import annotations

import pytest

from podcast_scraper.cli import _build_config, parse_args


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROK_API_KEY",
        "DEEPGRAM_API_KEY",
    ):
        monkeypatch.setenv(name, "test-" + name.lower().replace("_", "-") + "-dummy-key")


def _build_via_profile(name: str) -> object:
    args = parse_args(
        ["--profile", name, "https://example.com/feed.xml", "--output-dir", "/tmp/_t"]
    )
    return _build_config(args)


class TestCloudBalancedGatewayResilience:
    def test_declares_deepseek_failover_chain(self, _fake_keys: None) -> None:
        cfg = _build_via_profile("cloud_balanced")
        assert cfg.summary_fallback_providers == ["deepseek"]

    def test_deepseek_fallback_model_is_pinned(self, _fake_keys: None) -> None:
        cfg = _build_via_profile("cloud_balanced")
        assert cfg.deepseek_summary_model == "deepseek-v4-flash"
        assert cfg.deepseek_extra_body == {"reasoning_effort": "none"}

    def test_holds_and_retries_for_minutes_not_seconds(self, _fake_keys: None) -> None:
        """Resolved config values only — NOT a sleep-for-minutes test. 12 retries at 2s->120s
        backoff holds for several minutes before the gateway attempt gives up and the RFC-106
        failover chain (asserted above) takes over."""
        cfg = _build_via_profile("cloud_balanced")
        assert cfg.llm_retry_max_retries == 12
        assert cfg.llm_retry_initial_delay_seconds == 2.0
        assert cfg.llm_retry_max_delay_seconds == 120.0
        # And the short-window default, elsewhere, is untouched (regression guard for Part A).
        from podcast_scraper.config import Config

        default_cfg = Config(rss="https://example.com/feed.xml", output_dir="/tmp/_t")
        assert default_cfg.llm_retry_max_retries == 3
        assert default_cfg.llm_retry_initial_delay_seconds == 1.0
        assert default_cfg.llm_retry_max_delay_seconds == 30.0
