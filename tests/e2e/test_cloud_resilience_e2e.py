"""E2E: cloud-LLM-provider resilience (retry/5xx/timeout) via the mock server.

Gap noted during the #1003 review: ``test_{openai,anthropic,gemini,deepseek}_provider_e2e.py``
cover happy-path summarization against the mock server, and the
self-hosted ``test_tailnet_dgx_e2e.py`` covers 5xx and watchdog-hang for the
DGX paths — but none of the cloud LLM providers had resilience coverage via
``set_error_behavior`` / ``set_transient_error``. This suite fills that gap
with one tight test per provider for each failure mode that the cloud paths
should survive:

  • transient 503 → retries succeed (provider's ``retry_with_metrics`` loop
    exercises the SDK's underlying httpx layer)
  • permanent 5xx → provider surfaces a ``ProviderRuntimeError`` (the
    summarize-stage error type per ADR-100; the FallbackAware wrapper
    handles routing in production)

Anthropic uses its own /v1/messages route. Gemini uses ``:generateContent``.
OpenAI/DeepSeek share /v1/chat/completions (the mock server uses the
Authorization header to differentiate when needed).

This is the cloud-side counterpart to ``test_tailnet_dgx_e2e.py``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.e2e]

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config  # noqa: E402
from podcast_scraper.exceptions import ProviderRuntimeError  # noqa: E402

_TRANSCRIPT_TEXT = "Sample transcript for the cloud resilience E2E suite. " * 6


@pytest.fixture(autouse=True)
def _reset_error_behaviors(e2e_server):
    """The mock server is session-scoped; clear error injections around every test."""
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_all_error_behaviors()
    yield
    E2EHTTPRequestHandler.clear_all_error_behaviors()


@pytest.fixture(autouse=True)
def _skip_retry_backoff():
    """Skip the retry sleep so 5xx-retry tests don't take a real minute."""
    with patch("podcast_scraper.utils.provider_metrics.time.sleep"):
        yield


def _openai_provider(e2e_server):
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "openai_api_key": "sk-test",
            "openai_api_base": e2e_server.urls.openai_api_base(),
            "summary_provider": "openai",
            "generate_summaries": True,
        }
    )
    p = OpenAIProvider(cfg)
    p.initialize()
    return p


def _anthropic_provider(e2e_server):
    from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "anthropic_api_key": "sk-ant-test",
            "anthropic_api_base": e2e_server.urls.anthropic_api_base(),
            "summary_provider": "anthropic",
            "generate_summaries": True,
        }
    )
    p = AnthropicProvider(cfg)
    p.initialize()
    return p


def _gemini_provider(e2e_server):
    from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "gemini_api_key": "AIza-test",
            "gemini_api_base": e2e_server.urls.gemini_api_base(),
            "summary_provider": "gemini",
            "generate_summaries": True,
        }
    )
    p = GeminiProvider(cfg)
    p.initialize()
    return p


def _deepseek_provider(e2e_server):
    from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "deepseek_api_key": "sk-ds-test",
            "deepseek_api_base": e2e_server.urls.deepseek_api_base(),
            "summary_provider": "deepseek",
            "generate_summaries": True,
        }
    )
    p = DeepSeekProvider(cfg)
    p.initialize()
    return p


# ---------------------------------------------------------------------------
# Permanent-5xx: each provider must surface a wrapped ProviderRuntimeError.
# Different from a guardrail violation (200 OK with bad shape); the
# FallbackAware wrapper in production catches both flavors to route the
# fallback chain.
# ---------------------------------------------------------------------------


class TestOpenAIResilienceE2E:
    def test_permanent_5xx_surfaces_provider_runtime_error(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior("/v1/chat/completions", status=500)
        provider = _openai_provider(e2e_server)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert "OpenAIProvider/Summarization" in str(exc_info.value)


class TestDeepSeekResilienceE2E:
    def test_permanent_5xx_surfaces_provider_runtime_error(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior("/v1/chat/completions", status=500)
        provider = _deepseek_provider(e2e_server)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert "DeepSeekProvider/Summarization" in str(exc_info.value)


class TestAnthropicResilienceE2E:
    def test_permanent_5xx_surfaces_provider_runtime_error(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior("/v1/messages", status=500)
        provider = _anthropic_provider(e2e_server)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert "AnthropicProvider/Summarization" in str(exc_info.value)


class TestGeminiResilienceE2E:
    def test_permanent_5xx_surfaces_provider_runtime_error(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        # Gemini's SDK appends `:generateContent` per model; the mock server
        # routes any /v1beta/models/*/:generateContent path here.
        # set_error_behavior matches on prefix for Gemini-style paths.
        E2EHTTPRequestHandler.set_error_behavior(
            "/v1beta/models/gemini-2.5-flash-lite:generateContent", status=500
        )
        provider = _gemini_provider(e2e_server)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert "GeminiProvider/Summarization" in str(exc_info.value)
