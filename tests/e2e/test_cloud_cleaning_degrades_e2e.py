"""E2E regression guard: cleaning catch-and-degrade per ADR-100 (cloud).

The cloud providers' ``clean_transcript`` catches ``GuardrailViolation``
and returns the original transcript text (graceful degradation, distinct
from summarize/GI/KG fail-up). Ollama already has this test; the cloud
side didn't. Future refactors could regress to fail-up (or to silent
empty) — this guard catches it.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = [pytest.mark.e2e]

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config  # noqa: E402

_ORIGINAL = "Original transcript content that should pass through unchanged."


@pytest.fixture(autouse=True)
def _reset(e2e_server):
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_violations()
    yield
    E2EHTTPRequestHandler.clear_violations()


def _openai(e2e_server):
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


def _anthropic(e2e_server):
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


def _gemini(e2e_server):
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


def _deepseek(e2e_server):
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


class TestCleaningDegradesGracefullyOnGuardrailViolation:
    def test_openai(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
        provider = _openai(e2e_server)
        assert provider.clean_transcript(_ORIGINAL) == _ORIGINAL

    def test_anthropic(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/messages", "anthropic:thinking_prose")
        provider = _anthropic(e2e_server)
        assert provider.clean_transcript(_ORIGINAL) == _ORIGINAL

    def test_gemini(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1beta/generateContent", "gemini:thinking_prose")
        provider = _gemini(e2e_server)
        assert provider.clean_transcript(_ORIGINAL) == _ORIGINAL

    def test_deepseek(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
        provider = _deepseek(e2e_server)
        assert provider.clean_transcript(_ORIGINAL) == _ORIGINAL
