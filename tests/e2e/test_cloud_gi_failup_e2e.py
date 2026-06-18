"""E2E regression guard: GI fails up on guardrail violation (ADR-100).

GI was previously silently catching ``GuardrailViolation`` via the outer
``except Exception: return []`` on every cloud + Ollama provider —
ADR-100-incompatible (GI is fail-up). The fix made GuardrailViolation
propagate so FallbackAware can route. Without this test, a future
refactor could regress back to silent degradation and the guardrail
firing would be invisible at the call boundary.

One test per provider — parameterizing would be slightly nicer but the
provider-construction shapes differ enough that the explicit form is
clearer.
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
from podcast_scraper.providers import guardrails  # noqa: E402

_TRANSCRIPT_TEXT = "Sample transcript for GI fail-up E2E. " * 8


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


class TestGIFailUpOnGuardrailViolation:
    """ADR-100: GI must propagate ``GuardrailViolation``, NOT silently
    return an empty list."""

    def test_openai_gi_propagates(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _openai(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.generate_insights(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "openai"

    def test_anthropic_gi_propagates(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/messages", "anthropic:empty_content")
        provider = _anthropic(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.generate_insights(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "anthropic"

    def test_gemini_gi_propagates(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1beta/generateContent", "gemini:empty_content")
        provider = _gemini(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.generate_insights(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "gemini"

    def test_deepseek_gi_propagates(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _deepseek(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.generate_insights(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "deepseek"
