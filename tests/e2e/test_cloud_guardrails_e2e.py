"""E2E: cloud-LLM-provider response-shape guardrails (ADR-100 / #1003).

Each test boots the shared e2e mock server, points the real provider's SDK at it
via the ``<provider>_api_base`` config field, injects a structurally-valid HTTP
200 response that fails the per-service guardrail check, then calls
``provider.summarize()`` and asserts the call raises ``GuardrailViolation``.

This is the cloud-side counterpart to ``test_tailnet_dgx_e2e.py`` (self-hosted
guardrails for whisper/diarize). The unit-level helper coverage lives in
``tests/unit/podcast_scraper/providers/test_resilience_and_guardrails.py``; the
per-provider wiring smoke-tests live in ``test_cloud_guardrails_wiring.py``.
This suite is the integration-level confirmation: end-to-end HTTP round-trip,
real SDK, real provider class — the only path that catches "the wiring at
``check_chat_response`` actually reaches the call site under the real SDK
response shape".
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

_TRANSCRIPT_TEXT = "This is a short sample transcript used for the cloud guardrail E2E suite. " * 4


@pytest.fixture(autouse=True)
def _reset_violations(e2e_server):
    """Mock server is session-scoped; clear violation registry around each test."""
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_violations()
    yield
    E2EHTTPRequestHandler.clear_violations()


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


def _mistral_provider(e2e_server):
    from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "mistral_api_key": "test-mistral-key",
            "mistral_api_base": e2e_server.urls.mistral_api_base(),
            "summary_provider": "mistral",
            "generate_summaries": True,
        }
    )
    p = MistralProvider(cfg)
    p.initialize()
    return p


def _grok_provider(e2e_server):
    from podcast_scraper.providers.grok.grok_provider import GrokProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "grok_api_key": "xai-test",
            "grok_api_base": e2e_server.urls.grok_api_base(),
            "summary_provider": "grok",
            "generate_summaries": True,
        }
    )
    p = GrokProvider(cfg)
    p.initialize()
    return p


# ---------------------------------------------------------------------------
# OpenAI (and DeepSeek, which shares the chat-completions wire shape)
# ---------------------------------------------------------------------------


class TestOpenAIGuardrailE2E:
    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _openai_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "openai"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_thinking_prose_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
        provider = _openai_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE

    def test_finish_length_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:finish_length")
        provider = _openai_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH


class TestDeepSeekGuardrailE2E:
    """DeepSeek uses the OpenAI-compatible /v1/chat/completions wire shape."""

    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _deepseek_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "deepseek"

    def test_finish_length_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:finish_length")
        provider = _deepseek_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH


# ---------------------------------------------------------------------------
# Anthropic — separate path (/v1/messages, different response envelope)
# ---------------------------------------------------------------------------


class TestAnthropicGuardrailE2E:
    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/messages", "anthropic:empty_content")
        provider = _anthropic_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "anthropic"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_thinking_prose_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/messages", "anthropic:thinking_prose")
        provider = _anthropic_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE


# ---------------------------------------------------------------------------
# Gemini — separate path (:generateContent), separate response envelope
# ---------------------------------------------------------------------------


class TestGeminiGuardrailE2E:
    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1beta/generateContent", "gemini:empty_content")
        provider = _gemini_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "gemini"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_thinking_prose_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1beta/generateContent", "gemini:thinking_prose")
        provider = _gemini_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE


# ---------------------------------------------------------------------------
# Mistral + Grok — OpenAI-compatible wire shape, hit /v1/chat/completions
# ---------------------------------------------------------------------------


class TestMistralGuardrailE2E:
    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _mistral_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "mistral"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_thinking_prose_raises_guardrail_violation(self, e2e_server):
        # finish_reason-based tests aren't reliable for Mistral here because
        # the SDK's response object doesn't surface ``finish_reason`` through
        # ``response.choices[0]`` for all server payload shapes (verified at
        # E2E wiring time). The empty-content + thinking-prose detectors
        # don't depend on finish_reason and cover the common failure modes.
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
        provider = _mistral_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE


class TestGrokGuardrailE2E:
    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _grok_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "grok"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_finish_length_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:finish_length")
        provider = _grok_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH
