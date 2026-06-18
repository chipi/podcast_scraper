"""E2E: FallbackAware routes a real ``GuardrailViolation`` to the cloud fallback.

ADR-100's architectural claim is that a ``200 OK`` from a primary provider
that trips the response-shape guardrail flows through
``FallbackAwareSummarizationProvider`` to the configured
``degradation_policy.fallback_provider_on_failure`` — same path as a
connection-level failure. The unit-level fallback tests
(`tests/unit/podcast_scraper/summarization/test_fallback.py`) mock the
primary's exception; the cloud-guardrail E2E (`test_cloud_guardrails_e2e.py`)
asserts the raw ``GuardrailViolation`` propagates out of each provider.
Neither test verifies the full path together.

This file does. Boots the shared mock server, configures the primary
provider to point at the mock and the fallback provider to point at the
mock too, injects a guardrail-violating response on the primary, asserts
the fallback's response is what comes back.
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
from podcast_scraper.summarization.fallback import (  # noqa: E402
    FallbackAwareSummarizationProvider,
)

_TRANSCRIPT_TEXT = "Sample transcript for the FallbackAware-under-guardrail E2E suite. " * 4


@pytest.fixture(autouse=True)
def _reset_violations(e2e_server):
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_violations()
    yield
    E2EHTTPRequestHandler.clear_violations()


def _openai_primary_with_anthropic_fallback(e2e_server):
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "openai_api_key": "sk-test",
            "openai_api_base": e2e_server.urls.openai_api_base(),
            "anthropic_api_key": "sk-ant-test",
            "anthropic_api_base": e2e_server.urls.anthropic_api_base(),
            "summary_provider": "openai",
            "generate_summaries": True,
            "degradation_policy": {
                "fallback_provider_on_failure": "anthropic",
                "continue_on_stage_failure": True,
            },
        }
    )
    primary = OpenAIProvider(cfg)
    primary.initialize()
    wrapped = FallbackAwareSummarizationProvider(
        primary=primary,
        fallback_provider_name="anthropic",
        cfg=cfg,
    )
    return wrapped


def _gemini_primary_with_openai_fallback(e2e_server):
    from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "gemini_api_key": "AIza-test",
            "gemini_api_base": e2e_server.urls.gemini_api_base(),
            "openai_api_key": "sk-test",
            "openai_api_base": e2e_server.urls.openai_api_base(),
            "summary_provider": "gemini",
            "generate_summaries": True,
            "degradation_policy": {
                "fallback_provider_on_failure": "openai",
                "continue_on_stage_failure": True,
            },
        }
    )
    primary = GeminiProvider(cfg)
    primary.initialize()
    wrapped = FallbackAwareSummarizationProvider(
        primary=primary,
        fallback_provider_name="openai",
        cfg=cfg,
    )
    return wrapped


class TestFallbackUnderGuardrailViolation:
    """Inject a guardrail violation on the primary; assert the fallback's
    response is what reaches the caller."""

    def test_openai_empty_routes_to_anthropic(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        # Primary (OpenAI on /v1/chat/completions) returns an empty response.
        # Fallback (Anthropic on /v1/messages) returns the mock server's
        # canned valid summary.
        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        wrapped = _openai_primary_with_anthropic_fallback(e2e_server)
        result = wrapped.summarize(_TRANSCRIPT_TEXT)
        # The fallback's response shape — Anthropic-issued summary.
        assert result is not None
        assert "summary" in result
        assert result["metadata"]["provider"] == "anthropic"

    def test_gemini_thinking_prose_routes_to_openai(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1beta/generateContent", "gemini:thinking_prose")
        wrapped = _gemini_primary_with_openai_fallback(e2e_server)
        result = wrapped.summarize(_TRANSCRIPT_TEXT)
        assert result is not None
        assert "summary" in result
        assert result["metadata"]["provider"] == "openai"
