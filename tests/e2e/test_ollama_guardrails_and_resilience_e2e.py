"""E2E: Ollama guardrails + resilience via the mock server.

Ollama is the self-hosted-but-OpenAI-compatible third axis alongside the
4 cloud providers; the prod_dgx_full_with_fallback profile and the
freeze/ollama_qwen35 profile both wire it as the summary provider.
Symmetric coverage with the cloud suites:

- guardrail empty / thinking-prose / finish-length → ``GuardrailViolation``
- permanent 5xx → ``ProviderRuntimeError``
- transient 5xx → SDK retries recover
- tight client timeout against injected delay → ``ProviderRuntimeError``

Ollama's summarize hits ``/v1/chat/completions`` (the OpenAI-compatible
shim), so the existing chat-completions injection vocabulary applies.
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
from podcast_scraper.exceptions import ProviderRuntimeError  # noqa: E402
from podcast_scraper.providers import guardrails  # noqa: E402

_TRANSCRIPT_TEXT = "Sample transcript for the Ollama E2E suite. " * 6


@pytest.fixture(autouse=True)
def _reset_state(e2e_server):
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_violations()
    E2EHTTPRequestHandler.clear_all_error_behaviors()
    yield
    E2EHTTPRequestHandler.clear_violations()
    E2EHTTPRequestHandler.clear_all_error_behaviors()


def _ollama_provider(e2e_server, *, summarization_timeout: int = 120):
    from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "ollama_api_base": e2e_server.urls.ollama_api_base(),
            "summary_provider": "ollama",
            "generate_summaries": True,
            "summarization_timeout": summarization_timeout,
        }
    )
    p = OllamaProvider(cfg)
    # Skip the live-server health probe — the mock server doesn't implement
    # /api/version. Mark initialized so summarize() proceeds straight to the
    # chat-completions call we want to exercise.
    p._summarization_initialized = True
    return p


class TestOllamaGuardrailE2E:
    def test_empty_content_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:empty_content")
        provider = _ollama_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.service == "ollama"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_thinking_prose_raises_guardrail_violation(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
        provider = _ollama_provider(e2e_server)
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE

    def test_cleaning_thinking_prose_degrades_gracefully(self, e2e_server):
        # ADR-100: cleaning catches GuardrailViolation and returns original text.
        # NOT a propagation test — opposite policy from summarize.
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.inject_violation("/v1/chat/completions", "chat:thinking_prose")
        provider = _ollama_provider(e2e_server)
        original = "Original transcript content that should pass through unchanged."
        result = provider.clean_transcript(original)
        assert result == original


class TestOllamaResilienceE2E:
    def test_permanent_5xx_surfaces_provider_runtime_error(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior("/v1/chat/completions", status=500)
        provider = _ollama_provider(e2e_server)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert "Ollama" in str(exc_info.value)

    def test_transient_503_recovers(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_transient_error("/v1/chat/completions", status=503, fail_count=2)
        provider = _ollama_provider(e2e_server)
        result = provider.summarize(_TRANSCRIPT_TEXT)
        assert result is not None and "summary" in result

    def test_request_timeout_surfaces(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior("/v1/chat/completions", status=200, delay=3.0)
        provider = _ollama_provider(e2e_server, summarization_timeout=1)
        with pytest.raises(ProviderRuntimeError) as exc_info:
            provider.summarize(_TRANSCRIPT_TEXT)
        assert "Ollama" in str(exc_info.value)
