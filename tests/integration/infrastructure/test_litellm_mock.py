#!/usr/bin/env python3
"""Integration tests for the LiteLLM gateway provider against the E2E server mock (#1356).

The LiteLLM gateway is OpenAI-compatible, so the provider reuses the same mock chat endpoints as
openai/deepseek. These verify the provider routes chat (summary + speaker) through the configured
gateway base URL over real HTTP — component interaction, not a full workflow. The fail-closed
served-alias check is covered in the unit tests; it is disabled here (the mock advertises no
aliases) so these stay focused on the chat path.
"""

import os
import sys

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider


@pytest.mark.integration
@pytest.mark.llm
class TestLiteLLME2EServerIntegration:
    """The LiteLLM gateway provider routes chat through the configured (mock) gateway base URL."""

    def test_litellm_summarization_provider_uses_e2e_server(self, e2e_server):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            litellm_api_key="sk-test-gateway-key",
            litellm_api_base=e2e_server.urls.litellm_api_base(),
            litellm_summary_model="homelab-qwen",
            litellm_verify_served_model=False,  # served-alias check is unit-covered
            summary_provider="litellm",
            generate_summaries=True,
            generate_metadata=True,
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        assert str(provider.client.base_url).rstrip(
            "/"
        ) == e2e_server.urls.litellm_api_base().rstrip(
            "/"
        ), "Provider should use the E2E (mock gateway) base URL"

        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert "test summary" in result["summary"].lower()
        provider.cleanup()
