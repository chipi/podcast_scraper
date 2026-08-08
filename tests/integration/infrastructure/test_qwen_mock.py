#!/usr/bin/env python3
"""Integration tests for the native Qwen provider against the E2E server mock (ADR-147).

The Qwen provider is OpenAI-compatible, so it reuses the same mock chat endpoints as
openai/deepseek/litellm. These verify the provider routes chat (summary + speaker) through the
configured base URL over real HTTP — component interaction, not a full workflow. The fail-closed
served-model check is covered in the unit tests; it is disabled here (the mock advertises no models)
so these stay focused on the chat path. A qwen bearer is optional (local vLLM), so these also prove
the provider builds and routes WITHOUT any api key set.
"""

import os
import sys

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider


@pytest.mark.integration
@pytest.mark.llm
class TestQwenE2EServerIntegration:
    """The native Qwen provider routes chat through the configured (mock) base URL."""

    def test_qwen_summarization_provider_uses_e2e_server(self, e2e_server):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            qwen_api_base=e2e_server.urls.qwen_api_base(),
            qwen_summary_model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            qwen_verify_served_model=False,  # served-model check is unit-covered
            summary_provider="qwen",
            generate_summaries=True,
            generate_metadata=True,
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        assert str(provider.client.base_url).rstrip("/") == e2e_server.urls.qwen_api_base().rstrip(
            "/"
        ), "Provider should use the E2E (mock) base URL"

        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert "test summary" in result["summary"].lower()
        provider.cleanup()

    def test_qwen_speaker_detector_uses_e2e_server(self, e2e_server):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            qwen_api_base=e2e_server.urls.qwen_api_base(),
            qwen_speaker_model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            qwen_verify_served_model=False,
            speaker_detector_provider="qwen",
            auto_speakers=True,
        )

        detector = create_speaker_detector(cfg)
        detector.initialize()

        assert str(detector.client.base_url).rstrip("/") == e2e_server.urls.qwen_api_base().rstrip(
            "/"
        ), "Detector should use the E2E (mock) base URL"

        speakers, detected_hosts, success, _ = detector.detect_speakers(
            episode_title="Test Episode with Alice and Bob",
            episode_description="Alice interviews Bob about their work",
            known_hosts={"Alice"},
        )
        assert isinstance(speakers, list)
        assert isinstance(detected_hosts, set)
        assert isinstance(success, bool)
        detector.cleanup()
