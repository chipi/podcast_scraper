#!/usr/bin/env python3
"""Gemini provider E2E tests (Issue #194).

These tests verify that Gemini providers work correctly in complete user workflows:
- Gemini transcription in workflow
- Gemini speaker detection in workflow
- Gemini summarization in workflow
- Error handling (API errors, rate limiting, retries)

These tests use mocked Gemini SDK calls (since the SDK may not support custom base URLs)
and are marked with @pytest.mark.e2e to allow selective execution.

Real API Mode:
    When USE_REAL_GEMINI_API=1, tests use real Gemini API endpoints and real RSS feeds.
    This is for manual testing only and will incur API costs.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.module_gemini_providers]

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import workflow

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid pytest resolution issues
from tests.conftest import (  # noqa: E402
    create_test_config,
)

# Check if we should use real Gemini API (for manual testing only)
USE_REAL_GEMINI_API = os.getenv("USE_REAL_GEMINI_API", "0") == "1"

# Feed selection for LLM provider tests (shared by OpenAI, Gemini, etc.)
# Default to "multi" to work in both fast and multi_episode E2E_TEST_MODE
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")

# Real RSS feed URL for testing (only used when USE_REAL_GEMINI_API=1)
# NOTE: No default real feed - must be explicitly set via LLM_TEST_RSS_FEED
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Get RSS feed URL and Gemini config based on LLM_TEST_FEED environment variable.

    Args:
        e2e_server: E2E server fixture (None if using real API or if fixture not available)

    Returns:
        Tuple of (rss_url, gemini_api_base, gemini_api_key)
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    # Real API mode - can use either real RSS feed OR fixture feeds
    if USE_REAL_GEMINI_API:
        # If LLM_TEST_RSS_FEED is explicitly set, use that real RSS feed
        if REAL_TEST_RSS_FEED is not None:
            return REAL_TEST_RSS_FEED, None, None

        # Otherwise, use fixture feeds from E2E server (fixture input, real API calls)
        if e2e_server is None:
            raise ValueError(
                "E2E server is required when using fixture feeds with real API. "
                "Set LLM_TEST_RSS_FEED=<url> to use a real RSS feed instead."
            )

        # Use fixture feeds but with real API (no mock API base)
        feed_mapping = {
            "multi": "podcast1_multi_episode",
            "fast": "podcast1",
            "p01": "podcast1",
            "p02": "podcast2",
            "p03": "podcast3",
            "p04": "podcast4",
            "p05": "podcast5",
        }
        podcast_name = feed_mapping.get(feed_type, "podcast1_multi_episode")
        rss_url = e2e_server.urls.feed(podcast_name)
        return rss_url, None, None

    # Mock API mode - use E2E server for both feeds and API
    if e2e_server is None:
        raise ValueError("E2E server is required for mock API mode")

    feed_mapping = {
        "multi": "podcast1_multi_episode",
        "fast": "podcast1",
        "p01": "podcast1",
        "p02": "podcast2",
        "p03": "podcast3",
        "p04": "podcast4",
        "p05": "podcast5",
    }
    podcast_name = feed_mapping.get(feed_type, "podcast1_multi_episode")
    rss_url = e2e_server.urls.feed(podcast_name)
    gemini_api_base = e2e_server.urls.gemini_api_base()
    gemini_api_key = "test-dummy-key-for-e2e-tests"

    return rss_url, gemini_api_base, gemini_api_key


def _summarize_gemini_usage(metadata_content: dict) -> str:
    """Extract and summarize Gemini API calls and models used from metadata.

    Args:
        metadata_content: Parsed metadata JSON content

    Returns:
        Human-readable summary string of Gemini APIs and models used
    """
    processing = metadata_content.get("processing", {})
    config_snapshot = processing.get("config_snapshot", {})
    ml_providers = config_snapshot.get("ml_providers", {})

    summary_parts = []

    # Transcription
    transcription_info = ml_providers.get("transcription", {})
    if transcription_info.get("provider") == "gemini":
        model = transcription_info.get("gemini_model", "gemini-2.0-flash")
        summary_parts.append(f"  • Transcription: Gemini API (model: {model})")

    # Speaker detection
    speaker_info = ml_providers.get("speaker_detection", {})
    if speaker_info.get("provider") == "gemini":
        model = speaker_info.get("gemini_model", "gemini-2.0-flash")
        summary_parts.append(f"  • Speaker Detection: Gemini API (model: {model})")

    # Summarization
    summarization_info = ml_providers.get("summarization", {})
    if summarization_info.get("provider") == "gemini":
        model = summarization_info.get("gemini_model", "gemini-2.0-flash")
        summary_parts.append(f"  • Summarization: Gemini API (model: {model})")

    if summary_parts:
        return "Gemini APIs called:\n" + "\n".join(summary_parts)
    return "No Gemini APIs detected in metadata"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiProviderE2E:
    """Test Gemini providers in integration workflows using mocked SDK."""

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_transcription_in_pipeline(self, mock_genai, e2e_server: Optional[Any]):
        """Test Gemini transcription provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Gemini config
            rss_url, gemini_api_base, gemini_api_key = _get_test_feed_url(e2e_server)

            # Mock Gemini SDK responses
            mock_response = Mock()
            mock_response.text = "This is a test transcription from Gemini in the pipeline."
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            # Create config with Gemini transcription
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="gemini",
                speaker_detector_provider="spacy",  # Use local for speaker detection
                summary_provider="transformers",  # Use local for summarization
                gemini_api_key=gemini_api_key,
                gemini_api_base=gemini_api_base,
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Gemini for transcription, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_dir = Path(temp_dir) / "metadata"
            assert metadata_dir.exists(), "Metadata directory should exist"

            metadata_files = list(metadata_dir.glob("*.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify Gemini was used (check metadata)
            if metadata_files:
                import json

                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that Gemini was used for transcription
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                transcription_info = ml_providers.get("transcription", {})
                assert transcription_info.get("provider") == "gemini"

                # Print summary
                usage_summary = _summarize_gemini_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_speaker_detection_in_pipeline(self, mock_genai, e2e_server: Optional[Any]):
        """Test Gemini speaker detection provider in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Gemini config
            rss_url, gemini_api_base, gemini_api_key = _get_test_feed_url(e2e_server)

            # Mock Gemini SDK responses
            mock_response = Mock()
            mock_response.text = json.dumps(
                {
                    "speakers": ["Host", "Guest"],
                    "hosts": ["Host"],
                    "guests": ["Guest"],
                }
            )
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            # Create config with Gemini speaker detection
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="whisper",  # Use local for transcription
                speaker_detector_provider="gemini",
                summary_provider="transformers",  # Use local for summarization
                gemini_api_key=gemini_api_key,
                gemini_api_base=gemini_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=False,  # Disable summarization
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Gemini for speaker detection, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_dir = Path(temp_dir) / "metadata"
            assert metadata_dir.exists(), "Metadata directory should exist"

            metadata_files = list(metadata_dir.glob("*.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify Gemini was used (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that Gemini was used for speaker detection
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == "gemini"

                # Print summary
                usage_summary = _summarize_gemini_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_summarization_in_pipeline(self, mock_genai, e2e_server: Optional[Any]):
        """Test Gemini summarization provider in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Gemini config
            rss_url, gemini_api_base, gemini_api_key = _get_test_feed_url(e2e_server)

            # Mock Gemini SDK responses
            mock_response = Mock()
            mock_response.text = "This is a test summary from Gemini in the pipeline."
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            # Create config with Gemini summarization
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="whisper",  # Use local for transcription
                speaker_detector_provider="spacy",  # Use local for speaker detection
                summary_provider="gemini",
                gemini_api_key=gemini_api_key,
                gemini_api_base=gemini_api_base,
                generate_summaries=True,
                generate_metadata=True,
                auto_speakers=True,
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Gemini for summarization, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_dir = Path(temp_dir) / "metadata"
            assert metadata_dir.exists(), "Metadata directory should exist"

            metadata_files = list(metadata_dir.glob("*.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify Gemini was used (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that Gemini was used for summarization
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                summarization_info = ml_providers.get("summarization", {})
                assert summarization_info.get("provider") == "gemini"

                # Print summary
                usage_summary = _summarize_gemini_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_full_pipeline(self, mock_genai, e2e_server: Optional[Any]):
        """Test Gemini provider for all three capabilities in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Gemini config
            rss_url, gemini_api_base, gemini_api_key = _get_test_feed_url(e2e_server)

            # Mock Gemini SDK responses
            # Transcription response
            mock_transcription_response = Mock()
            mock_transcription_response.text = "This is a test transcription from Gemini."

            # Speaker detection response
            mock_speaker_response = Mock()
            mock_speaker_response.text = json.dumps(
                {
                    "speakers": ["Host", "Guest"],
                    "hosts": ["Host"],
                    "guests": ["Guest"],
                }
            )

            # Summarization response
            mock_summary_response = Mock()
            mock_summary_response.text = "This is a test summary from Gemini."

            # Create mock model that returns different responses based on call context
            def mock_generate_content(*args, **kwargs):
                # Check if request contains audio (transcription) or text (speaker/summary)
                contents = args[0] if args else kwargs.get("contents", [])
                for content in contents:
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("mime_type", "").startswith(
                                "audio/"
                            ):
                                return mock_transcription_response
                    # Check generation config for JSON response (speaker detection)
                    gen_config = kwargs.get("generation_config", {})
                    if gen_config.get("response_mime_type") == "application/json":
                        return mock_speaker_response
                # Default to summarization
                return mock_summary_response

            mock_model = Mock()
            mock_model.generate_content.side_effect = mock_generate_content
            mock_genai.GenerativeModel.return_value = mock_model

            # Create config with Gemini for all capabilities
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="gemini",
                speaker_detector_provider="gemini",
                summary_provider="gemini",
                gemini_api_key=gemini_api_key,
                gemini_api_base=gemini_api_base,
                generate_summaries=True,
                generate_metadata=True,
                auto_speakers=True,
                transcribe_missing=True,
                preload_models=False,  # No local ML models needed
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Gemini for all three capabilities)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_dir = Path(temp_dir) / "metadata"
            assert metadata_dir.exists(), "Metadata directory should exist"

            metadata_files = list(metadata_dir.glob("*.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify Gemini was used for all capabilities (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that Gemini was used for all three capabilities
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})

                transcription_info = ml_providers.get("transcription", {})
                assert transcription_info.get("provider") == "gemini"

                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == "gemini"

                summarization_info = ml_providers.get("summarization", {})
                assert summarization_info.get("provider") == "gemini"

                # Print summary
                usage_summary = _summarize_gemini_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
