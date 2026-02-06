#!/usr/bin/env python3
"""Grok provider E2E tests (Issue #1095).

These tests verify that Grok providers work correctly in complete user workflows:
- Grok speaker detection in workflow
- Grok summarization in workflow
- Error handling (API errors, rate limiting, retries)

These tests use the E2E server's OpenAI-compatible mock endpoints
(real HTTP requests to mock server) and are marked with @pytest.mark.e2e
to allow selective execution.

Note: Grok does NOT support transcription (no audio API).

Real API Mode:
    When USE_REAL_GROK_API=1, tests use real Grok API endpoints and real RSS feeds.
    This is for manual testing only and will incur API costs.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.module_grok_providers]

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

# Check if we should use real Grok API (for manual testing only)
USE_REAL_GROK_API = os.getenv("USE_REAL_GROK_API", "0") == "1"

# Feed selection for LLM provider tests (shared by OpenAI, Gemini, Grok, etc.)
# Default to "multi" to work in both fast and multi_episode E2E_TEST_MODE
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")

# Real RSS feed URL for testing (only used when USE_REAL_GROK_API=1)
# NOTE: No default real feed - must be explicitly set via LLM_TEST_RSS_FEED
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Get RSS feed URL and Grok config based on LLM_TEST_FEED environment variable.

    Args:
        e2e_server: E2E server fixture (None if using real API or if fixture not available)

    Returns:
        Tuple of (rss_url, grok_api_base, grok_api_key)
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    # Real API mode - can use either real RSS feed OR fixture feeds
    if USE_REAL_GROK_API:
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
    grok_api_base = e2e_server.urls.grok_api_base()
    grok_api_key = "test-dummy-key-for-e2e-tests"

    return rss_url, grok_api_base, grok_api_key


def _summarize_grok_usage(metadata_content: dict) -> str:
    """Extract and summarize Grok API calls and models used from metadata.

    Args:
        metadata_content: Parsed metadata JSON content

    Returns:
        Human-readable summary string of Grok APIs and models used
    """
    processing = metadata_content.get("processing", {})
    config_snapshot = processing.get("config_snapshot", {})
    ml_providers = config_snapshot.get("ml_providers", {})

    summary_parts = []

    # Speaker detection
    speaker_info = ml_providers.get("speaker_detection", {})
    if speaker_info.get("provider") == "grok":
        model = speaker_info.get("grok_model", "grok-2")
        summary_parts.append(f"  • Speaker Detection: Grok API (model: {model})")

    # Summarization
    summarization_info = ml_providers.get("summarization", {})
    if summarization_info.get("provider") == "grok":
        model = summarization_info.get("grok_model", "grok-2")
        summary_parts.append(f"  • Summarization: Grok API (model: {model})")

    if summary_parts:
        return "Grok APIs called:\n" + "\n".join(summary_parts)
    return "No Grok APIs detected in metadata"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.grok
class TestGrokProviderE2E:
    """Test Grok providers in integration workflows using E2E server mock endpoints."""

    def test_grok_speaker_detection_in_pipeline(self, e2e_server: Optional[Any]):
        """Test Grok speaker detection provider in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Grok config
            rss_url, grok_api_base, grok_api_key = _get_test_feed_url(e2e_server)

            # Create config with Grok speaker detection
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="whisper",  # Use local for transcription
                speaker_detector_provider="grok",
                summary_provider="transformers",  # Use local for summarization
                grok_api_key=grok_api_key,
                grok_api_base=grok_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=False,  # Disable summarization
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Grok for speaker detection, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            # Output directory includes run suffix, so find the actual run directory
            run_dirs = [
                d for d in Path(temp_dir).iterdir() if d.is_dir() and d.name.startswith("run_")
            ]
            assert (
                len(run_dirs) > 0
            ), f"Should have created at least one run directory in {temp_dir}"
            run_dir = run_dirs[0]
            metadata_dir = run_dir / "metadata"
            assert metadata_dir.exists(), f"Metadata directory should exist at {metadata_dir}"

            metadata_files = list(metadata_dir.glob("*.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify Grok was used (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that Grok was used for speaker detection
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == "grok"

                # Print summary
                usage_summary = _summarize_grok_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_grok_summarization_in_pipeline(self, e2e_server: Optional[Any]):
        """Test Grok summarization provider in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Grok config
            rss_url, grok_api_base, grok_api_key = _get_test_feed_url(e2e_server)

            # Create config with Grok summarization
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="whisper",  # Use local for transcription
                speaker_detector_provider="spacy",  # Use local for speaker detection
                summary_provider="grok",
                grok_api_key=grok_api_key,
                grok_api_base=grok_api_base,
                generate_summaries=True,
                generate_metadata=True,
                auto_speakers=True,
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Grok for summarization, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            # Output directory includes run suffix, so find the actual run directory
            run_dirs = [
                d for d in Path(temp_dir).iterdir() if d.is_dir() and d.name.startswith("run_")
            ]
            assert (
                len(run_dirs) > 0
            ), f"Should have created at least one run directory in {temp_dir}"
            run_dir = run_dirs[0]
            metadata_dir = run_dir / "metadata"
            assert metadata_dir.exists(), f"Metadata directory should exist at {metadata_dir}"

            metadata_files = list(metadata_dir.glob("*.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify Grok was used (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that Grok was used for summarization
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                summarization_info = ml_providers.get("summarization", {})
                assert summarization_info.get("provider") == "grok"

                # Print summary
                usage_summary = _summarize_grok_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_grok_full_pipeline(self, e2e_server: Optional[Any]):
        """Test all Grok providers together in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Grok config based on LLM_TEST_FEED
            rss_url, grok_api_base, grok_api_key = _get_test_feed_url(e2e_server)

            # Create config with ALL Grok providers (speaker detection + summarization)
            # Note: Grok does NOT support transcription
            # Use production model for real API tests
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Use local for transcription
                "speaker_detector_provider": "grok",
                "summary_provider": "grok",
                "grok_speaker_model": "grok-3-mini",  # Use grok-3-mini (faster, cheaper)
                "grok_summary_model": "grok-3-mini",  # Use grok-3-mini (faster, cheaper)
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": True,
                "preload_models": False,  # Disable model preloading (no local ML models)
                "transcribe_missing": True,
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            }
            # Only pass API key/base if provided (when None, Config loads from .env)
            # For real API mode, ensure API key is available from environment
            if grok_api_key is not None:
                config_kwargs["grok_api_key"] = grok_api_key
            elif USE_REAL_GROK_API:
                # In real API mode, try to load from environment if not provided
                env_key = os.getenv("GROK_API_KEY")
                if env_key:
                    config_kwargs["grok_api_key"] = env_key
            if grok_api_base is not None:
                config_kwargs["grok_api_base"] = grok_api_base

            cfg = create_test_config(**config_kwargs)

            # Run pipeline (uses E2E server Grok endpoints or real API)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata files should be created when generate_metadata=True"

            import json as json_module

            # Verify metadata structure for all episodes
            for metadata_file in sorted(metadata_files):
                metadata_content = json_module.loads(metadata_file.read_text())
                assert (
                    "content" in metadata_content or "episode" in metadata_content
                ), "Metadata should have content or episode section"

            # Verify Grok was used for both capabilities
            for metadata_file in sorted(metadata_files):
                metadata_content = json_module.loads(metadata_file.read_text())
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})

                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == "grok"

                summarization_info = ml_providers.get("summarization", {})
                assert summarization_info.get("provider") == "grok"

                # Print summary
                usage_summary = _summarize_grok_usage(metadata_content)
                print(f"\n{usage_summary}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
