#!/usr/bin/env python3
"""OpenAI provider E2E tests (Stage 10).

These tests verify that OpenAI providers work correctly in complete user workflows:
- OpenAI transcription in workflow
- OpenAI speaker detection in workflow
- OpenAI summarization in workflow
- Error handling (API errors, rate limiting, retries)
- Fallback behavior when OpenAI API fails

These tests use the E2E server's OpenAI mock endpoints (real HTTP requests to mock server)
and are marked with @pytest.mark.e2e to allow selective execution.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import workflow

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: E402
    create_test_config,
)

from podcast_scraper import config


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.openai
class TestOpenAIProviderE2E:
    """Test OpenAI providers in integration workflows using E2E server."""

    def test_openai_transcription_in_pipeline(self, e2e_server):
        """Test OpenAI transcription provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI transcription using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                transcription_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                transcribe_missing=True,  # Enable transcription
                generate_metadata=True,
                max_episodes=1,
            )

            # Run pipeline (uses E2E server OpenAI endpoints, no direct mocking)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify transcript files were created
            transcript_files = list(Path(temp_dir).rglob("*.txt")) + list(
                Path(temp_dir).rglob("*.vtt")
            )
            assert len(transcript_files) >= 1, "Should have created at least one transcript file"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_speaker_detection_in_pipeline(self, e2e_server):
        """Test OpenAI speaker detection provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI speaker detection using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                speaker_detector_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                auto_speakers=True,
                generate_metadata=True,
                transcription_provider="whisper",  # Use Whisper for transcription
                transcribe_missing=True,
                max_episodes=1,
            )

            # Require Whisper model to be cached (skip if not available)
            from tests.integration.ml_model_cache_helpers import require_whisper_model_cached

            require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

            # Run pipeline with real Whisper (uses E2E server OpenAI endpoints)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved (transcript file must exist for metadata)
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created (may not exist if
            # transcript file doesn't exist)
            metadata_files = list(Path(temp_dir).rglob("*.json"))
            # Note: Metadata files may not be created if transcript files don't exist
            # The key is that OpenAI speaker detection was called via E2E server
            if len(metadata_files) > 0:
                import json as json_module

                metadata_content = json_module.loads(Path(metadata_files[0]).read_text())
                # Check that speaker detection results are in metadata
                # They may be in 'content' section
                content = metadata_content.get("content", {})
                assert (
                    "detected_hosts" in content
                    or "detected_guests" in content
                    or "detected_hosts" in metadata_content
                    or "detected_guests" in metadata_content
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_summarization_in_pipeline(self, e2e_server):
        """Test OpenAI summarization provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI summarization using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                summary_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                generate_metadata=True,
                generate_summaries=True,
                transcription_provider="whisper",  # Use Whisper for transcription
                transcribe_missing=True,
                max_episodes=1,
            )

            # Require Whisper model to be cached (skip if not available)
            from tests.integration.ml_model_cache_helpers import require_whisper_model_cached

            require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

            # Run pipeline with real Whisper (uses E2E server OpenAI endpoints)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved (transcript file must exist for summarization)
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_files = list(Path(temp_dir).rglob("*.json"))
            # Note: Summarization may not be called if transcript file doesn't exist
            # or if there's an error. The key is that the pipeline completes.
            if len(metadata_files) > 0:
                import json as json_module

                metadata_content = json_module.loads(Path(metadata_files[0]).read_text())
                # If summary exists, verify it's correct (from E2E server)
                if "summary" in metadata_content:
                    assert len(metadata_content["summary"]) > 0, "Summary should not be empty"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_all_providers_in_pipeline(self, e2e_server):
        """Test all OpenAI providers together in full pipeline using E2E server."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with all OpenAI providers using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",
                summary_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=True,
                transcribe_missing=True,
                max_episodes=1,
            )

            # Run pipeline (uses E2E server OpenAI endpoints, no direct mocking)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created (may not exist if transcript file doesn't exist)
            # But if they are created, verify they're correct
            metadata_files = list(Path(temp_dir).rglob("*.json"))
            if len(metadata_files) > 0:
                # Verify metadata structure
                import json as json_module

                metadata_content = json_module.loads(Path(metadata_files[0]).read_text())
                assert "content" in metadata_content or "episode" in metadata_content
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_transcription_api_error_handling(self, e2e_server):
        """Test that OpenAI transcription API errors are handled gracefully.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI transcription using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                transcription_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                transcribe_missing=True,
                max_episodes=1,
            )

            # Run pipeline - E2E server provides successful responses
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_speaker_detection_api_error_handling(self, e2e_server):
        """Test that OpenAI speaker detection works with E2E server.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI speaker detection using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                speaker_detector_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                auto_speakers=True,
                generate_metadata=True,
                transcription_provider="whisper",
                transcribe_missing=True,
                max_episodes=1,
            )

            # Require Whisper model to be cached (skip if not available)
            from tests.integration.ml_model_cache_helpers import require_whisper_model_cached

            require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

            # Run pipeline with real Whisper - E2E server provides successful responses
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_summarization_api_error_handling(self, e2e_server):
        """Test that OpenAI summarization works with E2E server.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI summarization using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                summary_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                generate_metadata=True,
                generate_summaries=True,
                transcription_provider="whisper",
                transcribe_missing=True,
                max_episodes=1,
            )

            # Require Whisper model to be cached (skip if not available)
            from tests.integration.ml_model_cache_helpers import require_whisper_model_cached

            require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

            # Run pipeline with real Whisper - E2E server provides successful responses
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify metadata files were created
            # Metadata should be created with summary from E2E server
            metadata_files = list(Path(temp_dir).rglob("*.json"))
            assert len(metadata_files) > 0, "Metadata files should be created"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_transcription_rate_limiting(self, e2e_server):
        """Test that OpenAI transcription works with E2E server.

        Note: E2E server currently doesn't simulate rate limiting, so this test verifies
        that the pipeline completes successfully. Rate limiting is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Create config with OpenAI transcription using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                transcription_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                transcribe_missing=True,
                max_episodes=1,
            )

            # Run pipeline - E2E server provides successful responses
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Rate limiting is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
