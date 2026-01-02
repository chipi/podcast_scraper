#!/usr/bin/env python3
"""Multi-Episode E2E Tests.

These tests validate multi-episode processing logic using a multi-episode test feed
with 5 short episodes (10-15 seconds each). This tests episode iteration,
concurrent processing, job queues, and error handling across multiple episodes
without the overhead of long audio files.

These tests use the multi-episode feed automatically (all regular E2E tests use multi-episode feed).
They can process multiple episodes (5 episodes from the multi-episode feed) to validate
multi-episode processing logic.

Note: Fast E2E tests (marked with @pytest.mark.critical_path) use the fast feed
(1 episode). Regular E2E tests (like these) automatically use the multi-episode feed (5 episodes).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config, config, run_pipeline


@pytest.mark.e2e
class TestMultiEpisodeE2E:
    """Multi-episode tests for processing logic."""

    def test_multi_episode_processing(self, e2e_server):
        """Test that multi-episode processing works correctly with multi-episode feed.

        Validates:
        - All episodes from multi-episode feed are processed (5 in full mode, 1 in fast mode)
        - Episode iteration/looping logic works correctly
        - Concurrent processing handles multiple episodes
        - Job queues work correctly across episodes
        - Error handling doesn't break iteration

        Uses multi-episode feed with 5 short episodes (10-15 seconds each).
        Note: Regular E2E tests automatically use multi-episode feed (no marker needed).
        In fast mode, only 1 episode is processed.
        """
        import os

        # Determine expected episode count based on test mode
        test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
        expected_episodes = 1 if test_mode == "fast" else 5

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=5,  # Request 5 episodes (will be limited to 1 in fast mode)
                transcribe_missing=True,  # Enable transcription
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Validate processing (adjust expectations based on test mode)
            # In multi-episode mode, episodes 1 and 2 have transcripts (will be processed)
            # Episodes 3, 4, 5 need transcription (will only be processed if Whisper is cached)
            if test_mode == "fast":
                assert (
                    count == expected_episodes
                ), f"Should process {expected_episodes} episode(s) (mode: {test_mode}), got {count}"
            else:
                # In multi-episode mode, at least 2 episodes (with transcripts) should be processed
                # More if Whisper is cached
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) "
                    f"in multi-episode mode, got {count}"
                )

            # Verify transcript files were created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            if test_mode == "fast":
                assert len(transcript_files) >= expected_episodes, (
                    f"Should have at least {expected_episodes} transcript "
                    f"file(s), got {len(transcript_files)}"
                )
            else:
                assert len(transcript_files) >= 2, (
                    f"Should have at least 2 transcript file(s) in multi-episode mode, "
                    f"got {len(transcript_files)}"
                )

            # Verify metadata files were created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            if test_mode == "fast":
                assert len(metadata_files) == expected_episodes, (
                    f"Should have exactly {expected_episodes} metadata "
                    f"file(s), got {len(metadata_files)}"
                )
            else:
                assert len(metadata_files) >= 2, (
                    f"Should have at least 2 metadata file(s) in multi-episode mode, "
                    f"got {len(metadata_files)}"
                )

            # Validate that all episodes were processed correctly
            for metadata_file in sorted(metadata_files):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    assert metadata["content"] is not None, "Content should not be None"

    def test_multi_episode_concurrent_processing(self, e2e_server):
        """Test concurrent processing of multiple episodes with multi-episode feed.

        Validates:
        - Concurrent processing works correctly with multiple episodes
        - Thread pool handles multiple episodes without race conditions
        - Job queues are managed correctly across concurrent workers

        In fast mode, only 1 episode is processed.
        """
        import os

        # Determine expected episode count based on test mode
        test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
        expected_episodes = 1 if test_mode == "fast" else 5

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=5,  # Request 5 episodes (will be limited to 1 in fast mode)
                workers=3,  # Use concurrent processing
                transcribe_missing=True,
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Validate processing (adjust expectations based on test mode)
            # In multi-episode mode, episodes 1 and 2 have transcripts (will be processed)
            # Episodes 3, 4, 5 need transcription (will only be processed if Whisper is cached)
            if test_mode == "fast":
                assert count == expected_episodes, (
                    f"Should process {expected_episodes} episode(s) concurrently "
                    f"(mode: {test_mode}), got {count}"
                )
            else:
                # In multi-episode mode, at least 2 episodes (with transcripts) should be processed
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) "
                    f"concurrently in multi-episode mode, got {count}"
                )

            # Verify all episodes were processed
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            if test_mode == "fast":
                assert len(metadata_files) == expected_episodes, (
                    f"Should have exactly {expected_episodes} metadata file(s) "
                    f"after concurrent processing, got {len(metadata_files)}"
                )
            else:
                assert len(metadata_files) >= 2, (
                    f"Should have at least 2 metadata file(s) after concurrent "
                    f"processing in multi-episode mode, got {len(metadata_files)}"
                )

    @pytest.mark.serial  # Must run sequentially - MPS backend segfaults in parallel
    def test_multi_episode_with_summarization(self, e2e_server):
        """Test multi-episode processing with summarization enabled.

        Validates:
        - Summarization works correctly across multiple episodes
        - Processing jobs are queued and handled correctly
        - All episodes get summaries generated
        """
        # Require ML models to be cached
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
            require_whisper_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        # Require Whisper model to be cached - needed to transcribe episodes 3-5
        # Tests should use tiny.en (TEST_DEFAULT_WHISPER_MODEL), not base.en
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=5,  # Process all 5 multi-episode episodes
                transcribe_missing=True,
                generate_summaries=True,  # Enable summarization
                summary_provider="transformers",  # Use transformers (not deprecated "local")
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Determine expected episode count based on test mode
            test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
            expected_episodes = 1 if test_mode == "fast" else 5

            # Validate processing (adjust expectations based on test mode)
            # In multi-episode mode:
            # - Episodes 1 and 2 have transcripts (will be processed)
            # - Episodes 3, 4, 5 need transcription (will be processed if Whisper is cached)
            # Since we require Whisper to be cached, all 5 episodes should be processed
            if test_mode == "fast":
                assert count == expected_episodes, (
                    f"Should process {expected_episodes} episode(s) with "
                    f"summarization (mode: {test_mode}), got {count}"
                )
            else:
                # In multi-episode mode, all 5 episodes should be processed when Whisper is cached
                assert count == 5, (
                    f"Should process all 5 episodes with summarization when Whisper is cached, "
                    f"got {count}"
                )

            # Verify summaries were generated for all episodes
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            if test_mode == "fast":
                assert len(metadata_files) == expected_episodes, (
                    f"Should have exactly {expected_episodes} metadata "
                    f"file(s), got {len(metadata_files)}"
                )
            else:
                # When Whisper is cached, all 5 episodes should be processed and have metadata
                assert len(metadata_files) == 5, (
                    f"Should have exactly 5 metadata file(s) when Whisper is cached, "
                    f"got {len(metadata_files)}"
                )

            # Check that summaries are present for all episodes
            # When Whisper is cached, all 5 episodes should have transcripts and summaries
            for metadata_file in sorted(metadata_files):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    # All episodes should have content (transcript) when Whisper is cached
                    assert metadata.get(
                        "content"
                    ), f"Metadata should have content section (file: {metadata_file.name})"
                    assert metadata["content"].get(
                        "transcript_source"
                    ), f"Metadata should have transcript_source (file: {metadata_file.name})"
                    # All episodes should have summaries
                    assert (
                        "summary" in metadata
                    ), f"Metadata should have summary section (file: {metadata_file.name})"
                    assert (
                        metadata["summary"] is not None
                    ), f"Summary should not be None (file: {metadata_file.name})"
                    # Verify summary has content
                    assert metadata["summary"].get(
                        "short_summary"
                    ), f"Summary should have short_summary field (file: {metadata_file.name})"
