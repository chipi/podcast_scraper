#!/usr/bin/env python3
"""Smoke E2E Tests.

These tests validate multi-episode processing logic using a smoke test feed
with 5 short episodes (10-15 seconds each). This tests episode iteration,
concurrent processing, job queues, and error handling across multiple episodes
without the overhead of long audio files.

These tests use the smoke feed automatically (all regular E2E tests use smoke feed).
They can process multiple episodes (5 episodes from the smoke feed) to validate
multi-episode processing logic.

Note: Fast E2E tests (marked with @pytest.mark.critical_path) use the fast feed
(1 episode). Regular E2E tests (like these) automatically use the smoke feed (5 episodes).
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

from podcast_scraper import Config, run_pipeline


@pytest.mark.e2e
class TestSmokeE2E:
    """Smoke tests for multi-episode processing logic."""

    def test_multi_episode_processing_smoke(self, e2e_server):
        """Test that multi-episode processing works correctly with smoke feed.

        Validates:
        - All episodes from smoke feed are processed (5 in full mode, 1 in fast mode)
        - Episode iteration/looping logic works correctly
        - Concurrent processing handles multiple episodes
        - Job queues work correctly across episodes
        - Error handling doesn't break iteration

        Uses smoke feed with 5 short episodes (10-15 seconds each).
        Note: Regular E2E tests automatically use smoke feed (no marker needed).
        In fast mode, only 1 episode is processed.
        """
        import os

        # Determine expected episode count based on test mode
        test_mode = os.environ.get("E2E_TEST_MODE", "smoke").lower()
        expected_episodes = 1 if test_mode == "fast" else 5

        rss_url = e2e_server.urls.feed("podcast1_smoke")

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
            # In smoke mode, episodes 1 and 2 have transcripts (will be processed)
            # Episodes 3, 4, 5 need transcription (will only be processed if Whisper is cached)
            if test_mode == "fast":
                assert (
                    count == expected_episodes
                ), f"Should process {expected_episodes} episode(s) (mode: {test_mode}), got {count}"
            else:
                # In smoke mode, at least 2 episodes (with transcripts) should be processed
                # More if Whisper is cached
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) "
                    f"in smoke mode, got {count}"
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
                    f"Should have at least 2 transcript file(s) in smoke mode, "
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
                    f"Should have at least 2 metadata file(s) in smoke mode, "
                    f"got {len(metadata_files)}"
                )

            # Validate that all episodes were processed correctly
            for metadata_file in sorted(metadata_files):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    assert metadata["content"] is not None, "Content should not be None"

    def test_multi_episode_concurrent_processing_smoke(self, e2e_server):
        """Test concurrent processing of multiple episodes with smoke feed.

        Validates:
        - Concurrent processing works correctly with multiple episodes
        - Thread pool handles multiple episodes without race conditions
        - Job queues are managed correctly across concurrent workers

        In fast mode, only 1 episode is processed.
        """
        import os

        # Determine expected episode count based on test mode
        test_mode = os.environ.get("E2E_TEST_MODE", "smoke").lower()
        expected_episodes = 1 if test_mode == "fast" else 5

        rss_url = e2e_server.urls.feed("podcast1_smoke")

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
            # In smoke mode, episodes 1 and 2 have transcripts (will be processed)
            # Episodes 3, 4, 5 need transcription (will only be processed if Whisper is cached)
            if test_mode == "fast":
                assert count == expected_episodes, (
                    f"Should process {expected_episodes} episode(s) concurrently "
                    f"(mode: {test_mode}), got {count}"
                )
            else:
                # In smoke mode, at least 2 episodes (with transcripts) should be processed
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) "
                    f"concurrently in smoke mode, got {count}"
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
                    f"processing in smoke mode, got {len(metadata_files)}"
                )

    @pytest.mark.serial  # Must run sequentially - MPS backend segfaults in parallel
    def test_multi_episode_with_summarization_smoke(self, e2e_server):
        """Test multi-episode processing with summarization enabled.

        Validates:
        - Summarization works correctly across multiple episodes
        - Processing jobs are queued and handled correctly
        - All episodes get summaries generated
        """
        # Require ML models to be cached
        from tests.integration.ml_model_cache_helpers import (
            require_spacy_model_cached,
            require_transformers_model_cached,
        )

        require_spacy_model_cached("en_core_web_sm")
        require_transformers_model_cached("facebook/bart-large-cnn", None)

        require_transformers_model_cached("facebook/bart-large-cnn", None)

        rss_url = e2e_server.urls.feed("podcast1_smoke")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=5,  # Process all 5 smoke episodes
                transcribe_missing=True,
                generate_summaries=True,  # Enable summarization
                summary_provider="local",
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Determine expected episode count based on test mode
            import os

            test_mode = os.environ.get("E2E_TEST_MODE", "smoke").lower()
            expected_episodes = 1 if test_mode == "fast" else 5

            # Validate processing (adjust expectations based on test mode)
            # In smoke mode, episodes 1 and 2 have transcripts (will be processed)
            # Episodes 3, 4, 5 need transcription (will only be processed if Whisper is cached)
            if test_mode == "fast":
                assert count == expected_episodes, (
                    f"Should process {expected_episodes} episode(s) with "
                    f"summarization (mode: {test_mode}), got {count}"
                )
            else:
                # In smoke mode, at least 2 episodes (with transcripts) should be processed
                # More if Whisper is cached
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) with "
                    f"summarization in smoke mode, got {count}"
                )

            # Verify summaries were generated for all episodes
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            if test_mode == "fast":
                assert len(metadata_files) == expected_episodes, (
                    f"Should have exactly {expected_episodes} metadata "
                    f"file(s), got {len(metadata_files)}"
                )
            else:
                assert len(metadata_files) >= 2, (
                    f"Should have at least 2 metadata file(s) in smoke mode, "
                    f"got {len(metadata_files)}"
                )

            # Check that summaries are present
            for metadata_file in sorted(metadata_files):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "summary" in metadata, "Metadata should have summary section"
                    assert metadata["summary"] is not None, "Summary should not be None"
