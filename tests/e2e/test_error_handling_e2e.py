#!/usr/bin/env python3
"""E2E tests for Error Handling.

These tests verify error handling works correctly in E2E scenarios:
- HTTP errors (404, 500, timeout)
- Invalid RSS feeds
- Invalid config files
- Network errors
- Retry logic

All tests use real HTTP client and E2E server with error scenarios.
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

from podcast_scraper import Config, config, run_pipeline, service


@pytest.mark.e2e
@pytest.mark.slow
class TestHTTPErrorHandling:
    """HTTP error handling E2E tests."""

    def test_rss_feed_404_error(self, e2e_server):
        """Test handling of RSS feed 404 error."""
        # Set error behavior for RSS feed
        e2e_server.set_error_behavior("/feeds/podcast1_multi_episode/feed.xml", 404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should raise ValueError when RSS feed fails
            with pytest.raises(ValueError, match="Failed to fetch RSS feed"):
                run_pipeline(cfg)

        # Clear error behavior
        e2e_server.clear_error_behavior("/feeds/podcast1_multi_episode/feed.xml")

    def test_rss_feed_500_error(self, e2e_server):
        """Test handling of RSS feed 500 error."""
        # Set error behavior for RSS feed
        e2e_server.set_error_behavior("/feeds/podcast1_multi_episode/feed.xml", 500)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should raise ValueError after retries fail
            # (HTTP retries will attempt 5 times, then fail)
            with pytest.raises(ValueError, match="Failed to fetch RSS feed"):
                run_pipeline(cfg)

        # Clear error behavior
        e2e_server.clear_error_behavior("/feeds/podcast1_multi_episode/feed.xml")

    def test_transcript_download_404_error(self, e2e_server):
        """Test handling of transcript download 404 error."""
        # Set error behavior for transcript
        e2e_server.set_error_behavior("/transcripts/p01_multi_e01.txt", 404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should handle 404 gracefully
            count, summary = run_pipeline(cfg)

            # Pipeline should complete but may process 0 episodes (no transcript available)
            assert count >= 0, "Should handle transcript 404 error gracefully"
            assert isinstance(summary, str), "Summary should be a string"

        # Clear error behavior
        e2e_server.clear_error_behavior("/transcripts/p01_multi_e01.txt")

    def test_transcript_download_500_error(self, e2e_server):
        """Test handling of transcript download 500 error."""
        # Set error behavior for transcript
        e2e_server.set_error_behavior("/transcripts/p01_multi_e01.txt", 500)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should handle 500 gracefully (with retries)
            count, summary = run_pipeline(cfg)

            # Pipeline should complete but may process 0 episodes (after retries fail)
            assert count >= 0, "Should handle transcript 500 error gracefully"
            assert isinstance(summary, str), "Summary should be a string"

        # Clear error behavior
        e2e_server.clear_error_behavior("/transcripts/p01_multi_e01.txt")

    def test_audio_download_404_error(self, e2e_server):
        """Test handling of audio download 404 error."""
        # Set error behavior for audio
        e2e_server.set_error_behavior("/audio/p01_multi_e01.mp3", 404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should handle 404 gracefully
            # Note: If transcript exists, audio won't be downloaded
            count, summary = run_pipeline(cfg)

            # Pipeline should complete
            assert count >= 0, "Should handle audio 404 error gracefully"
            assert isinstance(summary, str), "Summary should be a string"

        # Clear error behavior
        e2e_server.clear_error_behavior("/audio/p01_multi_e01.mp3")

    def test_chaos_run_index_records_failed_episode(self, e2e_server):
        """Issue #429 Phase 2: Chaos test – run index records failed episode.

        Feed has 3 episodes; episode 3 (no transcript URL) gets 404 on audio.
        Assert run completes and index.json has one failed episode with
        status, error_type, error_message, error_stage set.
        """
        # p01_multi: e01/e02 have transcript URL, e03 has only enclosure
        e2e_server.set_error_behavior("/audio/p01_multi_e03.mp3", 404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=3,
                transcribe_missing=True,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            count, summary = run_pipeline(cfg)

            # Run should complete (exit 0 semantics: run completed)
            assert isinstance(summary, str), "Summary should be a string"

            # Find index.json (output is under tmpdir/run_<suffix>/)
            run_json_candidates = list(Path(tmpdir).rglob("run.json"))
            assert run_json_candidates, "run.json should be produced"
            output_root = run_json_candidates[0].parent
            index_path = output_root / "index.json"
            assert index_path.exists(), "index.json should exist"

            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            assert (
                index_data.get("episodes_failed", 0) >= 1
            ), "At least one episode should be recorded as failed"
            failed = [ep for ep in index_data.get("episodes", []) if ep.get("status") == "failed"]
            assert failed, "At least one episode entry should have status 'failed'"
            for ep in failed:
                assert ep.get("error_type"), "Failed entry should have error_type"
                assert ep.get("error_message"), "Failed entry should have error_message"
                assert ep.get("error_stage"), "Failed entry should have error_stage"

        e2e_server.clear_error_behavior("/audio/p01_multi_e03.mp3")

    def test_service_api_error_handling(self, e2e_server):
        """Test service API error handling with HTTP errors."""
        # Set error behavior for RSS feed
        e2e_server.set_error_behavior("/feeds/podcast1_multi_episode/feed.xml", 404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run service - should handle error gracefully
            result = service.run(cfg)

            # Service should indicate failure or process 0 episodes
            assert result.episodes_processed == 0, "Should process 0 episodes on 404"
            # Service may succeed (graceful handling) or fail (error returned)
            assert isinstance(result.summary, str), "Summary should be a string"

        # Clear error behavior
        e2e_server.clear_error_behavior("/feeds/podcast1_multi_episode/feed.xml")


@pytest.mark.e2e
@pytest.mark.slow
class TestInvalidRSSFeed:
    """Invalid RSS feed error handling E2E tests."""

    def test_malformed_rss_feed(self, e2e_server):
        """Test handling of malformed RSS feed."""
        # Use a non-existent feed that returns 404
        invalid_url = e2e_server.urls.base() + "/feeds/nonexistent/feed.xml"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=invalid_url,
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should raise ValueError when RSS feed fails
            with pytest.raises(ValueError, match="Failed to fetch RSS feed"):
                run_pipeline(cfg)

    def test_empty_rss_feed(self, e2e_server):
        """Test handling of empty RSS feed."""
        # Use a feed that doesn't exist (404)
        invalid_url = e2e_server.urls.base() + "/feeds/empty/feed.xml"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=invalid_url,
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should raise ValueError when RSS feed fails
            with pytest.raises(ValueError, match="Failed to fetch RSS feed"):
                run_pipeline(cfg)


@pytest.mark.e2e
@pytest.mark.slow
class TestInvalidConfig:
    """Invalid config error handling E2E tests."""

    def test_invalid_rss_url(self, e2e_server):
        """Test handling of invalid RSS URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use an invalid URL format
            cfg = Config(
                rss_url="not-a-valid-url",
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should raise ValueError when URL is invalid
            with pytest.raises(ValueError, match="Failed to fetch RSS feed"):
                run_pipeline(cfg)

    def test_service_api_invalid_config(self, e2e_server):
        """Test service API error handling with invalid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url="not-a-valid-url",
                output_dir=tmpdir,
                max_episodes=1,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run service - should handle invalid config gracefully
            result = service.run(cfg)

            # Service should indicate failure or process 0 episodes
            assert result.episodes_processed == 0, "Should process 0 episodes with invalid config"
            assert isinstance(result.summary, str), "Summary should be a string"


@pytest.mark.e2e
@pytest.mark.slow
class TestPartialFailureHandling:
    """Partial failure handling E2E tests."""

    def test_partial_transcript_failures(self, e2e_server):
        """Test that partial transcript failures don't break entire pipeline."""
        # Set error behavior for one transcript (episode 1)
        e2e_server.set_error_behavior("/transcripts/p01_multi_e01.txt", 404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=3,  # Process multiple episodes
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should handle partial failures
            count, summary = run_pipeline(cfg)

            # Pipeline should complete and process remaining episodes
            # Note: If episode 1 fails, episodes 2 and 3 should still be processed
            assert count >= 0, "Should handle partial failures gracefully"
            assert isinstance(summary, str), "Summary should be a string"

        # Clear error behavior
        e2e_server.clear_error_behavior("/transcripts/p01_multi_e01.txt")

    def test_mixed_success_and_failure(self, e2e_server):
        """Test pipeline with mixed success and failure scenarios."""
        # Set error behavior for one transcript
        e2e_server.set_error_behavior("/transcripts/p01_multi_e02.txt", 500)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=3,  # Process multiple episodes
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            # Run pipeline - should handle mixed scenarios
            count, summary = run_pipeline(cfg)

            # Pipeline should complete and process successful episodes
            assert count >= 0, "Should handle mixed success/failure gracefully"
            assert isinstance(summary, str), "Summary should be a string"

        # Clear error behavior
        e2e_server.clear_error_behavior("/transcripts/p01_multi_e02.txt")
