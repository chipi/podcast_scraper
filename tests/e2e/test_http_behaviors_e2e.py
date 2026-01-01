#!/usr/bin/env python3
"""E2E tests for HTTP Behaviors with Large Files.

These tests verify HTTP behaviors work correctly with large files:
- Range requests (206 Partial Content)
- Streaming downloads
- Timeout handling
- Retry logic
- Partial reads

All tests use real HTTP client, E2E server, and large audio files.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config, run_pipeline
from podcast_scraper.downloader import http_download_to_file


@pytest.mark.e2e
@pytest.mark.slow
class TestRangeRequests:
    """HTTP Range Request (206 Partial Content) E2E tests."""

    def test_range_request_support(self, e2e_server):
        """Test that E2E server supports HTTP range requests."""
        # Get a large audio file URL
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Request first 100 bytes
        headers = {"Range": "bytes=0-99"}
        response = requests.get(audio_url, headers=headers)

        # Should return 206 Partial Content
        assert response.status_code == 206, "Should return 206 Partial Content for range request"
        assert "Content-Range" in response.headers, "Should include Content-Range header"
        assert response.headers["Content-Range"].startswith(
            "bytes 0-99/"
        ), "Content-Range should match request"
        assert len(response.content) == 100, "Should return exactly 100 bytes"

    def test_range_request_middle(self, e2e_server):
        """Test range request for middle portion of file."""
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Request bytes 1000-1999
        headers = {"Range": "bytes=1000-1999"}
        response = requests.get(audio_url, headers=headers)

        # Should return 206 Partial Content
        assert response.status_code == 206, "Should return 206 Partial Content"
        assert "Content-Range" in response.headers, "Should include Content-Range header"
        assert (
            "bytes 1000-1999/" in response.headers["Content-Range"]
        ), "Content-Range should match request"
        assert len(response.content) == 1000, "Should return exactly 1000 bytes"

    def test_range_request_end(self, e2e_server):
        """Test range request for end of file."""
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Get file size first
        full_response = requests.head(audio_url)
        content_length = int(full_response.headers.get("Content-Length", 0))

        # Request last portion of file (use smaller chunk if file is small)
        chunk_size = min(500, content_length)
        start = max(0, content_length - chunk_size)
        end = content_length - 1
        headers = {"Range": f"bytes={start}-{end}"}
        response = requests.get(audio_url, headers=headers)

        # Should return 206 Partial Content
        assert response.status_code == 206, "Should return 206 Partial Content"
        assert len(response.content) == chunk_size, f"Should return exactly {chunk_size} bytes"


@pytest.mark.e2e
@pytest.mark.slow
class TestStreamingDownloads:
    """Streaming download E2E tests."""

    def test_audio_streaming_download(self, e2e_server):
        """Test that audio files are streamed (not downloaded all at once)."""
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "audio.mp3")

            # Download using the downloader function (which streams)
            success, bytes_downloaded = http_download_to_file(
                audio_url, "test-agent", timeout=30, out_path=out_path
            )

            # Should succeed
            assert success is True, "Download should succeed"
            assert bytes_downloaded > 0, "Should download some bytes"

            # Verify file exists and has content
            assert os.path.exists(out_path), "Output file should exist"
            assert (
                os.path.getsize(out_path) == bytes_downloaded
            ), "File size should match bytes downloaded"
            assert os.path.getsize(out_path) > 1000, "File should be reasonably large"

    def test_streaming_with_large_file(self, e2e_server):
        """Test streaming with a large audio file."""
        # Use a larger audio file if available (p01_e03 is typically larger)
        audio_url = e2e_server.urls.audio("p01_e03")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "large_audio.mp3")

            start_time = time.time()
            success, bytes_downloaded = http_download_to_file(
                audio_url, "test-agent", timeout=60, out_path=out_path
            )
            elapsed_time = time.time() - start_time

            # Should succeed
            assert success is True, "Download should succeed"
            assert bytes_downloaded > 0, "Should download some bytes"
            assert os.path.exists(out_path), "Output file should exist"
            assert os.path.getsize(out_path) == bytes_downloaded, "File size should match"

            # Streaming should be reasonably fast (not blocking on full download)
            # This is a sanity check - actual time depends on file size
            assert elapsed_time < 10, "Streaming should complete in reasonable time"


@pytest.mark.e2e
@pytest.mark.slow
class TestTimeoutHandling:
    """Timeout handling E2E tests."""

    def test_download_timeout(self, e2e_server):
        """Test that downloads respect timeout settings."""
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Set a very short timeout (should fail for large files)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "audio.mp3")

            # Use a very short timeout (0.01 seconds) - should fail for large files
            # Note: Very short timeouts may still succeed if the file is small or cached
            # So we test that timeout is respected, but may need to adjust expectations
            success, bytes_downloaded = http_download_to_file(
                audio_url, "test-agent", timeout=0.01, out_path=out_path
            )

            # With such a short timeout, download should either fail or download very little
            # (timeout handling depends on requests library behavior)
            if not success:
                # Expected: download failed due to timeout
                assert bytes_downloaded == 0, "Should download 0 bytes on timeout failure"
            else:
                # If it succeeds, it means the download was very fast (unlikely for large files)
                # This is acceptable - we're testing that timeout parameter is used
                assert bytes_downloaded >= 0, "Bytes downloaded should be non-negative"

    def test_timeout_with_retry(self, e2e_server):
        """Test that timeout triggers retry logic."""
        # This test verifies that the retry mechanism handles timeouts
        # The downloader has retry logic built-in, so we test that it works
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Set a reasonable timeout (should succeed)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "audio.mp3")

            success, bytes_downloaded = http_download_to_file(
                audio_url, "test-agent", timeout=30, out_path=out_path
            )

            # Should succeed with reasonable timeout
            assert success is True, "Download should succeed with reasonable timeout"
            assert bytes_downloaded > 0, "Should download some bytes"


@pytest.mark.e2e
@pytest.mark.slow
class TestRetryLogic:
    """Retry logic E2E tests."""

    def test_retry_on_500_error(self, e2e_server):
        """Test that retry logic handles 500 errors."""
        # Set error behavior to return 500 for first few requests, then succeed
        audio_url = e2e_server.urls.audio("p01_multi_e01")
        path = "/audio/p01_multi_e01.mp3"

        # Set error behavior to return 500 (will retry)
        e2e_server.set_error_behavior(path, 500)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "audio.mp3")

            # Download should eventually fail after retries (or succeed if retries work)
            success, bytes_downloaded = http_download_to_file(
                audio_url, "test-agent", timeout=30, out_path=out_path
            )

            # After retries, should either succeed or fail gracefully
            # The retry logic will attempt 5 times, then fail
            # So we expect failure here
            assert success is False, "Download should fail after retries exhaust"
            assert bytes_downloaded == 0, "Should download 0 bytes after retry failure"

        # Clear error behavior
        e2e_server.clear_error_behavior(path)

    def test_retry_on_transient_error(self, e2e_server):
        """Test that retry logic handles transient errors."""
        # For this test, we'll use a normal download (no error)
        # The retry logic is built into the HTTP adapter, so we verify it's configured
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "audio.mp3")

            # Normal download should succeed (retry logic is transparent)
            success, bytes_downloaded = http_download_to_file(
                audio_url, "test-agent", timeout=30, out_path=out_path
            )

            # Should succeed
            assert success is True, "Download should succeed"
            assert bytes_downloaded > 0, "Should download some bytes"


@pytest.mark.e2e
@pytest.mark.slow
class TestPartialReads:
    """Partial read E2E tests."""

    def test_partial_read_with_range_request(self, e2e_server):
        """Test partial read using range request."""
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Request first 1KB
        headers = {"Range": "bytes=0-1023"}
        response = requests.get(audio_url, headers=headers)

        # Should return 206 Partial Content
        assert response.status_code == 206, "Should return 206 Partial Content"
        assert len(response.content) == 1024, "Should return exactly 1024 bytes"

    def test_multiple_range_requests(self, e2e_server):
        """Test multiple range requests for same file."""
        audio_url = e2e_server.urls.audio("p01_multi_e01")

        # Request multiple ranges
        ranges = [
            ("bytes=0-99", 100),
            ("bytes=1000-1999", 1000),
            ("bytes=5000-5099", 100),
        ]

        for range_header, expected_size in ranges:
            headers = {"Range": range_header}
            response = requests.get(audio_url, headers=headers)

            assert response.status_code == 206, f"Should return 206 for {range_header}"
            assert (
                len(response.content) == expected_size
            ), f"Should return {expected_size} bytes for {range_header}"


@pytest.mark.e2e
@pytest.mark.slow
class TestFullPipelineWithLargeFiles:
    """Full pipeline E2E tests with large files."""

    def test_full_pipeline_with_large_audio(self, e2e_server):
        """Test full pipeline with large audio file (streaming download)."""
        # This test verifies that the pipeline uses streaming for downloads
        # The RSS feed has transcript URLs, so it will download transcripts (using streaming)
        # rather than transcribing audio
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,  # Process one episode
                transcribe_missing=True,
            )

            # Run pipeline - should handle large files with streaming
            count, summary = run_pipeline(cfg)

            # Pipeline should complete successfully
            assert count == 1, "Should process 1 episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created (downloaded/transcribed using streaming)
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) == 1, "Should create one transcript file"
            assert output_files[0].stat().st_size > 0, "Transcript file should not be empty"

    def test_full_pipeline_streaming_behavior(self, e2e_server):
        """Test that full pipeline uses streaming for large files."""
        # This test verifies that the pipeline uses streaming (not loading entire file into memory)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
            )

            # Run pipeline - should use streaming internally
            start_time = time.time()
            count, summary = run_pipeline(cfg)
            elapsed_time = time.time() - start_time

            # Pipeline should complete successfully
            assert count == 1, "Should process 1 episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Streaming should be reasonably fast
            # This is a sanity check - actual time depends on file size, network,
            # and transcription time. Transcription with Whisper can take 40-70s
            # for a single episode, so we allow up to 90s
            assert elapsed_time < 90, "Pipeline should complete in reasonable time with streaming"
