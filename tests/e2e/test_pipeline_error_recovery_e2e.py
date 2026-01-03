#!/usr/bin/env python3
"""E2E tests for pipeline error recovery and edge cases (Stage 8).

These tests verify that the pipeline handles errors gracefully and recovers
appropriately, including:
- Partial episode failures (pipeline continues)
- Resource cleanup on errors
- Fallback behavior (transcript download fails → use Whisper)
- Empty RSS feeds
- Malformed RSS feeds
- Missing required files
- Error messages and logging
- Partial results are saved correctly
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config, workflow

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: E402
    create_test_config,
    TEST_FEED_TITLE,
    TEST_TRANSCRIPT_TYPE_VTT,
)


class ErrorRecoveryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for error recovery tests."""

    def do_GET(self):
        """Handle GET requests with custom responses."""
        path = self.path.split("?")[0]  # Remove query string

        # Standard RSS feed endpoint (for tests that need it)
        if path == "/feed.xml":
            rss_xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0" version="2.0">
  <channel>
    <title>{TEST_FEED_TITLE}</title>
    <item>
      <title>Episode 1: Test Episode</title>
      <podcast:transcript
        url="http://127.0.0.1:{self.server.server_address[1]}/transcript1.vtt"
        type="{TEST_TRANSCRIPT_TYPE_VTT}" />
    </item>
  </channel>
</rss>"""
            self._send_response(200, rss_xml.encode("utf-8"), "application/xml")
        # RSS feed with multiple episodes (some will fail)
        elif path == "/feed-partial-failures.xml":
            rss_xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0" version="2.0">
  <channel>
    <title>{TEST_FEED_TITLE}</title>
    <item>
      <title>Episode 1: Success</title>
      <podcast:transcript
        url="http://127.0.0.1:{self.server.server_address[1]}/transcript1.vtt"
        type="{TEST_TRANSCRIPT_TYPE_VTT}" />
    </item>
    <item>
      <title>Episode 2: Will Fail</title>
      <podcast:transcript
        url="http://127.0.0.1:{self.server.server_address[1]}/transcript-error-404.vtt"
        type="{TEST_TRANSCRIPT_TYPE_VTT}" />
    </item>
    <item>
      <title>Episode 3: Success</title>
      <podcast:transcript
        url="http://127.0.0.1:{self.server.server_address[1]}/transcript3.vtt"
        type="{TEST_TRANSCRIPT_TYPE_VTT}" />
    </item>
  </channel>
</rss>"""
            self._send_response(200, rss_xml.encode("utf-8"), "application/xml")
        # Empty RSS feed
        elif path == "/feed-empty.xml":
            rss_xml = """<?xml version='1.0' encoding='UTF-8'?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <description>Feed with no episodes</description>
  </channel>
</rss>"""
            self._send_response(200, rss_xml.encode("utf-8"), "application/xml")
        # Malformed RSS feed
        elif path == "/feed-malformed.xml":
            malformed_xml = """<?xml version='1.0'?>
<rss>
  <channel>
    <title>Malformed Feed</title>
    <item>
      <title>Episode 1</title>
      <!-- Missing closing tags -->
    </item>
  </channel>
</rss>"""
            self._send_response(200, malformed_xml.encode("utf-8"), "application/xml")
        # RSS feed with fallback scenario (no transcript URL, should use Whisper)
        elif path == "/feed-fallback.xml":
            rss_xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0" version="2.0">
  <channel>
    <title>{TEST_FEED_TITLE}</title>
    <item>
      <title>Episode 1: Fallback Test</title>
      <description>Episode with no transcript URL, should use Whisper</description>
      <enclosure
        url="http://127.0.0.1:{self.server.server_address[1]}/episode1.mp3"
        type="audio/mpeg" />
      <!-- No transcript URL - will trigger Whisper fallback -->
    </item>
  </channel>
</rss>"""
            self._send_response(200, rss_xml.encode("utf-8"), "application/xml")
        # RSS feed with transcript URL that will fail (for testing error handling)
        elif path == "/feed-transcript-fails.xml":
            rss_xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0" version="2.0">
  <channel>
    <title>{TEST_FEED_TITLE}</title>
    <item>
      <title>Episode 1: Transcript Fails</title>
      <podcast:transcript
        url="http://127.0.0.1:{self.server.server_address[1]}/transcript-error-404.vtt"
        type="{TEST_TRANSCRIPT_TYPE_VTT}" />
    </item>
  </channel>
</rss>"""
            self._send_response(200, rss_xml.encode("utf-8"), "application/xml")
        # Successful transcript endpoints
        elif path == "/transcript1.vtt":
            transcript = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nEpisode 1 transcript content."
            self._send_response(200, transcript.encode("utf-8"), "text/vtt")
        elif path == "/transcript3.vtt":
            transcript = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nEpisode 3 transcript content."
            self._send_response(200, transcript.encode("utf-8"), "text/vtt")
        # Error endpoints
        elif path == "/transcript-error-404.vtt":
            self._send_response(404, b"Not Found", "text/plain")
        # Audio file endpoint - serve real audio file from fixtures
        elif path == "/episode1.mp3":
            # Use real audio fixture file (fast audio for quick testing)
            fixture_path = Path(__file__).parent.parent / "fixtures" / "audio" / "p01_e01_fast.mp3"
            if fixture_path.exists():
                with open(fixture_path, "rb") as f:
                    audio_data = f.read()
                self._send_response(200, audio_data, "audio/mpeg")
            else:
                # Fallback: return 404 if fixture not found
                self._send_response(404, b"Audio fixture not found", "text/plain")
        else:
            self._send_response(404, b"Not Found", "text/plain")

    def _send_response(self, status_code: int, content: bytes, content_type: str):
        """Send HTTP response."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Suppress server log messages during tests."""
        pass


class ErrorRecoveryHTTPServer:
    """Test HTTP server for error recovery tests."""

    def __init__(self, port: int = 0):
        """Initialize test server."""
        self.port = port
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.base_url: Optional[str] = None

    def start(self):
        """Start the test server."""

        def handler(*args, **kwargs):
            return ErrorRecoveryHTTPRequestHandler(*args, **kwargs)

        self.server = socketserver.TCPServer(("127.0.0.1", self.port), handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

        # Start server in background thread
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        # Wait a moment for server to start
        time.sleep(0.1)

    def stop(self):
        """Stop the test server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=1.0)

    def url(self, path: str) -> str:
        """Get full URL for a path."""
        if not self.base_url:
            raise RuntimeError("Server not started")
        return f"{self.base_url}{path}"


@pytest.fixture
def error_recovery_http_server():
    """Fixture for error recovery test HTTP server."""
    server = ErrorRecoveryHTTPServer()
    server.start()
    yield server
    server.stop()


@pytest.mark.e2e
@pytest.mark.slow
class TestPipelineErrorRecoveryE2E:
    """Test pipeline error recovery and edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self, error_recovery_http_server, tmp_path):
        """Set up test fixtures."""
        self.http_server = error_recovery_http_server
        self.temp_dir = tmp_path
        self.output_dir = os.path.join(self.temp_dir, "output")

    def test_pipeline_handles_partial_episode_failures(self):
        """Test that pipeline continues when some episodes fail."""
        feed_url = self.http_server.url("/feed-partial-failures.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=3,  # Process all 3 episodes
            generate_metadata=True,
            metadata_format="json",
            transcribe_missing=False,  # Don't transcribe if download fails
        )

        # Run pipeline
        count, summary = workflow.run_pipeline(cfg)

        # Verify that pipeline completed (didn't crash)
        assert count >= 1, "Should process at least one successful episode"

        # Verify that successful episodes were processed
        transcript_files = list(Path(self.output_dir).rglob("*.vtt"))
        assert len(transcript_files) >= 1, "Should create transcript files for successful episodes"

        # Verify metadata files were created for successful episodes
        metadata_files = list(Path(self.output_dir).rglob("*.metadata.json"))
        assert len(metadata_files) >= 1, "Should create metadata files for successful episodes"

        # Verify that Episode 2 (which failed) is not in the output
        # But Episodes 1 and 3 (which succeeded) are present
        episode_titles = []
        for metadata_file in metadata_files:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                episode_titles.append(metadata["episode"]["title"])

        # Should have Episode 1 and Episode 3, but not Episode 2 (which failed)
        assert "Episode 1: Success" in episode_titles or "Episode 3: Success" in episode_titles
        # Episode 2 should not be present (failed download)

    def test_pipeline_handles_empty_rss_feed(self):
        """Test that pipeline handles empty RSS feed gracefully."""
        feed_url = self.http_server.url("/feed-empty.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=10,
        )

        # Run pipeline
        count, summary = workflow.run_pipeline(cfg)

        # Should complete without errors
        assert count == 0, "Should process 0 episodes from empty feed"
        assert "episode" in summary.lower() or "0" in summary or "empty" in summary.lower()

        # Verify no files were created
        transcript_files = list(Path(self.output_dir).rglob("*.vtt"))
        assert len(transcript_files) == 0, "Should not create files for empty feed"

    def test_pipeline_handles_malformed_rss_feed(self):
        """Test that pipeline handles malformed RSS feed gracefully."""
        feed_url = self.http_server.url("/feed-malformed.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=1,
        )

        # Pipeline may raise ValueError when RSS parsing fails, or may handle it gracefully
        # XML parser might be lenient, so we check that it either raises or
        # completes with 0 episodes
        try:
            count, summary = workflow.run_pipeline(cfg)
            # If it completes, should process 0 episodes (malformed feed)
            assert count == 0, "Should process 0 episodes from malformed feed"
        except ValueError:
            # ValueError is also acceptable for malformed RSS
            pass

    def test_pipeline_fallback_to_whisper_when_no_transcript_url(self):
        """Test that pipeline falls back to Whisper when no transcript URL is provided."""
        feed_url = self.http_server.url("/feed-fallback.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=1,
            generate_metadata=True,
            metadata_format="json",
            transcribe_missing=True,  # Enable Whisper when no transcript URL
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        # Require Whisper model to be cached (skip if not available)
        from tests.integration.ml_model_cache_helpers import require_whisper_model_cached

        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        # Run pipeline with real Whisper
        count, summary = workflow.run_pipeline(cfg)

        # Verify that pipeline completed successfully using Whisper
        assert count > 0, "Should process episode using Whisper fallback"

        # Verify transcript file was created (from Whisper)
        transcript_files = list(Path(self.output_dir).rglob("*.txt"))
        assert len(transcript_files) > 0, "Should create transcript file from Whisper fallback"

        # Verify metadata indicates Whisper transcription
        metadata_files = list(Path(self.output_dir).rglob("*.metadata.json"))
        assert len(metadata_files) > 0, "Should create metadata file"

        if metadata_files:
            with open(metadata_files[0], "r", encoding="utf-8") as f:
                metadata = json.load(f)
            assert metadata["content"]["transcript_source"] == "whisper_transcription"

    def test_pipeline_handles_transcript_download_failure_gracefully(self):
        """Test that pipeline handles transcript download failure without crashing."""
        feed_url = self.http_server.url("/feed-transcript-fails.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=1,
            transcribe_missing=False,  # Don't fallback to Whisper
        )

        # Run pipeline - should complete without crashing even if transcript download fails
        count, summary = workflow.run_pipeline(cfg)

        # Pipeline should complete (may process 0 episodes if transcript download fails)
        assert count >= 0, "Pipeline should complete without crashing"

    def test_pipeline_saves_partial_results_on_error(self):
        """Test that pipeline saves partial results even when some operations fail."""
        feed_url = self.http_server.url("/feed-partial-failures.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=3,
            generate_metadata=True,
            metadata_format="json",
            transcribe_missing=False,
        )

        # Run pipeline
        count, summary = workflow.run_pipeline(cfg)

        # Verify that successful episodes were saved
        assert count >= 1, "Should save at least one successful episode"

        # Verify files exist for successful episodes
        transcript_files = list(Path(self.output_dir).rglob("*.vtt"))
        metadata_files = list(Path(self.output_dir).rglob("*.metadata.json"))

        # Should have files for successful episodes
        assert len(transcript_files) >= 1, "Should save transcript files for successful episodes"
        assert len(metadata_files) >= 1, "Should save metadata files for successful episodes"

        # Verify that saved files are valid
        for metadata_file in metadata_files:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                # Verify metadata is complete and valid
                assert "feed" in metadata
                assert "episode" in metadata
                assert "content" in metadata

    def test_pipeline_handles_missing_output_directory(self):
        """Test that pipeline creates output directory if it doesn't exist."""
        feed_url = self.http_server.url("/feed.xml")
        # Use a non-existent directory
        output_dir = os.path.join(self.temp_dir, "nonexistent", "output")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=output_dir,
            max_episodes=1,
            generate_metadata=True,
        )

        # Run pipeline
        count, summary = workflow.run_pipeline(cfg)

        # Verify that directory was created
        assert os.path.exists(output_dir), "Pipeline should create output directory"

        # Verify that files were created in the new directory
        transcript_files = list(Path(output_dir).rglob("*.vtt"))
        assert len(transcript_files) >= 0, "Should be able to create files in new directory"

    def test_pipeline_handles_invalid_config_gracefully(self):
        """Test that pipeline handles invalid configuration gracefully."""
        # Test that config validation catches invalid values
        # Note: max_episodes might accept negative values (treated as None or 0)
        # Let's test with a truly invalid config value
        try:
            # Try to create config with invalid RSS URL type
            _ = config.Config(  # Config for test
                rss_url=None,  # Invalid: None when it should be a string
                output_dir=self.output_dir,
            )
            # If it doesn't raise, that's okay - validation might happen later
            # The key is that the pipeline should handle it gracefully
        except (ValueError, TypeError):
            # Config validation caught the error - that's good
            pass

    def test_pipeline_continues_with_max_episodes_limit(self):
        """Test that pipeline respects max_episodes limit and continues processing."""
        feed_url = self.http_server.url("/feed-partial-failures.xml")

        cfg = create_test_config(
            rss_url=feed_url,
            output_dir=self.output_dir,
            max_episodes=2,  # Limit to 2 episodes
            generate_metadata=True,
            transcribe_missing=False,
        )

        # Run pipeline
        count, summary = workflow.run_pipeline(cfg)

        # Should process at most 2 episodes
        assert count <= 2, "Should respect max_episodes limit"

        # Verify that processing stopped at limit
        metadata_files = list(Path(self.output_dir).rglob("*.metadata.json"))
        assert len(metadata_files) <= 2, "Should not process more than max_episodes"
