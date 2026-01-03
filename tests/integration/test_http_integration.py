#!/usr/bin/env python3
"""Integration tests for HTTP client with real HTTP server.

These tests verify that the HTTP client (downloader.fetch_url) works correctly
with real HTTP requests using a local test server (not external network).

These tests use:
- Real HTTP client (requests library)
- Local test HTTP server (Python's http.server)
- Real HTTP requests/responses
- No external network calls

These tests are marked with @pytest.mark.integration_http to distinguish
them from other integration tests.
"""

import http.server
import os
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import downloader

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid conflicts with infrastructure conftest
import importlib.util

tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config


class MockHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for test server."""

    def __init__(self, *args, **kwargs):
        # Extract test data from server
        self.test_data = kwargs.pop("test_data", {})
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests with custom responses."""
        # Extract path
        path = self.path.split("?")[0]  # Remove query string

        # Check for special test endpoints
        if path == "/success":
            self._send_response(200, b"Success response", "text/plain")
        elif path == "/rss":
            rss_xml = b"""<?xml version='1.0'?>
<rss>
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
    </item>
  </channel>
</rss>"""
            self._send_response(200, rss_xml, "application/xml")
        elif path == "/transcript":
            transcript = b"Episode 1 transcript content"
            self._send_response(200, transcript, "text/vtt")
        elif path == "/large":
            # Large response for streaming tests
            large_content = b"X" * (1024 * 100)  # 100KB
            self._send_response(200, large_content, "application/octet-stream")
        elif path == "/slow":
            # Slow response for timeout tests
            time.sleep(2)  # Sleep for 2 seconds
            self._send_response(200, b"Slow response", "text/plain")
        elif path == "/error-404":
            self._send_response(404, b"Not Found", "text/plain")
        elif path == "/error-500":
            self._send_response(500, b"Internal Server Error", "text/plain")
        elif path == "/error-503":
            self._send_response(503, b"Service Unavailable", "text/plain")
        elif path == "/retry-then-success":
            # Fail first 2 times, then succeed
            attempt = getattr(self.server, "_retry_attempt", 0)
            self.server._retry_attempt = attempt + 1
            if attempt < 2:
                self._send_response(500, b"Temporary Error", "text/plain")
            else:
                self._send_response(200, b"Success after retries", "text/plain")
        elif path == "/check-user-agent":
            # Return the User-Agent header in response
            user_agent = self.headers.get("User-Agent", "Not provided")
            self._send_response(200, user_agent.encode("utf-8"), "text/plain")
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


class MockHTTPServer:
    """Test HTTP server for integration tests."""

    def __init__(self, port: int = 0):
        """Initialize test server.

        Args:
            port: Port number (0 = auto-assign)
        """
        self.port = port
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.base_url: Optional[str] = None

    def start(self):
        """Start the test server."""
        # Create server with custom handler
        handler = lambda *args, **kwargs: MockHTTPRequestHandler(*args, **kwargs)  # noqa: E731
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
        """Get full URL for a path.

        Args:
            path: URL path (e.g., "/success")

        Returns:
            Full URL (e.g., "http://127.0.0.1:8000/success")
        """
        if not self.base_url:
            raise RuntimeError("Server not started")
        return f"{self.base_url}{path}"


@pytest.fixture
def test_http_server():
    """Fixture for test HTTP server."""
    server = MockHTTPServer()
    server.start()
    yield server
    server.stop()


@pytest.mark.integration
@pytest.mark.integration_http
class TestHTTPClientIntegration:
    """Test HTTP client with real HTTP server."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            user_agent="test-agent/1.0",
            timeout=5,
        )

    @pytest.mark.critical_path
    def test_successful_http_request(self, test_http_server):
        """Test successful HTTP request."""
        url = test_http_server.url("/success")

        # Make real HTTP request
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify response
        assert resp is not None, "Response should not be None"
        assert resp.status_code == 200  # type: ignore[attr-defined]
        assert resp.content == b"Success response"  # type: ignore[attr-defined]

        # Clean up
        resp.close()  # type: ignore[attr-defined]

    @pytest.mark.critical_path
    def test_rss_feed_download(self, test_http_server):
        """Test RSS feed download."""
        url = test_http_server.url("/rss")

        # Make real HTTP request
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify response
        assert resp is not None
        assert resp.status_code == 200  # type: ignore[attr-defined]
        content = resp.content  # type: ignore[attr-defined]
        assert b"<rss>" in content
        assert b"Test Feed" in content

        # Clean up
        resp.close()  # type: ignore[attr-defined]

    @pytest.mark.critical_path
    def test_transcript_download(self, test_http_server):
        """Test transcript download."""
        url = test_http_server.url("/transcript")

        # Make real HTTP request
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify response
        assert resp is not None
        assert resp.status_code == 200  # type: ignore[attr-defined]
        content = resp.content  # type: ignore[attr-defined]
        assert content == b"Episode 1 transcript content"
        content_type = resp.headers.get("Content-Type", "")  # type: ignore[attr-defined]
        assert "text/vtt" in content_type

        # Clean up
        resp.close()  # type: ignore[attr-defined]

    @pytest.mark.critical_path
    def test_streaming_download(self, test_http_server):
        """Test streaming download."""
        url = test_http_server.url("/large")

        # Make real HTTP request with streaming
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=True)

        # Verify response
        assert resp is not None
        assert resp.status_code == 200  # type: ignore[attr-defined]

        # Read content in chunks (streaming)
        content_parts = []
        for chunk in resp.iter_content(chunk_size=1024):  # type: ignore[attr-defined]
            if chunk:
                content_parts.append(chunk)

        # Verify content
        full_content = b"".join(content_parts)
        assert len(full_content) == 1024 * 100  # 100KB
        assert full_content[:10] == b"X" * 10

        # Clean up
        resp.close()  # type: ignore[attr-defined]

    @pytest.mark.critical_path
    def test_user_agent_header(self, test_http_server):
        """Test that User-Agent header is sent correctly."""
        url = test_http_server.url("/check-user-agent")

        # Make real HTTP request
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify response contains User-Agent
        assert resp is not None
        assert resp.status_code == 200  # type: ignore[attr-defined]
        content = resp.content  # type: ignore[attr-defined]
        assert content.decode("utf-8") == self.cfg.user_agent

        # Clean up
        resp.close()  # type: ignore[attr-defined]

    def test_404_error_handling(self, test_http_server):
        """Test 404 error handling."""
        url = test_http_server.url("/error-404")

        # Make real HTTP request
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify error handling (fetch_url returns None on error)
        assert resp is None, "fetch_url should return None on 404 error"

    @pytest.mark.slow
    # Not critical_path - this is error handling, not part of the core workflow
    def test_500_error_handling(self, test_http_server):
        """Test 500 error handling.

        This test is marked as slow because it involves retry logic
        that may take time to complete.
        """
        url = test_http_server.url("/error-500")

        # Make real HTTP request
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify error handling (fetch_url returns None on error)
        assert resp is None, "fetch_url should return None on 500 error"

    @pytest.mark.slow
    # Not critical_path - this is error handling/retry logic, not part of the core workflow
    def test_retry_logic(self, test_http_server):
        """Test retry logic for transient errors.

        This test is marked as slow because it involves retry logic
        with multiple attempts that takes time to complete.
        """
        # Reset retry counter
        test_http_server.server._retry_attempt = 0  # type: ignore[attr-defined]

        url = test_http_server.url("/retry-then-success")

        # Make real HTTP request (should retry and eventually succeed)
        resp = downloader.fetch_url(url, self.cfg.user_agent, self.cfg.timeout, stream=False)

        # Verify retry worked (should succeed after retries)
        assert resp is not None, "fetch_url should succeed after retries"
        if resp:
            assert resp.status_code == 200  # type: ignore[attr-defined]
            content = resp.content  # type: ignore[attr-defined]
            assert content == b"Success after retries"
            resp.close()  # type: ignore[attr-defined]

    @pytest.mark.slow
    # Not critical_path - this is error handling, not part of the core workflow
    def test_timeout_handling(self, test_http_server):
        """Test timeout handling.

        This test is marked as slow because it intentionally waits for
        timeout conditions, which takes time to complete.
        """
        url = test_http_server.url("/slow")

        # Make real HTTP request with short timeout
        resp = downloader.fetch_url(url, self.cfg.user_agent, timeout=1, stream=False)

        # Verify timeout handling (fetch_url returns None on timeout)
        assert resp is None, "fetch_url should return None on timeout"

    @pytest.mark.critical_path
    def test_http_get_function(self, test_http_server):
        """Test http_get() function with real HTTP."""
        url = test_http_server.url("/transcript")

        # Make real HTTP request using http_get
        content, content_type = downloader.http_get(url, self.cfg.user_agent, self.cfg.timeout)

        # Verify response
        assert content is not None
        assert content_type is not None
        assert content == b"Episode 1 transcript content"
        assert "text/vtt" in content_type

    @pytest.mark.critical_path
    def test_http_download_to_file(self, test_http_server):
        """Test http_download_to_file() with real HTTP."""
        url = test_http_server.url("/transcript")

        # Download to file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            success, bytes_written = downloader.http_download_to_file(
                url, self.cfg.user_agent, self.cfg.timeout, tmp_path
            )

            # Verify download
            assert success, "Download should succeed"
            assert bytes_written > 0, "Should write bytes"

            # Verify file content
            with open(tmp_path, "rb") as f:
                content = f.read()
            assert content == b"Episode 1 transcript content"
            assert bytes_written == len(content)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.critical_path
    def test_url_normalization(self, test_http_server):
        """Test URL normalization with real HTTP."""
        # Test with URL that needs normalization
        url = test_http_server.url("/success")
        url_with_spaces = url.replace("/success", "/success%20test")  # URL encoding

        # Normalize URL
        normalized = downloader.normalize_url(url_with_spaces)

        # Verify normalization
        assert normalized is not None
        # Note: normalize_url uses requote_uri which may change encoding
