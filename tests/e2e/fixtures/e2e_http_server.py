"""E2E HTTP server for serving test fixtures.

This module provides a local HTTP server that serves RSS feeds, audio files,
and transcripts from the test fixtures directory. The server is designed for
E2E tests to use real HTTP clients with local fixture data.

Key Features:
- Serves RSS feeds, audio files, and transcripts
- Supports HTTP range requests (206 Partial Content) for streaming
- Path traversal protection
- URL mapping for flat fixture structure
- Configurable error scenarios
- Mock OpenAI API endpoints for E2E testing
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time
from email import message_from_bytes
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


class E2EServerURLs:
    """URL helper class for E2E server."""

    def __init__(self, base_url: str):
        """Initialize URL helper.

        Args:
            base_url: Base URL of the E2E server (e.g., "http://127.0.0.1:8000")
        """
        self.base_url = base_url.rstrip("/")

    def feed(self, podcast_name: str) -> str:
        """Get RSS feed URL for a podcast.

        Args:
            podcast_name: Podcast name (e.g., "podcast1")

        Returns:
            Full RSS feed URL (e.g., "http://127.0.0.1:8000/feeds/podcast1/feed.xml")
        """
        return f"{self.base_url}/feeds/{podcast_name}/feed.xml"

    def audio(self, episode_id: str) -> str:
        """Get audio file URL for an episode.

        Args:
            episode_id: Episode ID (e.g., "p01_e01")

        Returns:
            Full audio URL (e.g., "http://127.0.0.1:8000/audio/p01_e01.mp3")
        """
        return f"{self.base_url}/audio/{episode_id}.mp3"

    def transcript(self, episode_id: str) -> str:
        """Get transcript file URL for an episode.

        Args:
            episode_id: Episode ID (e.g., "p01_e01")

        Returns:
            Full transcript URL (e.g., "http://127.0.0.1:8000/transcripts/p01_e01.txt")
        """
        return f"{self.base_url}/transcripts/{episode_id}.txt"

    def base(self) -> str:
        """Get base URL of the server.

        Returns:
            Base URL (e.g., "http://127.0.0.1:8000")
        """
        return self.base_url

    def openai_api_base(self) -> str:
        """Get OpenAI API base URL (points to E2E server).

        Returns:
            OpenAI API base URL (e.g., "http://127.0.0.1:8000/v1")
        """
        return f"{self.base_url}/v1"


class E2EHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for E2E server.

    This handler serves files from the test fixtures directory with URL mapping
    to support the flat fixture structure. Supports error scenarios for testing.
    """

    # Mapping: podcast name -> RSS filename
    PODCAST_RSS_MAP = {
        "podcast1": "p01_mtb.xml",
        "podcast2": "p02_software.xml",
        "podcast3": "p03_scuba.xml",
        "podcast4": "p04_photo.xml",
        "podcast5": "p05_investing.xml",
        "edgecases": "p06_edge_cases.xml",
        # Smoke test feed with 5 short episodes (10-15 seconds each)
        "podcast1_smoke": "p01_smoke.xml",
    }

    # Fast mode RSS mapping (uses shorter episodes for faster tests)
    PODCAST_RSS_MAP_FAST = {
        # Fast version with 1-minute episode (Path 2: Transcription)
        "podcast1": "p01_fast.xml",
        # Fast version with transcript URL (Path 1: Download)
        "podcast1_with_transcript": "p01_fast_with_transcript.xml",
        # Smoke test feed (also available in fast mode)
        "podcast1_smoke": "p01_smoke.xml",
    }

    # Allowed podcasts for fast mode (shared across all handler instances)
    # If None, all podcasts are allowed. If set, only podcasts in this set are served.
    _allowed_podcasts: Optional[set[str]] = None
    _allowed_podcasts_lock = threading.Lock()

    # Error behavior registry (shared across all handler instances)
    # Format: {url_path: {"status": 404|500, "delay": seconds}}
    _error_behaviors: Dict[str, Dict[str, Any]] = {}
    _error_behaviors_lock = threading.Lock()

    @classmethod
    def set_error_behavior(cls, url_path: str, status: int, delay: float = 0.0):
        """Set error behavior for a specific URL path.

        Args:
            url_path: URL path to apply error to (e.g., "/feeds/podcast1/feed.xml")
            status: HTTP status code to return (e.g., 404, 500)
            delay: Optional delay in seconds before responding
        """
        with cls._error_behaviors_lock:
            cls._error_behaviors[url_path] = {"status": status, "delay": delay}

    @classmethod
    def clear_error_behavior(cls, url_path: str):
        """Clear error behavior for a specific URL path.

        Args:
            url_path: URL path to clear error behavior for
        """
        with cls._error_behaviors_lock:
            cls._error_behaviors.pop(url_path, None)

    @classmethod
    def clear_all_error_behaviors(cls):
        """Clear all error behaviors."""
        with cls._error_behaviors_lock:
            cls._error_behaviors.clear()

    @classmethod
    def set_allowed_podcasts(cls, podcasts: Optional[set[str]]):
        """Set allowed podcasts for serving RSS feeds.

        Args:
            podcasts: Set of podcast names to allow, or None to allow all podcasts.
                     In fast mode, this should be {"podcast1"} to limit to one feed.
                     In full mode, this should be None to allow all feeds.
        """
        with cls._allowed_podcasts_lock:
            cls._allowed_podcasts = podcasts

    @classmethod
    def get_allowed_podcasts(cls) -> Optional[set[str]]:
        """Get currently allowed podcasts.

        Returns:
            Set of allowed podcast names, or None if all podcasts are allowed.
        """
        with cls._allowed_podcasts_lock:
            return cls._allowed_podcasts

    @classmethod
    def get_fixture_root(cls) -> Path:
        """Get the root directory for test fixtures.

        Returns:
            Path to tests/fixtures directory
        """
        # This file is in tests/e2e/fixtures/
        # Fixtures are in tests/fixtures/
        # Path structure: tests/e2e/fixtures/e2e_http_server.py
        #                 -> tests/e2e/fixtures (parent)
        #                 -> tests/e2e (parent.parent)
        #                 -> tests (parent.parent.parent)
        #                 -> tests/fixtures (parent.parent.parent / "fixtures")
        current_file = Path(__file__).resolve()
        # Go up: e2e_http_server.py -> fixtures -> e2e -> tests
        tests_dir = current_file.parent.parent.parent
        fixture_root = tests_dir / "fixtures"
        return fixture_root

    def do_GET(self):
        """Handle GET requests with URL mapping."""
        path = self.path.split("?")[0]  # Remove query string

        # Check for error behavior first
        with self._error_behaviors_lock:
            error_behavior = self._error_behaviors.get(path)
        if error_behavior:
            # Apply delay if specified
            if error_behavior.get("delay", 0) > 0:
                time.sleep(error_behavior["delay"])
            # Return error status
            self.send_error(error_behavior["status"], f"Simulated {error_behavior['status']} error")
            return

        # Route 1: RSS feeds
        # /feeds/podcast1/feed.xml -> rss/p01_mtb.xml
        if path.startswith("/feeds/") and path.endswith("/feed.xml"):
            podcast_name = path.split("/")[2]  # Extract "podcast1"

            # Check if podcast is allowed (fast mode limitation)
            with self._allowed_podcasts_lock:
                allowed = self._allowed_podcasts
            if allowed is not None and podcast_name not in allowed:
                self.send_error(404, f"RSS feed not available in fast mode: {podcast_name}")
                return

            # Use fast RSS feed if in fast mode (allowed_podcasts is set)
            # Fast mode means we're limiting to specific podcasts, so use fast versions
            # Data quality mode (allowed_podcasts is None) uses original mock data
            is_fast_mode = allowed is not None

            # Only use fast fixtures in fast mode (when allowed_podcasts is set)
            # Data quality tests (allowed_podcasts is None) use original mock data
            if is_fast_mode and podcast_name in self.PODCAST_RSS_MAP_FAST:
                rss_file = self.PODCAST_RSS_MAP_FAST.get(podcast_name)
            else:
                # Use original mock data (for data quality tests or slow tests)
                rss_file = self.PODCAST_RSS_MAP.get(podcast_name)

            if rss_file:
                file_path = self.get_fixture_root() / "rss" / rss_file
                if file_path.exists():
                    self._serve_file(file_path, content_type="application/xml")
                    return
            self.send_error(404, "RSS feed not found")
            return

        # Route 2: Direct flat URLs for audio
        # /audio/p01_e01.mp3 -> audio/p01_e01.mp3
        if path.startswith("/audio/"):
            filename = path.split("/")[-1]  # Extract "p01_e01.mp3"
            file_path = self._get_safe_fixture_path("audio", filename)
            if file_path is None:
                self.send_error(403, "Path traversal not allowed")
                return
            self._serve_file(file_path, content_type="audio/mpeg", support_range=True)
            return

        # Route 3: Direct flat URLs for transcripts
        # /transcripts/p01_e01.txt -> transcripts/p01_e01.txt
        if path.startswith("/transcripts/"):
            filename = path.split("/")[-1]  # Extract "p01_e01.txt"
            file_path = self._get_safe_fixture_path("transcripts", filename)
            if file_path is None:
                self.send_error(403, "Path traversal not allowed")
                return
            self._serve_file(file_path, content_type="text/plain")
            return

        # 404 for all other paths
        self.send_error(404, "File not found")

    def do_POST(self):
        """Handle POST requests for OpenAI API endpoints."""
        path = self.path.split("?")[0]  # Remove query string

        # Check for error behavior first
        with self._error_behaviors_lock:
            error_behavior = self._error_behaviors.get(path)
        if error_behavior:
            # Apply delay if specified
            if error_behavior.get("delay", 0) > 0:
                time.sleep(error_behavior["delay"])
            # Return error status
            self.send_error(error_behavior["status"], f"Simulated {error_behavior['status']} error")
            return

        # Route: OpenAI API endpoints
        # /v1/chat/completions -> Mock chat completions (summarization, speaker detection)
        if path == "/v1/chat/completions":
            self._handle_chat_completions()
            return

        # Route: OpenAI Whisper API endpoint
        # /v1/audio/transcriptions -> Mock audio transcriptions
        if path == "/v1/audio/transcriptions":
            self._handle_audio_transcriptions()
            return

        # 404 for all other paths
        self.send_error(404, "OpenAI endpoint not found")

    def _handle_chat_completions(self):
        """Handle OpenAI chat completions API requests."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Request body required")
                return

            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode("utf-8"))

            # Extract request details
            messages = request_data.get("messages", [])
            user_message = next((m for m in messages if m.get("role") == "user"), {})
            user_content = user_message.get("content", "")
            response_format = request_data.get("response_format", {})

            # Determine response type based on response_format
            # If response_format is {"type": "json_object"}, it's speaker detection
            # Otherwise, it's summarization
            if response_format.get("type") == "json_object":
                # Speaker detection response
                response_data = {
                    "id": "chatcmpl-test-speaker",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "speakers": ["Host", "Guest"],
                                        "hosts": ["Host"],
                                        "guests": ["Guest"],
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }
            else:
                # Summarization response
                # Generate a simple summary based on text length
                summary_length = min(200, len(user_content) // 10)
                summary = (
                    f"This is a test summary of the transcript. {user_content[:summary_length]}..."
                )

                response_data = {
                    "id": "chatcmpl-test-summary",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": summary},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }

            # Send response
            response_json = json.dumps(response_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode("utf-8"))

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.send_error(500, f"Error handling chat completions: {e}")

    def _handle_audio_transcriptions(self):
        """Handle OpenAI audio transcriptions API requests."""
        try:
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_error(400, "Content-Type must be multipart/form-data")
                return

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Request body required")
                return

            body = self.rfile.read(content_length)

            # Parse multipart form data using email module
            filename = "unknown_audio.mp3"
            try:
                # Create a message from the multipart data
                msg = message_from_bytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
                # Extract filename from Content-Disposition header
                for part in msg.walk():
                    if part.get_content_disposition() == "form-data":
                        content_disposition = part.get("Content-Disposition", "")
                        if "filename=" in content_disposition:
                            # Extract filename from Content-Disposition header
                            # Format: filename="audio.mp3" or filename=audio.mp3
                            parts = content_disposition.split("filename=")
                            if len(parts) > 1:
                                filename_part = parts[1].strip().strip('"').strip("'")
                                if filename_part:
                                    filename = filename_part
                                    break
            except Exception:
                # If parsing fails, use default filename
                # This is acceptable for E2E tests
                pass

            # Generate a realistic transcription response
            transcript = (
                f"This is a test transcription of {filename}. "
                "The audio contains spoken content that has been transcribed."
            )

            # Send response (text format)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(transcript)))
            self.end_headers()
            self.wfile.write(transcript.encode("utf-8"))

        except Exception as e:
            self.send_error(500, f"Error handling audio transcriptions: {e}")

    def _get_safe_fixture_path(self, subdir: str, filename: str) -> Optional[Path]:
        """Safely construct and validate a fixture file path.

        This method prevents path traversal attacks by:
        1. Rejecting filenames with path separators or ".." segments
        2. Building the path relative to a fixed root directory
        3. Normalizing the path with resolve()
        4. Verifying the normalized path is within the intended root and is a file

        Args:
            subdir: Subdirectory name ("audio" or "transcripts")
            filename: Filename from request (untrusted input)

        Returns:
            Safe Path if validation passes, None if input is invalid
        """
        # Reject filenames containing path separators or ".." segments
        if "/" in filename or "\\" in filename or ".." in filename:
            return None

        # Build base directory
        base_dir = self.get_fixture_root() / subdir

        # Build candidate path and resolve to normalize
        try:
            candidate = (base_dir / filename).resolve()
        except (OSError, RuntimeError):
            # Path resolution failed (e.g., broken symlink, invalid path)
            return None

        # Verify candidate is a file and is within base_dir
        # Python 3.9+ has is_relative_to, we're on 3.10+ so it's available
        if not candidate.is_file() or not candidate.is_relative_to(base_dir):
            return None

        return candidate

    def _serve_file(self, file_path: Path, content_type: str, support_range: bool = False):
        """Serve a file with proper headers and range request support.

        Args:
            file_path: Path to file to serve
            content_type: Content-Type header value
            support_range: Whether to support HTTP range requests (206 Partial Content)
        """
        try:
            file_size = file_path.stat().st_size

            # Check for Range request
            range_header = self.headers.get("Range")
            if support_range and range_header:
                # Parse Range header (e.g., "bytes=0-1023")
                try:
                    range_spec = range_header.replace("bytes=", "")
                    parts = range_spec.split("-")
                    if len(parts) != 2:
                        raise ValueError("Invalid range format")
                    start_str, end_str = parts
                    start = int(start_str) if start_str else 0
                    end = int(end_str) if end_str else file_size - 1
                    end = min(end, file_size - 1)

                    # Send 206 Partial Content
                    self.send_response(206)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()

                    # Send partial content
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(end - start + 1))
                    return
                except (ValueError, IndexError):
                    # Invalid range header, fall through to full file
                    pass

            # Send full file (200 OK)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            if support_range:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                self.wfile.write(f.read())

        except Exception as e:
            self.send_error(500, f"Error serving file: {e}")

    def log_message(self, format, *args):
        """Suppress server log messages during tests."""
        pass


class E2EHTTPServer:
    """E2E HTTP server for serving test fixtures."""

    def __init__(self, port: int = 0):
        """Initialize E2E server.

        Args:
            port: Port number (0 = auto-assign)
        """
        self.port = port
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.base_url: Optional[str] = None
        self.urls: Optional[E2EServerURLs] = None

    def start(self):
        """Start the E2E server."""
        handler = lambda *args, **kwargs: E2EHTTPRequestHandler(*args, **kwargs)  # noqa: E731
        self.server = socketserver.TCPServer(("127.0.0.1", self.port), handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.urls = E2EServerURLs(self.base_url)

        # Start server in background thread
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        # Wait a moment for server to start
        time.sleep(0.1)

    def stop(self):
        """Stop the E2E server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=1.0)

    def reset(self):
        """Reset server state (for test isolation).

        Clears all error behaviors and resets allowed podcasts to None (all allowed).
        """
        E2EHTTPRequestHandler.clear_all_error_behaviors()
        E2EHTTPRequestHandler.set_allowed_podcasts(None)

    def set_allowed_podcasts(self, podcasts: Optional[set[str]]):
        """Set allowed podcasts for serving RSS feeds.

        Args:
            podcasts: Set of podcast names to allow, or None to allow all podcasts.
        """
        E2EHTTPRequestHandler.set_allowed_podcasts(podcasts)

    def set_error_behavior(self, url_path: str, status: int, delay: float = 0.0):
        """Set error behavior for a specific URL path.

        Args:
            url_path: URL path to apply error to (e.g., "/feeds/podcast1/feed.xml")
            status: HTTP status code to return (e.g., 404, 500)
            delay: Optional delay in seconds before responding
        """
        E2EHTTPRequestHandler.set_error_behavior(url_path, status, delay)

    def clear_error_behavior(self, url_path: str):
        """Clear error behavior for a specific URL path.

        Args:
            url_path: URL path to clear error behavior for
        """
        E2EHTTPRequestHandler.clear_error_behavior(url_path)


@pytest.fixture(scope="session")
def e2e_server():
    """E2E HTTP server fixture (session-scoped).

    This fixture provides a local HTTP server that serves test fixtures.
    The server is started once per test session and stopped after all tests.

    Usage:
        def test_something(e2e_server):
            rss_url = e2e_server.urls.feed("podcast1")
            audio_url = e2e_server.urls.audio("p01_e01")
            # Use URLs in tests...
    """
    server = E2EHTTPServer()
    server.start()
    yield server
    server.stop()
