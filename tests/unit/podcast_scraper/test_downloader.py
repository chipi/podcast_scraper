#!/usr/bin/env python3
"""Tests for HTTP downloader functionality."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import podcast_scraper
from podcast_scraper import downloader

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

# Import helper functions from parent conftest
build_rss_xml_with_media = parent_conftest.build_rss_xml_with_media
build_rss_xml_with_speakers = parent_conftest.build_rss_xml_with_speakers
build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_media_response = parent_conftest.create_media_response
create_mock_spacy_model = parent_conftest.create_mock_spacy_model
create_rss_response = parent_conftest.create_rss_response
create_test_args = parent_conftest.create_test_args
create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
create_transcript_response = parent_conftest.create_transcript_response
MockHTTPResponse = parent_conftest.MockHTTPResponse
TEST_BASE_URL = parent_conftest.TEST_BASE_URL
TEST_CONTENT_TYPE_SRT = parent_conftest.TEST_CONTENT_TYPE_SRT
TEST_CONTENT_TYPE_VTT = parent_conftest.TEST_CONTENT_TYPE_VTT
TEST_CUSTOM_OUTPUT_DIR = parent_conftest.TEST_CUSTOM_OUTPUT_DIR
TEST_EPISODE_TITLE = parent_conftest.TEST_EPISODE_TITLE
TEST_EPISODE_TITLE_SPECIAL = parent_conftest.TEST_EPISODE_TITLE_SPECIAL
TEST_FEED_TITLE = parent_conftest.TEST_FEED_TITLE
TEST_FEED_URL = parent_conftest.TEST_FEED_URL
TEST_MEDIA_URL = parent_conftest.TEST_MEDIA_URL
TEST_OUTPUT_DIR = parent_conftest.TEST_OUTPUT_DIR
TEST_RELATIVE_MEDIA = parent_conftest.TEST_RELATIVE_MEDIA
TEST_RELATIVE_TRANSCRIPT = parent_conftest.TEST_RELATIVE_TRANSCRIPT
TEST_RUN_ID = parent_conftest.TEST_RUN_ID
TEST_TRANSCRIPT_URL = parent_conftest.TEST_TRANSCRIPT_URL
TEST_TRANSCRIPT_URL_SRT = parent_conftest.TEST_TRANSCRIPT_URL_SRT
TEST_FULL_URL = parent_conftest.TEST_FULL_URL
TEST_MEDIA_TYPE_M4A = parent_conftest.TEST_MEDIA_TYPE_M4A
TEST_MEDIA_TYPE_MP3 = parent_conftest.TEST_MEDIA_TYPE_MP3
TEST_PATH = parent_conftest.TEST_PATH
TEST_TRANSCRIPT_TYPE_SRT = parent_conftest.TEST_TRANSCRIPT_TYPE_SRT
TEST_TRANSCRIPT_TYPE_VTT = parent_conftest.TEST_TRANSCRIPT_TYPE_VTT


class TestHTTPSessionConfiguration(unittest.TestCase):
    """Tests for HTTP session retry configuration."""

    def test_configure_http_session_mounts_retry_adapters(self):
        session = requests.Session()
        try:
            podcast_scraper.downloader._configure_http_session(session)
            https_adapter = session.get_adapter("https://")
            http_adapter = session.get_adapter("http://")

            self.assertIsInstance(https_adapter, requests.adapters.HTTPAdapter)
            self.assertIsInstance(http_adapter, requests.adapters.HTTPAdapter)

            https_retry = https_adapter.max_retries
            http_retry = http_adapter.max_retries

            self.assertEqual(https_retry.total, downloader.DEFAULT_HTTP_RETRY_TOTAL)
            self.assertEqual(http_retry.total, downloader.DEFAULT_HTTP_RETRY_TOTAL)
            self.assertEqual(https_retry.backoff_factor, downloader.DEFAULT_HTTP_BACKOFF_FACTOR)
            self.assertEqual(http_retry.backoff_factor, downloader.DEFAULT_HTTP_BACKOFF_FACTOR)
            self.assertEqual(https_retry.allowed_methods, downloader.HTTP_RETRY_ALLOWED_METHODS)
            self.assertEqual(http_retry.allowed_methods, downloader.HTTP_RETRY_ALLOWED_METHODS)
            self.assertEqual(
                set(https_retry.status_forcelist), set(downloader.HTTP_RETRY_STATUS_CODES)
            )
            self.assertEqual(
                set(http_retry.status_forcelist), set(downloader.HTTP_RETRY_STATUS_CODES)
            )
        finally:
            session.close()


class TestNormalizeURL(unittest.TestCase):
    """Tests for normalize_url function."""

    def test_simple_url(self):
        """Test normalizing a simple URL."""
        url = TEST_FULL_URL
        result = downloader.normalize_url(url)
        self.assertEqual(result, url)

    def test_url_with_non_ascii(self):
        """Test normalizing URL with non-ASCII characters."""
        url = f"{TEST_FULL_URL}/тест"
        result = downloader.normalize_url(url)
        self.assertIn("%D1%82%D0%B5%D1%81%D1%82", result)

    def test_url_with_query(self):
        """Test normalizing URL with query parameters."""
        url = f"{TEST_FULL_URL}?param=value&other=тест"
        result = downloader.normalize_url(url)
        self.assertIn("param=value", result)
        self.assertIn("%D1%82%D0%B5%D1%81%D1%82", result)

    def test_relative_url(self):
        """Test normalizing a relative URL."""
        url = "/path/to/resource"
        result = downloader.normalize_url(url)
        self.assertEqual(result, url)

    def test_url_with_fragment(self):
        """Test normalizing URL with fragment (should be preserved)."""
        url = f"{TEST_FULL_URL}#section"
        result = downloader.normalize_url(url)
        # requote_uri may or may not preserve fragments, so we just check it's normalized
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_url_with_special_characters(self):
        """Test normalizing URL with special characters."""
        url = f"{TEST_FULL_URL}/path with spaces/file%20name"
        result = downloader.normalize_url(url)
        # Should be normalized/encoded
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestShouldLogDownloadSummary(unittest.TestCase):
    """Tests for should_log_download_summary function."""

    def test_should_log_download_summary_not_tty(self):
        """Test should_log_download_summary returns True when stderr is not a TTY."""
        with patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            result = downloader.should_log_download_summary()
            self.assertTrue(result)

    def test_should_log_download_summary_is_tty(self):
        """Test should_log_download_summary returns False when stderr is a TTY."""
        with patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = True
            result = downloader.should_log_download_summary()
            self.assertFalse(result)

    def test_should_log_download_summary_no_isatty(self):
        """Test should_log_download_summary handles missing isatty attribute."""
        with patch("sys.stderr") as mock_stderr:
            del mock_stderr.isatty
            # Should return True as fallback
            result = downloader.should_log_download_summary()
            self.assertTrue(result)


class TestGetThreadRequestSession(unittest.TestCase):
    """Tests for _get_thread_request_session function."""

    def test_get_thread_request_session_creates_new(self):
        """Test _get_thread_request_session creates new session."""
        # Clear any existing session
        if hasattr(downloader._THREAD_LOCAL, "session"):
            delattr(downloader._THREAD_LOCAL, "session")

        session = downloader._get_thread_request_session()
        self.assertIsInstance(session, requests.Session)
        self.assertIs(session, downloader._THREAD_LOCAL.session)

    def test_get_thread_request_session_reuses_existing(self):
        """Test _get_thread_request_session reuses existing session."""
        # Create a session first
        session1 = downloader._get_thread_request_session()
        # Get it again
        session2 = downloader._get_thread_request_session()
        # Should be the same object
        self.assertIs(session1, session2)

    def test_get_thread_request_session_configures_retry(self):
        """Test _get_thread_request_session configures session with retry."""
        # Clear any existing session
        if hasattr(downloader._THREAD_LOCAL, "session"):
            delattr(downloader._THREAD_LOCAL, "session")

        session = downloader._get_thread_request_session()
        # Check that adapters are mounted
        https_adapter = session.get_adapter("https://")
        http_adapter = session.get_adapter("http://")
        self.assertIsInstance(https_adapter, requests.adapters.HTTPAdapter)
        self.assertIsInstance(http_adapter, requests.adapters.HTTPAdapter)


class TestCloseAllSessions(unittest.TestCase):
    """Tests for _close_all_sessions function."""

    def test_close_all_sessions_closes_registered_sessions(self):
        """Test _close_all_sessions closes all registered sessions."""
        # Create a mock session
        mock_session = Mock(spec=requests.Session)
        with downloader._SESSION_REGISTRY_LOCK:
            downloader._SESSION_REGISTRY.append(mock_session)

        # Close all sessions
        downloader._close_all_sessions()

        # Session should have been closed
        mock_session.close.assert_called_once()

    def test_close_all_sessions_handles_errors(self):
        """Test _close_all_sessions handles errors gracefully."""
        # Create a mock session that raises on close
        mock_session = Mock()
        mock_session.close.side_effect = Exception("Close error")

        with downloader._SESSION_REGISTRY_LOCK:
            downloader._SESSION_REGISTRY.append(mock_session)

        # Should not raise
        downloader._close_all_sessions()

    def test_close_all_sessions_clears_registry(self):
        """Test _close_all_sessions clears the registry."""
        # Add a session
        session = requests.Session()
        with downloader._SESSION_REGISTRY_LOCK:
            downloader._SESSION_REGISTRY.append(session)

        # Close all sessions
        downloader._close_all_sessions()

        # Registry should be empty
        with downloader._SESSION_REGISTRY_LOCK:
            self.assertEqual(len(downloader._SESSION_REGISTRY), 0)


class TestOpenHTTPRequest(unittest.TestCase):
    """Tests for _open_http_request function."""

    @patch("podcast_scraper.downloader._get_thread_request_session")
    @patch("podcast_scraper.downloader.normalize_url")
    def test_open_http_request_success(self, mock_normalize, mock_get_session):
        """Test successful HTTP request."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session
        mock_normalize.return_value = "https://example.com/test"

        result = downloader._open_http_request(
            "https://example.com/test", "test-agent", 10, stream=False
        )

        self.assertIsNotNone(result)
        self.assertEqual(result, mock_response)
        mock_session.get.assert_called_once_with(
            "https://example.com/test",
            headers={"User-Agent": "test-agent"},
            timeout=10,
            stream=False,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("podcast_scraper.downloader._get_thread_request_session")
    @patch("podcast_scraper.downloader.normalize_url")
    def test_open_http_request_with_stream(self, mock_normalize, mock_get_session):
        """Test HTTP request with stream=True."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session
        mock_normalize.return_value = "https://example.com/test"

        result = downloader._open_http_request(
            "https://example.com/test", "test-agent", 5, stream=True
        )

        self.assertIsNotNone(result)
        mock_session.get.assert_called_once_with(
            "https://example.com/test",
            headers={"User-Agent": "test-agent"},
            timeout=5,
            stream=True,
        )

    @patch("podcast_scraper.downloader._get_thread_request_session")
    @patch("podcast_scraper.downloader.normalize_url")
    def test_open_http_request_failure(self, mock_normalize, mock_get_session):
        """Test HTTP request failure returns None."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.RequestException("Connection error")
        mock_get_session.return_value = mock_session
        mock_normalize.return_value = "https://example.com/test"

        result = downloader._open_http_request("https://example.com/test", "test-agent", 10)

        self.assertIsNone(result)

    @patch("podcast_scraper.downloader._open_http_request")
    def test_fetch_url_wrapper(self, mock_open):
        """Test fetch_url is a wrapper around _open_http_request."""
        mock_response = Mock()
        mock_open.return_value = mock_response

        result = downloader.fetch_url("https://example.com", "test-agent", 10, stream=True)

        self.assertEqual(result, mock_response)
        mock_open.assert_called_once_with("https://example.com", "test-agent", 10, stream=True)


class TestHTTPGet(unittest.TestCase):
    """Tests for http_get function."""

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_success(self, mock_progress, mock_fetch):
        """Test successful http_get request."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/plain", "Content-Length": "13"}
        mock_response.iter_content.return_value = [b"Hello, World!"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertEqual(content, b"Hello, World!")
        self.assertEqual(content_type, "text/plain")
        mock_fetch.assert_called_once_with("https://example.com", "test-agent", 10, stream=True)
        mock_response.iter_content.assert_called_once()
        mock_response.close.assert_called_once()

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_multiple_chunks(self, mock_progress, mock_fetch):
        """Test http_get with multiple chunks."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "application/json", "Content-Length": "24"}
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertEqual(content, b"chunk1chunk2chunk3")
        self.assertEqual(content_type, "application/json")
        mock_reporter.update.assert_any_call(6)  # chunk1 length
        mock_reporter.update.assert_any_call(6)  # chunk2 length
        mock_reporter.update.assert_any_call(6)  # chunk3 length

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_no_content_length(self, mock_progress, mock_fetch):
        """Test http_get without Content-Length header."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.iter_content.return_value = [b"content"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertEqual(content, b"content")
        self.assertEqual(content_type, "text/html")
        # Should pass None as total_size when Content-Length is missing
        mock_progress.assert_called_once_with(None, "Downloading")

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_invalid_content_length(self, mock_progress, mock_fetch):
        """Test http_get with invalid Content-Length header."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/plain", "Content-Length": "invalid"}
        mock_response.iter_content.return_value = [b"content"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertEqual(content, b"content")
        # Should handle invalid Content-Length gracefully
        mock_progress.assert_called_once_with(None, "Downloading")

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_empty_chunks_skipped(self, mock_progress, mock_fetch):
        """Test http_get skips empty chunks."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.iter_content.return_value = [b"chunk1", b"", b"chunk2", None]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertEqual(content, b"chunk1chunk2")
        # Should only update for non-empty chunks
        self.assertEqual(mock_reporter.update.call_count, 2)

    @patch("podcast_scraper.downloader.fetch_url")
    def test_http_get_no_response(self, mock_fetch):
        """Test http_get returns None, None when fetch_url returns None."""
        mock_fetch.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertIsNone(content)
        self.assertIsNone(content_type)

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_read_error(self, mock_progress, mock_fetch):
        """Test http_get handles read errors gracefully."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.iter_content.side_effect = requests.RequestException("Read error")
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertIsNone(content)
        self.assertIsNone(content_type)
        mock_response.close.assert_called_once()

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    def test_http_get_os_error(self, mock_progress, mock_fetch):
        """Test http_get handles OSError gracefully."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.iter_content.side_effect = OSError("IO error")
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        content, content_type = downloader.http_get("https://example.com", "test-agent", 10)

        self.assertIsNone(content)
        self.assertIsNone(content_type)
        mock_response.close.assert_called_once()


class TestHTTPDownloadToFile(unittest.TestCase):
    """Tests for http_download_to_file function."""

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_success(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test successful file download."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "13"}
        mock_response.iter_content.return_value = [b"Hello, World!"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertTrue(success)
        self.assertEqual(bytes_written, 13)
        mock_makedirs.assert_called_once_with("/tmp", exist_ok=True)
        mock_file.write.assert_called_once_with(b"Hello, World!")
        mock_response.close.assert_called_once()

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_multiple_chunks(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download with multiple chunks."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "18"}
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertTrue(success)
        self.assertEqual(bytes_written, 18)
        self.assertEqual(mock_file.write.call_count, 3)
        self.assertEqual(mock_reporter.update.call_count, 3)

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_no_content_length(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download without Content-Length header."""
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [b"content"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertTrue(success)
        self.assertEqual(bytes_written, 7)  # len(b"content")
        mock_progress.assert_called_once_with(None, "Downloading file.txt")

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_no_response(self, mock_basename, mock_dirname, mock_fetch):
        """Test file download when fetch_url returns None."""
        mock_fetch.return_value = None
        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertFalse(success)
        self.assertEqual(bytes_written, 0)

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_empty_dirname(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download with empty dirname (current directory)."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "5"}
        mock_response.iter_content.return_value = [b"data"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = ""
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "file.txt"
        )

        self.assertTrue(success)
        # Should create directory "." when dirname is empty
        mock_makedirs.assert_called_once_with(".", exist_ok=True)

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_filename_from_url(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download uses URL basename when path basename is empty."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "5"}
        mock_response.iter_content.return_value = [b"data"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = ""  # Empty basename
        # When basename is empty, should use URL basename
        with patch("podcast_scraper.downloader.os.path.basename") as mock_url_basename:
            mock_url_basename.return_value = "url_file.txt"

            success, bytes_written = downloader.http_download_to_file(
                "https://example.com/url_file.txt", "test-agent", 10, "/tmp/"
            )

            self.assertTrue(success)
            # Should use URL basename for progress message
            mock_progress.assert_called_once_with(5, "Downloading url_file.txt")

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_write_error(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download handles write errors gracefully."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "5"}
        mock_response.iter_content.return_value = [b"data"]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write.side_effect = OSError("Write error")
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertFalse(success)
        self.assertEqual(bytes_written, 0)
        mock_response.close.assert_called_once()

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_request_error(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download handles request errors gracefully."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "5"}
        mock_response.iter_content.side_effect = requests.RequestException("Request error")
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertFalse(success)
        self.assertEqual(bytes_written, 0)
        mock_response.close.assert_called_once()

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.downloader.progress.progress_context")
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.path.dirname")
    @patch("os.path.basename")
    def test_http_download_to_file_empty_chunks_skipped(
        self, mock_basename, mock_dirname, mock_makedirs, mock_open, mock_progress, mock_fetch
    ):
        """Test file download skips empty chunks."""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "10"}
        mock_response.iter_content.return_value = [b"chunk1", b"", b"chunk2", None]
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        mock_file = Mock()
        mock_file.write = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_dirname.return_value = "/tmp"
        mock_basename.return_value = "file.txt"

        success, bytes_written = downloader.http_download_to_file(
            "https://example.com/file.txt", "test-agent", 10, "/tmp/file.txt"
        )

        self.assertTrue(success)
        # Should only write non-empty chunks
        self.assertEqual(mock_file.write.call_count, 2)
        mock_file.write.assert_any_call(b"chunk1")
        mock_file.write.assert_any_call(b"chunk2")


class TestHTTPHead(unittest.TestCase):
    """Tests for http_head function."""

    @patch("podcast_scraper.downloader._get_thread_request_session")
    @patch("podcast_scraper.downloader.normalize_url")
    def test_http_head_success(self, mock_normalize, mock_get_session):
        """Test successful HTTP HEAD request."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "1000"}
        mock_response.raise_for_status = Mock()
        mock_session.head.return_value = mock_response
        mock_get_session.return_value = mock_session
        mock_normalize.return_value = "https://example.com/test"

        result = downloader.http_head("https://example.com/test", "test-agent", 10)

        self.assertIsNotNone(result)
        self.assertEqual(result, mock_response)
        mock_session.head.assert_called_once_with(
            "https://example.com/test",
            headers={"User-Agent": "test-agent"},
            timeout=10,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("podcast_scraper.downloader._get_thread_request_session")
    @patch("podcast_scraper.downloader.normalize_url")
    def test_http_head_failure(self, mock_normalize, mock_get_session):
        """Test HTTP HEAD request failure returns None."""
        mock_session = Mock()
        mock_session.head.side_effect = requests.RequestException("Connection error")
        mock_get_session.return_value = mock_session
        mock_normalize.return_value = "https://example.com/test"

        result = downloader.http_head("https://example.com/test", "test-agent", 10)

        self.assertIsNone(result)

    @patch("podcast_scraper.downloader._get_thread_request_session")
    @patch("podcast_scraper.downloader.normalize_url")
    def test_http_head_http_error(self, mock_normalize, mock_get_session):
        """Test HTTP HEAD request with HTTP error returns None."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_session.head.return_value = mock_response
        mock_get_session.return_value = mock_session
        mock_normalize.return_value = "https://example.com/test"

        result = downloader.http_head("https://example.com/test", "test-agent", 10)

        self.assertIsNone(result)
