#!/usr/bin/env python3
"""Tests for HTTP downloader functionality."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import sys
import unittest
from pathlib import Path

import requests

import podcast_scraper
from podcast_scraper import downloader

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: F401, E402
    build_rss_xml_with_media,
    build_rss_xml_with_speakers,
    build_rss_xml_with_transcript,
    create_media_response,
    create_mock_spacy_model,
    create_rss_response,
    create_test_args,
    create_test_config,
    create_test_episode,
    create_test_feed,
    create_transcript_response,
    MockHTTPResponse,
    TEST_BASE_URL,
    TEST_CONTENT_TYPE_SRT,
    TEST_CONTENT_TYPE_VTT,
    TEST_CUSTOM_OUTPUT_DIR,
    TEST_EPISODE_TITLE,
    TEST_EPISODE_TITLE_SPECIAL,
    TEST_FEED_TITLE,
    TEST_FEED_URL,
    TEST_FULL_URL,
    TEST_MEDIA_TYPE_M4A,
    TEST_MEDIA_TYPE_MP3,
    TEST_MEDIA_URL,
    TEST_OUTPUT_DIR,
    TEST_PATH,
    TEST_RELATIVE_MEDIA,
    TEST_RELATIVE_TRANSCRIPT,
    TEST_RUN_ID,
    TEST_TRANSCRIPT_TYPE_SRT,
    TEST_TRANSCRIPT_TYPE_VTT,
    TEST_TRANSCRIPT_URL,
    TEST_TRANSCRIPT_URL_SRT,
)


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
