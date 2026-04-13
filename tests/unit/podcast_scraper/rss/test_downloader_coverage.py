#!/usr/bin/env python3
"""Additional unit tests for rss.downloader — patch coverage gaps.

Covers: configure_downloader, retry event counter, normalize_url,
fetch_rss_feed_url (conditional GET paths: 304 hit/miss, cache update),
http_get (streaming, malformed Content-Length), http_download_to_file
(success, failure with partial cleanup), http_head (circuit open, failure),
_open_http_request (circuit open, raise_for_status failure).
"""

from __future__ import annotations

import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from podcast_scraper.rss import downloader, http_policy

pytestmark = [pytest.mark.unit]


class TestConfigureDownloader(unittest.TestCase):
    def tearDown(self) -> None:
        downloader.configure_downloader()

    def test_resets_retry_counter(self) -> None:
        downloader._increment_http_retry_events()
        self.assertGreater(downloader.get_http_retry_event_count(), 0)
        downloader.configure_downloader()
        self.assertEqual(downloader.get_http_retry_event_count(), 0)

    def test_applies_overrides(self) -> None:
        downloader.configure_downloader(
            http_retry_total=3,
            http_backoff_factor=0.5,
            rss_retry_total=2,
            rss_backoff_factor=0.3,
        )
        self.assertEqual(downloader._effective_http_retry_total(), 3)
        self.assertAlmostEqual(downloader._effective_http_backoff_factor(), 0.5)
        self.assertEqual(downloader._effective_rss_retry_total(), 2)
        self.assertAlmostEqual(downloader._effective_rss_backoff_factor(), 0.3)

    def test_defaults_when_none(self) -> None:
        downloader._configured_http_retry_total = None
        downloader._configured_http_backoff_factor = None
        downloader._configured_rss_retry_total = None
        downloader._configured_rss_backoff_factor = None
        self.assertEqual(
            downloader._effective_http_retry_total(),
            downloader.DEFAULT_HTTP_RETRY_TOTAL,
        )


class TestRetryEventCounter(unittest.TestCase):
    def setUp(self) -> None:
        downloader.reset_http_retry_event_counter()

    def test_increment_and_get(self) -> None:
        self.assertEqual(downloader.get_http_retry_event_count(), 0)
        downloader._increment_http_retry_events()
        downloader._increment_http_retry_events()
        self.assertEqual(downloader.get_http_retry_event_count(), 2)


class TestNormalizeUrl(unittest.TestCase):
    def test_already_normalized(self) -> None:
        url = "https://example.com/path"
        self.assertEqual(downloader.normalize_url(url), url)

    def test_encodes_spaces(self) -> None:
        result = downloader.normalize_url("https://example.com/path with spaces")
        self.assertNotIn(" ", result)


class TestHttpHead(unittest.TestCase):
    def setUp(self) -> None:
        http_policy.configure_http_policy()

    @patch("podcast_scraper.rss.downloader._get_thread_request_session")
    def test_success(self, mock_session_fn: Mock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Length": "1024"}
        mock_resp.raise_for_status = Mock()
        mock_session = MagicMock()
        mock_session.head.return_value = mock_resp
        mock_session_fn.return_value = mock_session
        result = downloader.http_head("https://cdn.example/ep.mp3", "ua", 30)
        self.assertIsNotNone(result)

    @patch("podcast_scraper.rss.downloader._get_thread_request_session")
    def test_request_exception(self, mock_session_fn: Mock) -> None:
        mock_session = MagicMock()
        mock_session.head.side_effect = requests.ConnectionError("fail")
        mock_session_fn.return_value = mock_session
        result = downloader.http_head("https://cdn.example/ep.mp3", "ua", 30)
        self.assertIsNone(result)

    def test_circuit_open_returns_none(self) -> None:
        http_policy.configure_http_policy(circuit_breaker_enabled=True)
        cb = http_policy._STATE.circuit
        assert cb is not None
        url = "https://cdn.example/ep.mp3"
        for _ in range(5):
            cb.record_failure(url, 500)
        result = downloader.http_head(url, "ua", 30)
        self.assertIsNone(result)
        http_policy.configure_http_policy()


class TestOpenHttpRequest(unittest.TestCase):
    def setUp(self) -> None:
        http_policy.configure_http_policy()

    def test_circuit_open_returns_none(self) -> None:
        http_policy.configure_http_policy(circuit_breaker_enabled=True)
        cb = http_policy._STATE.circuit
        assert cb is not None
        url = "https://cdn.example/ep.mp3"
        for _ in range(5):
            cb.record_failure(url, 500)
        result = downloader._open_http_request(url, "ua", 30)
        self.assertIsNone(result)
        http_policy.configure_http_policy()

    @patch("podcast_scraper.rss.downloader._get_thread_request_session")
    def test_raise_for_status_failure_records_circuit(self, mock_session_fn: Mock) -> None:
        http_policy.configure_http_policy(circuit_breaker_enabled=True)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        err_resp = MagicMock()
        err_resp.status_code = 503
        exc = requests.HTTPError(response=err_resp)
        mock_resp.raise_for_status.side_effect = exc
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_fn.return_value = mock_session
        result = downloader._open_http_request("https://cdn.example/ep.mp3", "ua", 30)
        self.assertIsNone(result)
        http_policy.configure_http_policy()

    @patch("podcast_scraper.rss.downloader._get_thread_request_session")
    def test_connection_error_records_zero_status(self, mock_session_fn: Mock) -> None:
        http_policy.configure_http_policy(circuit_breaker_enabled=True)
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("fail")
        mock_session_fn.return_value = mock_session
        result = downloader._open_http_request("https://cdn.example/ep.mp3", "ua", 30)
        self.assertIsNone(result)
        http_policy.configure_http_policy()

    @patch("podcast_scraper.rss.downloader._get_thread_request_session")
    def test_304_with_accept_not_modified(self, mock_session_fn: Mock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 304
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_fn.return_value = mock_session
        result = downloader._open_http_request(
            "https://cdn.example/rss", "ua", 30, accept_not_modified=True
        )
        assert result is not None
        self.assertEqual(result.status_code, 304)


class TestFetchRssFeedUrl(unittest.TestCase):
    def setUp(self) -> None:
        http_policy.configure_http_policy()

    def tearDown(self) -> None:
        http_policy.configure_http_policy()

    @patch("podcast_scraper.rss.downloader._open_http_request")
    def test_returns_none_when_open_fails(self, mock_open: Mock) -> None:
        mock_open.return_value = None
        result = downloader.fetch_rss_feed_url("https://f.example/rss", "ua", 30)
        self.assertIsNone(result)

    @patch("podcast_scraper.rss.downloader._open_http_request")
    @patch("podcast_scraper.rss.downloader._get_thread_feed_request_session")
    def test_304_with_cached_body(self, mock_sess: Mock, mock_open: Mock) -> None:
        with TemporaryDirectory() as tmp:
            http_policy.configure_http_policy(rss_conditional_get=True, rss_cache_dir=tmp)
            cond = http_policy._STATE.conditional
            assert cond is not None
            url = "https://f.example/rss"
            cond.update_from_success(url, '"e"', "lm", b"<rss>cached</rss>")

            mock_resp = MagicMock()
            mock_resp.status_code = 304
            mock_resp.close = Mock()
            mock_open.return_value = mock_resp

            result = downloader.fetch_rss_feed_url(url, "ua", 30)
            assert result is not None
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.content, b"<rss>cached</rss>")

    @patch("podcast_scraper.rss.downloader._open_http_request")
    @patch("podcast_scraper.rss.downloader._get_thread_feed_request_session")
    def test_304_without_cached_body_returns_none(self, mock_sess: Mock, mock_open: Mock) -> None:
        with TemporaryDirectory() as tmp:
            http_policy.configure_http_policy(rss_conditional_get=True, rss_cache_dir=tmp)
            mock_resp = MagicMock()
            mock_resp.status_code = 304
            mock_resp.close = Mock()
            mock_open.return_value = mock_resp

            result = downloader.fetch_rss_feed_url("https://f.example/rss", "ua", 30)
            self.assertIsNone(result)

    @patch("podcast_scraper.rss.downloader._open_http_request")
    @patch("podcast_scraper.rss.downloader._get_thread_feed_request_session")
    def test_200_updates_cache(self, mock_sess: Mock, mock_open: Mock) -> None:
        with TemporaryDirectory() as tmp:
            http_policy.configure_http_policy(rss_conditional_get=True, rss_cache_dir=tmp)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"<rss>new</rss>"
            mock_resp.headers = CaseInsensitiveDict(
                {"ETag": '"new-etag"', "Last-Modified": "Mon, 01 Jan 2025 00:00:00 GMT"}
            )
            mock_open.return_value = mock_resp

            result = downloader.fetch_rss_feed_url("https://f.example/rss", "ua", 30)
            self.assertIsNotNone(result)
            cond = http_policy._STATE.conditional
            assert cond is not None
            h = cond.conditional_headers("https://f.example/rss")
            self.assertIn("If-None-Match", h)


class TestHttpGet(unittest.TestCase):
    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_returns_none_on_no_response(self, mock_fetch: Mock) -> None:
        mock_fetch.return_value = None
        data, ctype = downloader.http_get("https://x.example/t", "ua", 30)
        self.assertIsNone(data)
        self.assertIsNone(ctype)

    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_returns_body_and_content_type(self, mock_fetch: Mock) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = CaseInsensitiveDict(
            {"Content-Type": "text/plain", "Content-Length": "5"}
        )
        mock_resp.iter_content.return_value = [b"hello"]
        mock_resp.close = Mock()
        mock_fetch.return_value = mock_resp
        data, ctype = downloader.http_get("https://x.example/t", "ua", 30)
        self.assertEqual(data, b"hello")
        self.assertEqual(ctype, "text/plain")

    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_malformed_content_length(self, mock_fetch: Mock) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = CaseInsensitiveDict(
            {"Content-Type": "text/plain", "Content-Length": "not-a-number"}
        )
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.close = Mock()
        mock_fetch.return_value = mock_resp
        data, ctype = downloader.http_get("https://x.example/t", "ua", 30)
        self.assertEqual(data, b"data")

    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_read_error_returns_none(self, mock_fetch: Mock) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = CaseInsensitiveDict({"Content-Type": "text/plain"})
        mock_resp.iter_content.side_effect = requests.ConnectionError("fail")
        mock_resp.close = Mock()
        mock_fetch.return_value = mock_resp
        data, ctype = downloader.http_get("https://x.example/t", "ua", 30)
        self.assertIsNone(data)


class TestHttpDownloadToFile(unittest.TestCase):
    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_no_response_returns_false(self, mock_fetch: Mock) -> None:
        mock_fetch.return_value = None
        ok, size = downloader.http_download_to_file(
            "https://x.example/ep.mp3", "ua", 30, "/tmp/out.mp3"
        )
        self.assertFalse(ok)
        self.assertEqual(size, 0)

    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_success_writes_file(self, mock_fetch: Mock) -> None:
        with TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "ep.mp3")
            mock_resp = MagicMock()
            mock_resp.headers = CaseInsensitiveDict({"Content-Length": "5"})
            mock_resp.iter_content.return_value = [b"audio"]
            mock_resp.close = Mock()
            mock_fetch.return_value = mock_resp
            ok, size = downloader.http_download_to_file(
                "https://x.example/ep.mp3", "ua", 30, out_path
            )
            self.assertTrue(ok)
            self.assertEqual(size, 5)
            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), b"audio")

    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_write_error_removes_partial(self, mock_fetch: Mock) -> None:
        with TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "ep.mp3")
            mock_resp = MagicMock()
            mock_resp.headers = CaseInsensitiveDict({})
            mock_resp.iter_content.side_effect = OSError("disk full")
            mock_resp.close = Mock()
            mock_fetch.return_value = mock_resp
            ok, size = downloader.http_download_to_file(
                "https://x.example/ep.mp3", "ua", 30, out_path
            )
            self.assertFalse(ok)
            self.assertFalse(os.path.exists(out_path))

    @patch("podcast_scraper.rss.downloader.fetch_url")
    def test_malformed_content_length(self, mock_fetch: Mock) -> None:
        with TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "ep.mp3")
            mock_resp = MagicMock()
            mock_resp.headers = CaseInsensitiveDict({"Content-Length": "bad"})
            mock_resp.iter_content.return_value = [b"data"]
            mock_resp.close = Mock()
            mock_fetch.return_value = mock_resp
            ok, size = downloader.http_download_to_file(
                "https://x.example/ep.mp3", "ua", 30, out_path
            )
            self.assertTrue(ok)
            self.assertEqual(size, 4)


class TestSyntheticRssResponse(unittest.TestCase):
    def test_builds_200_response(self) -> None:
        resp = downloader._synthetic_rss_response("https://f.example/rss", b"<rss></rss>")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"<rss></rss>")
        self.assertEqual(resp.url, "https://f.example/rss")
        self.assertIn("Content-Type", resp.headers)
