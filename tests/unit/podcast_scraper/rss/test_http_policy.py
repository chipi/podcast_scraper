#!/usr/bin/env python3
"""Unit tests for rss.http_policy (Issue #522)."""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from podcast_scraper.rss import http_policy

pytestmark = [pytest.mark.unit]


class TestNetlocFromUrl(unittest.TestCase):
    def test_netloc_lowercase(self) -> None:
        self.assertEqual(
            http_policy.netloc_from_url("https://CDN.EXAMPLE.com/path"), "cdn.example.com"
        )


class TestRetryAfterTracker(unittest.TestCase):
    def test_wait_until_clear_when_deadline_passed(self) -> None:
        t = http_policy.RetryAfterTracker()
        with patch.object(time, "monotonic", return_value=0.0):
            t.set_deadline_from_header("example.com", "2")
        with patch.object(time, "monotonic", return_value=10.0):
            t.wait_until_clear("example.com")


class TestHostThrottle(unittest.TestCase):
    def test_interval_spaces_sequential_requests(self) -> None:
        tr = http_policy.RetryAfterTracker()
        th = http_policy.HostThrottle(interval_ms=100, max_concurrent=0, retry_after=tr)
        url = "https://h.example/a"
        th.acquire(url)
        th.release(url)
        t0 = time.monotonic()
        th.acquire(url)
        t1 = time.monotonic()
        th.release(url)
        self.assertGreaterEqual(t1 - t0, 0.08)


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self) -> None:
        http_policy.configure_http_policy()

    def test_trips_after_threshold(self) -> None:
        cb = http_policy.CircuitBreaker(
            True,
            3,
            60.0,
            10.0,
            "feed",
            "https://feed.example/rss.xml",
        )
        u = "https://cdn.example/ep.mp3"
        for _ in range(3):
            cb.record_failure(u, 503)
        with self.assertRaises(http_policy.CircuitOpenError):
            cb.check_allow(u)


class TestConditionalGetCache(unittest.TestCase):
    def test_roundtrip_meta_and_body(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            url = "https://feeds.example/podcast.xml"
            body = b"<rss></rss>"
            c.update_from_success(url, '"abc"', "Mon, 01 Jan 2024 00:00:00 GMT", body)
            h = c.conditional_headers(url)
            self.assertIn("If-None-Match", h)
            self.assertIn("If-Modified-Since", h)
            self.assertEqual(c.get_cached_body(url), body)


class TestConfigureHttpPolicy(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("PODCAST_SCRAPER_RSS_SKIP_CONDITIONAL", None)
        http_policy.configure_http_policy()

    def test_snapshot_resets(self) -> None:
        http_policy.record_host_throttle_wait(0.1)
        http_policy.configure_http_policy()
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["host_throttle_events"], 0)

    def test_skip_conditional_env_disables_flag(self) -> None:
        os.environ["PODCAST_SCRAPER_RSS_SKIP_CONDITIONAL"] = "1"
        http_policy.configure_http_policy(rss_conditional_get=True)
        self.assertFalse(http_policy._STATE.rss_conditional_get)

    def test_conditional_cache_oserror_continues_without_cache(self) -> None:
        with patch.object(
            http_policy.ConditionalGetCache,
            "__init__",
            side_effect=OSError("permission denied"),
        ):
            http_policy.configure_http_policy(rss_conditional_get=True, rss_cache_dir="/tmp/x")
        self.assertIsNone(http_policy._STATE.conditional)
