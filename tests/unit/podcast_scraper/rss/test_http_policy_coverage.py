#!/usr/bin/env python3
"""Additional unit tests for rss.http_policy — patch coverage gaps.

Covers: gated_http_request, note_retry_after_from_response, metrics helpers,
RetryAfterTracker (HTTP-date, edge cases), HostThrottle (concurrency),
CircuitBreaker (half-open, record_success, non-qualifying failures, host scope),
ConditionalGetCache (corrupt meta, missing body, OSError on write),
configure_http_policy (needs_policy paths, rss_cache_dir env, default cache).
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.rss import http_policy

pytestmark = [pytest.mark.unit]


class TestMetricsHelpers(unittest.TestCase):
    """Cover record_* / get_* metric helpers."""

    def setUp(self) -> None:
        http_policy.reset_http_policy_metrics()

    def test_record_host_throttle_wait_positive(self) -> None:
        http_policy.record_host_throttle_wait(1.5)
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["host_throttle_events"], 1)
        self.assertAlmostEqual(
            float(snap["host_throttle_wait_seconds"]), 1.5, places=3  # type: ignore[arg-type]
        )

    def test_record_host_throttle_wait_zero_ignored(self) -> None:
        http_policy.record_host_throttle_wait(0)
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["host_throttle_events"], 0)

    def test_record_host_throttle_wait_negative_ignored(self) -> None:
        http_policy.record_host_throttle_wait(-1.0)
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["host_throttle_events"], 0)

    def test_record_retry_after_sleep_positive(self) -> None:
        http_policy.record_retry_after_sleep(2.0)
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["retry_after_events"], 1)
        self.assertAlmostEqual(
            float(snap["retry_after_total_sleep_seconds"]),  # type: ignore[arg-type]
            2.0,
            places=3,
        )

    def test_record_retry_after_sleep_zero_ignored(self) -> None:
        http_policy.record_retry_after_sleep(0)
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["retry_after_events"], 0)

    def test_record_circuit_trip(self) -> None:
        http_policy.record_circuit_trip("feed:example")
        http_policy.record_circuit_trip("feed:example")
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["circuit_breaker_trips"], 2)
        self.assertEqual(snap["circuit_breaker_open_feeds"], ["feed:example"])

    def test_record_rss_conditional_hit_and_miss(self) -> None:
        http_policy.record_rss_conditional_hit()
        http_policy.record_rss_conditional_miss()
        http_policy.record_rss_conditional_miss()
        snap = http_policy.get_http_policy_metrics_snapshot()
        self.assertEqual(snap["rss_conditional_hit"], 1)
        self.assertEqual(snap["rss_conditional_miss"], 2)


class TestNetlocFromUrl(unittest.TestCase):
    def test_empty_url(self) -> None:
        self.assertEqual(http_policy.netloc_from_url(""), "unknown")

    def test_no_scheme(self) -> None:
        self.assertEqual(http_policy.netloc_from_url("not-a-url"), "unknown")

    def test_normal_url(self) -> None:
        self.assertEqual(
            http_policy.netloc_from_url("https://Example.COM:8080/path"),
            "example.com:8080",
        )


class TestCircuitOpenError(unittest.TestCase):
    def test_attributes(self) -> None:
        err = http_policy.CircuitOpenError("feed:rss.xml")
        self.assertEqual(err.scope_key, "feed:rss.xml")
        self.assertIn("feed:rss.xml", str(err))


class TestRetryAfterTrackerParsing(unittest.TestCase):
    """Cover _parse_retry_after edge cases."""

    def test_empty_string(self) -> None:
        result = http_policy.RetryAfterTracker._parse_retry_after("", 100.0)
        self.assertIsNone(result)

    def test_negative_seconds(self) -> None:
        result = http_policy.RetryAfterTracker._parse_retry_after("-5", 100.0)
        self.assertIsNone(result)

    def test_positive_seconds(self) -> None:
        result = http_policy.RetryAfterTracker._parse_retry_after("10", 100.0)
        assert result is not None
        self.assertAlmostEqual(result, 110.0, places=1)

    def test_http_date(self) -> None:
        future = "Mon, 01 Jan 2035 00:00:00 GMT"
        result = http_policy.RetryAfterTracker._parse_retry_after(future, 0.0)
        assert result is not None
        self.assertGreater(result, 0.0)

    def test_garbage_returns_none(self) -> None:
        result = http_policy.RetryAfterTracker._parse_retry_after("not-a-date-or-number", 0.0)
        self.assertIsNone(result)

    def test_set_deadline_extends_existing(self) -> None:
        t = http_policy.RetryAfterTracker()
        with patch.object(time, "monotonic", return_value=0.0):
            t.set_deadline_from_header("h.example", "5")
            t.set_deadline_from_header("h.example", "10")
        self.assertAlmostEqual(t._deadline["h.example"], 10.0, places=1)


class TestRetryAfterWaitLoop(unittest.TestCase):
    """Cover wait_until_clear with actual sleeping (mocked)."""

    def test_wait_sleeps_then_clears(self) -> None:
        t = http_policy.RetryAfterTracker()
        http_policy.reset_http_policy_metrics()
        mono_values = iter([0.0, 0.0, 1.0, 1.0])
        with (
            patch.object(time, "monotonic", side_effect=mono_values),
            patch.object(time, "sleep") as mock_sleep,
        ):
            t._deadline["host"] = 0.5
            t.wait_until_clear("host")
        mock_sleep.assert_called()
        self.assertNotIn("host", t._deadline)


class TestHostThrottleConcurrency(unittest.TestCase):
    """Cover max_concurrent semaphore path."""

    def test_semaphore_limits_concurrency(self) -> None:
        tr = http_policy.RetryAfterTracker()
        th = http_policy.HostThrottle(interval_ms=0, max_concurrent=2, retry_after=tr)
        url = "https://h.example/a"
        th.acquire(url)
        th.acquire(url)
        th.release(url)
        th.release(url)

    def test_no_throttle_when_both_zero(self) -> None:
        tr = http_policy.RetryAfterTracker()
        th = http_policy.HostThrottle(interval_ms=0, max_concurrent=0, retry_after=tr)
        url = "https://h.example/a"
        th.acquire(url)
        th.release(url)

    def test_release_unknown_host_is_noop(self) -> None:
        tr = http_policy.RetryAfterTracker()
        th = http_policy.HostThrottle(interval_ms=100, max_concurrent=0, retry_after=tr)
        th.release("https://never-acquired.example/x")


class TestCircuitBreakerStateMachine(unittest.TestCase):
    """Cover half-open, record_success, non-qualifying failures, host scope."""

    def setUp(self) -> None:
        http_policy.reset_http_policy_metrics()

    def test_half_open_success_closes(self) -> None:
        cb = http_policy.CircuitBreaker(True, 2, 60.0, 10.0, "feed", "https://f.example/rss")
        u = "https://cdn.example/ep.mp3"
        with patch.object(time, "monotonic", return_value=0.0):
            cb.record_failure(u, 500)
            cb.record_failure(u, 500)
        with patch.object(time, "monotonic", return_value=5.0):
            with self.assertRaises(http_policy.CircuitOpenError):
                cb.check_allow(u)
        with patch.object(time, "monotonic", return_value=15.0):
            cb.check_allow(u)
        entry = cb._entry(cb.scope_key(u))
        self.assertEqual(entry.state, "half_open")
        cb.record_success(u)
        self.assertEqual(entry.state, "closed")

    def test_half_open_failure_reopens(self) -> None:
        cb = http_policy.CircuitBreaker(True, 2, 60.0, 10.0, "feed", "https://f.example/rss")
        u = "https://cdn.example/ep.mp3"
        with patch.object(time, "monotonic", return_value=0.0):
            cb.record_failure(u, 500)
            cb.record_failure(u, 500)
        with patch.object(time, "monotonic", return_value=15.0):
            cb.check_allow(u)
        with patch.object(time, "monotonic", return_value=15.0):
            cb.record_failure(u, 503)
        entry = cb._entry(cb.scope_key(u))
        self.assertEqual(entry.state, "open")

    def test_non_qualifying_status_ignored(self) -> None:
        cb = http_policy.CircuitBreaker(True, 2, 60.0, 10.0, "feed", "https://f.example/rss")
        u = "https://cdn.example/ep.mp3"
        cb.record_failure(u, 404)
        cb.record_failure(u, 301)
        cb.check_allow(u)

    def test_qualifying_status_codes(self) -> None:
        u = "https://cdn.example/ep.mp3"
        for code in (0, 401, 403, 429, 500, 502, 503, 504):
            cb2 = http_policy.CircuitBreaker(
                True, 1, 60.0, 10.0, "feed", f"https://f{code}.example/rss"
            )
            cb2.record_failure(u, code)
            with self.assertRaises(http_policy.CircuitOpenError):
                cb2.check_allow(u)

    def test_host_scope_key(self) -> None:
        cb = http_policy.CircuitBreaker(True, 3, 60.0, 10.0, "host", "")
        key = cb.scope_key("https://cdn.example/ep.mp3")
        self.assertTrue(key.startswith("host:"))

    def test_feed_scope_key(self) -> None:
        cb = http_policy.CircuitBreaker(True, 3, 60.0, 10.0, "feed", "https://f.example/rss")
        key = cb.scope_key("https://cdn.example/ep.mp3")
        self.assertEqual(key, "feed:https://f.example/rss")

    def test_invalid_scope_defaults_to_feed(self) -> None:
        cb = http_policy.CircuitBreaker(True, 3, 60.0, 10.0, "bogus", "https://f.example/rss")
        self.assertEqual(cb._scope, "feed")

    def test_disabled_breaker_allows_everything(self) -> None:
        cb = http_policy.CircuitBreaker(False, 1, 60.0, 10.0, "feed", "https://f.example/rss")
        u = "https://cdn.example/ep.mp3"
        for _ in range(10):
            cb.record_failure(u, 500)
        cb.check_allow(u)

    def test_record_success_on_closed_clears_history(self) -> None:
        cb = http_policy.CircuitBreaker(True, 5, 60.0, 10.0, "feed", "https://f.example/rss")
        u = "https://cdn.example/ep.mp3"
        cb.record_failure(u, 500)
        cb.record_success(u)
        entry = cb._entry(cb.scope_key(u))
        self.assertEqual(len(entry.failure_times), 0)


class TestConditionalGetCacheEdgeCases(unittest.TestCase):
    def test_missing_meta_returns_empty_headers(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            h = c.conditional_headers("https://no-such-url.example/rss")
            self.assertEqual(h, {})

    def test_corrupt_meta_returns_empty_headers(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            meta_path, _ = c._paths("https://corrupt.example/rss")
            meta_path.write_text("not json", encoding="utf-8")
            h = c.conditional_headers("https://corrupt.example/rss")
            self.assertEqual(h, {})

    def test_missing_body_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            self.assertIsNone(c.get_cached_body("https://no-body.example/rss"))

    def test_partial_meta_only_etag(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            url = "https://partial.example/rss"
            c.update_from_success(url, '"etag-only"', None, b"body")
            h = c.conditional_headers(url)
            self.assertIn("If-None-Match", h)
            self.assertNotIn("If-Modified-Since", h)

    def test_partial_meta_only_last_modified(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            url = "https://partial.example/rss"
            c.update_from_success(url, None, "Mon, 01 Jan 2024 00:00:00 GMT", b"body")
            h = c.conditional_headers(url)
            self.assertNotIn("If-None-Match", h)
            self.assertIn("If-Modified-Since", h)

    def test_write_oserror_logs_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            url = "https://fail-write.example/rss"
            _, body_path = c._paths(url)
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(b"old")
            os.chmod(str(body_path), 0o000)
            try:
                c.update_from_success(url, '"e"', "lm", b"new")
            finally:
                os.chmod(str(body_path), 0o644)

    def test_body_read_oserror_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            c = http_policy.ConditionalGetCache(Path(tmp))
            url = "https://unreadable.example/rss"
            _, body_path = c._paths(url)
            body_path.write_bytes(b"data")
            os.chmod(str(body_path), 0o000)
            try:
                result = c.get_cached_body(url)
                self.assertIsNone(result)
            finally:
                os.chmod(str(body_path), 0o644)


class TestNoteRetryAfterFromResponse(unittest.TestCase):
    def setUp(self) -> None:
        http_policy.configure_http_policy(host_request_interval_ms=100)

    def tearDown(self) -> None:
        http_policy.configure_http_policy()

    def test_with_retry_after_header(self) -> None:
        resp = MagicMock()
        resp.headers = {"Retry-After": "5"}
        resp.url = "https://cdn.example/ep.mp3"
        http_policy.note_retry_after_from_response(resp)

    def test_no_headers_attr(self) -> None:
        resp = object()
        http_policy.note_retry_after_from_response(resp)

    def test_headers_not_callable_get(self) -> None:
        resp = MagicMock()
        resp.headers = "not-a-dict"
        http_policy.note_retry_after_from_response(resp)

    def test_no_retry_after_header(self) -> None:
        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.url = "https://cdn.example/ep.mp3"
        http_policy.note_retry_after_from_response(resp)

    def test_tracker_none_is_noop(self) -> None:
        http_policy.configure_http_policy()
        resp = MagicMock()
        resp.headers = {"Retry-After": "5"}
        http_policy.note_retry_after_from_response(resp)

    def test_url_from_request_url_fallback(self) -> None:
        resp = MagicMock()
        resp.headers = {"Retry-After": "5"}
        resp.url = None
        http_policy.note_retry_after_from_response(resp, request_url="https://fallback.example/rss")

    def test_exception_swallowed(self) -> None:
        resp = MagicMock()
        resp.headers = MagicMock()
        resp.headers.get = MagicMock(side_effect=RuntimeError("boom"))
        http_policy.note_retry_after_from_response(resp)


class TestGatedHttpRequest(unittest.TestCase):
    def test_noop_when_no_throttle(self) -> None:
        http_policy.configure_http_policy()
        with http_policy.gated_http_request("https://example.com/x"):
            pass

    def test_acquire_release_with_throttle(self) -> None:
        http_policy.configure_http_policy(host_request_interval_ms=10)
        with http_policy.gated_http_request("https://example.com/x"):
            pass
        http_policy.configure_http_policy()

    def test_release_on_exception(self) -> None:
        http_policy.configure_http_policy(host_request_interval_ms=10)
        try:
            with http_policy.gated_http_request("https://example.com/x"):
                raise ValueError("test")
        except ValueError:
            pass
        http_policy.configure_http_policy()


class TestConfigureHttpPolicyPaths(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("PODCAST_SCRAPER_RSS_SKIP_CONDITIONAL", None)
        os.environ.pop("PODCAST_SCRAPER_RSS_CACHE_DIR", None)
        http_policy.configure_http_policy()

    def test_needs_policy_false_when_all_zero(self) -> None:
        http_policy.configure_http_policy()
        self.assertIsNone(http_policy._STATE.throttle)
        self.assertIsNone(http_policy._STATE.circuit)

    def test_needs_policy_true_with_interval(self) -> None:
        http_policy.configure_http_policy(host_request_interval_ms=100)
        self.assertIsNotNone(http_policy._STATE.throttle)

    def test_needs_policy_true_with_concurrent(self) -> None:
        http_policy.configure_http_policy(host_max_concurrent=5)
        self.assertIsNotNone(http_policy._STATE.throttle)

    def test_circuit_breaker_created(self) -> None:
        http_policy.configure_http_policy(circuit_breaker_enabled=True)
        self.assertIsNotNone(http_policy._STATE.circuit)

    def test_conditional_get_with_explicit_cache_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            http_policy.configure_http_policy(rss_conditional_get=True, rss_cache_dir=tmp)
            self.assertIsNotNone(http_policy._STATE.conditional)

    def test_conditional_get_with_env_cache_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            os.environ["PODCAST_SCRAPER_RSS_CACHE_DIR"] = tmp
            http_policy.configure_http_policy(rss_conditional_get=True)
            self.assertIsNotNone(http_policy._STATE.conditional)

    def test_skip_conditional_env_values(self) -> None:
        for val in ("1", "true", "yes", "on", "TRUE", " Yes "):
            os.environ["PODCAST_SCRAPER_RSS_SKIP_CONDITIONAL"] = val
            http_policy.configure_http_policy(rss_conditional_get=True)
            self.assertFalse(http_policy._STATE.rss_conditional_get, f"val={val}")

    def test_skip_conditional_env_not_set(self) -> None:
        os.environ.pop("PODCAST_SCRAPER_RSS_SKIP_CONDITIONAL", None)
        with TemporaryDirectory() as tmp:
            http_policy.configure_http_policy(rss_conditional_get=True, rss_cache_dir=tmp)
            self.assertTrue(http_policy._STATE.rss_conditional_get)

    def test_rss_url_stored(self) -> None:
        http_policy.configure_http_policy(rss_url="https://f.example/rss")
        self.assertEqual(http_policy._STATE.rss_url, "https://f.example/rss")
