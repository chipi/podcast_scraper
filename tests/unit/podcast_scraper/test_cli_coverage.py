#!/usr/bin/env python3
"""Additional unit tests for cli.py — patch coverage for new functions.

Covers: _cli_iso_date, _load_rss_urls_from_file, collect_feed_urls,
_validate_download_resilience_cli.
"""

from __future__ import annotations

import argparse
import os
import unittest
from datetime import date
from tempfile import NamedTemporaryFile

import pytest

from podcast_scraper.cli import (
    _cli_iso_date,
    _load_rss_urls_from_file,
    _validate_download_resilience_cli,
    collect_feed_urls,
)

pytestmark = [pytest.mark.unit]


class TestCliIsoDate(unittest.TestCase):
    def test_valid_date(self) -> None:
        self.assertEqual(_cli_iso_date("2024-06-15"), date(2024, 6, 15))

    def test_whitespace_stripped(self) -> None:
        self.assertEqual(_cli_iso_date("  2024-01-01  "), date(2024, 1, 1))

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            _cli_iso_date("not-a-date")


class TestLoadRssUrlsFromFile(unittest.TestCase):
    def test_reads_lines(self) -> None:
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("https://a.example/rss\n")
            f.write("# comment\n")
            f.write("\n")
            f.write("https://b.example/rss\n")
            f.flush()
            path = f.name
        try:
            urls = _load_rss_urls_from_file(path)
            self.assertEqual(urls, ["https://a.example/rss", "https://b.example/rss"])
        finally:
            os.unlink(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _load_rss_urls_from_file("/nonexistent/file.txt")
        self.assertIn("not a readable file", str(ctx.exception))


class TestCollectFeedUrls(unittest.TestCase):
    def _args(self, **kwargs) -> argparse.Namespace:
        defaults = {
            "rss": None,
            "rss_extra": None,
            "rss_urls": None,
            "rss_file": None,
            "config": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_single_rss(self) -> None:
        args = self._args(rss="https://a.example/rss")
        urls = collect_feed_urls(args)
        self.assertEqual(urls, ["https://a.example/rss"])

    def test_deduplication(self) -> None:
        args = self._args(
            rss="https://a.example/rss",
            rss_extra=["https://a.example/rss", "https://b.example/rss"],
        )
        urls = collect_feed_urls(args)
        self.assertEqual(len(urls), 2)

    def test_rss_urls_from_args(self) -> None:
        args = self._args(rss_urls=["https://a.example/rss", "https://b.example/rss"])
        urls = collect_feed_urls(args)
        self.assertEqual(len(urls), 2)

    def test_preloaded_file_urls(self) -> None:
        args = self._args()
        urls = collect_feed_urls(
            args,
            preloaded_file_urls=["https://file.example/rss"],
        )
        self.assertIn("https://file.example/rss", urls)

    def test_empty_args_returns_empty(self) -> None:
        args = self._args()
        urls = collect_feed_urls(args)
        self.assertEqual(urls, [])


class TestValidateDownloadResilienceCli(unittest.TestCase):
    def _args(self, **kwargs) -> argparse.Namespace:
        return argparse.Namespace(**kwargs)

    def test_valid_values_no_errors(self) -> None:
        args = self._args(
            http_retry_total=5,
            http_backoff_factor=1.0,
            rss_retry_total=3,
            rss_backoff_factor=0.5,
            episode_retry_max=3,
            episode_retry_delay_sec=5.0,
            host_request_interval_ms=1000,
            host_max_concurrent=4,
            circuit_breaker_failure_threshold=5,
            circuit_breaker_window_seconds=60,
            circuit_breaker_cooldown_seconds=120,
        )
        errors: list[str] = []
        _validate_download_resilience_cli(args, errors)
        self.assertEqual(errors, [])

    def test_http_retry_total_out_of_range(self) -> None:
        args = self._args(
            http_retry_total=25,
            http_backoff_factor=None,
            rss_retry_total=None,
            rss_backoff_factor=None,
            episode_retry_max=None,
            episode_retry_delay_sec=None,
            host_request_interval_ms=None,
            host_max_concurrent=None,
            circuit_breaker_failure_threshold=None,
            circuit_breaker_window_seconds=None,
            circuit_breaker_cooldown_seconds=None,
        )
        errors: list[str] = []
        _validate_download_resilience_cli(args, errors)
        self.assertTrue(any("http-retry-total" in e for e in errors))

    def test_http_backoff_factor_out_of_range(self) -> None:
        args = self._args(
            http_retry_total=None,
            http_backoff_factor=15.0,
            rss_retry_total=None,
            rss_backoff_factor=None,
            episode_retry_max=None,
            episode_retry_delay_sec=None,
            host_request_interval_ms=None,
            host_max_concurrent=None,
            circuit_breaker_failure_threshold=None,
            circuit_breaker_window_seconds=None,
            circuit_breaker_cooldown_seconds=None,
        )
        errors: list[str] = []
        _validate_download_resilience_cli(args, errors)
        self.assertTrue(any("http-backoff-factor" in e for e in errors))

    def test_episode_retry_max_out_of_range(self) -> None:
        args = self._args(
            http_retry_total=None,
            http_backoff_factor=None,
            rss_retry_total=None,
            rss_backoff_factor=None,
            episode_retry_max=15,
            episode_retry_delay_sec=None,
            host_request_interval_ms=None,
            host_max_concurrent=None,
            circuit_breaker_failure_threshold=None,
            circuit_breaker_window_seconds=None,
            circuit_breaker_cooldown_seconds=None,
        )
        errors: list[str] = []
        _validate_download_resilience_cli(args, errors)
        self.assertTrue(any("episode-retry-max" in e for e in errors))

    def test_host_request_interval_out_of_range(self) -> None:
        args = self._args(
            http_retry_total=None,
            http_backoff_factor=None,
            rss_retry_total=None,
            rss_backoff_factor=None,
            episode_retry_max=None,
            episode_retry_delay_sec=None,
            host_request_interval_ms=700_000,
            host_max_concurrent=None,
            circuit_breaker_failure_threshold=None,
            circuit_breaker_window_seconds=None,
            circuit_breaker_cooldown_seconds=None,
        )
        errors: list[str] = []
        _validate_download_resilience_cli(args, errors)
        self.assertTrue(any("host-request-interval-ms" in e for e in errors))

    def test_none_values_no_errors(self) -> None:
        args = self._args(
            http_retry_total=None,
            http_backoff_factor=None,
            rss_retry_total=None,
            rss_backoff_factor=None,
            episode_retry_max=None,
            episode_retry_delay_sec=None,
            host_request_interval_ms=None,
            host_max_concurrent=None,
            circuit_breaker_failure_threshold=None,
            circuit_breaker_window_seconds=None,
            circuit_breaker_cooldown_seconds=None,
        )
        errors: list[str] = []
        _validate_download_resilience_cli(args, errors)
        self.assertEqual(errors, [])
