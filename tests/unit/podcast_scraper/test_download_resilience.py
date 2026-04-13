"""Unit tests for download resilience features.

Tests configurable HTTP retry policy, application-level episode retry,
and failure summary aggregation.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from podcast_scraper.config import Config
from podcast_scraper.rss.downloader import (
    _effective_http_backoff_factor,
    _effective_http_retry_total,
    _effective_rss_backoff_factor,
    _effective_rss_retry_total,
    configure_downloader,
    DEFAULT_HTTP_BACKOFF_FACTOR,
    DEFAULT_HTTP_RETRY_TOTAL,
    RSS_FEED_HTTP_BACKOFF_FACTOR,
    RSS_FEED_HTTP_RETRY_TOTAL,
)
from podcast_scraper.rss.http_policy import configure_http_policy
from podcast_scraper.workflow.run_index import (
    build_failure_summary,
    EpisodeIndexEntry,
    RunIndex,
)
from podcast_scraper.workflow.stages.processing import (
    _is_episode_retryable,
    _process_episode_with_retry,
)

pytestmark = [pytest.mark.unit]


# ── Layer 1: Configurable HTTP Retry Policy ──────────────────────


class TestConfigDefaultsMatchDownloaderConstants(unittest.TestCase):
    """Regression: Config defaults stay aligned with rss.downloader fallbacks."""

    def test_http_and_rss_defaults_match_module_constants(self):
        cfg = Config(rss_url="http://x.com/feed.xml")
        self.assertEqual(cfg.http_retry_total, DEFAULT_HTTP_RETRY_TOTAL)
        self.assertAlmostEqual(cfg.http_backoff_factor, DEFAULT_HTTP_BACKOFF_FACTOR)
        self.assertEqual(cfg.rss_retry_total, RSS_FEED_HTTP_RETRY_TOTAL)
        self.assertAlmostEqual(cfg.rss_backoff_factor, RSS_FEED_HTTP_BACKOFF_FACTOR)


class TestConfigRetryFields(unittest.TestCase):
    """Config fields for HTTP retry policy."""

    def test_defaults(self):
        cfg = Config(rss_url="http://x.com/feed.xml")
        self.assertEqual(cfg.http_retry_total, 8)
        self.assertEqual(cfg.http_backoff_factor, 1.0)
        self.assertEqual(cfg.rss_retry_total, 5)
        self.assertEqual(cfg.rss_backoff_factor, 1.0)

    def test_custom_values(self):
        cfg = Config(
            rss_url="http://x.com/feed.xml",
            http_retry_total=10,
            http_backoff_factor=1.5,
            rss_retry_total=3,
            rss_backoff_factor=0.25,
        )
        self.assertEqual(cfg.http_retry_total, 10)
        self.assertAlmostEqual(cfg.http_backoff_factor, 1.5)
        self.assertEqual(cfg.rss_retry_total, 3)
        self.assertAlmostEqual(cfg.rss_backoff_factor, 0.25)

    def test_http_retry_total_bounds(self):
        with self.assertRaises(ValidationError):
            Config(rss_url="http://x.com/f.xml", http_retry_total=-1)
        with self.assertRaises(ValidationError):
            Config(rss_url="http://x.com/f.xml", http_retry_total=21)

    def test_backoff_factor_bounds(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="http://x.com/f.xml",
                http_backoff_factor=-0.1,
            )
        with self.assertRaises(ValidationError):
            Config(
                rss_url="http://x.com/f.xml",
                http_backoff_factor=11.0,
            )

    def test_zero_retries_allowed(self):
        cfg = Config(
            rss_url="http://x.com/feed.xml",
            http_retry_total=0,
            rss_retry_total=0,
        )
        self.assertEqual(cfg.http_retry_total, 0)
        self.assertEqual(cfg.rss_retry_total, 0)


class TestConfigIssue522Fields(unittest.TestCase):
    """Config validation for fair HTTP (Issue #522)."""

    def test_host_interval_and_concurrency_bounds(self):
        with self.assertRaises(ValidationError):
            Config(rss_url="http://x.com/f.xml", host_request_interval_ms=-1)
        with self.assertRaises(ValidationError):
            Config(rss_url="http://x.com/f.xml", host_request_interval_ms=600_001)
        with self.assertRaises(ValidationError):
            Config(rss_url="http://x.com/f.xml", host_max_concurrent=65)

    def test_circuit_breaker_scope_literal(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="http://x.com/f.xml",
                circuit_breaker_scope="invalid",  # type: ignore[arg-type]
            )

    def test_defaults_off(self):
        cfg = Config(rss_url="http://x.com/f.xml")
        self.assertEqual(cfg.host_request_interval_ms, 0)
        self.assertFalse(cfg.circuit_breaker_enabled)
        self.assertFalse(cfg.rss_conditional_get)


class TestConfigureDownloader(unittest.TestCase):
    """configure_downloader sets module-level effective values."""

    def tearDown(self):
        configure_http_policy()
        configure_downloader(
            http_retry_total=None,
            http_backoff_factor=None,
            rss_retry_total=None,
            rss_backoff_factor=None,
        )
        import podcast_scraper.rss.downloader as dl

        dl._configured_http_retry_total = None
        dl._configured_http_backoff_factor = None
        dl._configured_rss_retry_total = None
        dl._configured_rss_backoff_factor = None

    def test_effective_defaults_without_configure(self):
        import podcast_scraper.rss.downloader as dl

        dl._configured_http_retry_total = None
        dl._configured_http_backoff_factor = None
        self.assertEqual(_effective_http_retry_total(), 8)
        self.assertAlmostEqual(_effective_http_backoff_factor(), 1.0)
        self.assertEqual(_effective_rss_retry_total(), 5)
        self.assertAlmostEqual(_effective_rss_backoff_factor(), 1.0)

    def test_configure_overrides_effective(self):
        configure_downloader(
            http_retry_total=2,
            http_backoff_factor=0.1,
            rss_retry_total=4,
            rss_backoff_factor=2.0,
        )
        self.assertEqual(_effective_http_retry_total(), 2)
        self.assertAlmostEqual(_effective_http_backoff_factor(), 0.1)
        self.assertEqual(_effective_rss_retry_total(), 4)
        self.assertAlmostEqual(_effective_rss_backoff_factor(), 2.0)

    def test_partial_configure(self):
        configure_downloader(http_retry_total=3)
        self.assertEqual(_effective_http_retry_total(), 3)
        self.assertAlmostEqual(_effective_http_backoff_factor(), 1.0)


# ── Layer 2: Application-Level Episode Retry ─────────────────────


class TestConfigEpisodeRetryFields(unittest.TestCase):
    """Config fields for episode-level retry."""

    def test_defaults(self):
        cfg = Config(rss_url="http://x.com/feed.xml")
        self.assertEqual(cfg.episode_retry_max, 1)
        self.assertAlmostEqual(cfg.episode_retry_delay_sec, 10.0)

    def test_custom_values(self):
        cfg = Config(
            rss_url="http://x.com/feed.xml",
            episode_retry_max=3,
            episode_retry_delay_sec=10.0,
        )
        self.assertEqual(cfg.episode_retry_max, 3)
        self.assertAlmostEqual(cfg.episode_retry_delay_sec, 10.0)

    def test_bounds(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="http://x.com/f.xml",
                episode_retry_max=-1,
            )
        with self.assertRaises(ValidationError):
            Config(
                rss_url="http://x.com/f.xml",
                episode_retry_max=11,
            )
        with self.assertRaises(ValidationError):
            Config(
                rss_url="http://x.com/f.xml",
                episode_retry_delay_sec=121.0,
            )


class TestIsEpisodeRetryable(unittest.TestCase):
    """_is_episode_retryable classifies exceptions."""

    def test_connection_error(self):
        self.assertTrue(_is_episode_retryable(ConnectionError("reset")))

    def test_timeout_error(self):
        self.assertTrue(_is_episode_retryable(TimeoutError("timed out")))

    def test_os_error(self):
        self.assertTrue(_is_episode_retryable(OSError("broken pipe")))

    def test_value_error_not_retryable(self):
        self.assertFalse(_is_episode_retryable(ValueError("bad config")))

    def test_message_based_detection(self):
        self.assertTrue(_is_episode_retryable(Exception("HTTP 429 rate limited")))
        self.assertTrue(_is_episode_retryable(Exception("503 service unavailable")))

    def test_generic_exception_not_retryable(self):
        self.assertFalse(_is_episode_retryable(Exception("something else")))


class TestProcessEpisodeWithRetry(unittest.TestCase):
    """_process_episode_with_retry wraps episode downloads."""

    def _make_cfg(self, retry_max=0, delay=0.01):
        return SimpleNamespace(
            episode_retry_max=retry_max,
            episode_retry_delay_sec=delay,
        )

    def _make_args(self, idx=1):
        episode = SimpleNamespace(idx=idx)
        return (episode, None, None, None, None, None, None, None)

    def test_no_retry_passthrough(self):
        """With retry_max=0, calls function once."""
        fn = Mock(return_value=(True, "/path", "direct_download", 100))
        cfg = self._make_cfg(retry_max=0)
        metrics = Mock()
        result = _process_episode_with_retry(fn, self._make_args(), cfg, metrics)
        self.assertEqual(result, (True, "/path", "direct_download", 100))
        fn.assert_called_once()

    @patch("podcast_scraper.workflow.stages.processing.time.sleep")
    def test_retry_on_transient_error(self, mock_sleep):
        """Retries on ConnectionError, succeeds on second attempt."""
        fn = Mock(
            side_effect=[
                ConnectionError("reset"),
                (True, "/path", "direct_download", 200),
            ]
        )
        cfg = self._make_cfg(retry_max=1, delay=0.01)
        metrics = Mock(spec=["record_episode_download_retry"])
        result = _process_episode_with_retry(fn, self._make_args(), cfg, metrics)
        self.assertEqual(result[0], True)
        self.assertEqual(fn.call_count, 2)
        mock_sleep.assert_called_once()
        metrics.record_episode_download_retry.assert_called_once()

    def test_non_retryable_error_raises_immediately(self):
        """ValueError is not retryable, raises on first attempt."""
        fn = Mock(side_effect=ValueError("bad config"))
        cfg = self._make_cfg(retry_max=3)
        metrics = Mock()
        with self.assertRaises(ValueError):
            _process_episode_with_retry(fn, self._make_args(), cfg, metrics)
        fn.assert_called_once()

    @patch("podcast_scraper.workflow.stages.processing.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep):
        """All retries fail, raises the last exception."""
        fn = Mock(side_effect=ConnectionError("down"))
        cfg = self._make_cfg(retry_max=2, delay=0.01)
        metrics = Mock(spec=["record_episode_download_retry"])
        with self.assertRaises(ConnectionError):
            _process_episode_with_retry(fn, self._make_args(), cfg, metrics)
        self.assertEqual(fn.call_count, 3)
        self.assertEqual(metrics.record_episode_download_retry.call_count, 2)


# ── Layer 3: Failure Summary ─────────────────────────────────────


class TestBuildFailureSummary(unittest.TestCase):
    """build_failure_summary aggregates failures by error type."""

    def test_no_failures(self):
        index = RunIndex(
            episodes=[
                EpisodeIndexEntry(episode_id="ep1", status="ok"),
                EpisodeIndexEntry(episode_id="ep2", status="skipped"),
            ]
        )
        result = build_failure_summary(index)
        self.assertEqual(result["total_failed"], 0)
        self.assertEqual(result["by_error_type"], {})
        self.assertEqual(result["failed_episode_ids"], [])

    def test_failures_grouped_by_type(self):
        index = RunIndex(
            episodes=[
                EpisodeIndexEntry(
                    episode_id="ep1",
                    status="failed",
                    error_type="HTTPError",
                ),
                EpisodeIndexEntry(
                    episode_id="ep2",
                    status="ok",
                ),
                EpisodeIndexEntry(
                    episode_id="ep3",
                    status="failed",
                    error_type="HTTPError",
                ),
                EpisodeIndexEntry(
                    episode_id="ep4",
                    status="failed",
                    error_type="TimeoutError",
                ),
            ]
        )
        result = build_failure_summary(index)
        self.assertEqual(result["total_failed"], 3)
        self.assertEqual(result["by_error_type"]["HTTPError"], 2)
        self.assertEqual(result["by_error_type"]["TimeoutError"], 1)
        self.assertIn("ep1", result["failed_episode_ids"])
        self.assertIn("ep3", result["failed_episode_ids"])
        self.assertIn("ep4", result["failed_episode_ids"])

    def test_unknown_error_type(self):
        index = RunIndex(
            episodes=[
                EpisodeIndexEntry(
                    episode_id="ep1",
                    status="failed",
                    error_type=None,
                ),
            ]
        )
        result = build_failure_summary(index)
        self.assertEqual(result["by_error_type"]["unknown"], 1)

    def test_sorted_by_count_descending(self):
        index = RunIndex(
            episodes=[
                EpisodeIndexEntry(
                    episode_id=f"ep{i}",
                    status="failed",
                    error_type="Timeout",
                )
                for i in range(5)
            ]
            + [
                EpisodeIndexEntry(
                    episode_id="epX",
                    status="failed",
                    error_type="HTTP404",
                ),
            ]
        )
        result = build_failure_summary(index)
        keys = list(result["by_error_type"].keys())
        self.assertEqual(keys[0], "Timeout")
        self.assertEqual(keys[1], "HTTP404")


class TestRunSummaryIncludesFailures(unittest.TestCase):
    """create_run_summary includes failure_summary when present."""

    def test_failure_summary_included(self):
        from podcast_scraper.workflow.run_summary import (
            create_run_summary,
        )

        fs = {
            "total_failed": 2,
            "by_error_type": {"HTTPError": 2},
            "failed_episode_ids": ["ep1", "ep2"],
        }
        summary = create_run_summary(
            run_manifest=None,
            pipeline_metrics=None,
            output_dir="/tmp/test",
            failure_summary=fs,
        )
        self.assertIn("failure_summary", summary)
        self.assertEqual(summary["failure_summary"]["total_failed"], 2)

    def test_no_failure_summary_when_zero(self):
        from podcast_scraper.workflow.run_summary import (
            create_run_summary,
        )

        fs = {
            "total_failed": 0,
            "by_error_type": {},
            "failed_episode_ids": [],
        }
        summary = create_run_summary(
            run_manifest=None,
            pipeline_metrics=None,
            output_dir="/tmp/test",
            failure_summary=fs,
        )
        self.assertNotIn("failure_summary", summary)

    def test_no_failure_summary_when_none(self):
        from podcast_scraper.workflow.run_summary import (
            create_run_summary,
        )

        summary = create_run_summary(
            run_manifest=None,
            pipeline_metrics=None,
            output_dir="/tmp/test",
            failure_summary=None,
        )
        self.assertNotIn("failure_summary", summary)


if __name__ == "__main__":
    unittest.main()
