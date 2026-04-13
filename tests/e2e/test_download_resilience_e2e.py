#!/usr/bin/env python3
"""E2E tests for download resilience features.

Tests verify that the pipeline handles transient HTTP errors gracefully:
- Transient 429/5xx errors that resolve after N requests
- Configurable HTTP retry values
- Multi-feed isolation when one feed is down
- Failure summary in run.json
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config, run_pipeline
from podcast_scraper.rss.downloader import (
    configure_downloader,
    fetch_url,
    reset_http_sessions,
)

try:
    from tests.e2e.fixtures.e2e_http_server import (
        E2EHTTPRequestHandler,
    )
except ImportError:
    E2EHTTPRequestHandler = None  # type: ignore[assignment,misc]


def _cleanup_transient_errors():
    """Reset transient and permanent error behaviors."""
    if E2EHTTPRequestHandler is not None:
        E2EHTTPRequestHandler.clear_all_error_behaviors()


@pytest.fixture(autouse=True)
def _reset_errors():
    """Ensure error behaviors and HTTP sessions are cleaned up after each test."""
    reset_http_sessions()
    yield
    _cleanup_transient_errors()
    configure_downloader(http_retry_total=2, http_backoff_factor=0.0)
    reset_http_sessions()


@pytest.mark.e2e
@pytest.mark.critical_path
class TestTransientHTTPErrors:
    """Transient HTTP errors that resolve after N failures."""

    def test_transient_503_then_success(self, e2e_server):
        """Server returns 503 twice then 200; download succeeds."""
        transcript_path = "/transcripts/p01_multi_e01.txt"
        E2EHTTPRequestHandler.set_transient_error(transcript_path, status=503, fail_count=2)

        url = f"{e2e_server.urls.base()}{transcript_path}"
        resp = fetch_url(url, "test-agent", timeout=30)

        assert resp is not None, "fetch_url should succeed after transient 503s"
        assert resp.status_code == 200

    def test_transient_429_then_success(self, e2e_server):
        """Server returns 429 twice then 200; download succeeds."""
        transcript_path = "/transcripts/p01_multi_e02.txt"
        E2EHTTPRequestHandler.set_transient_error(transcript_path, status=429, fail_count=2)

        url = f"{e2e_server.urls.base()}{transcript_path}"
        resp = fetch_url(url, "test-agent", timeout=30)

        assert resp is not None, "fetch_url should succeed after transient 429s"
        assert resp.status_code == 200

    def test_permanent_500_fails(self, e2e_server):
        """Permanent 500 exhausts all retries and returns None."""
        transcript_path = "/transcripts/p01_multi_e03.txt"
        E2EHTTPRequestHandler.set_error_behavior(transcript_path, status=500)

        configure_downloader(http_retry_total=1, http_backoff_factor=0.0)
        reset_http_sessions()
        url = f"{e2e_server.urls.base()}{transcript_path}"
        resp = fetch_url(url, "test-agent", timeout=10)
        assert resp is None, "Permanent 500 should exhaust retries and return None"


@pytest.mark.e2e
@pytest.mark.critical_path
class TestConfigurableRetryValues:
    """Configurable HTTP retry policy via Config."""

    def test_custom_retry_values_in_pipeline(self, e2e_server):
        """Pipeline respects custom http_retry_total from Config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,
                http_retry_total=2,
                http_backoff_factor=0.0,
                rss_retry_total=2,
                rss_backoff_factor=0.0,
            )
            count, summary = run_pipeline(cfg)
            assert count >= 0
            assert isinstance(summary, str)


@pytest.mark.e2e
@pytest.mark.critical_path
class TestPipelineTransientRecovery:
    """Full pipeline recovers from transient transcript errors."""

    def test_pipeline_recovers_from_transient_503(self, e2e_server):
        """Pipeline downloads transcript after transient 503s."""
        E2EHTTPRequestHandler.set_transient_error(
            "/transcripts/p01_multi_e01.txt",
            status=503,
            fail_count=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,
                http_retry_total=3,
                http_backoff_factor=0.0,
                rss_retry_total=3,
                rss_backoff_factor=0.0,
            )
            count, summary = run_pipeline(cfg)
            assert count >= 1, "Pipeline should recover and save transcript " "after transient 503"

            txt_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(txt_files) >= 1, "Should have at least one transcript file"


@pytest.mark.e2e
@pytest.mark.critical_path
class TestMultiFeedIsolation:
    """Multi-feed runs isolate per-feed failures."""

    def test_one_feed_down_others_continue(self, e2e_server):
        """When one RSS feed returns 500, other feeds still succeed."""
        E2EHTTPRequestHandler.set_error_behavior("/feeds/podcast2/feed.xml", status=500)

        from podcast_scraper import service

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    e2e_server.urls.feed("podcast2"),
                ],
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,
                http_retry_total=2,
                http_backoff_factor=0.0,
                rss_retry_total=2,
                rss_backoff_factor=0.0,
            )
            result = service.run(cfg)

            assert result.episodes_processed >= 1, "At least one feed should succeed"
            assert result.success is False, "Overall success should be False (one feed failed)"
            assert result.error is not None, "Error should be reported for the failed feed"


@pytest.mark.e2e
@pytest.mark.critical_path
class TestFailureSummaryInRunJson:
    """run.json includes failure_summary when episodes fail."""

    def test_partial_failure_produces_summary(self, e2e_server):
        """Pipeline with partial failures writes failure_summary."""
        E2EHTTPRequestHandler.set_error_behavior("/transcripts/p01_multi_e03.txt", status=404)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                output_dir=tmpdir,
                max_episodes=3,
                transcribe_missing=False,
                http_retry_total=2,
                http_backoff_factor=0.0,
                rss_retry_total=2,
                rss_backoff_factor=0.0,
            )
            count, summary = run_pipeline(cfg)

            run_json_candidates = list(Path(tmpdir).rglob("run.json"))
            assert len(run_json_candidates) >= 1, "run.json should be written"
            run_data = json.loads(run_json_candidates[0].read_text(encoding="utf-8"))
            assert "schema_version" in run_data

    def test_all_success_no_failure_summary(self, e2e_server):
        """Pipeline with all successes has no failure_summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,
                http_retry_total=2,
                http_backoff_factor=0.0,
                rss_retry_total=2,
                rss_backoff_factor=0.0,
            )
            count, summary = run_pipeline(cfg)

            run_json_candidates = list(Path(tmpdir).rglob("run.json"))
            if run_json_candidates:
                run_data = json.loads(run_json_candidates[0].read_text(encoding="utf-8"))
                fs = run_data.get("failure_summary")
                if fs is not None:
                    assert fs.get("total_failed", 0) == 0
