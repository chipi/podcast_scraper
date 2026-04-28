#!/usr/bin/env python3
"""Tests for ``emit_job_state_change`` job-webhook fire-and-forget (#682).

The webhook emitter wraps an ``httpx.AsyncClient.post`` in a timeout +
broad-except so a misbehaving sink (slow, 5xx, DNS-failure) cannot break
the api's job-state finalisation. Tests cover:

* No-op path: env var unset -> no HTTP call, no error.
* Happy path: env var set + 2xx response -> single POST with the
  expected payload shape (``{"event": "job_state_changed", "job": ...}``).
* Bad-status path: 4xx/5xx -> warning logged, no exception bubbled.
* Network failure path: ``httpx`` raises -> warning logged, no exception.
* Timeout config: ``PODCAST_JOB_WEBHOOK_TIMEOUT_SEC`` parsed; bad
  values fall back to the default; floor at 0.5s.
* CancelledError propagates (so the api's task cancellation works).
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from podcast_scraper.server.job_webhook import (
    _DEFAULT_TIMEOUT_SEC,
    _timeout_sec,
    _webhook_url,
    emit_job_state_change,
)


@pytest.mark.unit
class TestWebhookHelpers(unittest.TestCase):
    """``_webhook_url`` and ``_timeout_sec`` env parsing."""

    def setUp(self) -> None:
        self._prior = {
            k: os.environ.get(k)
            for k in ("PODCAST_JOB_WEBHOOK_URL", "PODCAST_JOB_WEBHOOK_TIMEOUT_SEC")
        }
        for k in self._prior:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_webhook_url_unset_returns_none(self) -> None:
        self.assertIsNone(_webhook_url())

    def test_webhook_url_blank_returns_none(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_URL"] = "   "
        self.assertIsNone(_webhook_url())

    def test_webhook_url_set_returns_stripped(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_URL"] = "  https://example.com/hook  "
        self.assertEqual(_webhook_url(), "https://example.com/hook")

    def test_timeout_unset_returns_default(self) -> None:
        self.assertEqual(_timeout_sec(), _DEFAULT_TIMEOUT_SEC)

    def test_timeout_set_returns_parsed(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_TIMEOUT_SEC"] = "12.5"
        self.assertEqual(_timeout_sec(), 12.5)

    def test_timeout_floor_is_half_second(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_TIMEOUT_SEC"] = "0.1"
        self.assertEqual(_timeout_sec(), 0.5)

    def test_timeout_invalid_falls_back_to_default(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_TIMEOUT_SEC"] = "not-a-number"
        self.assertEqual(_timeout_sec(), _DEFAULT_TIMEOUT_SEC)


def _run(coro):
    """Drive an async coroutine to completion in a unittest.TestCase."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.unit
class TestEmitJobStateChange(unittest.TestCase):
    """High-level emit_job_state_change behaviour."""

    def setUp(self) -> None:
        self._prior_url = os.environ.get("PODCAST_JOB_WEBHOOK_URL")
        os.environ.pop("PODCAST_JOB_WEBHOOK_URL", None)

    def tearDown(self) -> None:
        if self._prior_url is None:
            os.environ.pop("PODCAST_JOB_WEBHOOK_URL", None)
        else:
            os.environ["PODCAST_JOB_WEBHOOK_URL"] = self._prior_url

    def test_unset_url_is_noop(self) -> None:
        # Patch httpx so we can assert it was never imported / used.
        mock_httpx = MagicMock()
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            _run(emit_job_state_change({"job_id": "noop", "state": "succeeded"}))
        mock_httpx.AsyncClient.assert_not_called()

    def test_happy_path_posts_expected_payload(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_URL"] = "https://example.com/hook"
        record = {"job_id": "abc-123", "state": "succeeded", "exit_code": 0}

        mock_resp = MagicMock(status_code=200)
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            _run(emit_job_state_change(record))

        mock_client.post.assert_awaited_once()
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://example.com/hook")
        self.assertEqual(
            kwargs["json"],
            {"event": "job_state_changed", "job": record},
        )

    def test_bad_status_logs_warning_no_raise(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_URL"] = "https://example.com/hook"

        mock_resp = MagicMock(status_code=503)
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with self.assertLogs("podcast_scraper.server.job_webhook", level="WARNING") as logs:
                _run(emit_job_state_change({"job_id": "abc", "state": "failed"}))
        self.assertTrue(any("503" in line for line in logs.output))

    def test_httpx_unimportable_logs_warning_no_raise(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_URL"] = "https://example.com/hook"
        # Force ``import httpx`` inside emit_job_state_change to raise
        # ImportError without uninstalling the real package.
        with patch.dict("sys.modules", {"httpx": None}):
            with self.assertLogs("podcast_scraper.server.job_webhook", level="WARNING") as logs:
                _run(emit_job_state_change({"job_id": "abc", "state": "succeeded"}))
        self.assertTrue(any("httpx is not installed" in line for line in logs.output))

    def test_cancelled_error_propagates(self) -> None:
        os.environ["PODCAST_JOB_WEBHOOK_URL"] = "https://example.com/hook"

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=asyncio.CancelledError())

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with self.assertRaises(asyncio.CancelledError):
                _run(emit_job_state_change({"job_id": "abc"}))


if __name__ == "__main__":
    unittest.main()
