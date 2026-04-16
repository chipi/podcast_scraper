"""Unit tests for multi-feed failure classification (GitHub #559)."""

import pytest

from podcast_scraper.exceptions import ProviderRuntimeError
from podcast_scraper.workflow.corpus_operations import classify_multi_feed_feed_exception

pytestmark = [pytest.mark.unit, pytest.mark.module_workflow]


def test_classify_rss_fetch_valueerror_soft() -> None:
    assert classify_multi_feed_feed_exception(ValueError("Failed to fetch RSS feed.")) == "soft"


def test_classify_rss_parse_valueerror_soft() -> None:
    assert classify_multi_feed_feed_exception(ValueError("Failed to parse RSS XML: boom")) == "soft"


def test_classify_other_valueerror_hard() -> None:
    assert classify_multi_feed_feed_exception(ValueError("feed b failed")) == "hard"


def test_classify_unicode_decode_error_soft() -> None:
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "reason")
    assert classify_multi_feed_feed_exception(exc) == "soft"


def test_classify_provider_runtime_413_soft() -> None:
    exc = ProviderRuntimeError(
        message="OpenAI transcription failed: Error code: 413 - Maximum content size limit",
        provider="OpenAIProvider/Transcription",
    )
    assert classify_multi_feed_feed_exception(exc) == "soft"


def test_classify_provider_runtime_other_hard() -> None:
    exc = ProviderRuntimeError(message="rate limit", provider="OpenAIProvider/Transcription")
    assert classify_multi_feed_feed_exception(exc) == "hard"


def test_classify_runtime_error_hard() -> None:
    assert classify_multi_feed_feed_exception(RuntimeError("boom")) == "hard"
