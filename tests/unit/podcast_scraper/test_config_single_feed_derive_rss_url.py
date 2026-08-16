"""Unit tests for the _derive_rss_url_from_single_feed model validator (#1542).

Covers the belt-and-suspenders that promotes rss_urls[0].url → rss_url when
exactly one feed is supplied via rss_urls and rss_url is not already set.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config import Config

pytestmark = [pytest.mark.unit]

_FEED_A = "https://a.example/feed.xml"
_FEED_B = "https://b.example/feed.xml"

# Config has extra="forbid" + populate_by_name=True.  Keyword args bypass the
# populate_by_name path on some pydantic versions; use model_validate (dict) to
# avoid [extra_forbidden] errors when passing field names alongside aliases.


def test_single_rss_urls_derives_rss_url() -> None:
    """One entry in rss_urls, no rss_url set → rss_url promoted from rss_urls[0]."""
    cfg = Config.model_validate({"rss_urls": [_FEED_A], "output_dir": "/tmp/out"})
    assert cfg.rss_url == _FEED_A


def test_explicit_rss_url_is_not_overridden() -> None:
    """When rss_url is already set, the validator must not touch it."""
    cfg = Config.model_validate({"rss": _FEED_A, "output_dir": "/tmp/out"})
    assert cfg.rss_url == _FEED_A


def test_rss_and_rss_urls_single_different_entry_becomes_batch() -> None:
    """When rss + rss_urls[0] are different URLs the pre-validator (_normalize_multi_rss_input)
    merges them into a 2-entry batch before _derive_rss_url_from_single_feed runs.
    Consequence: rss_url stays None and both appear in rss_urls.  This ensures
    _derive_rss_url_from_single_feed is not reachable when an explicit rss_url is
    present alongside a conflicting single-entry rss_urls."""
    cfg = Config.model_validate({"rss": _FEED_A, "rss_urls": [_FEED_B], "output_dir": "/tmp/out"})
    # Pre-validator merged the two URLs into a 2-entry batch.
    assert cfg.rss_url is None
    assert cfg.rss_urls is not None
    assert len(cfg.rss_urls) == 2


def test_two_feeds_rss_url_stays_none() -> None:
    """Two entries in rss_urls (genuine batch) → rss_url remains None."""
    cfg = Config.model_validate({"rss_urls": [_FEED_A, _FEED_B], "output_dir": "/tmp/out"})
    assert cfg.rss_url is None


def test_zero_feeds_rss_url_stays_none() -> None:
    """No rss_url and no rss_urls → rss_url stays None (no crash)."""
    cfg = Config.model_validate({"output_dir": "/tmp/out"})
    assert cfg.rss_url is None
